from crypto_data_engine.services.tick_data_scraper.extractor.convert import extract_archive, convert_dir_to_parquet
from crypto_data_engine.services.tick_data_scraper.tick_worker import run_download
from crypto_data_engine.db.repository.download import DownloadTaskRepository
from crypto_data_engine.db.constants import TaskStatus
import logging

logger = logging.getLogger(__name__)


def register_tasks(celery_app):
    """register your celery tasks here"""

    @celery_app.task(name="tick.download", bind=True)
    def dispatch_tick_download(self, cfg: dict):
        """Distributed tick data download task."""
        task_id = cfg.get('task_id')
        try:
            logger.info(f"Starting download task: {cfg}")
            if task_id:
                DownloadTaskRepository.update_status(task_id, TaskStatus.DOWNLOADING)
            exchange_name = cfg.get('exchange_name')
            self.update_state(
                state='PROGRESS',
                meta={'status': 'start downloading', 'exchange': exchange_name}
            )
            result = run_download(
                exchange_name=exchange_name,
                symbols=cfg.get('symbols'),
                start_date=cfg.get('start_date'),
                end_date=cfg.get('end_date'),
                max_threads=cfg.get('max_threads', 8)
            )
            logger.info(f"Download task finished: {exchange_name}")
            symbols = cfg.get('symbols')
            return {
                'status': 'SUCCESS',
                'exchange': exchange_name,
                'symbols_count': (len(symbols) if isinstance(symbols, (list, tuple)) else 0),
                'task_id': task_id,
                'result': result
            }

        except Exception as e:
            logger.error(f"Download task failed: {str(e)}")
            if task_id:
                DownloadTaskRepository.update(
                    task_id,
                    status=TaskStatus.FAILED,
                    error_message=str(e)
                )
            self.update_state(
                state='FAILURE',
                meta={'error': str(e), 'exchange': cfg.get('exchange_name', 'unknown')}
            )
            return {
                'status': 'FAILED',
                'error': str(e),
                'task_id': task_id
            }

    @celery_app.task(name="tick.extract_task", bind=True)
    def extract_task(self, directory: str, file_name: str):
        """Unpack an archive, convert CSVs to parquet, and update task status."""
        from datetime import datetime as _dt
        try:
            # Mark EXTRACTING if we can locate the task by (local_path, file_name)
            try:
                task = DownloadTaskRepository.get_by_kwargs(local_path=directory, file_name=file_name)
                if task:
                    DownloadTaskRepository.update(task.id, status=TaskStatus.EXTRACTING)
            except Exception as _:
                pass

            result = extract_archive(directory, file_name)
            out_dir = result["out_dir"]
            parquet_files = convert_dir_to_parquet(out_dir, pattern="*.csv")

            # Mark EXTRACTED on success
            try:
                task = DownloadTaskRepository.get_by_kwargs(local_path=directory, file_name=file_name)
                if task:
                    DownloadTaskRepository.update(
                        task.id,
                        status=TaskStatus.EXTRACTED,
                        task_end=_dt.now(),
                    )
            except Exception as _:
                pass

            response = {
                "archive": result["archive"],
                "out_dir": out_dir,
                "files": result["files"],
                "parquet_files": parquet_files,
            }

            # Auto-dispatch aggregation for the exchange inferred from directory path
            try:
                # Heuristic: directory like data/tick_data/<exchange>/...
                from pathlib import Path as _P
                parts = _P(directory).parts
                exchange_name = None
                for i, p in enumerate(parts):
                    if p.lower() in ("binance", "okx", "bybit", "huobi"):
                        exchange_name = p.lower()
                        break
                if exchange_name:
                    from task_manager.celery_app import celery_app as _app
                    _app.send_task(
                        "bar.aggregate",
                        kwargs={
                            "exchange": exchange_name,
                            "symbols": None,  # let aggregator derive from DB
                            "bar_type": "volume_bar",
                            "threshold": None,
                        },
                        queue="cpu",
                    )
            except Exception:
                # best-effort; do not fail extract task
                pass

            return response
        except Exception as e:
            # Mark FAILED if possible
            try:
                task = DownloadTaskRepository.get_by_kwargs(local_path=directory, file_name=file_name)
                if task:
                    DownloadTaskRepository.update(task.id, status=TaskStatus.FAILED)
            except Exception as _:
                pass
            raise


    @celery_app.task(name="bar.aggregate")
    def aggregate_bars(cfg: dict):
        """Bar aggregation task dispatched from API gateway.

        cfg: {
            "exchange": str,
            "symbols": list[str] | None,
            "bar_type": str,
            "threshold": int | None,
        }

        Missing parameters are filled from `AggregationConfig` defaults.
        """
        from crypto_data_engine.common.config.config_settings import settings
        from crypto_data_engine.services.bar_aggregator.bar_processor import (
            BarProcessorContext, BarProcessor,
        )

        exchange = cfg.get("exchange")
        symbols = cfg.get("symbols",None)
        bar_type = cfg.get("bar_type", "volume_bar")
        user_threshold = cfg.get("threshold")

        # Resolve defaults
        agg_cfg = settings.aggregator_cfg
        merged = agg_cfg.resolve_defaults(bar_type)
        threshold = user_threshold or merged.get("threshold")
        bar_type_norm = merged["bar_type"]

        # Resolve directories
        raw_exchange_cfg = settings.downloader_cfg.get_exchange_config(exchange)
        raw_data_dir = str(raw_exchange_cfg.data_dir)
        output_dir = str(agg_cfg.make_output_dir(exchange))

        # Build processor context
        context = BarProcessorContext(
            raw_data_dir=raw_data_dir,
            output_dir=output_dir,
            bar_type=bar_type_norm,
            threshold=threshold,
        )
        processor = BarProcessor(context)
        result = processor.run_bar_generation_pipeline({
            "exchange": exchange,
            "symbols": symbols,
        })
        # Persist aggregation artifacts into AggregateTask table to prevent reprocessing
        try:
            from datetime import date as _date, datetime as _dt
            from pathlib import Path
            from crypto_data_engine.db.repository.aggregate import AggregateTaskRepository as _AggRepo
            for item in (result or {}).get("results", []) or []:
                symbol = item.get("symbol")
                output_file = item.get("output_file")
                if not symbol or not output_file:
                    continue
                # create or update
                existing = _AggRepo.get_by_kwargs(
                    exchange=exchange, symbol=symbol, bar_type=bar_type_norm, part_date=_date.today()
                )
                if existing:
                    _AggRepo.update(
                        existing.id,
                        status=TaskStatus.COMPLETED,
                        file_name=str(Path(output_file).name),
                        file_path=str(output_file),
                        task_end=_dt.now(),
                    )
                else:
                    _AggRepo.create_task(
                        exchange=exchange,
                        symbol=symbol,
                        bar_type=bar_type_norm,
                        part_date=_date.today(),
                        status=TaskStatus.COMPLETED,
                        file_name=str(Path(output_file).name),
                        file_path=str(output_file),
                        task_end=_dt.now(),
                    )
        except Exception as _:
            # Best-effort; do not fail the whole task if DB write fails
            logger.warning("aggregate_bars: failed to persist aggregation records", exc_info=True)

        return result

    @celery_app.task(name="tick.health_check")
    def health_check():
        """Celery worker health check task."""
        import time
        import socket

        worker_info = {
            'hostname': socket.gethostname(),
            'timestamp': time.time(),
            'status': 'healthy'
        }

        logger.info(f"Health check: {worker_info}")
        return worker_info


    return {
        'dispatch_tick_download': dispatch_tick_download,
        'extract': extract_task,
        'aggregate_bars': aggregate_bars,
        'health_check': health_check
    }
