from crypto_data_engine.services.tick_data_scraper.extractor.convert import extract_archive, convert_dir_to_parquet
from crypto_data_engine.services.tick_data_scraper.tick_worker import run_download
from crypto_data_engine.db.repository.download import DownloadTaskRepository
from crypto_data_engine.db.models.download import TaskStatus
import logging

logger = logging.getLogger(__name__)


def register_tasks(celery_app):
    """register your celery tasks here"""

    @celery_app.task(name="tick.download", bind=True)
    def dispatch_tick_download(self, cfg: dict):
        """
        分布式tick数据下载任务
        """
        task_id = cfg.get('task_id')
        try:
            logger.info(f"开始执行下载任务: {cfg}")
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
            logger.info(f"download tasks finished: {exchange_name}")
            return {
                'status': 'SUCCESS',
                'exchange': exchange_name,
                'symbols_count': len(cfg.get('symbols')),
                'task_id': task_id,
                'result': result
            }

        except Exception as e:
            logger.error(f"下载任务失败: {str(e)}")
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

    @celery_app.task(name="tick.extract_task",bind=True)
    def extract_task(self, directory: str, file_name: str):
        result = extract_archive(directory, file_name)
        out_dir = result["out_dir"]
        parquet_files = convert_dir_to_parquet(out_dir, pattern="*.csv")
        return {
            "archive": result["archive"],
            "out_dir": out_dir,
            "files": result["files"],
            "parquet_files": parquet_files,
        }


    @celery_app.task(name="bars.aggregate")
    def aggregate_bars(cfg: dict):
        """Bar aggregation task dispatched from API gateway.
        cfg: {
            "exchange": str,
            "symbols": list[str] | None,
            "bar_type": str,
            "threshold": int | None,
        }
        Missing params are filled from AggregationConfig defaults.
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
        return result

    @celery_app.task(name="tick.health_check")
    def health_check():
        """健康检查任务"""
        import time
        import socket

        worker_info = {
            'hostname': socket.gethostname(),
            'timestamp': time.time(),
            'status': 'healthy'
        }

        logger.info(f"健康检查: {worker_info}")
        return worker_info


    return {
        'dispatch_tick_download': dispatch_tick_download,
        'extract': extract_task,
        'health_check': health_check
    }
