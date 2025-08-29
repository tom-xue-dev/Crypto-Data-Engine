
from crypto_data_engine.services.tick_data_scraper.tick_worker import run_simple_download
from crypto_data_engine.db.repository.download import DownloadTaskRepository
from crypto_data_engine.db.models.download import TaskStatus
import logging

logger = logging.getLogger(__name__)


def register_tasks(celery_app):
    """注册所有Celery任务"""

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
            exchange_name = cfg.get('exchange_name', 'binance')
            symbols = cfg.get('symbols', None)
            start_date = cfg.get('start_date', '2020-01')
            end_date = cfg.get('end_date', '2022-03')
            max_threads = cfg.get('max_threads', 8)
            self.update_state(
                state='PROGRESS',
                meta={'status': '开始下载', 'exchange': exchange_name}
            )
            result = run_simple_download(
                exchange_name=exchange_name,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                max_threads=max_threads
            )
            if task_id:
                DownloadTaskRepository.update_status(task_id, TaskStatus.COMPLETED)
            logger.info(f"下载任务完成: {exchange_name}")
            return {
                'status': 'SUCCESS',
                'exchange': exchange_name,
                'symbols_count': len(symbols) if symbols else 0,
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

    # @celery_app.task(name="bar.generate", bind=True)
    # def dispatch_bar_generation(self, cfg: dict):
    # from crypto_data_engine.services.bar_aggregator.bar_processor import run_simple_bar_generation
    #     """
    #     分布式 Bar 生成任务
    #     """
    #     task_id = cfg.get('task_id')
    #
    #     try:
    #         logger.info(f"开始执行 Bar 生成任务: {cfg}")
    #
    #         # 更新任务状态
    #         if task_id:
    #             DownloadTaskRepository.update_status(task_id, TaskStatus.PROCESSING)
    #
    #         # 提取参数
    #         exchange_name = cfg.get('exchange_name', 'binance')
    #         symbols = cfg.get('symbols', None)
    #         bar_type = cfg.get('bar_type', 'volume_bar')
    #         threshold = cfg.get('threshold', 10000000)
    #         raw_data_dir = cfg.get('raw_data_dir', './data/tick_data')
    #         output_dir = cfg.get('output_dir', './data/bar_data')
    #         # 更新任务进度
    #         self.update_state(
    #             state='PROGRESS',
    #             meta={
    #                 'status': f'开始生成 {bar_type}',
    #                 'exchange': exchange_name,
    #                 'symbols_count': len(symbols) if symbols else 0
    #             }
    #         )
    #         # 执行 Bar 生成
    #         result = run_simple_bar_generation(
    #             exchange_name=exchange_name,
    #             symbols=symbols,
    #             bar_type=bar_type,
    #             threshold=threshold,
    #             raw_data_dir=raw_data_dir,
    #             output_dir=output_dir
    #         )
    #
    #         # 更新任务状态为完成
    #         if task_id:
    #             DownloadTaskRepository.update_status(task_id, TaskStatus.COMPLETED)
    #
    #         logger.info(f"Bar 生成任务完成: {exchange_name}")
    #
    #         return {
    #             'status': 'SUCCESS',
    #             'exchange': exchange_name,
    #             'bar_type': bar_type,
    #             'threshold': threshold,
    #             'processed_count': result.get('processed', 0),
    #             'total_symbols': result.get('total_symbols', 0),
    #             'task_id': task_id,
    #             'result': result
    #         }
    #
    #     except Exception as e:
    #         logger.error(f"Bar 生成任务失败: {str(e)}")
    #
    #         # 更新任务状态为失败
    #         if task_id:
    #             DownloadTaskRepository.update_status(task_id, TaskStatus.FAILED)
    #
    #         # 更新任务状态
    #         self.update_state(
    #             state='FAILURE',
    #             meta={
    #                 'error': str(e),
    #                 'exchange': cfg.get('exchange_name', 'unknown'),
    #                 'bar_type': cfg.get('bar_type', 'unknown')
    #             }
    #         )
    #
    #         return {
    #             'status': 'FAILED',
    #             'error': str(e),
    #             'task_id': task_id,
    #             'exchange': cfg.get('exchange_name', 'unknown')
    #         }

    return {
        'dispatch_tick_download': dispatch_tick_download,
        'health_check': health_check
    }