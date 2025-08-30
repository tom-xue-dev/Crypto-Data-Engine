import logging
import os
import requests
import hashlib
import concurrent.futures
import multiprocessing
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import time
from crypto_data_engine.db.repository.download import DownloadTaskRepository
from typing import Dict, List, Optional, Tuple
from crypto_data_engine.db.constants import TaskStatus

from .exchange_factory import ExchangeFactory
from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

class DownloadContext:
    """initiates all the necessary download context"""

    def __init__(self, config: Dict,start_date:str,end_date:str,symbols:List[str] = None):
        config.update({
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
        })
        self.exchange_name = config['exchange_name']
        self.save_dir = config['data_dir']
        self.exchange_adapter = ExchangeFactory.create_adapter(
            self.exchange_name, config
        )


class FileDownloader:
    """专注于文件下载的下载器"""

    def __init__(self, context: DownloadContext):
        self.context = context
        self.adapter = context.exchange_adapter

    @staticmethod
    def verify_checksum(file_path: str, expected_checksum: str) -> bool:
        """验证文件校验和"""
        expected_checksum = expected_checksum.split()[0].strip()
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        computed_checksum = sha256.hexdigest()
        return computed_checksum == expected_checksum

    def download_file(self, symbol: str, year: int, month: int) -> Optional[str]:
        """下载单个文件，返回下载后的文件路径"""
        file_name = self.adapter.get_file_name(symbol, year, month)
        url = self.adapter.build_download_url(symbol, year, month)
        os.makedirs(os.path.join(self.context.save_dir,symbol), exist_ok=True)
        local_file_path = os.path.join(self.context.save_dir,symbol,file_name)
        logger.info(f"local_path: {local_file_path}")
        if os.path.exists(local_file_path):
            return None
        try:
            response = requests.head(url, timeout=5)
            if response.status_code != 200:
                return None
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(local_file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # 过滤空块
                        f.write(chunk)
            if self._verify_download(symbol, year, month, local_file_path):
                return local_file_path
            else:
                os.remove(local_file_path)
                return None
        except Exception as e:
            print(f"Download failed for {symbol} {year}-{month:02d}: {e}")
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            return None

    def _verify_download(self, symbol: str, year: int, month: int, file_path: str) -> bool:
        """验证下载的文件（校验和等）"""
        try:
            if hasattr(self.adapter, 'supports_checksum') and self.adapter.supports_checksum:
                checksum_url = self.adapter.build_checksum_url(symbol, year, month)
                if checksum_url:
                    checksum_response = requests.get(checksum_url, timeout=10)
                    if checksum_response.status_code == 200:
                        expected_checksum = checksum_response.text
                        if not self.verify_checksum(file_path, expected_checksum):
                            print(f"Checksum verification failed for {symbol} {year}-{month:02d}")
                            return False

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                print(f"Downloaded file is empty: {symbol} {year}-{month:02d}")
                return False

            return True

        except Exception as e:
            print(f"Download verification failed for {symbol} {year}-{month:02d}: {e}")
            return False

    def run_download_pipeline(self, config: Dict):
        """运行下载流水线 - 只负责下载"""
        max_threads = config['max_threads']
        start_date = config['start_date']
        end_date = config['end_date']
        save_dir = config['data_dir']
        os.makedirs(save_dir, exist_ok=True)
        download_counter = multiprocessing.Value('i', 0)
        failed_counter = multiprocessing.Value('i', 0)
        # 获取交易对列表
        if config['symbols'] == "auto":
            symbols = self.adapter.get_all_symbols(config.get('filter_suffix'))
        else:
            symbols = config['symbols']

        completed_downloads = DownloadTaskRepository.get_all_tasks(exchange=self.adapter.name,status=TaskStatus.EXTRACTED)
        download_tasks = self._generate_download_tasks(symbols, start_date, end_date, completed_downloads)
        total_tasks = len(download_tasks)
        logger.info(f"📊 Exchange: {self.adapter.name}")
        logger.info(f"📥 Total download tasks: {total_tasks}")

        if total_tasks == 0:
            logger.info("✅ No new files to download")
            return

        start_time = time.time()

        # 使用线程池进行并发下载
        with tqdm(total=total_tasks, desc=f"[{self.adapter.name} Download]") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                future_to_task = {
                    executor.submit(self._download_task, symbol, year, month, pbar, download_counter, failed_counter):
                        (symbol, year, month)
                    for (symbol, year, month) in download_tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    symbol, year, month = future_to_task[future]
                    try:
                        self._set_start_status(self.context.exchange_name, symbol, year, month)
                        success = future.result()
                        if success:
                            self._set_complete_status(self.context.exchange_name, symbol, year, month, save_dir)
                    except Exception as e:
                        logger.warning(f"Task failed for {symbol} {year}-{month}: {e}")

        end_time = time.time()

        logger.info(f"\n✅ {self.adapter.name} 下载完成:")
        logger.info(f"   📥 成功下载: {download_counter.value}")
        logger.info(f"   ❌ 失败/跳过: {failed_counter.value}")
        logger.info(f"   ⏰ 耗时: {end_time - start_time:.2f} 秒")

    def _download_task(self, symbol: str, year: int, month: int, pbar, success_counter, failed_counter) -> bool:
        """单个下载任务"""
        try:
            result = self.download_file(symbol, year, month)
            pbar.update(1)

            if result:
                with success_counter.get_lock():
                    success_counter.value += 1
                return True
            else:
                with failed_counter.get_lock():
                    failed_counter.value += 1
                return False

        except Exception as e:
            print(f"Download task error for {symbol} {year}-{month}: {e}")
            pbar.update(1)
            with failed_counter.get_lock():
                failed_counter.value += 1
            return False


    def _set_complete_status(self, exchange, symbol, year, month, save_dir):
        """追加完成的下载任务到文件,并发起解压任务"""
        task_id = DownloadTaskRepository.get_task_id(exchange,symbol,year,month)
        DownloadTaskRepository.update_file_info(task_id=task_id,local_path=str(os.path.join(save_dir,symbol)))
        DownloadTaskRepository.update_status(task_id=task_id,status=TaskStatus.EXTRACTED)
        DownloadTaskRepository.update(record_id=task_id,task_end = datetime.now())
        task = DownloadTaskRepository.get_by_kwargs(exchange=exchange,symbol=symbol,year=year,month=month)
        from task_manager.celery_app import celery_app
        celery_app.send_task(
            "tick.extract_task",
            kwargs={"directory": str(task.local_path), "file_name": task.file_name},
            queue="extract",
        )

    def _set_start_status(self, exchange, symbol, year, month):
        task_id = DownloadTaskRepository.get_task_id(exchange, symbol, year, month)
        DownloadTaskRepository.update_status(task_id=task_id, status=TaskStatus.DOWNLOADING)
        DownloadTaskRepository.update(record_id=task_id, task_start=datetime.now())


    def _generate_download_tasks(self, symbols: List[str], start_date: str, end_date: str,completed: list) -> List[Tuple[str, int, int]]:
        """生成下载任务列表"""
        tasks = []

        for symbol in symbols:
            current_date = datetime.strptime(start_date, "%Y-%m")
            end_dt = datetime.strptime(end_date, "%Y-%m")

            while current_date <= end_dt:
                task_id = f"{self.context.exchange_name}/{symbol}/{current_date.year}-{current_date.month}"
                if task_id not in completed:
                    tasks.append((symbol, current_date.year, current_date.month))
                current_date += relativedelta(months=1)

        return tasks


# 保持向后兼容性的别名
MultiExchangeDownloadContext = DownloadContext
MultiExchangeDownloader = FileDownloader