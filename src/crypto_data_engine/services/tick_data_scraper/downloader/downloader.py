import os
import requests
import hashlib
import concurrent.futures
import multiprocessing
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .exchange_factory import ExchangeFactory


class DownloadContext:
    """下载上下文（重命名，专注于下载）"""

    def __init__(self, config: Dict):
        self.save_dir = config['data_dir']
        self.completed_downloads_file = config.get('completed_downloads_file',
                                                   Path(config['data_dir']) / "completed_downloads.txt")
        self.exchange_name = config['exchange_name']

        # 创建交易所适配器
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

        # 创建交易所特定的保存目录
        exchange_dir = os.path.join(self.context.save_dir, self.adapter.name)
        os.makedirs(exchange_dir, exist_ok=True)

        local_file_path = os.path.join(exchange_dir, file_name)

        # 检查是否已经下载过
        if os.path.exists(local_file_path):
            return None  # 已存在，跳过下载

        try:
            # 检查远程文件是否存在
            response = requests.head(url, timeout=5)
            if response.status_code != 200:
                return None

            # 下载文件
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()

            with open(local_file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # 过滤空块
                        f.write(chunk)

            # 验证校验和（如果支持）
            if self._verify_download(symbol, year, month, local_file_path):
                return local_file_path
            else:
                # 校验和验证失败，删除文件
                os.remove(local_file_path)
                return None

        except Exception as e:
            print(f"Download failed for {symbol} {year}-{month:02d}: {e}")
            # 清理可能的不完整文件
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            return None

    def _verify_download(self, symbol: str, year: int, month: int, file_path: str) -> bool:
        """验证下载的文件（校验和等）"""
        try:
            # 如果交易所支持校验和验证
            if hasattr(self.adapter, 'supports_checksum') and self.adapter.supports_checksum:
                checksum_url = self.adapter.build_checksum_url(symbol, year, month)
                if checksum_url:
                    checksum_response = requests.get(checksum_url, timeout=10)
                    if checksum_response.status_code == 200:
                        expected_checksum = checksum_response.text
                        if not self.verify_checksum(file_path, expected_checksum):
                            print(f"Checksum verification failed for {symbol} {year}-{month:02d}")
                            return False

            # 基本的文件大小检查
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
        queue_size = config.get('queue_size', 100)
        start_date = config['start_date']
        end_date = config['end_date']
        os.makedirs(self.context.save_dir, exist_ok=True)

        download_counter = multiprocessing.Value('i', 0)
        failed_counter = multiprocessing.Value('i', 0)

        # 获取交易对列表
        if config['symbols'] == "auto":
            symbols = self.adapter.get_all_symbols(config.get('filter_suffix'))
        else:
            symbols = config['symbols']

        # 读取已完成下载
        completed_downloads = self._read_completed_file(self.context.completed_downloads_file)
        download_tasks = self._generate_download_tasks(symbols, start_date, end_date, completed_downloads)

        total_tasks = len(download_tasks)
        print(f"📊 Exchange: {self.adapter.name}")
        print(f"📥 Total download tasks: {total_tasks}")

        if total_tasks == 0:
            print("✅ No new files to download")
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
                        success = future.result()
                        if success:
                            task_id = f"{self.context.exchange_name}/{symbol}/{year}-{month}"
                            self._append_completed_file(self.context.completed_downloads_file, task_id)
                    except Exception as e:
                        print(f"Task failed for {symbol} {year}-{month}: {e}")

        end_time = time.time()

        print(f"\n✅ {self.adapter.name} 下载完成:")
        print(f"   📥 成功下载: {download_counter.value}")
        print(f"   ❌ 失败/跳过: {failed_counter.value}")
        print(f"   ⏰ 耗时: {end_time - start_time:.2f} 秒")

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

    def _read_completed_file(self, path: str) -> set:
        """读取已完成下载文件"""
        if not os.path.exists(path):
            return set()
        with open(path, "r") as f:
            return set(line.strip() for line in f.readlines())

    def _append_completed_file(self, path: str, task: str):
        """追加完成的下载任务到文件"""
        with open(path, "a") as f:
            f.write(task + "\n")

    def _generate_download_tasks(self, symbols: List[str], start_date: str, end_date: str,
                                 completed: set) -> List[Tuple[str, int, int]]:
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

    def get_downloaded_files(self, symbol: str = None) -> List[str]:
        """获取已下载的文件列表"""
        exchange_dir = os.path.join(self.context.save_dir, self.adapter.name)
        if not os.path.exists(exchange_dir):
            return []

        files = []
        for file in os.listdir(exchange_dir):
            if file.endswith('.zip'):  # 假设下载的是zip文件
                if symbol is None or symbol.upper() in file.upper():
                    files.append(os.path.join(exchange_dir, file))

        return sorted(files)


# 保持向后兼容性的别名
MultiExchangeDownloadContext = DownloadContext
MultiExchangeDownloader = FileDownloader