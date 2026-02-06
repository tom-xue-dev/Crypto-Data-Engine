"""
Enhanced file downloader with:
- Retry logic with exponential backoff
- Automatic extraction (zip -> csv) and conversion (csv -> parquet)
- Configurable thread/process count
- No Celery dependency (uses local multiprocessing)
- Integrity check on existing files to handle interrupted downloads
"""
import hashlib
import os
import time
import zipfile
import concurrent.futures
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from crypto_data_engine.common.logger.logger import get_logger
from .exchange_factory import ExchangeFactory

logger = get_logger(__name__)


class DownloadContext:
    """Initiates all the necessary download context."""

    def __init__(
        self,
        config: Dict,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ):
        config.update({
            "start_date": start_date,
            "end_date": end_date,
            "symbols": symbols,
        })
        self.exchange_name: str = config["exchange_name"]
        self.save_dir: Path = Path(str(config["data_dir"]))
        self.exchange_adapter = ExchangeFactory.create_adapter(
            self.exchange_name, config
        )
        self.config = config


class FileDownloader:
    """Enhanced downloader with retry, extraction, and conversion."""

    def __init__(self, context: DownloadContext):
        self.context = context
        self.adapter = context.exchange_adapter
        self.max_retries: int = context.config.get("max_retries", 3)
        self.base_retry_delay: float = context.config.get("base_retry_delay", 1.0)
        self.exponential_backoff: bool = context.config.get("exponential_backoff", True)
        self.http_timeout: float = context.config.get("http_timeout", 60.0)
        self.convert_processes: int = context.config.get("convert_processes", 4)

    @staticmethod
    def verify_checksum(file_path: str, expected_checksum: str) -> bool:
        """Validate file checksum (SHA256)."""
        expected_checksum = expected_checksum.split()[0].strip()
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_checksum

    def download_file(self, symbol: str, year: int, month: int) -> Optional[str]:
        """Download a single file with retry logic. Returns local path on success."""
        file_name = self.adapter.get_file_name(symbol, year, month)
        url = self.adapter.build_download_url(symbol, year, month)
        symbol_dir = self.context.save_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        local_file_path = symbol_dir / file_name

        # Skip if already downloaded
        if local_file_path.exists() and local_file_path.stat().st_size > 0:
            return None  # Already exists

        for attempt in range(1, self.max_retries + 1):
            try:
                # HEAD check first
                head_response = requests.head(url, timeout=10)
                if head_response.status_code == 404:
                    logger.debug(f"File not found (404): {symbol} {year}-{month:02d}")
                    return None
                if head_response.status_code != 200:
                    logger.warning(
                        f"HEAD returned {head_response.status_code} for "
                        f"{symbol} {year}-{month:02d}, attempt {attempt}/{self.max_retries}"
                    )
                    self._wait_before_retry(attempt)
                    continue

                # Download with streaming
                response = requests.get(url, stream=True, timeout=self.http_timeout)
                response.raise_for_status()

                with open(local_file_path, "wb") as file_handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file_handle.write(chunk)

                # Verify download
                if self._verify_download(symbol, year, month, str(local_file_path)):
                    return str(local_file_path)
                else:
                    local_file_path.unlink(missing_ok=True)
                    logger.warning(
                        f"Verification failed for {symbol} {year}-{month:02d}, "
                        f"attempt {attempt}/{self.max_retries}"
                    )

            except requests.RequestException as error:
                logger.warning(
                    f"Download failed for {symbol} {year}-{month:02d} "
                    f"(attempt {attempt}/{self.max_retries}): {error}"
                )
                local_file_path.unlink(missing_ok=True)

            self._wait_before_retry(attempt)

        logger.error(f"All {self.max_retries} attempts failed for {symbol} {year}-{month:02d}")
        return None

    def _wait_before_retry(self, attempt: int) -> None:
        """Wait before retrying with exponential backoff."""
        if attempt >= self.max_retries:
            return
        if self.exponential_backoff:
            delay = self.base_retry_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_retry_delay
        time.sleep(delay)

    def _verify_download(self, symbol: str, year: int, month: int, file_path: str) -> bool:
        """Validate the downloaded file (checksum + size)."""
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.warning(f"Downloaded file is empty: {symbol} {year}-{month:02d}")
                return False

            # Checksum verification
            if getattr(self.adapter, "supports_checksum", False):
                checksum_url = self.adapter.build_checksum_url(symbol, year, month)
                if checksum_url:
                    try:
                        checksum_response = requests.get(checksum_url, timeout=10)
                        if checksum_response.status_code == 200:
                            expected = checksum_response.text
                            if not self.verify_checksum(file_path, expected):
                                logger.warning(
                                    f"Checksum mismatch for {symbol} {year}-{month:02d}"
                                )
                                return False
                    except requests.RequestException:
                        logger.debug(f"Checksum fetch failed for {symbol}, skipping verification")

            return True
        except Exception as error:
            logger.warning(f"Download verification error for {symbol}: {error}")
            return False

    def run_download_pipeline(self, config: Optional[Dict] = None):
        """Run the full download pipeline: download -> extract -> convert to parquet."""
        effective_config = config or self.context.config
        max_threads = effective_config.get("max_threads", 16)
        start_date = effective_config["start_date"]
        end_date = effective_config["end_date"]

        self.context.save_dir.mkdir(parents=True, exist_ok=True)

        download_counter = multiprocessing.Value("i", 0)
        failed_counter = multiprocessing.Value("i", 0)

        # Fetch symbols
        symbols_config = effective_config.get("symbols")
        if symbols_config is None or symbols_config == "auto":
            symbols = self.adapter.get_all_symbols(effective_config.get("filter_suffix"))
        else:
            symbols = symbols_config

        # Generate download tasks (skip already-completed ones)
        download_tasks = self._generate_download_tasks(symbols, start_date, end_date)
        total_tasks = len(download_tasks)
        logger.info(f"Exchange: {self.adapter.name} | Symbols: {len(symbols)} | Tasks: {total_tasks}")

        if total_tasks == 0:
            logger.info("No new files to download")
            return

        pipeline_start = time.time()

        # Phase 1: Download
        downloaded_files: List[str] = []
        with tqdm(total=total_tasks, desc=f"[{self.adapter.name} Download]") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                future_to_task = {
                    executor.submit(
                        self._download_task, symbol, year, month,
                        progress_bar, download_counter, failed_counter,
                    ): (symbol, year, month)
                    for symbol, year, month in download_tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    symbol, year, month = future_to_task[future]
                    try:
                        result_path = future.result()
                        if result_path:
                            downloaded_files.append(result_path)
                    except Exception as error:
                        logger.warning(f"Task failed for {symbol} {year}-{month}: {error}")

        # Phase 2: Extract and convert to Parquet
        if downloaded_files:
            logger.info(f"Extracting and converting {len(downloaded_files)} files to Parquet...")
            self._extract_and_convert_batch(downloaded_files)

        elapsed = time.time() - pipeline_start
        logger.info(
            f"{self.adapter.name} pipeline finished: "
            f"downloaded={download_counter.value}, failed={failed_counter.value}, "
            f"elapsed={elapsed:.1f}s"
        )

    def _download_task(
        self,
        symbol: str,
        year: int,
        month: int,
        progress_bar,
        success_counter,
        failed_counter,
    ) -> Optional[str]:
        """Single download unit. Returns file path on success."""
        try:
            result = self.download_file(symbol, year, month)
            progress_bar.update(1)
            if result:
                with success_counter.get_lock():
                    success_counter.value += 1
                return result
            else:
                with failed_counter.get_lock():
                    failed_counter.value += 1
                return None
        except Exception as error:
            logger.warning(f"Download task error for {symbol} {year}-{month}: {error}")
            progress_bar.update(1)
            with failed_counter.get_lock():
                failed_counter.value += 1
            return None

    def _extract_and_convert_batch(self, zip_file_paths: List[str]) -> None:
        """Extract zip files and convert CSV contents to Parquet in parallel."""
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.convert_processes
        ) as executor:
            results = list(
                tqdm(
                    executor.map(_process_single_zip, zip_file_paths),
                    total=len(zip_file_paths),
                    desc="[Extract & Convert]",
                )
            )

        successful = sum(1 for r in results if r is not None)
        logger.info(f"Extract & convert: {successful}/{len(zip_file_paths)} succeeded")

    @staticmethod
    def _is_valid_zip(file_path: Path) -> bool:
        """Lightweight check: verify the ZIP central directory is readable.

        This catches truncated / incomplete downloads without reading the
        entire file content (unlike ``testzip()`` which does full CRC
        verification and is too slow for bulk checks).
        """
        try:
            with zipfile.ZipFile(file_path, "r") as zip_handle:
                # Reading the name list forces parsing of the central
                # directory at the end of the file. An incomplete download
                # will fail here because the central directory is missing.
                if len(zip_handle.namelist()) == 0:
                    return False
            return True
        except (zipfile.BadZipFile, OSError):
            return False

    def _generate_download_tasks(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> List[Tuple[str, int, int]]:
        """Generate download task list, skipping already-downloaded files.

        Performs a lightweight integrity check on existing ZIP files:
        corrupt or incomplete archives (e.g. from interrupted downloads)
        are deleted and re-queued for download.
        """
        tasks = []
        cleaned_count = 0
        existing_count = 0

        for symbol in symbols:
            current_date = datetime.strptime(start_date, "%Y-%m")
            end_dt = datetime.strptime(end_date, "%Y-%m")

            while current_date <= end_dt:
                file_name = self.adapter.get_file_name(
                    symbol, current_date.year, current_date.month
                )
                local_path = self.context.save_dir / symbol / file_name

                if local_path.exists():
                    existing_count += 1
                    if not self._is_valid_zip(local_path):
                        file_size = local_path.stat().st_size
                        logger.warning(
                            f"Corrupt/incomplete file detected, removing and "
                            f"re-queuing: {local_path} ({file_size} bytes)"
                        )
                        local_path.unlink(missing_ok=True)
                        tasks.append((symbol, current_date.year, current_date.month))
                        cleaned_count += 1
                else:
                    tasks.append((symbol, current_date.year, current_date.month))

                current_date += relativedelta(months=1)

        if existing_count > 0:
            logger.info(
                f"Integrity check: scanned {existing_count} existing file(s), "
                f"removed {cleaned_count} corrupt/incomplete"
            )

        return tasks


def _process_single_zip(zip_path: str) -> Optional[str]:
    """Extract a single zip file and convert CSV contents to Parquet.

    Defined at module level so it can be pickled by ProcessPoolExecutor on Windows.
    """
    from crypto_data_engine.services.tick_data_scraper.extractor.convert import (
        extract_archive,
        convert_dir_to_parquet,
    )

    try:
        zip_file = Path(zip_path)
        extract_result = extract_archive(
            directory=str(zip_file.parent),
            file_name=zip_file.name,
        )
        extracted_dir = extract_result["out_dir"]
        parquet_files = convert_dir_to_parquet(
            extracted_dir=extracted_dir,
            pattern="*.csv",
        )
        return f"{zip_path}: {len(parquet_files)} parquet files"
    except Exception as error:
        logger.warning(f"Extract/convert failed for {zip_path}: {error}")
        return None


# Backward compatibility aliases
MultiExchangeDownloadContext = DownloadContext
MultiExchangeDownloader = FileDownloader
