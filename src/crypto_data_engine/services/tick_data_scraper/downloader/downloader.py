"""
Enhanced file downloader with:
- Redis-backed queue for download → convert pipeline
- Retry logic with exponential backoff
- Resume support (checks parquet existence, not ZIP)
- Deletes ZIP after successful Parquet conversion
- API-compatible with TaskManager progress tracking
"""
import hashlib
import os
import shutil
import threading
import time
import zipfile
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import redis
import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from crypto_data_engine.common.logger.logger import get_logger
from .exchange_factory import ExchangeFactory

logger = get_logger(__name__)

# Redis key prefixes for download pipeline
CONVERT_QUEUE_KEY = "download:convert_queue"
PROGRESS_KEY_PREFIX = "download:progress"


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
    """Enhanced downloader with Redis-backed pipeline, retry, and resume."""

    def __init__(self, context: DownloadContext, redis_url: Optional[str] = None):
        self.context = context
        self.adapter = context.exchange_adapter
        self.max_retries: int = context.config.get("max_retries", 3)
        self.base_retry_delay: float = context.config.get("base_retry_delay", 1.0)
        self.exponential_backoff: bool = context.config.get("exponential_backoff", True)
        self.http_timeout: float = context.config.get("http_timeout", 60.0)
        self.convert_workers: int = context.config.get("convert_processes", 4)
        self._redis_url = redis_url
        self._redis: Optional[redis.Redis] = None

    @property
    def redis_client(self) -> redis.Redis:
        """Lazy-connect to Redis using config settings.
        
        Raises:
            redis.ConnectionError: If Redis is not running or unreachable.
        """
        if self._redis is None:
            url = self._redis_url
            if url is None:
                from crypto_data_engine.common.config.config_settings import settings
                url = settings.task_cfg.redis_url
            try:
                self._redis = redis.from_url(url, decode_responses=True)
                self._redis.ping()
                logger.info(f"Connected to Redis at {url}")
            except redis.ConnectionError as error:
                logger.error(f"Failed to connect to Redis at {url}: {error}")
                raise ConnectionError(
                    f"Redis is required for download pipeline but connection failed.\n"
                    f"Redis URL: {url}\n"
                    f"Error: {error}\n\n"
                    f"Please start Redis:\n"
                    f"  - Docker: docker run -d -p 6379:6379 redis:7.2\n"
                    f"  - Or: docker-compose -f deploy/docker-compose.yml up redis -d"
                ) from error
        return self._redis

    # =========================================================================
    # Download logic
    # =========================================================================

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
        symbol_dir = self.context.save_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = symbol_dir / (Path(file_name).stem + ".parquet")

        if parquet_path.exists():
            return None

        if getattr(self.adapter, "supports_api_fetch", False):
            result = self.adapter.fetch_via_api(
                symbol, year, month, str(self.context.save_dir)
            )
            return result

        url = self.adapter.build_download_url(symbol, year, month)
        local_file_path = symbol_dir / file_name

        if local_file_path.exists() and local_file_path.stat().st_size > 0:
            return None

        for attempt in range(1, self.max_retries + 1):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                head_response = requests.head(url, timeout=10, headers=headers)
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

                response = requests.get(url, stream=True, timeout=self.http_timeout, headers=headers)
                response.raise_for_status()

                with open(local_file_path, "wb") as file_handle:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file_handle.write(chunk)

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
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.warning(f"Downloaded file is empty: {symbol} {year}-{month:02d}")
                return False

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

    # =========================================================================
    # Pipeline: Redis-backed download → convert
    # =========================================================================

    def run_download_pipeline(
        self,
        config: Optional[Dict] = None,
        task_id: Optional[str] = None,
        task_manager=None,
    ):
        """Run pipelined download → extract → convert with Redis queue.

        - Downloads push completed ZIP paths to a Redis List.
        - Conversion worker threads BRPOP from the list and process immediately.
        - Progress is tracked in a Redis Hash for API queries.
        - Supports resume: ZIPs without parquet are re-queued for conversion.

        Args:
            config: Override effective config.
            task_id: TaskManager task ID for API progress updates.
            task_manager: TaskManager instance for API integration.
        """
        effective_config = config or self.context.config
        max_threads = effective_config.get("max_threads", 16)
        convert_workers = effective_config.get("convert_processes", self.convert_workers)
        start_date = effective_config["start_date"]
        end_date = effective_config["end_date"]

        self.context.save_dir.mkdir(parents=True, exist_ok=True)

        # Fetch symbols
        symbols_config = effective_config.get("symbols")
        if symbols_config is None or symbols_config == "auto":
            symbols = self.adapter.get_all_symbols(effective_config.get("filter_suffix"))
        else:
            symbols = symbols_config

        # Scan: categorize into download-needed vs convert-only (resume)
        download_tasks, resume_zips = self._scan_pending_tasks(symbols, start_date, end_date)

        total_download = len(download_tasks)
        total_convert = total_download + len(resume_zips)
        logger.info(
            f"Exchange: {self.adapter.name} | Symbols: {len(symbols)} | "
            f"To download: {total_download} | To convert (resume): {len(resume_zips)} | "
            f"Total convert: {total_convert}"
        )

        if total_convert == 0:
            logger.info("No new files to process")
            return

        pipeline_start = time.time()
        job_id = task_id or f"dl_{self.adapter.name}_{int(time.time())}"
        convert_queue_key = f"{CONVERT_QUEUE_KEY}:{job_id}"
        progress_key = f"{PROGRESS_KEY_PREFIX}:{job_id}"

        redis_client = self.redis_client

        # Clean up any leftover queue from a previous run with same job_id
        redis_client.delete(convert_queue_key)

        # Push resume ZIPs directly into convert queue
        if resume_zips:
            redis_client.lpush(convert_queue_key, *resume_zips)
            logger.info(f"Resumed {len(resume_zips)} ZIPs from previous incomplete run")

        # Initialize progress hash in Redis
        redis_client.hset(progress_key, mapping={
            "job_id": job_id,
            "exchange": self.adapter.name,
            "total_download": total_download,
            "total_convert": total_convert,
            "downloaded": 0,
            "download_failed": 0,
            "download_skipped": 0,
            "converted": 0,
            "convert_failed": 0,
            "status": "running",
            "started_at": datetime.now().isoformat(),
        })
        redis_client.expire(progress_key, 86400)  # TTL 24h

        # Sentinel flag: downloads done
        downloads_complete = threading.Event()

        # Progress bars (CLI only, safe if stdout isn't a terminal)
        download_bar = tqdm(
            total=total_download, desc=f"[{self.adapter.name} Download]", position=0
        ) if total_download > 0 else None
        convert_bar = tqdm(
            total=total_convert, desc=f"[{self.adapter.name} Convert]", position=1
        )

        # ------- Consumer: Conversion workers -------
        def conversion_worker():
            while True:
                result = redis_client.brpop(convert_queue_key, timeout=2)
                if result is None:
                    # Queue empty — check if downloads are done and queue drained
                    if downloads_complete.is_set() and redis_client.llen(convert_queue_key) == 0:
                        break
                    continue
                _, zip_path = result
                try:
                    convert_result = _process_single_zip(zip_path)
                    if convert_result:
                        redis_client.hincrby(progress_key, "converted", 1)
                    else:
                        redis_client.hincrby(progress_key, "convert_failed", 1)
                except Exception as error:
                    logger.warning(f"Conversion failed for {zip_path}: {error}")
                    redis_client.hincrby(progress_key, "convert_failed", 1)
                finally:
                    convert_bar.update(1)
                    self._sync_task_manager_progress(
                        task_manager, task_id, redis_client, progress_key, total_convert,
                    )

        conversion_threads = []
        for _ in range(convert_workers):
            thread = threading.Thread(target=conversion_worker, daemon=True)
            thread.start()
            conversion_threads.append(thread)

        # ------- Producer: Download and push to Redis queue -------
        def download_and_enqueue(symbol: str, year: int, month: int):
            try:
                result_path = self.download_file(symbol, year, month)
                if result_path:
                    redis_client.hincrby(progress_key, "downloaded", 1)
                    redis_client.lpush(convert_queue_key, result_path)
                else:
                    redis_client.hincrby(progress_key, "download_skipped", 1)
                    convert_bar.update(1)  # No conversion needed
            except Exception as error:
                logger.warning(f"Download failed for {symbol} {year}-{month}: {error}")
                redis_client.hincrby(progress_key, "download_failed", 1)
                convert_bar.update(1)
            finally:
                if download_bar:
                    download_bar.update(1)

        # Execute all downloads in thread pool
        if download_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = [
                    executor.submit(download_and_enqueue, symbol, year, month)
                    for symbol, year, month in download_tasks
                ]
                concurrent.futures.wait(futures)

        # Signal conversion workers that no more items will arrive
        downloads_complete.set()

        # Wait for all conversions to finish
        for thread in conversion_threads:
            thread.join()

        if download_bar:
            download_bar.close()
        convert_bar.close()

        # Finalize progress
        elapsed = time.time() - pipeline_start
        progress = redis_client.hgetall(progress_key)
        redis_client.hset(progress_key, mapping={
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "elapsed_seconds": f"{elapsed:.1f}",
        })

        logger.info(
            f"{self.adapter.name} pipeline finished: "
            f"downloaded={progress.get('downloaded', 0)}, "
            f"skipped={progress.get('download_skipped', 0)}, "
            f"failed_dl={progress.get('download_failed', 0)}, "
            f"converted={progress.get('converted', 0)}, "
            f"failed_cv={progress.get('convert_failed', 0)}, "
            f"elapsed={elapsed:.1f}s"
        )

        # Final TaskManager update
        if task_manager and task_id:
            from crypto_data_engine.common.task_manager import TaskStatus
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=1.0,
                message=f"Done: {progress.get('converted', 0)}/{total_convert} converted",
                result={
                    "downloaded": int(progress.get("downloaded", 0)),
                    "converted": int(progress.get("converted", 0)),
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

        # Cleanup Redis queue key (progress key kept for API queries)
        redis_client.delete(convert_queue_key)

    # =========================================================================
    # Task scanning (replaces _generate_download_tasks)
    # =========================================================================

    def _scan_pending_tasks(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> Tuple[List[Tuple[str, int, int]], List[str]]:
        """Scan data directory and categorize tasks for the pipeline.

        Returns:
            download_tasks: (symbol, year, month) tuples that need downloading.
            resume_zips: ZIP file paths that exist but lack a parquet (need conversion only).
        """
        download_tasks: List[Tuple[str, int, int]] = []
        resume_zips: List[str] = []
        parquet_found_count = 0

        for symbol in symbols:
            # Determine start date
            if start_date.lower() == "auto":
                list_time_ms = hasattr(self.adapter, "get_symbol_list_time") and self.adapter.get_symbol_list_time(symbol)
                if list_time_ms:
                    sym_start = datetime.fromtimestamp(list_time_ms / 1000.0)
                else:
                    logger.warning(f"Adapter doesn't support auto start date for {symbol}, defaulting to 2020-01")
                    sym_start = datetime(2020, 1, 1)
            else:
                sym_start = datetime.strptime(start_date, "%Y-%m")

            # Determine end date
            if end_date.lower() == "auto":
                # Current month
                now = datetime.now()
                sym_end = datetime(now.year, now.month, 1)
            else:
                sym_end = datetime.strptime(end_date, "%Y-%m")

            # Force dates to start-of-month for clean iteration
            current_date = datetime(sym_start.year, sym_start.month, 1)
            end_dt = datetime(sym_end.year, sym_end.month, 1)

            if current_date > end_dt:
                logger.warning(f"Start date {current_date.strftime('%Y-%m')} is after end date {end_dt.strftime('%Y-%m')} for {symbol}")
                continue

            while current_date <= end_dt:
                file_name = self.adapter.get_file_name(
                    symbol, current_date.year, current_date.month
                )
                symbol_dir = self.context.save_dir / symbol
                zip_path = symbol_dir / file_name
                parquet_path = symbol_dir / (Path(file_name).stem + ".parquet")

                if parquet_path.exists():
                    # Already converted — skip entirely
                    parquet_found_count += 1
                elif zip_path.exists() and self._is_valid_zip(zip_path):
                    # ZIP downloaded but not yet converted — queue for conversion
                    resume_zips.append(str(zip_path))
                else:
                    # Remove corrupt/partial ZIP if present, then add to download list
                    if zip_path.exists():
                        logger.warning(
                            f"Corrupt/incomplete ZIP removed: {zip_path} "
                            f"({zip_path.stat().st_size} bytes)"
                        )
                        zip_path.unlink(missing_ok=True)
                    download_tasks.append((symbol, current_date.year, current_date.month))

                current_date += relativedelta(months=1)

        if parquet_found_count > 0:
            logger.info(f"Skipped {parquet_found_count} already-converted parquet files")
        if resume_zips:
            logger.info(f"Found {len(resume_zips)} ZIPs pending conversion (resume)")

        return download_tasks, resume_zips

    @staticmethod
    def _is_valid_zip(file_path: Path) -> bool:
        """Lightweight check: verify the ZIP central directory is readable."""
        try:
            with zipfile.ZipFile(file_path, "r") as zip_handle:
                if len(zip_handle.namelist()) == 0:
                    return False
            return True
        except (zipfile.BadZipFile, OSError):
            return False

    @staticmethod
    def _sync_task_manager_progress(
        task_manager,
        task_id: Optional[str],
        redis_client: redis.Redis,
        progress_key: str,
        total_convert: int,
    ) -> None:
        """Push current Redis progress to TaskManager (for API polling)."""
        if not task_manager or not task_id:
            return
        try:
            converted = int(redis_client.hget(progress_key, "converted") or 0)
            failed = int(redis_client.hget(progress_key, "convert_failed") or 0)
            done = converted + failed
            progress_ratio = done / total_convert if total_convert > 0 else 0.0
            task_manager.update_task(
                task_id,
                progress=progress_ratio,
                message=f"Converted: {converted}/{total_convert}, failed: {failed}",
            )
        except Exception:
            pass  # Non-critical, don't block conversion

    def get_pipeline_progress(self, job_id: str) -> Optional[Dict]:
        """Query live pipeline progress from Redis.

        Returns dict with all progress counters, or None if job not found.
        """
        progress_key = f"{PROGRESS_KEY_PREFIX}:{job_id}"
        progress = self.redis_client.hgetall(progress_key)
        return progress if progress else None


# =============================================================================
# Module-level conversion function (used by both pipeline and CLI `data convert`)
# =============================================================================

def _process_single_zip(zip_path: str) -> Optional[str]:
    """Extract ZIP, convert CSV to Parquet, then delete ZIP and temp files.

    If zip_path is a .parquet file (API-fetched, e.g. OKX), no-op and return success.
    The .parquet is placed directly in the symbol directory (same level as ZIP).
    After successful conversion, both the extracted directory and the original
    ZIP are removed.

    Defined at module level for ProcessPoolExecutor pickle compatibility.
    """
    try:
        file_path = Path(zip_path)
        if file_path.suffix.lower() == ".parquet" and file_path.exists():
            return f"{file_path.name}: already parquet (API fetch)"
    except Exception:
        pass

    from crypto_data_engine.services.tick_data_scraper.extractor.convert import (
        extract_archive,
        convert_dir_to_parquet,
    )

    try:
        zip_file = Path(zip_path)
        symbol_dir = zip_file.parent

        # Step 1: Extract ZIP → temporary sub-directory
        extract_result = extract_archive(
            directory=str(symbol_dir),
            file_name=zip_file.name,
        )
        extracted_dir = extract_result["out_dir"]

        # Step 2: Convert CSV → Parquet, output to symbol directory
        parquet_files = convert_dir_to_parquet(
            extracted_dir=extracted_dir,
            pattern="*.csv",
            output_dir=str(symbol_dir),
        )

        # Step 3: Remove extracted sub-directory (CSV no longer needed)
        extracted_path = Path(extracted_dir)
        if extracted_path.exists() and extracted_path.is_dir():
            shutil.rmtree(extracted_path)

        # Step 4: Remove original ZIP (parquet is the final artifact)
        if zip_file.exists() and parquet_files:
            zip_file.unlink()
            logger.debug(f"Deleted ZIP after conversion: {zip_file.name}")

        return f"{zip_file.name}: {len(parquet_files)} parquet files"
    except Exception as error:
        logger.warning(f"Extract/convert failed for {zip_path}: {error}")
        return None


# Backward compatibility aliases
MultiExchangeDownloadContext = DownloadContext
MultiExchangeDownloader = FileDownloader
