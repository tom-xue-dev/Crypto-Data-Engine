"""
Batch bar aggregation service with Redis-backed pipeline.

Features:
- Scan tick data directory for all parquet files
- Skip already-aggregated bars (resume support)
- Redis queue for scalable parallel aggregation
- Real-time progress tracking via Redis Hash
- Dynamic dollar bar thresholds (threshold=auto)
- Supports multiple bar types and thresholds
"""
import concurrent.futures
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import redis
from tqdm import tqdm

from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.services.bar_aggregator import aggregate_bars

logger = get_logger(__name__)

# Redis key prefixes
AGGREGATE_QUEUE_KEY = "aggregate:task_queue"
AGGREGATE_PROGRESS_KEY = "aggregate:progress"


class AggregationTask:
    """Single aggregation task descriptor."""

    def __init__(
        self,
        tick_file: Path,
        symbol: str,
        bar_type: str,
        threshold: str,
        output_dir: Path,
    ):
        self.tick_file = tick_file
        self.symbol = symbol
        self.bar_type = bar_type
        self.threshold = threshold
        self.output_dir = output_dir

    def get_output_path(self) -> Path:
        """Generate output file path for the aggregated bars."""
        stem = self.tick_file.stem
        parts = stem.split("-")
        if len(parts) >= 4:
            year_month = f"{parts[-2]}-{parts[-1]}"
        else:
            year_month = "unknown"

        threshold_label = self.threshold

        output_file = (
            self.output_dir
            / self.symbol
            / f"{self.symbol}_{self.bar_type}_{threshold_label}_{year_month}.parquet"
        )
        return output_file

    def to_dict(self) -> Dict:
        """Serialize task for Redis queue."""
        return {
            "tick_file": str(self.tick_file),
            "symbol": self.symbol,
            "bar_type": self.bar_type,
            "threshold": str(self.threshold),
            "output_dir": str(self.output_dir),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AggregationTask":
        """Deserialize task from Redis queue."""
        return cls(
            tick_file=Path(data["tick_file"]),
            symbol=data["symbol"],
            bar_type=data["bar_type"],
            threshold=data["threshold"],
            output_dir=Path(data["output_dir"]),
        )


class BatchAggregator:
    """Batch aggregation service with Redis-backed pipeline.

    Supports two threshold modes:
    - Fixed: threshold is a static value (e.g. "1000000")
    - Dynamic: threshold="auto", computes per-month thresholds from
      rolling average daily dollar volume
    """

    def __init__(
        self,
        tick_data_dir: str = "E:/data/binance_futures",
        output_dir: str = "E:/data/dollar_bar/bars",
        bar_type: str = "dollar_bar",
        threshold: str = "1000000",
        lookback_days: int = 10,
        bars_per_day: int = 50,
        discard_months: int = 1,
        use_ema: bool = False,
        redis_url: Optional[str] = None,
    ):
        self.tick_data_dir = Path(tick_data_dir)
        self.output_dir = Path(output_dir)
        self.bar_type = bar_type
        self.threshold = threshold
        self.lookback_days = lookback_days
        self.bars_per_day = bars_per_day
        self.discard_months = discard_months
        self.use_ema = use_ema
        self._redis_url = redis_url
        self._redis: Optional[redis.Redis] = None

    @property
    def is_dynamic_threshold(self) -> bool:
        """Whether this aggregator uses dynamic (auto) thresholds."""
        return self.threshold.lower() == "auto" and self.bar_type == "dollar_bar"

    @property
    def redis_client(self) -> redis.Redis:
        """Lazy-connect to Redis."""
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
                    f"Redis is required for batch aggregation but connection failed.\n"
                    f"Redis URL: {url}\n"
                    f"Error: {error}\n\n"
                    f"Please start Redis:\n"
                    f"  - Docker: docker run -d -p 6379:6379 redis:7.2\n"
                    f"  - Or: docker-compose -f deploy/docker-compose.yml up redis -d"
                ) from error
        return self._redis

    # =========================================================================
    # Fixed threshold scanning (original logic)
    # =========================================================================

    def scan_tasks(
        self, symbols: Optional[List[str]] = None, force: bool = False
    ) -> Tuple[List[AggregationTask], int]:
        """Scan tick data directory and identify tasks for fixed threshold mode."""
        tasks: List[AggregationTask] = []
        skipped = 0

        if not self.tick_data_dir.exists():
            logger.warning(f"Tick data directory not found: {self.tick_data_dir}")
            return tasks, skipped

        symbol_dirs = sorted([d for d in self.tick_data_dir.iterdir() if d.is_dir()])
        if symbols:
            symbol_set = set(s.upper() for s in symbols)
            symbol_dirs = [d for d in symbol_dirs if d.name in symbol_set]

        for symbol_dir in symbol_dirs:
            symbol = symbol_dir.name
            tick_files = sorted(symbol_dir.glob("*.parquet"))

            for tick_file in tick_files:
                task = AggregationTask(
                    tick_file=tick_file,
                    symbol=symbol,
                    bar_type=self.bar_type,
                    threshold=self.threshold,
                    output_dir=self.output_dir,
                )
                if task.get_output_path().exists() and not force:
                    skipped += 1
                else:
                    tasks.append(task)

        return tasks, skipped

    # =========================================================================
    # Dynamic threshold scanning (dollar_bar auto mode)
    # =========================================================================

    def scan_dynamic_tasks(
        self,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ) -> Tuple[Dict[str, List[AggregationTask]], int, int]:
        """Scan and build tasks for dynamic dollar bar threshold mode.

        Phase 1: Builds daily dollar volume profiles (parallelizable).
        Phase 2: For each symbol, determines per-month thresholds and creates tasks.

        Returns:
            symbol_tasks: Dict mapping symbol → ordered list of tasks.
            total_tasks: Total number of tasks to process.
            skipped: Number of already-completed files.
        """
        from crypto_data_engine.services.bar_aggregator.dollar_profile import (
            build_symbol_profile,
            get_dynamic_threshold,
            get_first_valid_date,
            extract_file_start_date,
        )

        if not self.tick_data_dir.exists():
            logger.warning(f"Tick data directory not found: {self.tick_data_dir}")
            return {}, 0, 0

        symbol_dirs = sorted([d for d in self.tick_data_dir.iterdir() if d.is_dir()])
        if symbols:
            symbol_set = set(s.upper() for s in symbols)
            symbol_dirs = [d for d in symbol_dirs if d.name in symbol_set]

        avg_method = "ema" if self.use_ema else "sma"
        threshold_label = f"auto_K{self.bars_per_day}_{avg_method}"
        profile_cache_dir = self.output_dir.parent / "profiles"

        symbol_tasks: Dict[str, List[AggregationTask]] = {}
        total_tasks = 0
        skipped = 0
        discarded_symbols = 0

        logger.info(
            f"Phase 1: Building daily dollar volume profiles for {len(symbol_dirs)} symbols..."
        )

        # --- Parallel profile building ---
        # Use ThreadPoolExecutor to build profiles concurrently (I/O bound)
        max_profile_workers = min(8, len(symbol_dirs))
        profiles_map: Dict[str, Tuple[Path, pd.DataFrame]] = {}

        def _build_one_profile(symbol_directory: Path) -> Tuple[str, pd.DataFrame]:
            """Build profile for a single symbol (runs in thread pool)."""
            return symbol_directory.name, build_symbol_profile(
                symbol_directory, cache_dir=profile_cache_dir, force_rebuild=force,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_profile_workers) as executor:
            futures = {
                executor.submit(_build_one_profile, symbol_dir): symbol_dir
                for symbol_dir in symbol_dirs
            }
            progress_bar = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="[Building profiles]",
            )
            for future in progress_bar:
                try:
                    symbol_name, profile = future.result()
                    if not profile.empty:
                        profiles_map[symbol_name] = (futures[future], profile)
                except Exception as error:
                    failed_dir = futures[future]
                    logger.debug(f"Profile build failed for {failed_dir.name}: {error}")
            progress_bar.close()

        logger.info(
            f"Phase 1 done: {len(profiles_map)} profiles built out of {len(symbol_dirs)} symbols"
        )

        # --- Phase 2: Generate tasks from profiles (lightweight, single-threaded) ---
        for symbol_name, (symbol_dir, profile) in sorted(profiles_map.items()):
            tick_files = sorted(symbol_dir.glob("*.parquet"))
            if not tick_files:
                continue

            # Determine first valid date (respects lookback + discard)
            first_valid = get_first_valid_date(
                profile,
                lookback_days=self.lookback_days,
                discard_months=self.discard_months,
            )
            if first_valid is None:
                discarded_symbols += 1
                continue

            # Create tasks for each file after the first valid date
            tasks_for_symbol: List[AggregationTask] = []
            for tick_file in tick_files:
                file_date = extract_file_start_date(tick_file)
                if file_date is None or file_date < first_valid:
                    continue

                # Get dynamic threshold for this month
                month_threshold = get_dynamic_threshold(
                    profile,
                    target_date=file_date,
                    lookback_days=self.lookback_days,
                    bars_per_day=self.bars_per_day,
                    use_ema=self.use_ema,
                )
                if month_threshold is None:
                    continue

                task = AggregationTask(
                    tick_file=tick_file,
                    symbol=symbol_name,
                    bar_type=self.bar_type,
                    threshold=threshold_label,
                    output_dir=self.output_dir,
                )
                # Store the actual numeric threshold for processing
                task._dynamic_threshold = month_threshold

                if task.get_output_path().exists() and not force:
                    skipped += 1
                else:
                    tasks_for_symbol.append(task)

            if tasks_for_symbol:
                symbol_tasks[symbol_name] = tasks_for_symbol
                total_tasks += len(tasks_for_symbol)

        if discarded_symbols > 0:
            logger.info(
                f"Discarded {discarded_symbols} symbols with insufficient history "
                f"(need {self.lookback_days} days + {self.discard_months} month(s))"
            )

        return symbol_tasks, total_tasks, skipped

    # =========================================================================
    # Pipeline execution
    # =========================================================================

    def run_aggregation_pipeline(
        self,
        symbols: Optional[List[str]] = None,
        workers: int = 4,
        force: bool = False,
        task_id: Optional[str] = None,
        task_manager=None,
    ):
        """Run batch aggregation pipeline.

        Routes to the appropriate strategy:
        - Dynamic threshold: per-symbol sequential, cross-symbol parallel
        - Fixed threshold: all tasks via Redis queue
        """
        if self.is_dynamic_threshold:
            self._run_dynamic_pipeline(
                symbols=symbols, workers=workers, force=force,
                task_id=task_id, task_manager=task_manager,
            )
        else:
            self._run_fixed_pipeline(
                symbols=symbols, workers=workers, force=force,
                task_id=task_id, task_manager=task_manager,
            )

    def _run_dynamic_pipeline(
        self,
        symbols: Optional[List[str]] = None,
        workers: int = 4,
        force: bool = False,
        task_id: Optional[str] = None,
        task_manager=None,
    ):
        """Execute dynamic threshold pipeline.

        Each symbol is processed sequentially (monthly order matters),
        but different symbols are processed in parallel.
        """
        symbol_tasks, total_tasks, skipped = self.scan_dynamic_tasks(
            symbols=symbols, force=force,
        )

        logger.info(
            f"Dynamic dollar bar: {len(symbol_tasks)} symbols, "
            f"{total_tasks} tasks to process, {skipped} already aggregated"
        )

        if total_tasks == 0:
            logger.info("No aggregation tasks to process")
            return

        pipeline_start = time.time()
        job_id = task_id or f"agg_{self.bar_type}_auto_{int(time.time())}"
        progress_key = f"{AGGREGATE_PROGRESS_KEY}:{job_id}"

        redis_client = self.redis_client
        redis_client.hset(
            progress_key,
            mapping={
                "job_id": job_id,
                "bar_type": self.bar_type,
                "threshold": f"auto_K{self.bars_per_day}_{'ema' if self.use_ema else 'sma'}",
                "total_tasks": total_tasks,
                "completed": 0,
                "failed": 0,
                "status": "running",
                "started_at": datetime.now().isoformat(),
            },
        )
        redis_client.expire(progress_key, 86400)

        progress_bar = tqdm(total=total_tasks, desc=f"[Aggregate {self.bar_type} auto]")

        # Process each symbol sequentially, symbols in parallel
        def process_symbol(symbol: str, tasks: List[AggregationTask]):
            for task in tasks:
                try:
                    success = self._process_single_task_dynamic(task)
                    if success:
                        redis_client.hincrby(progress_key, "completed", 1)
                    else:
                        redis_client.hincrby(progress_key, "failed", 1)
                except Exception as error:
                    logger.warning(f"Task failed for {task.tick_file.name}: {error}")
                    redis_client.hincrby(progress_key, "failed", 1)
                finally:
                    progress_bar.update(1)
                    self._sync_task_manager_progress(
                        task_manager, task_id, redis_client, progress_key, total_tasks,
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_symbol, symbol, tasks)
                for symbol, tasks in symbol_tasks.items()
            ]
            concurrent.futures.wait(futures)

        progress_bar.close()
        self._finalize_pipeline(
            redis_client, progress_key, pipeline_start, total_tasks,
            task_manager, task_id,
        )

    def _run_fixed_pipeline(
        self,
        symbols: Optional[List[str]] = None,
        workers: int = 4,
        force: bool = False,
        task_id: Optional[str] = None,
        task_manager=None,
    ):
        """Execute fixed threshold pipeline via Redis queue."""
        tasks, skipped = self.scan_tasks(symbols=symbols, force=force)
        total_tasks = len(tasks)

        logger.info(
            f"Batch aggregation: {total_tasks} tasks to process, {skipped} already aggregated"
        )

        if total_tasks == 0:
            logger.info("No aggregation tasks to process")
            return

        pipeline_start = time.time()
        job_id = task_id or f"agg_{self.bar_type}_{int(time.time())}"
        queue_key = f"{AGGREGATE_QUEUE_KEY}:{job_id}"
        progress_key = f"{AGGREGATE_PROGRESS_KEY}:{job_id}"

        redis_client = self.redis_client
        redis_client.delete(queue_key)

        for task in tasks:
            redis_client.lpush(queue_key, json.dumps(task.to_dict()))

        redis_client.hset(
            progress_key,
            mapping={
                "job_id": job_id,
                "bar_type": self.bar_type,
                "threshold": str(self.threshold),
                "total_tasks": total_tasks,
                "completed": 0,
                "failed": 0,
                "status": "running",
                "started_at": datetime.now().isoformat(),
            },
        )
        redis_client.expire(progress_key, 86400)

        workers_done = threading.Event()
        progress_bar = tqdm(total=total_tasks, desc=f"[Aggregate {self.bar_type}]")

        def aggregation_worker():
            while True:
                result = redis_client.brpop(queue_key, timeout=2)
                if result is None:
                    if workers_done.is_set() and redis_client.llen(queue_key) == 0:
                        break
                    continue

                _, task_json = result
                try:
                    task_data = json.loads(task_json)
                    task = AggregationTask.from_dict(task_data)
                    success = self._process_single_task(task)
                    if success:
                        redis_client.hincrby(progress_key, "completed", 1)
                    else:
                        redis_client.hincrby(progress_key, "failed", 1)
                except Exception as error:
                    logger.warning(f"Task failed: {error}")
                    redis_client.hincrby(progress_key, "failed", 1)
                finally:
                    progress_bar.update(1)
                    self._sync_task_manager_progress(
                        task_manager, task_id, redis_client, progress_key, total_tasks,
                    )

        worker_threads = []
        for _ in range(workers):
            thread = threading.Thread(target=aggregation_worker, daemon=True)
            thread.start()
            worker_threads.append(thread)

        workers_done.set()
        for thread in worker_threads:
            thread.join()

        progress_bar.close()
        redis_client.delete(queue_key)
        self._finalize_pipeline(
            redis_client, progress_key, pipeline_start, total_tasks,
            task_manager, task_id,
        )

    # =========================================================================
    # Task processing
    # =========================================================================

    def _process_single_task(self, task: AggregationTask) -> bool:
        """Execute a single aggregation task (fixed threshold)."""
        try:
            from crypto_data_engine.services.bar_aggregator.tick_normalizer import (
                normalize_tick_data,
            )

            tick_data = pd.read_parquet(task.tick_file)
            tick_data = normalize_tick_data(tick_data, source_hint=task.tick_file.name)

            bars = aggregate_bars(
                tick_data, task.bar_type, task.threshold, use_numba=True,
            )

            output_path = task.get_output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            bars.to_parquet(output_path)

            logger.debug(
                f"Aggregated {task.tick_file.name} -> {len(bars)} bars -> {output_path.name}"
            )
            return True
        except Exception as error:
            logger.warning(f"Aggregation failed for {task.tick_file.name}: {error}")
            return False

    def _process_single_task_dynamic(self, task: AggregationTask) -> bool:
        """Execute a single aggregation task with dynamic threshold.

        Uses the pre-computed _dynamic_threshold stored on the task object.
        """
        try:
            from crypto_data_engine.services.bar_aggregator.tick_normalizer import (
                normalize_tick_data,
            )

            dynamic_threshold = getattr(task, "_dynamic_threshold", None)
            if dynamic_threshold is None:
                logger.warning(f"No dynamic threshold for {task.tick_file.name}, skipping")
                return False

            tick_data = pd.read_parquet(task.tick_file)
            tick_data = normalize_tick_data(tick_data, source_hint=task.tick_file.name)

            bars = aggregate_bars(
                tick_data, task.bar_type, dynamic_threshold, use_numba=True,
            )

            output_path = task.get_output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            bars.to_parquet(output_path)

            logger.debug(
                f"Aggregated {task.tick_file.name} -> {len(bars)} bars "
                f"(threshold={dynamic_threshold:,.0f}) -> {output_path.name}"
            )
            return True
        except Exception as error:
            logger.warning(f"Aggregation failed for {task.tick_file.name}: {error}")
            return False

    # =========================================================================
    # Helpers
    # =========================================================================

    def _finalize_pipeline(
        self,
        redis_client: redis.Redis,
        progress_key: str,
        pipeline_start: float,
        total_tasks: int,
        task_manager=None,
        task_id: Optional[str] = None,
    ):
        """Finalize pipeline: update progress, log summary, update TaskManager."""
        elapsed = time.time() - pipeline_start
        progress = redis_client.hgetall(progress_key)
        redis_client.hset(
            progress_key,
            mapping={
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "elapsed_seconds": f"{elapsed:.1f}",
            },
        )

        logger.info(
            f"Batch aggregation finished: "
            f"completed={progress.get('completed', 0)}, "
            f"failed={progress.get('failed', 0)}, "
            f"elapsed={elapsed:.1f}s"
        )

        if task_manager and task_id:
            from crypto_data_engine.common.task_manager import TaskStatus
            task_manager.update_task(
                task_id,
                status=TaskStatus.COMPLETED,
                progress=1.0,
                message=f"Done: {progress.get('completed', 0)}/{total_tasks} aggregated",
                result={
                    "completed": int(progress.get("completed", 0)),
                    "failed": int(progress.get("failed", 0)),
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

    @staticmethod
    def _sync_task_manager_progress(
        task_manager,
        task_id: Optional[str],
        redis_client: redis.Redis,
        progress_key: str,
        total_tasks: int,
    ) -> None:
        """Push current progress to TaskManager (for API polling)."""
        if not task_manager or not task_id:
            return
        try:
            completed = int(redis_client.hget(progress_key, "completed") or 0)
            failed = int(redis_client.hget(progress_key, "failed") or 0)
            done = completed + failed
            progress_ratio = done / total_tasks if total_tasks > 0 else 0.0
            task_manager.update_task(
                task_id,
                progress=progress_ratio,
                message=f"Aggregated: {completed}/{total_tasks}, failed: {failed}",
            )
        except Exception:
            pass

    def get_pipeline_progress(self, job_id: str) -> Optional[Dict]:
        """Query live pipeline progress from Redis."""
        progress_key = f"{AGGREGATE_PROGRESS_KEY}:{job_id}"
        progress = self.redis_client.hgetall(progress_key)
        return progress if progress else None
