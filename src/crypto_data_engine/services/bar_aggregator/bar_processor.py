import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from tqdm import tqdm
from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.db.constants import TaskStatus
from crypto_data_engine.db.repository.aggregate import AggregateTaskRepository
from crypto_data_engine.db.repository.download import DownloadTaskRepository
from crypto_data_engine.services.bar_aggregator import BarConstructor

logger = get_logger(__name__)


@dataclass
class BarProcessorContext:
    """Configuration required by :class:`BarProcessor`.
    Parameters correspond to the keys used in the configuration dictionaries
    passed around the project.  Only a subset is required for the simplified
    implementation, remaining values are stored for completeness.
    """
    raw_data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    bar_type: str = "volume"
    threshold: int = 10000000
    process_num_limit: int = 4
    suffix_filter: Optional[str] = None
    adaptive: bool = False
    sample_days: int = 1
    target_bars: int = 300
    batch_size: int = 10


class BarProcessor:
    """Process tick data and generate bar files."""

    def __init__(self, context: BarProcessorContext) -> None:
        self.context = context
        self.processed_files: set[str] = set()

    # ------------------------------------------------------------------
    def run_bar_generation_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Entry point for bar generation.
        The method discovers symbols in the ``raw_data_dir`` and processes
        them one by one.  Results for each symbol are collected and returned
        as a dictionary for easy introspection.
        """

        logger.info("🚀 Starting bar generation pipeline")
        logger.info("📊 Bar type: %s", self.context.bar_type)
        logger.info("📂 Input directory: %s", self.context.raw_data_dir)
        logger.info("📂 Output directory: %s", self.context.output_dir)
        exchange_name = config.get("exchange")
        symbols = self._get_symbols_to_process(exchange_name)
        if not symbols:
            logger.warning("⚠️  No symbols found for processing")
            return {"status": "completed", "processed": 0, "message": "No files to process"}

        os.makedirs(self.context.output_dir, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for symbol in tqdm(symbols, desc="Processing symbols"):
            try:
                result = self._process_symbol(symbol)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("❌ %s processing failed: %s", symbol, exc)
                continue

            if result:
                results.append(result)
                self.processed_files.add(symbol)
                logger.info("✅ %s processed: %s bars", symbol, result["bars_count"])

        return {
            "status": "completed",
            "processed": len(results),
            "total_symbols": len(symbols),
            "results": results,
            "bar_type": self.context.bar_type,
            "threshold": self.context.threshold,
        }

    # ------------------------------------------------------------------
    def _get_symbols_to_process(self,exchange_name:str) -> List[str]:
        """Discover which symbols should be processed."""
        task_extracted = DownloadTaskRepository.get_all_tasks(exchange=exchange_name,status=TaskStatus.COMPLETED)
        task_aggregated = AggregateTaskRepository.get_all_tasks(exchange=exchange_name,status=TaskStatus.COMPLETED)
        tasks = task_extracted - task_aggregated
        symbols = None
        if self.context.suffix_filter:
            symbols = [task.symbol for task in tasks if task.symbol.endswith(self.context.suffix_filter)]
        logger.info("📋 %d symbols to process: %s", len(symbols), symbols[:10])
        return symbols

    # ------------------------------------------------------------------
    def _process_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load all parquet tick files for ``symbol`` and build bars."""
        symbol_path = Path(self.context.raw_data_dir) / symbol
        if not symbol_path.exists() or not symbol_path.is_dir():
            logger.warning("⚠️  Symbol directory missing: %s", symbol_path)
            return None

        parquet_files = sorted(symbol_path.glob("*.parquet"))
        if not parquet_files:
            logger.warning("⚠️  No parquet files found for %s", symbol)
            return None

        constructor = BarConstructor(
            folder_path=parquet_files,
            threshold=self.context.threshold,
            bar_type=self.context.bar_type,
        )

        bars_df = constructor.process_asset_data()
        if bars_df is None or bars_df.empty:
            logger.warning("⚠️  %s produced no bars", symbol)
            return None

        output_file = self._save_bars(symbol, bars_df)
        return {
            "symbol": symbol,
            "bars_count": len(bars_df),
            "output_file": str(output_file),
            "bar_type": self.context.bar_type,
            "threshold": self.context.threshold,
            "processed_at": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    def _save_bars(self, symbol: str, bars_df: pd.DataFrame) -> Path:
        """Persist the generated bars to ``data/bar_data/<symbol>/``."""

        symbol_output_dir = Path(self.context.output_dir) / symbol
        symbol_output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{symbol}_{self.context.bar_type}_{self.context.threshold}.parquet"
        output_file = symbol_output_dir / filename
        bars_df.to_parquet(output_file, compression="brotli", index=False)
        logger.info("💾 %s bars saved to %s", symbol, output_file)
        return output_file

    # ------------------------------------------------------------------
    def get_processing_stats(self) -> Dict[str, Any]:
        """Return statistics about the processed symbols."""

        return {
            "processed_files": len(self.processed_files),
            "context": {
                "bar_type": self.context.bar_type,
                "threshold": self.context.threshold,
                "raw_data_dir": self.context.raw_data_dir,
                "output_dir": self.context.output_dir,
            },
        }


# ---------------------------------------------------------------------------
# Convenience wrapper used by tests / scripts
# ---------------------------------------------------------------------------


def run_simple_bar_generation(
    exchange_name: str = "binance",
    symbols: Optional[List[str]] = None,
    bar_type: str = "volume",
    threshold: int = 10_000_000,
    raw_data_dir: str = "./data/tick_data",
    output_dir: str = "./data/bar_data",
) -> Dict[str, Any]:
    """Simplified interface similar to ``run_simple_download``.

    Parameters mirror those used in the original project.  A configuration
    dictionary is created and passed to :class:`BarProcessor`.
    """

    logger.info("\n🚀 Starting %s bar generation", exchange_name.upper())
    logger.info("📊 Bar type: %s", bar_type)
    logger.info("🎯 Threshold: %s", threshold)

    config = {
        "raw_data_dir": f"{raw_data_dir}/{exchange_name}",
        "output_dir": f"{output_dir}/{exchange_name}",
        "bar_type": bar_type,
        "threshold": threshold,
        "symbols": symbols,
        "process_num_limit": 4,
        "batch_size": 10,
    }

    try:
        context = BarProcessorContext(**config)
        processor = BarProcessor(context)
        result = processor.run_bar_generation_pipeline(config)
        logger.info("\n🎉 %s bar generation finished", exchange_name.upper())
        logger.info("📊 Result: %s", result)
        return result
    except Exception as exc:  # pragma: no cover - logging
        logger.error("❌ Bar generation failed: %s", exc)
        raise


__all__ = [
    "BarConstructor",
    "BarProcessor",
    "BarProcessorContext",
    "run_simple_bar_generation",
]

if __name__ == "__main__":
    run_simple_bar_generation(exchange_name="binance", threshold=10000000)