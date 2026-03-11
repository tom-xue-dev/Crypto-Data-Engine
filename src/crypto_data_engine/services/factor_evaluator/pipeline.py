"""
Factor pipeline: orchestrates load → compute → analyze → report.

This is the only module that knows about the full data flow.
Individual components (Calculator, Analyzer, Reporter) are decoupled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.services.data_manager.data_loader import BarDataLoader
from crypto_data_engine.services.data_manager.data_transforms import resample_to_rebalance
from crypto_data_engine.common.config.factor_config import FactorConfig
from crypto_data_engine.services.factor_evaluator.calculator import FactorCalculator
from crypto_data_engine.services.factor_evaluator.analyzer import FactorAnalyzer
from crypto_data_engine.services.factor_evaluator.reporter import FactorReporter
from crypto_data_engine.services.factor_evaluator.backtester import (
    BacktestConfig,
    backtest_long_short,
)

logger = get_logger(__name__)


class FactorPipeline:
    """Orchestrates the full factor evaluation workflow.

    load_data  → panel DataFrame
    compute    → Dict[name, Series]
    analyze    → Dict[name, metrics]
    report     → CSV + PNG
    """

    def __init__(
        self,
        loader: Optional[BarDataLoader] = None,
        calculator: Optional[FactorCalculator] = None,
        analyzer: Optional[FactorAnalyzer] = None,
        reporter: Optional[FactorReporter] = None,
    ):
        self.loader = loader or BarDataLoader()
        self.calculator = calculator or FactorCalculator()
        self.analyzer = analyzer or FactorAnalyzer()
        self.reporter = reporter or FactorReporter()

    def run(
        self,
        start_date: str,
        end_date: str,
        factor_configs: List[FactorConfig],
        symbols: Optional[List[str]] = None,
        warmup_periods: int = 60,
        bar_type: str = "time",
        interval: str = "1h",
        columns: Optional[List[str]] = None,
        workers: int = 4,
        output_dir: Optional[Path] = None,
        charts: bool = True,
        backtest: Optional[BacktestConfig] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
        """End-to-end: load → compute → analyze → [backtest] → report.

        Args:
            columns: Extra columns to load (on top of those inferred from
                     factor_configs). Useful for CUSTOM factors whose
                     required columns can't be auto-detected.
            backtest: If provided, run a simple long-short backtest per factor.
                      Results are added to each factor's metrics dict under
                      the key ``"backtest"``.

        Returns:
            (batch_results, summary_table)
        """
        # 1. Load
        panel, price_df = self.load_data(
            start_date, end_date, factor_configs,
            symbols=symbols, warmup_periods=warmup_periods,
            bar_type=bar_type, interval=interval,
            extra_columns=columns,
        )
        if panel.empty:
            return {}, pd.DataFrame()

        # 2. Compute
        factors = self.calculator.compute(panel, factor_configs, workers=workers)

        # 3. Trim warmup
        factors, price_df = self._trim_warmup(
            factors, price_df, start_date, warmup_periods
        )

        if not factors:
            return {}, pd.DataFrame()

        # 4. Resample to rebalancing frequency (if configured)
        rebal_freq = self.analyzer.config.rebalance_freq
        if rebal_freq:
            factors, price_df = resample_to_rebalance(
                factors, price_df, rebal_freq
            )

        # 5. Analyze
        batch_results = self.analyzer.analyze_batch(factors, price_df)

        # 6. Backtest (optional)
        if backtest and batch_results:
            for name, result in batch_results.items():
                factor_series = factors.get(name)
                if factor_series is None:
                    continue
                try:
                    factor_data = self.analyzer.prepare_data(
                        factor_series, price_df
                    )
                    result["backtest"] = backtest_long_short(
                        factor_data, price_df, backtest
                    )
                except Exception as exc:
                    logger.error(f"Backtest failed for '{name}': {exc}")

        # 7. Report
        summary = pd.DataFrame()
        if batch_results and output_dir:
            out = Path(output_dir)
            self.reporter.export(batch_results, out, charts=charts)
            summary = self.reporter.summary_table(batch_results)

        return batch_results, summary

    def load_data(
        self,
        start_date: str,
        end_date: str,
        factor_configs: List[FactorConfig],
        symbols: Optional[List[str]] = None,
        warmup_periods: int = 60,
        bar_type: str = "time",
        interval: str = "1h",
        extra_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load panel data and extract price matrix.

        Returns:
            (panel, price_df) — panel includes warmup rows, price_df does NOT.
        """
        needed_cols = FactorCalculator.collect_columns(factor_configs)
        needed_cols.add("close")
        if extra_columns:
            needed_cols.update(extra_columns)
        col_list = sorted(needed_cols)

        warmup_start = self._shift_start(start_date, warmup_periods, interval)

        logger.info(
            f"Loading {'all' if not symbols else len(symbols)} symbols, "
            f"{warmup_start} -> {end_date}, {len(col_list)} columns"
        )
        panel = self.loader.load_panel(
            symbols, warmup_start, end_date,
            bar_type=bar_type, interval=interval, columns=col_list,
        )

        if panel.empty:
            logger.warning("No data loaded")
            return pd.DataFrame(), pd.DataFrame()

        # Extract price matrix (trimmed to actual date range, no warmup)
        close_wide = panel["close"].unstack("symbol")
        cutoff = pd.Timestamp(start_date, tz="UTC")
        price_df = close_wide[close_wide.index >= cutoff].sort_index()

        return panel, price_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_warmup(
        factors: Dict[str, pd.Series],
        price_df: pd.DataFrame,
        start_date: str,
        warmup_periods: int,
    ) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
        """Remove warmup rows from computed factors."""
        if warmup_periods <= 0:
            return factors, price_df

        cutoff = pd.Timestamp(start_date, tz="UTC")
        trimmed = {}
        for name, s in factors.items():
            trimmed[name] = s[s.index.get_level_values("timestamp") >= cutoff]
        return trimmed, price_df

    @staticmethod
    def _shift_start(start_date: str, periods: int, interval: str) -> str:
        """Shift start date back by warmup periods."""
        ts = pd.Timestamp(start_date)
        if interval.endswith("h"):
            shifted = ts - pd.Timedelta(hours=int(interval[:-1]) * periods)
        elif interval.endswith("m"):
            shifted = ts - pd.Timedelta(minutes=int(interval[:-1]) * periods)
        else:
            shifted = ts - pd.Timedelta(days=periods)
        return shifted.strftime("%Y-%m")
