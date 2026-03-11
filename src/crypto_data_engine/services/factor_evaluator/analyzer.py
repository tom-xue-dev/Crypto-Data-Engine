"""
Pure factor analysis. No visualization, no file IO.

Input:  factor Series + price DataFrame
Output: metrics dict (IC, ICIR, quantile returns, turnover)

Uses alphalens for quantile binning and turnover.
IC / forward returns / quantile returns computed manually
to avoid alphalens compatibility issues with pandas 2.x + intraday data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import alphalens.performance as al_perf
import alphalens.utils as al_utils

from crypto_data_engine.common.config.factor_config import AnalysisConfig
from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)


class FactorAnalyzer:
    """Compute factor metrics. No charts, no file IO."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    def prepare_data(
        self,
        factor_series: pd.Series,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert factor + prices into alphalens factor_data format.

        Handles intraday data by computing forward returns manually
        to bypass alphalens' trading calendar inference.
        """
        # alphalens expects tz-naive datetimes
        factor = factor_series.copy()
        if hasattr(factor.index, "levels"):
            ts_level = factor.index.get_level_values(0)
            if hasattr(ts_level, "tz") and ts_level.tz is not None:
                factor.index = factor.index.set_levels(
                    factor.index.levels[0].tz_localize(None), level=0
                )

        prices = price_df.copy()
        if hasattr(prices.index, "tz") and prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        factor = factor.replace([np.inf, -np.inf], np.nan).dropna()
        factor.index = factor.index.set_names(["date", "asset"])

        forward_returns = self._compute_forward_returns(factor, prices)

        quantile_kw = {}
        if self.config.bins is not None:
            quantile_kw["bins"] = self.config.bins
        else:
            quantile_kw["quantiles"] = self.config.quantiles

        factor_data = al_utils.get_clean_factor(
            factor, forward_returns,
            max_loss=self.config.max_loss,
            **quantile_kw,
        )
        return factor_data

    def analyze(
        self,
        factor_name: str,
        factor_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Analyze a single factor. Returns metrics dict.

        Keys: factor_name, ic, mean_ic, ic_ir, quantile_returns, turnover
        """
        results: Dict[str, Any] = {"factor_name": factor_name}
        period_cols = [
            c for c in factor_data.columns
            if c not in ("factor", "factor_quantile", "group")
        ]

        ic = self._compute_ic(factor_data)
        results["ic"] = ic
        results["cumulative_ic"] = ic.cumsum()
        results["mean_ic"] = ic.mean()
        ic_std = ic.std()
        results["ic_ir"] = ic.mean() / ic_std.replace(0, np.nan)

        results["quantile_returns"] = self._compute_quantile_returns(
            factor_data, period_cols
        )

        turnover = {}
        for p in period_cols:
            try:
                turnover[p] = al_perf.quantile_turnover(factor_data, p)
            except Exception:
                pass
        results["turnover"] = turnover

        return results

    def analyze_batch(
        self,
        factors: Dict[str, pd.Series],
        price_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple factors."""
        results = {}
        total = len(factors)
        for i, (name, series) in enumerate(factors.items(), 1):
            logger.info(f"[{i}/{total}] Analyzing factor: {name}")
            try:
                factor_data = self.prepare_data(series, price_df)
                results[name] = self.analyze(name, factor_data)
            except Exception as exc:
                logger.error(f"Failed to analyze factor '{name}': {exc}")
        return results

    # ------------------------------------------------------------------
    # Internal computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ic(factor_data: pd.DataFrame) -> pd.DataFrame:
        """Spearman IC per period per timestamp."""
        from scipy.stats import spearmanr

        period_cols = [
            c for c in factor_data.columns
            if c not in ("factor", "factor_quantile", "group")
        ]
        ic_data = {}
        grouped = factor_data.groupby(level="date")
        for period_col in period_cols:
            ics = {}
            for dt, group in grouped:
                f = group["factor"]
                r = group[period_col]
                mask = f.notna() & r.notna()
                if mask.sum() >= 3:
                    corr, _ = spearmanr(f[mask], r[mask])
                    ics[dt] = corr
            ic_data[period_col] = pd.Series(ics)
        return pd.DataFrame(ic_data)

    def _compute_forward_returns(
        self,
        factor: pd.Series,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Manual forward returns (bypasses alphalens calendar inference)."""
        forward_returns_dict = {}
        for period in self.config.periods:
            fwd = prices.shift(-period)
            ret = (fwd / prices) - 1.0
            ret_stacked = ret.stack()
            ret_stacked.index.names = ["date", "asset"]
            forward_returns_dict[f"{period}"] = ret_stacked

        forward_returns = pd.DataFrame(forward_returns_dict)
        common_idx = factor.index.intersection(forward_returns.index)
        return forward_returns.loc[common_idx]

    @staticmethod
    def _compute_quantile_returns(
        factor_data: pd.DataFrame, period_cols: List[str]
    ) -> pd.DataFrame:
        """Mean forward return by quantile."""
        return factor_data.groupby("factor_quantile")[period_cols].mean()
