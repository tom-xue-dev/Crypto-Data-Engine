"""
Pure factor computation. No data loading, no analysis, no IO.

Input:  panel DataFrame with MultiIndex (timestamp, symbol)
Output: Dict[factor_name, Series with same MultiIndex]

Supports three factor types:
- RAW: directly use a bar data column
- ROLLING: vectorized matrix rolling (zscore, rank, mean, std, change, diff)
- CUSTOM: user-defined function (DataFrame -> Series)

Performance:
- ROLLING uses unstack -> rolling -> stack (vectorized across all symbols)
- Numba JIT for zscore if available
- ThreadPoolExecutor for parallel multi-factor computation
"""
from __future__ import annotations

import concurrent.futures
from typing import Dict, List

import numpy as np
import pandas as pd

from crypto_data_engine.common.config.factor_config import (
    BUILTIN_FACTORS,
    FactorConfig,
    FactorType,
    RollingMethod,
)
from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _nb_rolling_zscore(arr2d: np.ndarray, window: int) -> np.ndarray:
        """Rolling z-score on 2D array (rows=time, cols=symbol), parallel over cols."""
        n_rows, n_cols = arr2d.shape
        out = np.full((n_rows, n_cols), np.nan)
        for j in prange(n_cols):
            for i in range(n_rows):
                start = max(0, i - window + 1)
                count = 0
                s = 0.0
                for k in range(start, i + 1):
                    v = arr2d[k, j]
                    if not np.isnan(v):
                        s += v
                        count += 1
                if count < 2:
                    continue
                mean = s / count
                var = 0.0
                for k in range(start, i + 1):
                    v = arr2d[k, j]
                    if not np.isnan(v):
                        var += (v - mean) ** 2
                std = np.sqrt(var / (count - 1))
                cur = arr2d[i, j]
                if std > 0.0 and not np.isnan(cur):
                    out[i, j] = (cur - mean) / std
        return out


# ---------------------------------------------------------------------------
# Calculator — pure computation, no IO
# ---------------------------------------------------------------------------

class FactorCalculator:
    """Compute factors from a panel DataFrame.

    Does NOT load data. Receives panel, returns factors.
    """

    def compute(
        self,
        panel: pd.DataFrame,
        factor_configs: List[FactorConfig],
        workers: int = 4,
    ) -> Dict[str, pd.Series]:
        """Compute multiple factors in parallel.

        Args:
            panel: MultiIndex (timestamp, symbol) DataFrame with bar data.
            factor_configs: List of factor definitions.
            workers: Thread count for parallel computation.

        Returns:
            Dict mapping factor_name -> Series with same MultiIndex.
        """
        results: Dict[str, pd.Series] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.compute_single, panel, config): config.name
                for config in factor_configs
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.error(f"Failed to compute factor '{name}': {exc}")

        logger.info(f"Computed {len(results)} factors (workers={workers}, numba={NUMBA_AVAILABLE})")
        return results

    def compute_single(
        self, panel: pd.DataFrame, config: FactorConfig
    ) -> pd.Series:
        """Compute one factor from panel data."""
        if config.factor_type == FactorType.RAW:
            return self._compute_raw(panel, config)
        elif config.factor_type == FactorType.ROLLING:
            return self._compute_rolling(panel, config)
        elif config.factor_type == FactorType.CUSTOM:
            return self._compute_custom(panel, config)
        else:
            raise ValueError(f"Unknown factor type: {config.factor_type}")

    # ------------------------------------------------------------------
    # Built-in factor presets
    # ------------------------------------------------------------------

    @staticmethod
    def builtin_factors() -> List[FactorConfig]:
        """Return a list of built-in factor configs."""
        return list(BUILTIN_FACTORS)

    @staticmethod
    def collect_columns(configs: List[FactorConfig]) -> set:
        """Gather all bar columns needed by factor configs."""
        return {c.column for c in configs if c.column}

    # ------------------------------------------------------------------
    # Internal computation methods
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_raw(panel: pd.DataFrame, config: FactorConfig) -> pd.Series:
        if config.column not in panel.columns:
            raise KeyError(f"Column '{config.column}' not found in data")
        series = panel[config.column].copy()
        series.name = config.name
        return series

    @staticmethod
    def _compute_rolling(panel: pd.DataFrame, config: FactorConfig) -> pd.Series:
        if config.column not in panel.columns:
            raise KeyError(f"Column '{config.column}' not found in data")
        if config.rolling_method is None:
            raise ValueError("rolling_method must be set for ROLLING factors")

        method = config.rolling_method
        window = config.window

        # Vectorized: pivot to (timestamp x symbol) matrix, apply rolling, stack back
        wide = panel[config.column].unstack(level="symbol")

        if method == RollingMethod.ZSCORE:
            if NUMBA_AVAILABLE:
                arr = wide.to_numpy(dtype=np.float64)
                result_arr = _nb_rolling_zscore(arr, window)
                result_wide = pd.DataFrame(result_arr, index=wide.index, columns=wide.columns)
            else:
                roll_mean = wide.rolling(window, min_periods=2).mean()
                roll_std = wide.rolling(window, min_periods=2).std()
                result_wide = (wide - roll_mean) / roll_std.replace(0, np.nan)
        elif method == RollingMethod.MEAN:
            result_wide = wide.rolling(window, min_periods=1).mean()
        elif method == RollingMethod.STD:
            result_wide = wide.rolling(window, min_periods=2).std()
        elif method == RollingMethod.CHANGE:
            result_wide = wide.pct_change(window)
        elif method == RollingMethod.DIFF:
            result_wide = wide.diff(window)
        elif method == RollingMethod.RANK:
            result_wide = wide.apply(
                lambda col: col.rolling(window, min_periods=1).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
            )
        else:
            raise ValueError(f"Unknown rolling method: {method}")

        result = result_wide.stack(future_stack=True)
        result.index.names = ["timestamp", "symbol"]
        result.name = config.name
        return result.reindex(panel.index)

    @staticmethod
    def _compute_custom(panel: pd.DataFrame, config: FactorConfig) -> pd.Series:
        if config.func is None:
            raise ValueError("func must be set for CUSTOM factors")
        result = panel.groupby(level="symbol", group_keys=False).apply(config.func)
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
        result.name = config.name
        return result
