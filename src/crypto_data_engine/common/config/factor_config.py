"""
Factor definitions: types, methods, configs, and built-in presets.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import pandas as pd


class FactorType(Enum):
    RAW = "raw"
    ROLLING = "rolling"
    CUSTOM = "custom"


class RollingMethod(Enum):
    ZSCORE = "zscore"
    RANK = "rank"
    MEAN = "mean"
    STD = "std"
    CHANGE = "change"
    DIFF = "diff"


@dataclass
class FactorConfig:
    """Configuration for a single factor.

    Examples:
        FactorConfig(name="amihud", column="amihud")
        FactorConfig(name="amihud_z60", column="amihud",
                     factor_type=FactorType.ROLLING,
                     rolling_method=RollingMethod.ZSCORE, window=60)
        FactorConfig(name="custom1", factor_type=FactorType.CUSTOM,
                     func=lambda df: df["close"].pct_change(20) * df["volume"])
    """
    name: str
    factor_type: FactorType = FactorType.RAW
    column: Optional[str] = None
    rolling_method: Optional[RollingMethod] = None
    window: int = 20
    func: Optional[Callable[[pd.DataFrame], pd.Series]] = None


@dataclass
class AnalysisConfig:
    """Configuration for factor analysis.

    Args:
        periods: Forward return periods (in rebalancing units if rebalance_freq set,
                 otherwise in bar units).
        quantiles: Number of quantile groups for factor binning.
        bins: If set, use fixed bins instead of quantiles.
        max_loss: Max fraction of data allowed to be dropped by alphalens.
        rebalance_freq: Resample factor/price to every N bars before analysis.
                        e.g. 24 with 1h bars = daily rebalancing.
                        None = use every bar (default).
    """
    periods: Tuple[int, ...] = (1, 5, 10, 20)
    quantiles: int = 5
    bins: Optional[int] = None
    max_loss: float = 0.5
    rebalance_freq: Optional[int] = None


# ---------------------------------------------------------------------------
# Built-in factor presets
# ---------------------------------------------------------------------------

BUILTIN_RAW_FACTORS = [
    FactorConfig(name="amihud", column="amihud"),
    FactorConfig(name="rv_amihud", column="rv_amihud"),
    FactorConfig(name="path_efficiency", column="path_efficiency"),
    FactorConfig(name="vwap_gap", column="vwap_gap"),
    FactorConfig(name="signed_volume_imbalance", column="signed_volume_imbalance"),
    FactorConfig(name="aggressive_buy_ratio", column="aggressive_buy_ratio"),
    FactorConfig(name="trade_intensity", column="trade_intensity"),
    FactorConfig(name="return_1h", column="return"),
    FactorConfig(name="realized_vol", column="realized_vol"),
    FactorConfig(name="buy_volume_share", column="buy_volume_share"),
    FactorConfig(name="impact_imbalance", column="impact_imbalance"),
    FactorConfig(name="close_location", column="close_location"),
    FactorConfig(name="downside_upside_vol_ratio", column="downside_upside_vol_ratio"),
    FactorConfig(name="path_churn", column="path_churn"),
]

BUILTIN_ROLLING_FACTORS = [
    FactorConfig(name="amihud_z60", column="amihud",
                 factor_type=FactorType.ROLLING,
                 rolling_method=RollingMethod.ZSCORE, window=60),
    FactorConfig(name="momentum_20", column="close",
                 factor_type=FactorType.ROLLING,
                 rolling_method=RollingMethod.CHANGE, window=20),
    FactorConfig(name="momentum_60", column="close",
                 factor_type=FactorType.ROLLING,
                 rolling_method=RollingMethod.CHANGE, window=60),
    FactorConfig(name="vol_z20", column="realized_vol",
                 factor_type=FactorType.ROLLING,
                 rolling_method=RollingMethod.ZSCORE, window=20),
    FactorConfig(name="imbalance_ma20", column="signed_volume_imbalance",
                 factor_type=FactorType.ROLLING,
                 rolling_method=RollingMethod.MEAN, window=20),
    FactorConfig(name="imbalance_ma60", column="signed_volume_imbalance",
                 factor_type=FactorType.ROLLING,
                 rolling_method=RollingMethod.MEAN, window=60),
    FactorConfig(name="buy_ratio_z20", column="aggressive_buy_ratio",
                 factor_type=FactorType.ROLLING,
                 rolling_method=RollingMethod.ZSCORE, window=20),
    FactorConfig(name="intensity_z60", column="trade_intensity",
                 factor_type=FactorType.ROLLING,
                 rolling_method=RollingMethod.ZSCORE, window=60),
]

BUILTIN_FACTORS: List[FactorConfig] = BUILTIN_RAW_FACTORS + BUILTIN_ROLLING_FACTORS
