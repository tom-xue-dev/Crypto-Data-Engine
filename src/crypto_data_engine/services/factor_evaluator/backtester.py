"""
Simple long-short factor backtest.

Long top quantile, short bottom quantile, equal weight, with transaction costs.

Uses actual prices to compute holding-period returns (not pre-computed
forward returns from factor_data), avoiding overlap and look-ahead issues.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from crypto_data_engine.common.logger.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for long-short backtest."""
    cost_bps: float = 5.0         # one-way transaction cost in basis points
    long_quantile: Optional[int] = None   # default: max quantile
    short_quantile: int = 1       # default: bottom quantile


def backtest_long_short(
    factor_data: pd.DataFrame,
    price_df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
) -> Dict[str, Any]:
    """Run a simple long-short factor backtest.

    At each rebalance date t[i], we observe factor quantiles and form a
    portfolio that is held until the next rebalance date t[i+1].

    Logic:
      1. At t[i], long assets in top quantile, short assets in bottom quantile
         (equal weight within each leg).
      2. Hold from t[i] to t[i+1].
      3. Holding-period return computed from price_df (actual close prices).
      4. Transaction cost = one-way cost_bps × turnover (weight changes).

    Args:
        factor_data: Output of FactorAnalyzer.prepare_data().
                     MultiIndex (date, asset) with 'factor_quantile' column.
        price_df: Wide DataFrame (index=timestamps, columns=symbols) of
                  close prices, aligned with factor_data dates.
        config: Backtest parameters.

    Returns:
        Dict with keys: nav, returns, turnover, performance, config.
    """
    cfg = config or BacktestConfig()

    q_max = int(factor_data["factor_quantile"].max())
    long_q = cfg.long_quantile if cfg.long_quantile is not None else q_max
    short_q = cfg.short_quantile
    cost_frac = cfg.cost_bps / 10_000

    dates = factor_data.index.get_level_values("date").unique().sort_values()

    # Align timezone: factor_data dates are tz-naive (alphalens strips tz),
    # but price_df may be tz-aware (UTC).
    if price_df.index.tz is not None and dates.tz is None:
        dates = dates.tz_localize(price_df.index.tz)
    elif price_df.index.tz is None and dates.tz is not None:
        dates = dates.tz_localize(None)

    nav = 1.0
    nav_records = []
    return_records = []
    turnover_records = []
    prev_weights: Dict[str, float] = {}

    # tz-naive dates for indexing into factor_data (alphalens format)
    dates_naive = dates.tz_localize(None) if dates.tz is not None else dates

    for i in range(len(dates) - 1):
        dt = dates[i]           # for price_df (may be tz-aware)
        next_dt = dates[i + 1]
        dt_naive = dates_naive[i]  # for factor_data (always tz-naive)

        cs = factor_data.loc[dt_naive]
        if isinstance(cs, pd.Series):
            continue

        long_mask = cs["factor_quantile"] == long_q
        short_mask = cs["factor_quantile"] == short_q

        long_assets = cs.index[long_mask].tolist()
        short_assets = cs.index[short_mask].tolist()

        # Target weights: long leg +1 total, short leg -1 total
        new_weights: Dict[str, float] = {}
        if long_assets:
            w = 1.0 / len(long_assets)
            for a in long_assets:
                new_weights[a] = w
        if short_assets:
            w = 1.0 / len(short_assets)
            for a in short_assets:
                new_weights[a] = new_weights.get(a, 0) - w

        # Turnover
        all_assets = set(prev_weights) | set(new_weights)
        turnover = sum(
            abs(new_weights.get(a, 0.0) - prev_weights.get(a, 0.0))
            for a in all_assets
        )
        cost = turnover * cost_frac

        # Holding-period return from actual prices: t[i] → t[i+1]
        long_ret = _holding_return(price_df, long_assets, dt, next_dt)
        short_ret = _holding_return(price_df, short_assets, dt, next_dt)
        port_ret = long_ret - short_ret

        nav *= (1 + port_ret - cost)
        nav_records.append((next_dt, nav))
        return_records.append((dt, port_ret, cost, long_ret, short_ret))
        turnover_records.append((dt, turnover))

        prev_weights = new_weights

    # Build result DataFrames
    nav_series = pd.Series(
        dict(nav_records), name="nav"
    )
    returns_df = pd.DataFrame(
        return_records,
        columns=["date", "gross_return", "cost", "long_return", "short_return"],
    ).set_index("date")
    returns_df["net_return"] = returns_df["gross_return"] - returns_df["cost"]

    turnover_series = pd.Series(
        dict(turnover_records), name="turnover"
    )

    # Performance metrics
    from crypto_data_engine.core.base import calculate_performance_metrics
    perf = calculate_performance_metrics(nav_series)
    perf["total_cost"] = float(returns_df["cost"].sum())
    perf["avg_turnover"] = float(turnover_series.mean())
    perf["n_rebalances"] = len(nav_records)
    perf["long_quantile"] = long_q
    perf["short_quantile"] = short_q

    logger.info(
        f"Backtest done: {len(nav_records)} rebalances, "
        f"return={perf['total_return']:.2%}, "
        f"sharpe={perf['sharpe_ratio']:.2f}, "
        f"mdd={perf['max_drawdown']:.2%}, "
        f"total_cost={perf['total_cost']:.4f}"
    )

    return {
        "nav": nav_series,
        "returns": returns_df,
        "turnover": turnover_series,
        "performance": perf,
        "config": cfg,
    }


def _holding_return(
    price_df: pd.DataFrame,
    assets: list,
    entry_dt: pd.Timestamp,
    exit_dt: pd.Timestamp,
) -> float:
    """Equal-weight average return of *assets* from entry_dt to exit_dt."""
    if not assets:
        return 0.0

    valid_assets = [a for a in assets if a in price_df.columns]
    if not valid_assets:
        return 0.0

    p0 = price_df.loc[entry_dt, valid_assets]
    p1 = price_df.loc[exit_dt, valid_assets]

    rets = (p1 / p0 - 1).dropna()
    return float(rets.mean()) if len(rets) > 0 else 0.0
