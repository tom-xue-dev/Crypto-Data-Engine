"""
Visualization API router.

Endpoints for retrieving chart data from backtest results and bar OHLCV for charts.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/viz", tags=["visualization"])

DEFAULT_BAR_DIR = "E:/data/dollar_bar/bars"

# Shared task storage (no circular imports)
from ..storage import backtest_tasks as _tasks


def _get_task_storage():
    """Get reference to shared task storage."""
    return _tasks


@router.get("/nav/{task_id}")
async def get_nav_chart(
    task_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    resample: str = Query(default=None, description="Resample frequency: D, W, M"),
):
    """
    Get NAV time series data for charting.

    Returns data suitable for line charts.
    """
    tasks = _get_task_storage()

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    if task["status"].value != "completed":
        raise HTTPException(status_code=400, detail="Backtest not completed")

    nav_series = task["result"]["nav_series"]

    # Filter by date range if specified
    if start or end:
        filtered = []
        for point in nav_series:
            ts = datetime.fromisoformat(point["timestamp"].replace("Z", "+00:00"))
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            filtered.append(point)
        nav_series = filtered

    # Calculate additional metrics
    if nav_series:
        initial = nav_series[0]["nav"]
        data = []
        for point in nav_series:
            nav = point["nav"]
            data.append({
                "timestamp": point["timestamp"],
                "nav": nav,
                "return_pct": (nav / initial - 1) * 100,
            })
    else:
        data = []

    return {
        "task_id": task_id,
        "series": data,
        "total_points": len(data),
    }


@router.get("/drawdown/{task_id}")
async def get_drawdown_chart(task_id: str):
    """
    Get drawdown time series data for charting.

    Returns data suitable for area charts.
    """
    tasks = _get_task_storage()

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    if task["status"].value != "completed":
        raise HTTPException(status_code=400, detail="Backtest not completed")

    drawdown_series = task["result"]["drawdown_series"]

    # Convert to percentage
    data = [
        {
            "timestamp": point["timestamp"],
            "drawdown_pct": point["drawdown"] * 100,
        }
        for point in drawdown_series
    ]

    # Find max drawdown
    max_dd = max((d["drawdown_pct"] for d in data), default=0)

    return {
        "task_id": task_id,
        "series": data,
        "max_drawdown_pct": max_dd,
        "total_points": len(data),
    }


@router.get("/returns/{task_id}")
async def get_returns_distribution(task_id: str, bins: int = Query(default=50, ge=10, le=200)):
    """
    Get returns distribution data for histogram.

    Returns binned return frequencies.
    """
    import numpy as np

    tasks = _get_task_storage()

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    if task["status"].value != "completed":
        raise HTTPException(status_code=400, detail="Backtest not completed")

    nav_series = task["result"]["nav_series"]

    if len(nav_series) < 2:
        return {"task_id": task_id, "histogram": [], "statistics": {}}

    # Calculate returns
    navs = [p["nav"] for p in nav_series]
    returns = np.diff(navs) / navs[:-1]

    # Create histogram
    counts, bin_edges = np.histogram(returns, bins=bins)

    histogram = [
        {
            "bin_start": float(bin_edges[i]),
            "bin_end": float(bin_edges[i + 1]),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    # Statistics
    statistics = {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "skew": float(_calculate_skew(returns)),
        "kurtosis": float(_calculate_kurtosis(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "positive_pct": float(np.sum(returns > 0) / len(returns) * 100),
    }

    return {
        "task_id": task_id,
        "histogram": histogram,
        "statistics": statistics,
    }


def _calculate_skew(returns) -> float:
    """Calculate skewness."""
    import numpy as np

    n = len(returns)
    if n < 3:
        return 0.0

    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return 0.0

    return float(np.mean(((returns - mean) / std) ** 3))


def _calculate_kurtosis(returns) -> float:
    """Calculate excess kurtosis."""
    import numpy as np

    n = len(returns)
    if n < 4:
        return 0.0

    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return 0.0

    return float(np.mean(((returns - mean) / std) ** 4) - 3)


@router.get("/heatmap/{task_id}")
async def get_monthly_returns_heatmap(task_id: str):
    """
    Get monthly returns heatmap data.

    Returns matrix of year x month returns.
    """
    import numpy as np

    tasks = _get_task_storage()

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    if task["status"].value != "completed":
        raise HTTPException(status_code=400, detail="Backtest not completed")

    nav_series = task["result"]["nav_series"]

    if len(nav_series) < 2:
        return {"task_id": task_id, "heatmap": [], "years": [], "months": list(range(1, 13))}

    # Group by month
    monthly_data: Dict[tuple, List[float]] = {}

    prev_nav = None
    for point in nav_series:
        ts = datetime.fromisoformat(point["timestamp"].replace("Z", "+00:00"))
        nav = point["nav"]

        if prev_nav is not None:
            ret = (nav / prev_nav - 1)
            key = (ts.year, ts.month)
            if key not in monthly_data:
                monthly_data[key] = []
            monthly_data[key].append(ret)

        prev_nav = nav

    # Aggregate to monthly returns
    monthly_returns: Dict[tuple, float] = {}
    for key, rets in monthly_data.items():
        # Compound returns for the month
        monthly_returns[key] = np.prod([1 + r for r in rets]) - 1

    # Build heatmap matrix
    if not monthly_returns:
        return {"task_id": task_id, "heatmap": [], "years": [], "months": list(range(1, 13))}

    years = sorted(set(k[0] for k in monthly_returns.keys()))
    months = list(range(1, 13))

    heatmap = []
    for year in years:
        row = {"year": year}
        for month in months:
            key = (year, month)
            if key in monthly_returns:
                row[f"m{month}"] = round(monthly_returns[key] * 100, 2)
            else:
                row[f"m{month}"] = None
        heatmap.append(row)

    return {
        "task_id": task_id,
        "heatmap": heatmap,
        "years": years,
        "months": months,
    }


@router.get("/trades/{task_id}/summary")
async def get_trade_summary(task_id: str):
    """
    Get trade analysis summary.

    Includes win/loss statistics, holding periods, etc.
    """
    tasks = _get_task_storage()

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    if task["status"].value != "completed":
        raise HTTPException(status_code=400, detail="Backtest not completed")

    trades = task["result"]["trades"]

    if not trades:
        return {
            "task_id": task_id,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "profit_factor": None,
            "by_asset": {},
        }

    # Analyze trades
    pnls = [t.get("pnl", 0) or 0 for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_wins = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 0

    # By asset breakdown
    by_asset: Dict[str, dict] = {}
    for t in trades:
        asset = t.get("asset", "unknown")
        if asset not in by_asset:
            by_asset[asset] = {"trades": 0, "pnl": 0, "volume": 0}
        by_asset[asset]["trades"] += 1
        by_asset[asset]["pnl"] += t.get("pnl", 0) or 0
        by_asset[asset]["volume"] += t.get("value", 0) or 0

    return {
        "task_id": task_id,
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "avg_win": sum(wins) / len(wins) if wins else 0,
        "avg_loss": sum(losses) / len(losses) if losses else 0,
        "largest_win": max(wins) if wins else 0,
        "largest_loss": min(losses) if losses else 0,
        "profit_factor": total_wins / total_losses if total_losses > 0 else None,
        "by_asset": by_asset,
    }


@router.get("/performance/{task_id}")
async def get_performance_summary(task_id: str):
    """
    Get comprehensive performance summary.

    Combines metrics, risk measures, and key statistics.
    """
    tasks = _get_task_storage()

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = tasks[task_id]

    if task["status"].value != "completed":
        raise HTTPException(status_code=400, detail="Backtest not completed")

    result = task["result"]
    metrics = result["metrics"]

    return {
        "task_id": task_id,
        "summary": {
            "total_return": f"{metrics['total_return'] * 100:.2f}%",
            "annualized_return": f"{metrics['annualized_return'] * 100:.2f}%",
            "sharpe_ratio": f"{metrics['sharpe_ratio']:.2f}",
            "max_drawdown": f"{metrics['max_drawdown'] * 100:.2f}%",
            "win_rate": f"{metrics['win_rate'] * 100:.1f}%",
            "total_trades": metrics["total_trades"],
        },
        "returns": {
            "total": metrics["total_return"],
            "annualized": metrics["annualized_return"],
            "volatility": metrics["volatility"],
        },
        "risk": {
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics.get("sortino_ratio"),
            "max_drawdown": metrics["max_drawdown"],
            "calmar_ratio": metrics.get("calmar_ratio"),
        },
        "trading": {
            "total_trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics.get("profit_factor"),
            "avg_trade_return": metrics["avg_trade_return"],
        },
        "exposure": {
            "avg_long": metrics["avg_long_exposure"],
            "avg_short": metrics["avg_short_exposure"],
            "avg_gross": metrics["avg_gross_exposure"],
            "avg_net": metrics["avg_net_exposure"],
        },
        "costs": {
            "total_commission": metrics["total_commission"],
            "total_slippage": metrics["total_slippage"],
            "total_funding_fees": metrics["total_funding_fees"],
            "total_leverage_fees": metrics["total_leverage_fees"],
            "total_costs": (
                metrics["total_commission"]
                + metrics["total_slippage"]
                + metrics["total_funding_fees"]
                + metrics["total_leverage_fees"]
            ),
        },
    }


@router.get("/bars")
async def get_bars(
    symbol: str = Query(..., description="Symbol, e.g. BTCUSDT"),
    bar_dir: Optional[str] = Query(default=None, description="Bar directory (default: E:/data/dollar_bar/bars)"),
    limit: int = Query(default=500, ge=1, le=5000, description="Max number of bars to return (most recent)"),
):
    """
    Get dollar bar OHLCV data for visualization (candlestick chart).

    Reads parquet from bar_dir/symbol/*.parquet, sorts by start_time, returns last `limit` bars.
    """
    import pandas as pd

    root = Path(bar_dir or DEFAULT_BAR_DIR)
    symbol_dir = root / symbol
    if not symbol_dir.exists() or not symbol_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Symbol directory not found: {symbol_dir}")

    files = sorted(symbol_dir.glob("*.parquet"))
    if not files:
        raise HTTPException(status_code=404, detail=f"No parquet files in {symbol_dir}")

    frames = []
    for path in files:
        df = pd.read_parquet(path)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        raise HTTPException(status_code=404, detail=f"No rows in parquet files for {symbol}")

    combined = pd.concat(frames, ignore_index=True)
    if "start_time" in combined.columns:
        combined["start_time"] = pd.to_datetime(combined["start_time"], utc=True)
        combined = combined.sort_values("start_time").reset_index(drop=True)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in combined.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing columns: {missing}")

    combined = combined.tail(limit)

    time_col = "start_time" if "start_time" in combined.columns else combined.index
    times = combined["start_time"] if "start_time" in combined.columns else combined.index.astype(str)

    bars: List[Dict[str, Any]] = []
    for i in range(len(combined)):
        ts = times.iloc[i]
        if hasattr(ts, "isoformat"):
            ts_str = ts.isoformat().replace("+00:00", "Z")
        else:
            ts_str = str(ts)
        row = {
            "time": ts_str,
            "open": float(combined["open"].iloc[i]),
            "high": float(combined["high"].iloc[i]),
            "low": float(combined["low"].iloc[i]),
            "close": float(combined["close"].iloc[i]),
        }
        if "volume" in combined.columns:
            row["volume"] = float(combined["volume"].iloc[i])
        if "dollar_volume" in combined.columns:
            row["dollar_volume"] = float(combined["dollar_volume"].iloc[i])
        bars.append(row)

    return {
        "symbol": symbol,
        "bar_dir": str(root),
        "bars": bars,
        "count": len(bars),
    }
