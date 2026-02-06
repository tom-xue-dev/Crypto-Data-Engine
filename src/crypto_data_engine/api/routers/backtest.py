"""
Backtest API router.

Endpoints for running backtests, checking status, retrieving results, and accessing logs.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from ..schemas.backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestResult,
    BacktestStatus,
    PerformanceMetrics,
    TaskStatus,
    TradeRecord,
)

router = APIRouter(prefix="/backtest", tags=["backtest"])

# Shared task storage (eliminates circular import with visualization router)
from ..storage import backtest_tasks as _tasks

# Log storage directory
LOG_DIR = Path("data/backtest_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _create_strategy(strategy_config: dict):
    """Create strategy instance from config."""
    from crypto_data_engine.services.back_test.strategies.base_strategies import (
        EqualWeightStrategy,
        FactorStrategy,
        LongShortStrategy,
        MomentumStrategy,
        VolumeWeightedStrategy,
    )

    strategy_map = {
        "equal_weight": EqualWeightStrategy,
        "momentum": MomentumStrategy,
        "factor": FactorStrategy,
        "long_short": LongShortStrategy,
        "volume_weighted": VolumeWeightedStrategy,
    }

    name = strategy_config["name"].lower()
    if name not in strategy_map:
        raise ValueError(f"Unknown strategy: {name}")

    params = strategy_config.get("params", {})
    return strategy_map[name](**params)


def _run_backtest_task(task_id: str, request: BacktestRequest):
    """Run backtest in background."""
    import traceback

    from crypto_data_engine.services.back_test import (
        BacktestConfig,
        BacktestMode,
        CostConfigModel,
        MultiAssetDataLoader,
        RiskConfigModel,
        TradingLogger,
        create_backtest_engine,
    )

    task = _tasks[task_id]
    task["status"] = TaskStatus.RUNNING
    task["started_at"] = datetime.now()

    try:
        # Create config
        mode_map = {
            "cross_sectional": BacktestMode.CROSS_SECTIONAL,
            "time_series": BacktestMode.TIME_SERIES,
            "multi_asset": BacktestMode.MULTI_ASSET_TIME_SERIES,
        }

        risk_config = None
        if request.risk_config:
            risk_config = RiskConfigModel(
                max_position_size=request.risk_config.max_position_size,
                max_total_exposure=request.risk_config.max_total_exposure,
                max_leverage=request.risk_config.max_leverage,
                max_drawdown=request.risk_config.max_drawdown,
                daily_loss_limit=request.risk_config.daily_loss_limit,
                stop_loss_strategies=request.risk_config.stop_loss_strategies,
                stop_loss_params=request.risk_config.stop_loss_params,
            )

        cost_config = None
        if request.cost_config:
            cost_config = CostConfigModel(
                commission_rate=request.cost_config.commission_rate,
                maker_rate=request.cost_config.maker_rate,
                taker_rate=request.cost_config.taker_rate,
                slippage_rate=request.cost_config.slippage_rate,
                funding_enabled=request.cost_config.funding_enabled,
                leverage_enabled=request.cost_config.leverage_enabled,
                default_funding_rate=request.cost_config.default_funding_rate,
                default_leverage_rate=request.cost_config.default_leverage_rate,
            )

        config = BacktestConfig(
            mode=mode_map.get(request.mode.value, BacktestMode.CROSS_SECTIONAL),
            initial_capital=request.initial_capital,
            start_date=request.start_date,
            end_date=request.end_date,
            price_col=request.price_col,
            time_col=request.time_col,
            rebalance_frequency=request.rebalance_frequency,
            risk_config=risk_config,
            cost_config=cost_config,
        )

        # Create strategy
        strategy = _create_strategy(request.strategy.model_dump())

        # Create trading logger
        logger = TradingLogger(
            task_id=task_id,
            log_signals=True,
            log_snapshots=True,
            snapshot_frequency=1,
        )

        # Create engine with logger
        engine = create_backtest_engine(config, strategy, logger=logger)

        # Load data
        loader = MultiAssetDataLoader()
        if request.data_dir:
            loader.config.data_dir = request.data_dir

        assets = request.assets or []
        if not assets and request.use_asset_pool:
            # Load from asset pool config or use defaults
            assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Default assets

        start = request.start_date or datetime(2024, 1, 1)
        end = request.end_date or datetime.now()

        data = loader.load_bars(assets, start, end, request.bar_type)

        # Add features
        data = loader.load_features(data)

        # Run backtest
        result = engine.run(data)

        # Extract results
        nav_history = engine.get_nav_history()
        trades = engine.get_trades()

        # Calculate metrics
        metrics = _calculate_metrics(nav_history, trades, config.initial_capital)

        # Export logs to local storage
        task_log_dir = LOG_DIR / task_id
        task_log_dir.mkdir(parents=True, exist_ok=True)
        log_paths = logger.export_all(task_log_dir, prefix="backtest")

        # Store result with log info
        task["status"] = TaskStatus.COMPLETED
        task["completed_at"] = datetime.now()
        task["result"] = {
            "config": config.to_dict(),
            "metrics": metrics,
            "nav_series": [
                {"timestamp": t.isoformat(), "nav": v}
                for t, v in nav_history.items()
            ],
            "drawdown_series": _calculate_drawdown_series(nav_history),
            "trades": [_trade_to_dict(t) for t in trades],
            "final_positions": [],
        }
        task["log_summary"] = logger.get_summary()
        task["trade_summary"] = logger.get_trade_summary()
        task["log_paths"] = log_paths
        task["logger"] = logger  # Store logger reference for API queries
        task["progress"] = 1.0

    except Exception as e:
        task["status"] = TaskStatus.FAILED
        task["error"] = str(e)
        task["traceback"] = traceback.format_exc()
        task["completed_at"] = datetime.now()


def _calculate_metrics(nav_history: dict, trades: list, initial_capital: float) -> dict:
    """Calculate performance metrics from backtest results."""
    import numpy as np

    if not nav_history:
        return _empty_metrics()

    navs = list(nav_history.values())
    returns = np.diff(navs) / navs[:-1] if len(navs) > 1 else []

    total_return = (navs[-1] / initial_capital - 1) if navs else 0
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
    sharpe = (np.mean(returns) * 252) / volatility if volatility > 0 else 0

    # Max drawdown
    peak = initial_capital
    max_dd = 0
    for nav in navs:
        if nav > peak:
            peak = nav
        dd = (peak - nav) / peak
        if dd > max_dd:
            max_dd = dd

    # Trade statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if getattr(t, "pnl", 0) > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    total_commission = sum(getattr(t, "commission", 0) for t in trades)
    total_slippage = sum(getattr(t, "slippage", 0) for t in trades)

    return {
        "total_return": total_return,
        "annualized_return": total_return,  # Simplified
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": None,
        "max_drawdown": max_dd,
        "calmar_ratio": total_return / max_dd if max_dd > 0 else None,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": None,
        "avg_trade_return": total_return / total_trades if total_trades > 0 else 0,
        "avg_long_exposure": 0.5,  # Placeholder
        "avg_short_exposure": 0.0,
        "avg_gross_exposure": 0.5,
        "avg_net_exposure": 0.5,
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "total_funding_fees": 0,
        "total_leverage_fees": 0,
    }


def _empty_metrics() -> dict:
    """Return empty metrics."""
    return {
        "total_return": 0,
        "annualized_return": 0,
        "volatility": 0,
        "sharpe_ratio": 0,
        "sortino_ratio": None,
        "max_drawdown": 0,
        "calmar_ratio": None,
        "total_trades": 0,
        "win_rate": 0,
        "profit_factor": None,
        "avg_trade_return": 0,
        "avg_long_exposure": 0,
        "avg_short_exposure": 0,
        "avg_gross_exposure": 0,
        "avg_net_exposure": 0,
        "total_commission": 0,
        "total_slippage": 0,
        "total_funding_fees": 0,
        "total_leverage_fees": 0,
    }


def _calculate_drawdown_series(nav_history: dict) -> list:
    """Calculate drawdown time series."""
    result = []
    peak = 0
    for timestamp, nav in nav_history.items():
        if nav > peak:
            peak = nav
        dd = (peak - nav) / peak if peak > 0 else 0
        result.append({
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "drawdown": dd,
        })
    return result


def _trade_to_dict(trade) -> dict:
    """Convert trade object to dict."""
    if isinstance(trade, dict):
        return trade
    return {
        "timestamp": getattr(trade, "timestamp", datetime.now()).isoformat(),
        "asset": getattr(trade, "asset", ""),
        "direction": getattr(trade, "direction", "BUY"),
        "quantity": getattr(trade, "quantity", 0),
        "price": getattr(trade, "price", 0),
        "value": getattr(trade, "value", 0),
        "commission": getattr(trade, "commission", 0),
        "slippage": getattr(trade, "slippage", 0),
        "pnl": getattr(trade, "pnl", None),
    }


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a backtest task for execution.

    Returns a task_id that can be used to check status and retrieve results.
    """
    task_id = str(uuid.uuid4())

    _tasks[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "progress": 0.0,
        "created_at": datetime.now(),
        "started_at": None,
        "completed_at": None,
        "request": request.model_dump(),
        "result": None,
        "error": None,
    }

    # Run in background
    background_tasks.add_task(_run_backtest_task, task_id, request)

    return BacktestResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Backtest task submitted successfully",
        created_at=datetime.now(),
    )


@router.get("/status/{task_id}", response_model=BacktestStatus)
async def get_backtest_status(task_id: str):
    """Get the status of a backtest task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    return BacktestStatus(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0),
        message=task.get("message"),
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        error=task.get("error"),
    )


@router.get("/result/{task_id}", response_model=BacktestResult)
async def get_backtest_result(task_id: str):
    """Get the result of a completed backtest task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] == TaskStatus.PENDING:
        raise HTTPException(status_code=400, detail="Backtest not started yet")

    if task["status"] == TaskStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Backtest still running")

    if task["status"] == TaskStatus.FAILED:
        raise HTTPException(
            status_code=500,
            detail=f"Backtest failed: {task.get('error', 'Unknown error')}",
        )

    result = task["result"]

    return BacktestResult(
        task_id=task_id,
        status=task["status"],
        config=result["config"],
        metrics=PerformanceMetrics(**result["metrics"]),
        nav_series=result["nav_series"],
        drawdown_series=result["drawdown_series"],
        trades=[TradeRecord(**t) for t in result["trades"]],
        final_positions=[],
        started_at=task["started_at"],
        completed_at=task["completed_at"],
        duration_seconds=(task["completed_at"] - task["started_at"]).total_seconds(),
    )


@router.get("/trades/{task_id}")
async def get_backtest_trades(
    task_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Get trades from a completed backtest with pagination."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Backtest not completed")

    trades = task["result"]["trades"]
    total = len(trades)
    paginated = trades[offset : offset + limit]

    return {
        "task_id": task_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "trades": [TradeRecord(**t) for t in paginated],
    }


@router.delete("/{task_id}")
async def cancel_backtest(task_id: str):
    """Cancel a running backtest or delete a completed task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] == TaskStatus.RUNNING:
        # In a real implementation, we would signal the background task to stop
        task["status"] = TaskStatus.FAILED
        task["error"] = "Cancelled by user"
        task["completed_at"] = datetime.now()

    del _tasks[task_id]

    return {"message": f"Task {task_id} deleted"}


@router.get("/list")
async def list_backtests(
    status: Optional[TaskStatus] = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
):
    """List all backtest tasks with optional filtering."""
    tasks = list(_tasks.values())

    if status:
        tasks = [t for t in tasks if t["status"] == status]

    # Sort by created_at descending
    tasks.sort(key=lambda x: x["created_at"], reverse=True)

    total = len(tasks)
    paginated = tasks[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "tasks": [
            BacktestStatus(
                task_id=t["task_id"],
                status=t["status"],
                progress=t.get("progress", 0),
                started_at=t.get("started_at"),
                completed_at=t.get("completed_at"),
                error=t.get("error"),
            )
            for t in paginated
        ],
    }


# ============================================================================
# Trading Log Endpoints
# ============================================================================

@router.get("/logs/{task_id}")
async def get_backtest_logs(
    task_id: str,
    event_type: Optional[str] = None,
    asset: Optional[str] = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Get trading logs from a completed backtest.
    
    Supports filtering by event_type (TRADE, SIGNAL, RISK_TRIGGER, SNAPSHOT, REBALANCE)
    and by asset.
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Backtest not completed")

    logger = task.get("logger")
    if not logger:
        raise HTTPException(status_code=404, detail="Logs not available for this task")

    # Get all entries
    entries = logger.get_all_entries()

    # Filter by event type
    if event_type:
        event_type_upper = event_type.upper()
        entries = [e for e in entries if e.event_type.value == event_type_upper]

    # Filter by asset
    if asset:
        entries = [e for e in entries if hasattr(e, 'asset') and e.asset == asset]

    total = len(entries)
    paginated = entries[offset : offset + limit]

    return {
        "task_id": task_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "summary": task.get("log_summary", {}),
        "entries": [e.to_dict() for e in paginated],
    }


@router.get("/logs/{task_id}/trades")
async def get_trade_logs(
    task_id: str,
    asset: Optional[str] = None,
    action: Optional[str] = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Get trade execution logs from a completed backtest.
    
    Supports filtering by asset and action (OPEN, CLOSE).
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Backtest not completed")

    logger = task.get("logger")
    if not logger:
        raise HTTPException(status_code=404, detail="Logs not available for this task")

    trades = logger.get_trades()

    # Filter by asset
    if asset:
        trades = [t for t in trades if t.asset == asset]

    # Filter by action
    if action:
        action_upper = action.upper()
        trades = [t for t in trades if t.action == action_upper]

    total = len(trades)
    paginated = trades[offset : offset + limit]

    return {
        "task_id": task_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "trade_summary": task.get("trade_summary", {}),
        "trades": [t.to_dict() for t in paginated],
    }


@router.get("/logs/{task_id}/signals")
async def get_signal_logs(
    task_id: str,
    asset: Optional[str] = None,
    executed_only: bool = False,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Get strategy signal logs from a completed backtest.
    
    Supports filtering by asset and whether signals were executed.
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Backtest not completed")

    logger = task.get("logger")
    if not logger:
        raise HTTPException(status_code=404, detail="Logs not available for this task")

    signals = logger.get_signals()

    # Filter by asset
    if asset:
        signals = [s for s in signals if s.asset == asset]

    # Filter by executed
    if executed_only:
        signals = [s for s in signals if s.was_executed]

    total = len(signals)
    paginated = signals[offset : offset + limit]

    return {
        "task_id": task_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "signals": [s.to_dict() for s in paginated],
    }


@router.get("/logs/{task_id}/snapshots")
async def get_portfolio_snapshots(
    task_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Get portfolio snapshots from a completed backtest.
    
    Returns NAV, positions, weights, and exposure at each snapshot point.
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Backtest not completed")

    logger = task.get("logger")
    if not logger:
        raise HTTPException(status_code=404, detail="Logs not available for this task")

    snapshots = logger.get_snapshots()
    total = len(snapshots)
    paginated = snapshots[offset : offset + limit]

    return {
        "task_id": task_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "snapshots": [s.to_dict() for s in paginated],
    }


@router.get("/logs/{task_id}/risk-triggers")
async def get_risk_trigger_logs(
    task_id: str,
    asset: Optional[str] = None,
    trigger_type: Optional[str] = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Get risk management trigger logs from a completed backtest.
    
    Supports filtering by asset and trigger_type (stop_loss, trailing_stop, etc).
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Backtest not completed")

    logger = task.get("logger")
    if not logger:
        raise HTTPException(status_code=404, detail="Logs not available for this task")

    triggers = logger.get_risk_triggers()

    # Filter by asset
    if asset:
        triggers = [t for t in triggers if t.asset == asset]

    # Filter by trigger type
    if trigger_type:
        triggers = [t for t in triggers if t.trigger_type == trigger_type]

    total = len(triggers)
    paginated = triggers[offset : offset + limit]

    return {
        "task_id": task_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "triggers": [t.to_dict() for t in paginated],
    }


@router.get("/logs/{task_id}/rebalances")
async def get_rebalance_logs(
    task_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """
    Get rebalancing event logs from a completed backtest.
    
    Returns details of each rebalancing including weight changes and turnover.
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Backtest not completed")

    logger = task.get("logger")
    if not logger:
        raise HTTPException(status_code=404, detail="Logs not available for this task")

    rebalances = logger.get_rebalances()
    total = len(rebalances)
    paginated = rebalances[offset : offset + limit]

    return {
        "task_id": task_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "rebalances": [r.to_dict() for r in paginated],
    }


@router.get("/logs/{task_id}/export")
async def export_logs(
    task_id: str,
    format: str = Query(default="json", enum=["json", "csv"]),
    log_type: str = Query(default="full", enum=["full", "trades", "snapshots"]),
):
    """
    Export logs to a file.
    
    Supports JSON (full logs) and CSV (trades or snapshots) formats.
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]

    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Backtest not completed")

    log_paths = task.get("log_paths", {})

    if format == "json":
        file_path = log_paths.get("full_log")
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Log file not found")
        return FileResponse(
            file_path,
            media_type="application/json",
            filename=f"backtest_{task_id}_log.json",
        )
    elif format == "csv":
        if log_type == "trades":
            file_path = log_paths.get("trades")
        elif log_type == "snapshots":
            file_path = log_paths.get("snapshots")
        else:
            raise HTTPException(
                status_code=400,
                detail="CSV export only supports 'trades' or 'snapshots' log_type",
            )

        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Log file not found")
        return FileResponse(
            file_path,
            media_type="text/csv",
            filename=f"backtest_{task_id}_{log_type}.csv",
        )

    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
