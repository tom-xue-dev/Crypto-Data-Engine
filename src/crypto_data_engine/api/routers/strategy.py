"""
Strategy API router.

Endpoints for listing strategies, getting parameters, and validation.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ..schemas.strategy import (
    ParamType,
    StrategyInfo,
    StrategyListResponse,
    StrategyParam,
    StrategyPresetResponse,
    StrategyValidateRequest,
    StrategyValidateResponse,
)

router = APIRouter(prefix="/strategy", tags=["strategy"])

# Strategy registry with metadata
STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "equal_weight": {
        "display_name": "Equal Weight",
        "description": "Allocates equal weight to all assets in the universe",
        "category": "cross_sectional",
        "supports_long": True,
        "supports_short": False,
        "params": [
            {
                "name": "max_positions",
                "type": ParamType.INT,
                "default": 20,
                "description": "Maximum number of positions to hold",
                "min_value": 1,
                "max_value": 500,
            },
        ],
    },
    "momentum": {
        "display_name": "Momentum Strategy",
        "description": "Long winners and short losers based on past returns",
        "category": "cross_sectional",
        "supports_long": True,
        "supports_short": True,
        "params": [
            {
                "name": "lookback_period",
                "type": ParamType.INT,
                "default": 20,
                "description": "Lookback period for return calculation",
                "min_value": 1,
                "max_value": 252,
            },
            {
                "name": "long_count",
                "type": ParamType.INT,
                "default": 10,
                "description": "Number of assets to long",
                "min_value": 1,
                "max_value": 100,
            },
            {
                "name": "short_count",
                "type": ParamType.INT,
                "default": 10,
                "description": "Number of assets to short",
                "min_value": 0,
                "max_value": 100,
            },
            {
                "name": "return_col",
                "type": ParamType.STRING,
                "default": "return_20",
                "description": "Column name for returns data",
            },
            {
                "name": "gross_exposure",
                "type": ParamType.FLOAT,
                "default": 2.0,
                "description": "Total gross exposure (long + abs(short))",
                "min_value": 0.1,
                "max_value": 10.0,
            },
        ],
    },
    "factor": {
        "display_name": "Factor Strategy",
        "description": "Generic factor-based strategy with customizable factor column",
        "category": "cross_sectional",
        "supports_long": True,
        "supports_short": True,
        "params": [
            {
                "name": "factor_col",
                "type": ParamType.STRING,
                "default": "factor",
                "description": "Column name containing factor values",
                "required": True,
            },
            {
                "name": "long_count",
                "type": ParamType.INT,
                "default": 10,
                "description": "Number of assets to long",
                "min_value": 1,
                "max_value": 100,
            },
            {
                "name": "short_count",
                "type": ParamType.INT,
                "default": 0,
                "description": "Number of assets to short (0 for long-only)",
                "min_value": 0,
                "max_value": 100,
            },
            {
                "name": "long_direction",
                "type": ParamType.CHOICE,
                "default": "low",
                "description": "Direction for long positions",
                "choices": ["low", "high"],
            },
            {
                "name": "gross_exposure",
                "type": ParamType.FLOAT,
                "default": 1.0,
                "description": "Total gross exposure",
                "min_value": 0.1,
                "max_value": 10.0,
            },
        ],
    },
    "long_short": {
        "display_name": "Long/Short Strategy",
        "description": "Dollar-neutral long/short strategy based on signal",
        "category": "cross_sectional",
        "supports_long": True,
        "supports_short": True,
        "params": [
            {
                "name": "signal_col",
                "type": ParamType.STRING,
                "default": "signal",
                "description": "Column name containing signal values",
                "required": True,
            },
            {
                "name": "long_count",
                "type": ParamType.INT,
                "default": 10,
                "description": "Number of assets to long",
                "min_value": 1,
                "max_value": 100,
            },
            {
                "name": "short_count",
                "type": ParamType.INT,
                "default": 10,
                "description": "Number of assets to short",
                "min_value": 1,
                "max_value": 100,
            },
            {
                "name": "gross_exposure",
                "type": ParamType.FLOAT,
                "default": 2.0,
                "description": "Total gross exposure",
                "min_value": 0.1,
                "max_value": 10.0,
            },
        ],
    },
    "volume_weighted": {
        "display_name": "Volume Weighted Strategy",
        "description": "Weight positions by volume for mean reversion",
        "category": "cross_sectional",
        "supports_long": True,
        "supports_short": True,
        "params": [
            {
                "name": "signal_col",
                "type": ParamType.STRING,
                "default": "signal",
                "description": "Column name containing signal values",
                "required": True,
            },
            {
                "name": "volume_col",
                "type": ParamType.STRING,
                "default": "volume",
                "description": "Column name containing volume data",
            },
            {
                "name": "long_count",
                "type": ParamType.INT,
                "default": 10,
                "description": "Number of assets to long",
                "min_value": 1,
                "max_value": 100,
            },
            {
                "name": "short_count",
                "type": ParamType.INT,
                "default": 10,
                "description": "Number of assets to short",
                "min_value": 0,
                "max_value": 100,
            },
            {
                "name": "weight_by_volume",
                "type": ParamType.BOOL,
                "default": True,
                "description": "True = more weight to high volume assets",
            },
            {
                "name": "gross_exposure",
                "type": ParamType.FLOAT,
                "default": 2.0,
                "description": "Total gross exposure",
                "min_value": 0.1,
                "max_value": 10.0,
            },
        ],
    },
    "trend_following": {
        "display_name": "Trend Following",
        "description": "Simple trend following based on moving average crossover",
        "category": "time_series",
        "supports_long": True,
        "supports_short": False,
        "params": [
            {
                "name": "ma_col",
                "type": ParamType.STRING,
                "default": "sma_20",
                "description": "Column name for moving average",
            },
            {
                "name": "price_col",
                "type": ParamType.STRING,
                "default": "close",
                "description": "Column name for price",
            },
        ],
    },
    "rsi_mean_reversion": {
        "display_name": "RSI Mean Reversion",
        "description": "Mean reversion strategy based on RSI levels",
        "category": "time_series",
        "supports_long": True,
        "supports_short": False,
        "params": [
            {
                "name": "rsi_col",
                "type": ParamType.STRING,
                "default": "rsi_14",
                "description": "Column name for RSI values",
            },
            {
                "name": "oversold",
                "type": ParamType.FLOAT,
                "default": 30,
                "description": "RSI oversold threshold (buy signal)",
                "min_value": 0,
                "max_value": 50,
            },
            {
                "name": "overbought",
                "type": ParamType.FLOAT,
                "default": 70,
                "description": "RSI overbought threshold (sell signal)",
                "min_value": 50,
                "max_value": 100,
            },
        ],
    },
    "momentum_crossover": {
        "display_name": "Momentum Crossover",
        "description": "Trade on short-term vs long-term momentum crossovers",
        "category": "time_series",
        "supports_long": True,
        "supports_short": False,
        "params": [
            {
                "name": "short_momentum_col",
                "type": ParamType.STRING,
                "default": "momentum_5",
                "description": "Column for short-term momentum",
            },
            {
                "name": "long_momentum_col",
                "type": ParamType.STRING,
                "default": "momentum_20",
                "description": "Column for long-term momentum",
            },
        ],
    },
}

# Strategy presets
STRATEGY_PRESETS = {
    "conservative_momentum": {
        "description": "Conservative momentum with lower exposure",
        "config": {
            "name": "momentum",
            "params": {
                "lookback_period": 60,
                "long_count": 20,
                "short_count": 0,
                "gross_exposure": 1.0,
            },
        },
    },
    "aggressive_long_short": {
        "description": "Aggressive dollar-neutral long/short",
        "config": {
            "name": "momentum",
            "params": {
                "lookback_period": 20,
                "long_count": 10,
                "short_count": 10,
                "gross_exposure": 2.0,
            },
        },
    },
    "value_factor": {
        "description": "Long low-factor assets (e.g., low P/E)",
        "config": {
            "name": "factor",
            "params": {
                "factor_col": "pe_ratio",
                "long_count": 20,
                "short_count": 0,
                "long_direction": "low",
                "gross_exposure": 1.0,
            },
        },
    },
}


@router.get("/list", response_model=StrategyListResponse)
async def list_strategies(category: str = None):
    """
    List all available strategies.

    Optionally filter by category: cross_sectional, time_series, multi_asset
    """
    strategies = []

    for name, meta in STRATEGY_REGISTRY.items():
        if category and meta["category"] != category:
            continue

        params = [StrategyParam(**p) for p in meta["params"]]

        strategies.append(
            StrategyInfo(
                name=name,
                display_name=meta["display_name"],
                description=meta["description"],
                category=meta["category"],
                supports_long=meta["supports_long"],
                supports_short=meta["supports_short"],
                params=params,
            )
        )

    return StrategyListResponse(strategies=strategies, total=len(strategies))


@router.get("/{name}", response_model=StrategyInfo)
async def get_strategy(name: str):
    """Get detailed information about a specific strategy."""
    if name not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    meta = STRATEGY_REGISTRY[name]
    params = [StrategyParam(**p) for p in meta["params"]]

    return StrategyInfo(
        name=name,
        display_name=meta["display_name"],
        description=meta["description"],
        category=meta["category"],
        supports_long=meta["supports_long"],
        supports_short=meta["supports_short"],
        params=params,
    )


@router.get("/{name}/params")
async def get_strategy_params(name: str):
    """Get parameter definitions for a strategy."""
    if name not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")

    meta = STRATEGY_REGISTRY[name]
    return {"name": name, "params": [StrategyParam(**p) for p in meta["params"]]}


@router.post("/validate", response_model=StrategyValidateResponse)
async def validate_strategy(request: StrategyValidateRequest):
    """
    Validate strategy parameters.

    Checks that all required parameters are provided and values are within bounds.
    """
    if request.name not in STRATEGY_REGISTRY:
        return StrategyValidateResponse(
            valid=False,
            errors=[f"Unknown strategy: {request.name}"],
        )

    meta = STRATEGY_REGISTRY[request.name]
    errors = []
    warnings = []
    normalized = {}

    for param_def in meta["params"]:
        param_name = param_def["name"]
        param_type = param_def["type"]
        default = param_def.get("default")
        required = param_def.get("required", False)

        if param_name in request.params:
            value = request.params[param_name]

            # Type validation
            if param_type == ParamType.INT:
                if not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        errors.append(f"{param_name}: expected int, got {type(value).__name__}")
                        continue

                # Range validation
                if "min_value" in param_def and value < param_def["min_value"]:
                    errors.append(f"{param_name}: value {value} is below minimum {param_def['min_value']}")
                if "max_value" in param_def and value > param_def["max_value"]:
                    errors.append(f"{param_name}: value {value} exceeds maximum {param_def['max_value']}")

            elif param_type == ParamType.FLOAT:
                if not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{param_name}: expected float, got {type(value).__name__}")
                        continue

                if "min_value" in param_def and value < param_def["min_value"]:
                    errors.append(f"{param_name}: value {value} is below minimum {param_def['min_value']}")
                if "max_value" in param_def and value > param_def["max_value"]:
                    errors.append(f"{param_name}: value {value} exceeds maximum {param_def['max_value']}")

            elif param_type == ParamType.BOOL:
                if not isinstance(value, bool):
                    errors.append(f"{param_name}: expected bool, got {type(value).__name__}")
                    continue

            elif param_type == ParamType.CHOICE:
                if "choices" in param_def and value not in param_def["choices"]:
                    errors.append(f"{param_name}: value '{value}' not in choices {param_def['choices']}")
                    continue

            normalized[param_name] = value

        elif required:
            errors.append(f"Missing required parameter: {param_name}")

        else:
            normalized[param_name] = default

    return StrategyValidateResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        normalized_params=normalized if not errors else None,
    )


@router.get("/presets/list")
async def list_presets():
    """List all strategy presets."""
    return {
        "presets": [
            {"name": name, "description": preset["description"]}
            for name, preset in STRATEGY_PRESETS.items()
        ]
    }


@router.get("/presets/{name}", response_model=StrategyPresetResponse)
async def get_preset(name: str):
    """Get a strategy preset configuration."""
    if name not in STRATEGY_PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")

    preset = STRATEGY_PRESETS[name]
    return StrategyPresetResponse(
        name=name,
        description=preset["description"],
        config=preset["config"],
    )
