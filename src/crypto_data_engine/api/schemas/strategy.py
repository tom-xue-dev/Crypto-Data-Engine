"""
Pydantic schemas for strategy API.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ParamType(str, Enum):
    """Parameter type for strategy parameters."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    LIST = "list"
    CHOICE = "choice"


class StrategyParam(BaseModel):
    """Strategy parameter definition."""
    name: str
    type: ParamType
    default: Any
    description: str
    required: bool = False

    # Validation constraints
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "name": "lookback_period",
                "type": "int",
                "default": 20,
                "description": "Number of periods to look back for momentum calculation",
                "required": False,
                "min_value": 1,
                "max_value": 252
            }
        }


class StrategyInfo(BaseModel):
    """Information about a strategy."""
    name: str
    display_name: str
    description: str
    category: str  # "cross_sectional", "time_series", "multi_asset"
    supports_long: bool = True
    supports_short: bool = False
    params: List[StrategyParam]

    class Config:
        json_schema_extra = {
            "example": {
                "name": "momentum",
                "display_name": "Momentum Strategy",
                "description": "Long winners, short losers based on past returns",
                "category": "cross_sectional",
                "supports_long": True,
                "supports_short": True,
                "params": [
                    {
                        "name": "lookback_period",
                        "type": "int",
                        "default": 20,
                        "description": "Lookback period for returns",
                        "min_value": 1,
                        "max_value": 252
                    }
                ]
            }
        }


class StrategyListResponse(BaseModel):
    """Response for listing available strategies."""
    strategies: List[StrategyInfo]
    total: int


class StrategyValidateRequest(BaseModel):
    """Request for validating strategy parameters."""
    name: str
    params: Dict[str, Any]


class StrategyValidateResponse(BaseModel):
    """Response for strategy validation."""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    normalized_params: Optional[Dict[str, Any]] = None


class StrategyPresetResponse(BaseModel):
    """Response for getting strategy presets."""
    name: str
    description: str
    config: Dict[str, Any]
