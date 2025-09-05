from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field
from crypto_data_engine.common.config.paths import DATA_ROOT


class AggregationConfig(BaseModel):
    """Default parameters for bar aggregation per bar type.
    If a user request omits parameters, these defaults are used.
    Also defines the root output directory for aggregated bars.
    """

    # Root dir for bar outputs, final path = output_root/<exchange>/<symbol>/...
    output_root: Path = Field(default=DATA_ROOT / "bar_data")

    # Normalized bar types: tick_bar | volume_bar | dollar_bar | time | imbalance
    default_thresholds: Dict[str, int] = Field(
        default={
            "tick_bar": 2_000,
            "volume_bar": 10_000_000,
            "dollar_bar": 5_000_000,
        }
    )

    # Additional bar-type specific defaults
    default_params: Dict[str, Dict[str, Any]] = Field(
        default={
            "time": {"interval": "1m"},
            "imbalance": {"ema_window": 50},
        }
    )

    synonyms: Dict[str, str] = Field(
        default={
            "tick": "tick_bar",
            "volume": "volume_bar",
            "dollar": "dollar_bar",
        }
    )

    def normalize_bar_type(self, bar_type: str) -> str:
        return self.synonyms.get(bar_type, bar_type)

    def resolve_defaults(self, bar_type: str) -> Dict[str, Any]:
        """Return default parameters (including threshold if applicable)."""
        bt = self.normalize_bar_type(bar_type)
        params: Dict[str, Any] = {}
        if bt in self.default_thresholds:
            params["threshold"] = self.default_thresholds[bt]
        params.update(self.default_params.get(bt, {}))
        return {"bar_type": bt, **params}

    def make_output_dir(self, exchange: str) -> Path:
        return self.output_root / exchange
