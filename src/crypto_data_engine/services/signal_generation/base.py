"""
Base classes for signal generation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from crypto_data_engine.core.base import SignalType


@dataclass
class SignalOutput:
    """
    Output from a signal generator.
    
    Can represent:
    - Discrete signals (BUY/SELL/HOLD)
    - Continuous signals (-1 to 1 strength)
    - Target weights (for cross-sectional strategies)
    """
    timestamp: datetime
    
    # Discrete signals per asset
    signals: Dict[str, SignalType] = field(default_factory=dict)
    
    # Continuous signal strength (-1 to 1)
    strengths: Dict[str, float] = field(default_factory=dict)
    
    # Target portfolio weights
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Confidence scores (0 to 1)
    confidence: Dict[str, float] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "signals": {k: v.name for k, v in self.signals.items()},
            "strengths": self.strengths,
            "weights": self.weights,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_weights(
        cls,
        timestamp: datetime,
        weights: Dict[str, float],
        confidence: Optional[Dict[str, float]] = None,
    ) -> SignalOutput:
        """Create SignalOutput from target weights."""
        signals = {}
        strengths = {}
        
        for asset, weight in weights.items():
            if weight > 0.001:
                signals[asset] = SignalType.BUY
                strengths[asset] = min(weight * 2, 1.0)  # Scale weight to strength
            elif weight < -0.001:
                signals[asset] = SignalType.SELL
                strengths[asset] = max(weight * 2, -1.0)
            else:
                signals[asset] = SignalType.HOLD
                strengths[asset] = 0.0
        
        return cls(
            timestamp=timestamp,
            signals=signals,
            strengths=strengths,
            weights=weights,
            confidence=confidence or {},
        )
    
    @classmethod
    def from_strengths(
        cls,
        timestamp: datetime,
        strengths: Dict[str, float],
        long_threshold: float = 0.1,
        short_threshold: float = -0.1,
    ) -> SignalOutput:
        """Create SignalOutput from signal strengths."""
        signals = {}
        
        for asset, strength in strengths.items():
            if strength > long_threshold:
                signals[asset] = SignalType.BUY
            elif strength < short_threshold:
                signals[asset] = SignalType.SELL
            else:
                signals[asset] = SignalType.HOLD
        
        return cls(
            timestamp=timestamp,
            signals=signals,
            strengths=strengths,
        )


class BaseSignalGenerator(ABC):
    """
    Abstract base class for signal generators.
    
    Signal generators transform features/factors into trading signals.
    They can operate in two modes:
    
    1. Time-series mode: Generate signal for a single asset over time
    2. Cross-sectional mode: Generate signals for multiple assets at a point in time
    """
    
    def __init__(self, name: str = "BaseSignal"):
        self.name = name
        self._params: Dict[str, Any] = {}
    
    @property
    def params(self) -> Dict[str, Any]:
        """Get generator parameters."""
        return self._params.copy()
    
    def set_params(self, **kwargs) -> None:
        """Set generator parameters."""
        self._params.update(kwargs)
    
    @abstractmethod
    def generate(
        self,
        data: Union[pd.DataFrame, pd.Series],
        timestamp: Optional[datetime] = None,
    ) -> SignalOutput:
        """
        Generate signals from input data.
        
        Args:
            data: Input features/factors
                  - For time-series: DataFrame with time index
                  - For cross-sectional: DataFrame with assets as index
            timestamp: Current timestamp (optional, inferred from data)
            
        Returns:
            SignalOutput with signals, strengths, and/or weights
        """
        pass
    
    def generate_batch(
        self,
        data: pd.DataFrame,
        timestamps: Optional[List[datetime]] = None,
    ) -> List[SignalOutput]:
        """
        Generate signals for multiple timestamps.
        
        Args:
            data: Multi-index DataFrame (timestamp, asset)
            timestamps: Timestamps to generate signals for
            
        Returns:
            List of SignalOutput objects
        """
        if timestamps is None:
            timestamps = data.index.get_level_values(0).unique()
        
        outputs = []
        for ts in timestamps:
            try:
                cross_section = data.loc[ts]
                output = self.generate(cross_section, ts)
                outputs.append(output)
            except KeyError:
                continue
        
        return outputs
    
    def to_dataframe(self, outputs: List[SignalOutput]) -> pd.DataFrame:
        """
        Convert list of SignalOutput to DataFrame.
        
        Args:
            outputs: List of signal outputs
            
        Returns:
            DataFrame with timestamps as index, assets as columns
        """
        if not outputs:
            return pd.DataFrame()
        
        records = []
        for out in outputs:
            for asset in set(out.signals.keys()) | set(out.weights.keys()):
                records.append({
                    "timestamp": out.timestamp,
                    "asset": asset,
                    "signal": out.signals.get(asset, SignalType.HOLD).value,
                    "strength": out.strengths.get(asset, 0.0),
                    "weight": out.weights.get(asset, 0.0),
                    "confidence": out.confidence.get(asset, 1.0),
                })
        
        return pd.DataFrame(records)
