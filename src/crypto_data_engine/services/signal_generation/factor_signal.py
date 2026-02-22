"""
Factor-based signal generators.

Generate trading signals based on factor values through:
- Ranking (top/bottom N)
- Thresholds (factor value above/below)
- Z-score normalization
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from crypto_data_engine.core.base import SignalType
from crypto_data_engine.services.signal_generation.base import (
    BaseSignalGenerator,
    SignalOutput,
)


@dataclass
class FactorConfig:
    """Configuration for a single factor in signal generation."""
    
    name: str
    """Factor column name in data."""
    
    direction: int = 1
    """1 for higher is better, -1 for lower is better."""
    
    weight: float = 1.0
    """Weight when combining multiple factors."""
    
    normalize: bool = True
    """Whether to z-score normalize the factor."""
    
    winsorize: Optional[float] = 3.0
    """Winsorize extreme values at N standard deviations."""
    
    fillna_method: str = "median"
    """How to fill NaN: 'median', 'mean', 'zero', 'drop'."""


class FactorSignalGenerator(BaseSignalGenerator):
    """
    Generate signals from a combination of factors.
    
    Supports:
    - Multiple factors with configurable weights
    - Z-score normalization
    - Winsorization of extreme values
    - Long/short signal generation based on composite score
    """
    
    def __init__(
        self,
        factors: List[FactorConfig],
        long_threshold: float = 0.5,
        short_threshold: float = -0.5,
        top_n_long: int = 0,
        top_n_short: int = 0,
        name: str = "FactorSignal",
    ):
        """
        Initialize factor signal generator.
        
        Args:
            factors: List of factor configurations
            long_threshold: Composite score above this → long
            short_threshold: Composite score below this → short
            top_n_long: If > 0, long only top N by score
            top_n_short: If > 0, short only bottom N by score
            name: Generator name
        """
        super().__init__(name)
        self.factors = factors
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.top_n_long = top_n_long
        self.top_n_short = top_n_short

    def generate(
        self,
        data: Union[pd.DataFrame, pd.Series],
        timestamp: Optional[datetime] = None,
    ) -> SignalOutput:
        """
        Generate signals from factor data.
        
        Args:
            data: DataFrame with assets as index, factors as columns
            timestamp: Current timestamp
            
        Returns:
            SignalOutput with signals and weights
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        
        # Calculate composite score
        scores = self._calculate_composite_score(data)
        
        if scores.empty:
            return SignalOutput(timestamp=timestamp)
        
        # Generate weights based on scores
        weights = self._scores_to_weights(scores)
        
        # Generate signals
        signals = {}
        strengths = {}
        
        for asset, score in scores.items():
            if score > self.long_threshold:
                signals[asset] = SignalType.BUY
            elif score < self.short_threshold:
                signals[asset] = SignalType.SELL
            else:
                signals[asset] = SignalType.HOLD
            
            strengths[asset] = float(np.clip(score, -1, 1))
        
        return SignalOutput(
            timestamp=timestamp,
            signals=signals,
            strengths=strengths,
            weights=weights,
            confidence={a: 1.0 for a in scores.index},
            metadata={"composite_scores": scores.to_dict()},
        )

    def _calculate_composite_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate weighted composite score from factors."""
        scores = pd.DataFrame(index=data.index)
        total_weight = 0
        
        for factor_cfg in self.factors:
            if factor_cfg.name not in data.columns:
                continue
            
            factor_values = data[factor_cfg.name].copy()
            
            # Fill NaN
            if factor_cfg.fillna_method == "median":
                factor_values = factor_values.fillna(factor_values.median())
            elif factor_cfg.fillna_method == "mean":
                factor_values = factor_values.fillna(factor_values.mean())
            elif factor_cfg.fillna_method == "zero":
                factor_values = factor_values.fillna(0)
            elif factor_cfg.fillna_method == "drop":
                pass  # Keep NaN, they'll be dropped later
            
            # Normalize
            if factor_cfg.normalize and factor_values.std() > 0:
                factor_values = (factor_values - factor_values.mean()) / factor_values.std()
            
            # Winsorize
            if factor_cfg.winsorize:
                factor_values = factor_values.clip(
                    lower=-factor_cfg.winsorize,
                    upper=factor_cfg.winsorize,
                )
            
            # Apply direction and weight
            scores[factor_cfg.name] = (
                factor_values * factor_cfg.direction * factor_cfg.weight
            )
            total_weight += factor_cfg.weight
        
        if scores.empty:
            return pd.Series(dtype=float)
        
        # Weighted average
        composite = scores.sum(axis=1) / total_weight if total_weight > 0 else scores.sum(axis=1)
        return composite.dropna()

    def _scores_to_weights(self, scores: pd.Series) -> Dict[str, float]:
        """Convert composite scores to portfolio weights."""
        if scores.empty:
            return {}
        
        weights = {}
        
        # If using top_n selection
        if self.top_n_long > 0 or self.top_n_short > 0:
            sorted_scores = scores.sort_values(ascending=False)
            
            # Long positions
            if self.top_n_long > 0:
                long_assets = sorted_scores.head(self.top_n_long).index.tolist()
                long_weight = 0.5 / self.top_n_long if self.top_n_long > 0 else 0
                for asset in long_assets:
                    weights[asset] = long_weight
            
            # Short positions
            if self.top_n_short > 0:
                short_assets = sorted_scores.tail(self.top_n_short).index.tolist()
                short_weight = -0.5 / self.top_n_short if self.top_n_short > 0 else 0
                for asset in short_assets:
                    weights[asset] = short_weight
        
        else:
            # Proportional weights based on scores
            long_scores = scores[scores > self.long_threshold]
            short_scores = scores[scores < self.short_threshold]
            
            if len(long_scores) > 0:
                total_long = long_scores.sum()
                for asset, score in long_scores.items():
                    weights[asset] = 0.5 * score / total_long if total_long > 0 else 0
            
            if len(short_scores) > 0:
                total_short = abs(short_scores.sum())
                for asset, score in short_scores.items():
                    weights[asset] = 0.5 * score / total_short if total_short > 0 else 0
        
        return weights


class RankSignalGenerator(BaseSignalGenerator):
    """
    Generate signals based on factor ranking.
    
    Simple strategy:
    - Long top N by factor
    - Short bottom N by factor
    """
    
    def __init__(
        self,
        factor_col: str,
        top_n_long: int = 10,
        top_n_short: int = 10,
        ascending: bool = False,
        equal_weight: bool = True,
        name: str = "RankSignal",
    ):
        """
        Initialize rank signal generator.
        
        Args:
            factor_col: Column to rank by
            top_n_long: Number of assets to long
            top_n_short: Number of assets to short
            ascending: If True, low values are better
            equal_weight: Use equal weights (vs. proportional)
            name: Generator name
        """
        super().__init__(name)
        self.factor_col = factor_col
        self.top_n_long = top_n_long
        self.top_n_short = top_n_short
        self.ascending = ascending
        self.equal_weight = equal_weight

    def generate(
        self,
        data: Union[pd.DataFrame, pd.Series],
        timestamp: Optional[datetime] = None,
    ) -> SignalOutput:
        """Generate signals based on ranking."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if self.factor_col not in data.columns:
            return SignalOutput(timestamp=timestamp)
        
        # Get valid data
        valid_data = data[data[self.factor_col].notna()]
        
        if len(valid_data) < self.top_n_long + self.top_n_short:
            return SignalOutput(timestamp=timestamp)
        
        # Rank
        ranked = valid_data[self.factor_col].sort_values(ascending=self.ascending)
        
        signals = {}
        weights = {}
        strengths = {}
        
        # For ascending=False (descending sort): head = highest, tail = lowest
        # For ascending=True (ascending sort): head = lowest, tail = highest
        top_assets = ranked.head(self.top_n_long).index.tolist()
        bottom_assets = ranked.tail(self.top_n_short).index.tolist()
        
        # Assign signals and weights
        if not self.ascending:
            # High is good → long top (head of descending), short bottom (tail of descending)
            long_assets = top_assets
            short_assets = bottom_assets
        else:
            # Low is good → long bottom, short top
            long_assets = bottom_assets
            short_assets = top_assets
        
        if self.equal_weight:
            long_weight = 0.5 / len(long_assets) if long_assets else 0
            short_weight = -0.5 / len(short_assets) if short_assets else 0
        else:
            # Proportional to rank
            long_total = sum(range(1, len(long_assets) + 1))
            short_total = sum(range(1, len(short_assets) + 1))
        
        for i, asset in enumerate(long_assets):
            signals[asset] = SignalType.BUY
            if self.equal_weight:
                weights[asset] = long_weight
            else:
                weights[asset] = 0.5 * (i + 1) / long_total
            strengths[asset] = 0.5 + 0.5 * (i + 1) / len(long_assets)
        
        for i, asset in enumerate(short_assets):
            signals[asset] = SignalType.SELL
            if self.equal_weight:
                weights[asset] = short_weight
            else:
                weights[asset] = -0.5 * (len(short_assets) - i) / short_total
            strengths[asset] = -0.5 - 0.5 * (len(short_assets) - i) / len(short_assets)
        
        return SignalOutput(
            timestamp=timestamp,
            signals=signals,
            strengths=strengths,
            weights=weights,
        )


class ThresholdSignalGenerator(BaseSignalGenerator):
    """
    Generate signals based on factor thresholds.
    
    Simple threshold-based signals:
    - Factor above upper threshold → long
    - Factor below lower threshold → short
    """
    
    def __init__(
        self,
        factor_col: str,
        long_threshold: float,
        short_threshold: float,
        use_zscore: bool = True,
        zscore_window: int = 20,
        name: str = "ThresholdSignal",
    ):
        """
        Initialize threshold signal generator.
        
        Args:
            factor_col: Column to check thresholds
            long_threshold: Factor above this → long
            short_threshold: Factor below this → short
            use_zscore: Apply z-score normalization
            zscore_window: Window for rolling z-score
            name: Generator name
        """
        super().__init__(name)
        self.factor_col = factor_col
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.use_zscore = use_zscore
        self.zscore_window = zscore_window

    def generate(
        self,
        data: Union[pd.DataFrame, pd.Series],
        timestamp: Optional[datetime] = None,
    ) -> SignalOutput:
        """Generate signals based on thresholds."""
        if timestamp is None:
            timestamp = datetime.now()
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if self.factor_col not in data.columns:
            return SignalOutput(timestamp=timestamp)
        
        values = data[self.factor_col].copy()
        
        # Apply z-score if requested
        if self.use_zscore and len(values) > 0:
            mean = values.mean()
            std = values.std()
            if std > 0:
                values = (values - mean) / std
        
        signals = {}
        strengths = {}
        weights = {}
        
        long_count = 0
        short_count = 0
        
        # First pass: count signals
        for asset, val in values.items():
            if pd.isna(val):
                continue
            if val > self.long_threshold:
                long_count += 1
            elif val < self.short_threshold:
                short_count += 1
        
        # Second pass: assign signals and weights
        for asset, val in values.items():
            if pd.isna(val):
                continue
            
            if val > self.long_threshold:
                signals[asset] = SignalType.BUY
                strengths[asset] = float(np.clip(val, 0, 1))
                weights[asset] = 0.5 / long_count if long_count > 0 else 0
            elif val < self.short_threshold:
                signals[asset] = SignalType.SELL
                strengths[asset] = float(np.clip(val, -1, 0))
                weights[asset] = -0.5 / short_count if short_count > 0 else 0
            else:
                signals[asset] = SignalType.HOLD
                strengths[asset] = 0.0
        
        return SignalOutput(
            timestamp=timestamp,
            signals=signals,
            strengths=strengths,
            weights=weights,
        )
