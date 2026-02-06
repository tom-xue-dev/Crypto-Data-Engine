"""
Ensemble signal generators.

Combine multiple signal sources for more robust signals.
"""
from __future__ import annotations

from dataclasses import dataclass
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


class EnsembleMethod(str, Enum):
    """Methods for combining signals."""
    
    AVERAGE = "average"
    """Simple average of signal strengths."""
    
    WEIGHTED_AVERAGE = "weighted_average"
    """Weighted average based on generator weights."""
    
    VOTING = "voting"
    """Majority voting (discrete signals)."""
    
    UNANIMOUS = "unanimous"
    """All generators must agree."""
    
    MAX = "max"
    """Use strongest signal."""
    
    MIN = "min"
    """Use weakest signal (most conservative)."""


@dataclass
class GeneratorConfig:
    """Configuration for a generator in the ensemble."""
    
    generator: BaseSignalGenerator
    """The signal generator."""
    
    weight: float = 1.0
    """Weight in the ensemble."""
    
    min_confidence: float = 0.0
    """Minimum confidence to include signal."""


class EnsembleSignalGenerator(BaseSignalGenerator):
    """
    Combine multiple signal generators into an ensemble.
    
    Supports various combination methods:
    - Average: Simple average of signal strengths
    - Weighted average: Weighted by generator importance
    - Voting: Majority vote on discrete signals
    - Unanimous: All generators must agree
    
    Examples:
        # Create ensemble from multiple generators
        ensemble = EnsembleSignalGenerator(
            generators=[
                GeneratorConfig(momentum_gen, weight=2.0),
                GeneratorConfig(mean_rev_gen, weight=1.0),
                GeneratorConfig(rule_gen, weight=1.5),
            ],
            method=EnsembleMethod.WEIGHTED_AVERAGE,
        )
        
        output = ensemble.generate(data)
    """
    
    def __init__(
        self,
        generators: List[GeneratorConfig],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        long_threshold: float = 0.3,
        short_threshold: float = -0.3,
        min_agreement: float = 0.5,
        name: str = "EnsembleSignal",
    ):
        """
        Initialize ensemble signal generator.
        
        Args:
            generators: List of generator configurations
            method: How to combine signals
            long_threshold: Combined strength above this → long
            short_threshold: Combined strength below this → short
            min_agreement: Minimum fraction of generators agreeing (for voting)
            name: Generator name
        """
        super().__init__(name)
        self.generators = generators
        self.method = method
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.min_agreement = min_agreement

    def generate(
        self,
        data: Union[pd.DataFrame, pd.Series],
        timestamp: Optional[datetime] = None,
    ) -> SignalOutput:
        """
        Generate combined signals from all generators.
        
        Args:
            data: Input data for all generators
            timestamp: Current timestamp
            
        Returns:
            Combined SignalOutput
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Collect outputs from all generators
        outputs = []
        for gen_config in self.generators:
            try:
                output = gen_config.generator.generate(data, timestamp)
                outputs.append((gen_config, output))
            except Exception:
                continue
        
        if not outputs:
            return SignalOutput(timestamp=timestamp)
        
        # Combine based on method
        if self.method == EnsembleMethod.AVERAGE:
            return self._combine_average(timestamp, outputs)
        elif self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._combine_weighted_average(timestamp, outputs)
        elif self.method == EnsembleMethod.VOTING:
            return self._combine_voting(timestamp, outputs)
        elif self.method == EnsembleMethod.UNANIMOUS:
            return self._combine_unanimous(timestamp, outputs)
        elif self.method == EnsembleMethod.MAX:
            return self._combine_max(timestamp, outputs)
        elif self.method == EnsembleMethod.MIN:
            return self._combine_min(timestamp, outputs)
        else:
            return self._combine_average(timestamp, outputs)

    def _combine_average(
        self,
        timestamp: datetime,
        outputs: List[tuple[GeneratorConfig, SignalOutput]],
    ) -> SignalOutput:
        """Combine using simple average."""
        all_assets = set()
        for _, output in outputs:
            all_assets.update(output.strengths.keys())
            all_assets.update(output.weights.keys())
        
        combined_strengths = {}
        combined_weights = {}
        
        for asset in all_assets:
            strengths = [
                out.strengths.get(asset, 0)
                for _, out in outputs
                if asset in out.strengths or asset in out.weights
            ]
            weights = [
                out.weights.get(asset, 0)
                for _, out in outputs
                if asset in out.weights
            ]
            
            if strengths:
                combined_strengths[asset] = np.mean(strengths)
            if weights:
                combined_weights[asset] = np.mean(weights)
        
        return self._create_output(timestamp, combined_strengths, combined_weights)

    def _combine_weighted_average(
        self,
        timestamp: datetime,
        outputs: List[tuple[GeneratorConfig, SignalOutput]],
    ) -> SignalOutput:
        """Combine using weighted average."""
        all_assets = set()
        for _, output in outputs:
            all_assets.update(output.strengths.keys())
            all_assets.update(output.weights.keys())
        
        combined_strengths = {}
        combined_weights = {}
        
        for asset in all_assets:
            weighted_strength_sum = 0
            weighted_weight_sum = 0
            total_weight = 0
            
            for config, output in outputs:
                if asset in output.strengths:
                    weighted_strength_sum += output.strengths[asset] * config.weight
                    total_weight += config.weight
                
                if asset in output.weights:
                    weighted_weight_sum += output.weights[asset] * config.weight
            
            if total_weight > 0:
                combined_strengths[asset] = weighted_strength_sum / total_weight
            
            total_gen_weight = sum(c.weight for c, _ in outputs)
            if total_gen_weight > 0:
                combined_weights[asset] = weighted_weight_sum / total_gen_weight
        
        return self._create_output(timestamp, combined_strengths, combined_weights)

    def _combine_voting(
        self,
        timestamp: datetime,
        outputs: List[tuple[GeneratorConfig, SignalOutput]],
    ) -> SignalOutput:
        """Combine using majority voting."""
        all_assets = set()
        for _, output in outputs:
            all_assets.update(output.signals.keys())
        
        signals = {}
        strengths = {}
        weights = {}
        
        for asset in all_assets:
            votes = {SignalType.BUY: 0, SignalType.SELL: 0, SignalType.HOLD: 0}
            weight_sum = {SignalType.BUY: 0, SignalType.SELL: 0, SignalType.HOLD: 0}
            
            for config, output in outputs:
                if asset in output.signals:
                    signal = output.signals[asset]
                    votes[signal] += 1
                    weight_sum[signal] += config.weight
            
            total_votes = sum(votes.values())
            if total_votes == 0:
                continue
            
            # Find majority
            max_signal = max(votes.keys(), key=lambda s: (votes[s], weight_sum[s]))
            agreement = votes[max_signal] / total_votes
            
            if agreement >= self.min_agreement:
                signals[asset] = max_signal
                strengths[asset] = agreement if max_signal == SignalType.BUY else (-agreement if max_signal == SignalType.SELL else 0)
                
                # Average weights from agreeing generators
                agreeing_weights = [
                    out.weights.get(asset, 0)
                    for cfg, out in outputs
                    if out.signals.get(asset) == max_signal
                ]
                weights[asset] = np.mean(agreeing_weights) if agreeing_weights else 0
            else:
                signals[asset] = SignalType.HOLD
                strengths[asset] = 0
        
        return SignalOutput(
            timestamp=timestamp,
            signals=signals,
            strengths=strengths,
            weights=weights,
        )

    def _combine_unanimous(
        self,
        timestamp: datetime,
        outputs: List[tuple[GeneratorConfig, SignalOutput]],
    ) -> SignalOutput:
        """Combine requiring unanimous agreement."""
        all_assets = set()
        for _, output in outputs:
            all_assets.update(output.signals.keys())
        
        signals = {}
        strengths = {}
        weights = {}
        
        for asset in all_assets:
            asset_signals = [
                out.signals.get(asset)
                for _, out in outputs
                if asset in out.signals
            ]
            
            if not asset_signals:
                continue
            
            # Check if all non-HOLD signals agree
            non_hold_signals = [s for s in asset_signals if s != SignalType.HOLD]
            
            if len(non_hold_signals) == 0:
                signals[asset] = SignalType.HOLD
                strengths[asset] = 0
            elif len(set(non_hold_signals)) == 1:
                # Unanimous non-HOLD signal
                signals[asset] = non_hold_signals[0]
                
                asset_strengths = [
                    out.strengths.get(asset, 0)
                    for _, out in outputs
                    if asset in out.strengths
                ]
                strengths[asset] = np.mean(asset_strengths) if asset_strengths else 0
                
                asset_weights = [
                    out.weights.get(asset, 0)
                    for _, out in outputs
                    if asset in out.weights
                ]
                weights[asset] = np.mean(asset_weights) if asset_weights else 0
            else:
                # Disagreement → HOLD
                signals[asset] = SignalType.HOLD
                strengths[asset] = 0
        
        return SignalOutput(
            timestamp=timestamp,
            signals=signals,
            strengths=strengths,
            weights=weights,
        )

    def _combine_max(
        self,
        timestamp: datetime,
        outputs: List[tuple[GeneratorConfig, SignalOutput]],
    ) -> SignalOutput:
        """Use the strongest signal for each asset."""
        all_assets = set()
        for _, output in outputs:
            all_assets.update(output.strengths.keys())
        
        combined_strengths = {}
        combined_weights = {}
        
        for asset in all_assets:
            max_strength = 0
            max_weight = 0
            
            for _, output in outputs:
                if asset in output.strengths:
                    if abs(output.strengths[asset]) > abs(max_strength):
                        max_strength = output.strengths[asset]
                        max_weight = output.weights.get(asset, 0)
            
            combined_strengths[asset] = max_strength
            combined_weights[asset] = max_weight
        
        return self._create_output(timestamp, combined_strengths, combined_weights)

    def _combine_min(
        self,
        timestamp: datetime,
        outputs: List[tuple[GeneratorConfig, SignalOutput]],
    ) -> SignalOutput:
        """Use the weakest signal for each asset (conservative)."""
        all_assets = set()
        for _, output in outputs:
            all_assets.update(output.strengths.keys())
        
        combined_strengths = {}
        combined_weights = {}
        
        for asset in all_assets:
            min_abs_strength = float('inf')
            min_strength = 0
            min_weight = 0
            
            for _, output in outputs:
                if asset in output.strengths:
                    strength = output.strengths[asset]
                    if abs(strength) < min_abs_strength:
                        min_abs_strength = abs(strength)
                        min_strength = strength
                        min_weight = output.weights.get(asset, 0)
            
            if min_abs_strength != float('inf'):
                combined_strengths[asset] = min_strength
                combined_weights[asset] = min_weight
        
        return self._create_output(timestamp, combined_strengths, combined_weights)

    def _create_output(
        self,
        timestamp: datetime,
        strengths: Dict[str, float],
        weights: Dict[str, float],
    ) -> SignalOutput:
        """Create output from strengths and weights."""
        signals = {}
        
        for asset, strength in strengths.items():
            if strength > self.long_threshold:
                signals[asset] = SignalType.BUY
            elif strength < self.short_threshold:
                signals[asset] = SignalType.SELL
            else:
                signals[asset] = SignalType.HOLD
        
        return SignalOutput(
            timestamp=timestamp,
            signals=signals,
            strengths=strengths,
            weights=weights,
        )
