"""
Rule-based signal generators.

Generate trading signals based on explicit rules/conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from crypto_data_engine.core.base import SignalType
from crypto_data_engine.services.signal_generation.base import (
    BaseSignalGenerator,
    SignalOutput,
)


class ComparisonOperator(str, Enum):
    """Comparison operators for rule conditions."""
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    BETWEEN = "between"
    NOT_BETWEEN = "not_between"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"


@dataclass
class RuleCondition:
    """
    A single rule condition for signal generation.
    
    Examples:
        # Simple threshold
        RuleCondition("close", ComparisonOperator.GREATER, "sma_20")
        
        # Numeric comparison
        RuleCondition("rsi", ComparisonOperator.LESS, 30)
        
        # Between range
        RuleCondition("zscore", ComparisonOperator.BETWEEN, (-2, 2))
    """
    
    column: str
    """Column to evaluate."""
    
    operator: ComparisonOperator
    """Comparison operator."""
    
    value: Union[float, str, tuple]
    """
    Value to compare against:
    - float: Direct comparison
    - str: Another column name
    - tuple: Range for BETWEEN operators
    """
    
    weight: float = 1.0
    """Weight of this condition in composite signals."""
    
    signal_on_true: SignalType = SignalType.BUY
    """Signal to generate when condition is True."""
    
    signal_on_false: SignalType = SignalType.HOLD
    """Signal to generate when condition is False."""

    def evaluate(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series] = None,
    ) -> bool:
        """
        Evaluate the condition for a single row.
        
        Args:
            row: Current data row
            prev_row: Previous row (for crossover conditions)
            
        Returns:
            True if condition is met
        """
        if self.column not in row.index:
            return False
        
        current_value = row[self.column]
        
        if pd.isna(current_value):
            return False
        
        # Get comparison value
        if isinstance(self.value, str):
            if self.value not in row.index:
                return False
            compare_value = row[self.value]
        else:
            compare_value = self.value
        
        # Evaluate based on operator
        if self.operator == ComparisonOperator.GREATER:
            return current_value > compare_value
        
        elif self.operator == ComparisonOperator.GREATER_EQUAL:
            return current_value >= compare_value
        
        elif self.operator == ComparisonOperator.LESS:
            return current_value < compare_value
        
        elif self.operator == ComparisonOperator.LESS_EQUAL:
            return current_value <= compare_value
        
        elif self.operator == ComparisonOperator.EQUAL:
            return current_value == compare_value
        
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return current_value != compare_value
        
        elif self.operator == ComparisonOperator.BETWEEN:
            if not isinstance(compare_value, (tuple, list)) or len(compare_value) != 2:
                return False
            return compare_value[0] <= current_value <= compare_value[1]
        
        elif self.operator == ComparisonOperator.NOT_BETWEEN:
            if not isinstance(compare_value, (tuple, list)) or len(compare_value) != 2:
                return False
            return current_value < compare_value[0] or current_value > compare_value[1]
        
        elif self.operator == ComparisonOperator.CROSS_ABOVE:
            if prev_row is None or self.column not in prev_row.index:
                return False
            prev_value = prev_row[self.column]
            if isinstance(self.value, str):
                if self.value not in prev_row.index:
                    return False
                prev_compare = prev_row[self.value]
            else:
                prev_compare = self.value
            return prev_value <= prev_compare and current_value > compare_value
        
        elif self.operator == ComparisonOperator.CROSS_BELOW:
            if prev_row is None or self.column not in prev_row.index:
                return False
            prev_value = prev_row[self.column]
            if isinstance(self.value, str):
                if self.value not in prev_row.index:
                    return False
                prev_compare = prev_row[self.value]
            else:
                prev_compare = self.value
            return prev_value >= prev_compare and current_value < compare_value
        
        return False


class RuleSignalGenerator(BaseSignalGenerator):
    """
    Generate signals based on rule conditions.
    
    Supports:
    - Multiple conditions with AND/OR logic
    - Weighted condition evaluation
    - Custom signal mapping
    
    Examples:
        # RSI oversold strategy
        generator = RuleSignalGenerator(
            long_conditions=[
                RuleCondition("rsi", ComparisonOperator.LESS, 30),
            ],
            short_conditions=[
                RuleCondition("rsi", ComparisonOperator.GREATER, 70),
            ],
        )
        
        # Moving average crossover
        generator = RuleSignalGenerator(
            long_conditions=[
                RuleCondition("sma_10", ComparisonOperator.CROSS_ABOVE, "sma_50"),
            ],
            short_conditions=[
                RuleCondition("sma_10", ComparisonOperator.CROSS_BELOW, "sma_50"),
            ],
        )
    """
    
    def __init__(
        self,
        long_conditions: List[RuleCondition],
        short_conditions: Optional[List[RuleCondition]] = None,
        require_all_long: bool = True,
        require_all_short: bool = True,
        min_conditions_long: int = 1,
        min_conditions_short: int = 1,
        name: str = "RuleSignal",
    ):
        """
        Initialize rule signal generator.
        
        Args:
            long_conditions: Conditions for long signals
            short_conditions: Conditions for short signals
            require_all_long: Require ALL long conditions (AND logic)
            require_all_short: Require ALL short conditions (AND logic)
            min_conditions_long: Minimum conditions to trigger long (OR logic)
            min_conditions_short: Minimum conditions to trigger short (OR logic)
            name: Generator name
        """
        super().__init__(name)
        self.long_conditions = long_conditions
        self.short_conditions = short_conditions or []
        self.require_all_long = require_all_long
        self.require_all_short = require_all_short
        self.min_conditions_long = min_conditions_long
        self.min_conditions_short = min_conditions_short

    def generate(
        self,
        data: Union[pd.DataFrame, pd.Series],
        timestamp: Optional[datetime] = None,
        prev_data: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ) -> SignalOutput:
        """
        Generate signals based on rule conditions.
        
        Args:
            data: Current data (cross-section with assets as index)
            timestamp: Current timestamp
            prev_data: Previous period data (for crossover conditions)
            
        Returns:
            SignalOutput with rule-based signals
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        
        if prev_data is not None and isinstance(prev_data, pd.Series):
            prev_data = prev_data.to_frame().T
        
        signals = {}
        strengths = {}
        weights = {}
        
        long_count = 0
        short_count = 0
        
        # First pass: count signals
        for asset in data.index:
            row = data.loc[asset]
            prev_row = prev_data.loc[asset] if prev_data is not None and asset in prev_data.index else None
            
            is_long, long_strength = self._evaluate_conditions(
                row, prev_row, self.long_conditions, self.require_all_long, self.min_conditions_long
            )
            is_short, short_strength = self._evaluate_conditions(
                row, prev_row, self.short_conditions, self.require_all_short, self.min_conditions_short
            )
            
            if is_long and not is_short:
                long_count += 1
            elif is_short and not is_long:
                short_count += 1
        
        # Second pass: assign signals and weights
        for asset in data.index:
            row = data.loc[asset]
            prev_row = prev_data.loc[asset] if prev_data is not None and asset in prev_data.index else None
            
            is_long, long_strength = self._evaluate_conditions(
                row, prev_row, self.long_conditions, self.require_all_long, self.min_conditions_long
            )
            is_short, short_strength = self._evaluate_conditions(
                row, prev_row, self.short_conditions, self.require_all_short, self.min_conditions_short
            )
            
            if is_long and not is_short:
                signals[asset] = SignalType.BUY
                strengths[asset] = long_strength
                weights[asset] = 0.5 / long_count if long_count > 0 else 0
            elif is_short and not is_long:
                signals[asset] = SignalType.SELL
                strengths[asset] = -short_strength
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

    def _evaluate_conditions(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        conditions: List[RuleCondition],
        require_all: bool,
        min_conditions: int,
    ) -> tuple[bool, float]:
        """
        Evaluate a list of conditions.
        
        Returns:
            Tuple of (is_triggered, strength)
        """
        if not conditions:
            return False, 0.0
        
        results = []
        total_weight = 0
        weighted_sum = 0
        
        for condition in conditions:
            is_met = condition.evaluate(row, prev_row)
            results.append(is_met)
            
            if is_met:
                weighted_sum += condition.weight
            total_weight += condition.weight
        
        met_count = sum(results)
        
        if require_all:
            is_triggered = all(results)
        else:
            is_triggered = met_count >= min_conditions
        
        strength = weighted_sum / total_weight if total_weight > 0 else 0
        
        return is_triggered, strength


class TechnicalRuleGenerator(RuleSignalGenerator):
    """
    Pre-built technical analysis rule generator.
    
    Common technical analysis strategies.
    """
    
    @classmethod
    def rsi_strategy(
        cls,
        oversold: float = 30,
        overbought: float = 70,
        rsi_col: str = "rsi",
    ) -> RuleSignalGenerator:
        """Create RSI mean-reversion strategy."""
        return cls(
            long_conditions=[
                RuleCondition(rsi_col, ComparisonOperator.LESS, oversold),
            ],
            short_conditions=[
                RuleCondition(rsi_col, ComparisonOperator.GREATER, overbought),
            ],
            name="RSI_Strategy",
        )
    
    @classmethod
    def ma_crossover(
        cls,
        fast_ma: str = "sma_10",
        slow_ma: str = "sma_50",
    ) -> RuleSignalGenerator:
        """Create moving average crossover strategy."""
        return cls(
            long_conditions=[
                RuleCondition(fast_ma, ComparisonOperator.CROSS_ABOVE, slow_ma),
            ],
            short_conditions=[
                RuleCondition(fast_ma, ComparisonOperator.CROSS_BELOW, slow_ma),
            ],
            name="MA_Crossover",
        )
    
    @classmethod
    def bollinger_bands(
        cls,
        price_col: str = "close",
        lower_band: str = "bb_lower",
        upper_band: str = "bb_upper",
    ) -> RuleSignalGenerator:
        """Create Bollinger Bands mean-reversion strategy."""
        return cls(
            long_conditions=[
                RuleCondition(price_col, ComparisonOperator.LESS, lower_band),
            ],
            short_conditions=[
                RuleCondition(price_col, ComparisonOperator.GREATER, upper_band),
            ],
            name="BB_Strategy",
        )
    
    @classmethod
    def momentum(
        cls,
        momentum_col: str = "momentum_20",
        threshold: float = 0.05,
    ) -> RuleSignalGenerator:
        """Create momentum breakout strategy."""
        return cls(
            long_conditions=[
                RuleCondition(momentum_col, ComparisonOperator.GREATER, threshold),
            ],
            short_conditions=[
                RuleCondition(momentum_col, ComparisonOperator.LESS, -threshold),
            ],
            name="Momentum_Strategy",
        )
