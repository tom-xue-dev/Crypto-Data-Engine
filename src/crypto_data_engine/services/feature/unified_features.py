"""
Unified feature calculation system.

Combines:
- Rolling features (from feature_calculator.py)
- Alpha factors (from Factor.py)
- Cross-sectional features
- Custom feature functions

Usage:
    from crypto_data_engine.services.feature import calculate_features
    
    # Simple usage
    features = calculate_features(bar_data)
    
    # With configuration
    features = calculate_features(
        bar_data,
        windows=[5, 10, 20, 60],
        include_alphas=True,
        include_microstructure=True,
    )
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Try to import talib (optional)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


@dataclass
class UnifiedFeatureConfig:
    """Configuration for unified feature calculation."""
    
    # Rolling windows
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60, 120])
    
    # Feature groups
    include_returns: bool = True
    include_volatility: bool = True
    include_momentum: bool = True
    include_volume: bool = True
    include_microstructure: bool = True
    include_alphas: bool = True
    include_technical: bool = True
    
    # Alpha factors to calculate
    alpha_list: Optional[List[str]] = None  # None = all available
    
    # Cross-sectional
    include_cross_sectional: bool = False
    rank_columns: List[str] = field(default_factory=lambda: ["return_20", "volatility_20"])
    
    # Processing
    n_jobs: int = 1  # Number of parallel jobs for alpha calculation
    drop_na: bool = False
    
    # Normalization
    normalize: bool = False
    winsorize_std: Optional[float] = 3.0  # Winsorize at N standard deviations


class UnifiedFeatureCalculator:
    """
    Unified feature calculator combining multiple feature sources.
    
    Features:
    - Rolling returns, volatility, momentum
    - Volume and microstructure features
    - Alpha factors (26+ available)
    - Technical indicators (if talib available)
    - Cross-sectional ranks and z-scores
    """
    
    def __init__(self, config: Optional[UnifiedFeatureConfig] = None):
        self.config = config or UnifiedFeatureConfig()
    
    def calculate(
        self,
        data: pd.DataFrame,
        asset: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate all configured features.
        
        Args:
            data: Bar data with OHLCV columns
            asset: Asset identifier (for multi-asset data)
            
        Returns:
            DataFrame with original data and calculated features
        """
        df = data.copy()
        
        # Basic features
        if self.config.include_returns:
            df = self._calc_returns(df)
        
        if self.config.include_volatility:
            df = self._calc_volatility(df)
        
        if self.config.include_momentum:
            df = self._calc_momentum(df)
        
        if self.config.include_volume:
            df = self._calc_volume_features(df)
        
        if self.config.include_microstructure:
            df = self._calc_microstructure(df)
        
        # Alpha factors
        if self.config.include_alphas:
            df = self._calc_alpha_factors(df)
        
        # Technical indicators
        if self.config.include_technical and TALIB_AVAILABLE:
            df = self._calc_technical_indicators(df)
        
        # Post-processing
        if self.config.normalize:
            df = self._normalize_features(df)
        
        if self.config.drop_na:
            df = df.dropna()
        
        return df
    
    def calculate_multi_asset(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate features for multi-asset data.
        
        Args:
            data: DataFrame with MultiIndex (timestamp, asset) or 'asset' column
            
        Returns:
            DataFrame with features calculated per asset
        """
        # Prepare data
        if isinstance(data.index, pd.MultiIndex):
            df = data.reset_index()
        else:
            df = data.copy()
        
        if "asset" not in df.columns:
            raise ValueError("Data must have 'asset' column or MultiIndex with asset level")
        
        # Calculate features per asset
        results = []
        for asset, group in df.groupby("asset"):
            group_features = self.calculate(group, asset=asset)
            group_features["asset"] = asset
            results.append(group_features)
        
        result = pd.concat(results, ignore_index=True)
        
        # Cross-sectional features
        if self.config.include_cross_sectional:
            result = self._calc_cross_sectional(result)
        
        return result
    
    def _calc_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate return features."""
        if "close" not in df.columns:
            return df
        
        # Single period returns
        df["return_1"] = df["close"].pct_change()
        df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
        
        for window in self.config.windows:
            df[f"return_{window}"] = df["close"].pct_change(window)
            df[f"log_return_{window}"] = np.log(df["close"] / df["close"].shift(window))
        
        return df
    
    def _calc_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features."""
        if "return_1" not in df.columns:
            if "close" in df.columns:
                df["return_1"] = df["close"].pct_change()
            else:
                return df
        
        for window in self.config.windows:
            df[f"volatility_{window}"] = df["return_1"].rolling(window).std()
            df[f"volatility_ann_{window}"] = df[f"volatility_{window}"] * np.sqrt(252)
            
            # Parkinson volatility: sqrt(1/(4*n*ln(2)) * sum(ln(H/L)^2))
            if "high" in df.columns and "low" in df.columns:
                log_hl = np.log(df["high"] / df["low"])
                df[f"parkinson_{window}"] = log_hl.rolling(window).apply(
                    lambda x: np.sqrt(np.sum(x ** 2) / (4 * len(x) * np.log(2))),
                    raw=True,
                )
        
        return df
    
    def _calc_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features."""
        if "close" not in df.columns:
            return df
        
        for window in self.config.windows:
            # Price momentum
            df[f"momentum_{window}"] = df["close"] / df["close"].shift(window) - 1
            
            # Moving averages
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
            
            # Price relative to SMA
            df[f"price_sma_{window}"] = df["close"] / df[f"sma_{window}"]
            
            # Bollinger position
            bb_std = df["close"].rolling(window).std()
            bb_upper = df[f"sma_{window}"] + 2 * bb_std
            bb_lower = df[f"sma_{window}"] - 2 * bb_std
            bb_range = bb_upper - bb_lower
            df[f"bb_pos_{window}"] = np.where(
                bb_range > 0,
                (df["close"] - bb_lower) / bb_range,
                0.5
            )
        
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        for window in [14] + [w for w in self.config.windows if w not in [14]]:
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calc_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume features."""
        if "volume" not in df.columns:
            return df
        
        for window in self.config.windows:
            df[f"volume_sma_{window}"] = df["volume"].rolling(window).mean()
            df[f"volume_ratio_{window}"] = df["volume"] / df[f"volume_sma_{window}"]
            
            if "dollar_volume" in df.columns:
                df[f"dollar_vol_sma_{window}"] = df["dollar_volume"].rolling(window).mean()
        
        # Buy/sell imbalance
        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            df["buy_ratio"] = df["buy_volume"] / df["volume"]
            df["imbalance"] = (df["buy_volume"] - df["sell_volume"]) / df["volume"]
        
        return df
    
    def _calc_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate microstructure features."""
        # Up move ratio
        if "up_move_ratio" in df.columns:
            for window in self.config.windows:
                df[f"up_ratio_sma_{window}"] = df["up_move_ratio"].rolling(window).mean()
        
        # Reversal features
        if "reversals" in df.columns:
            for window in self.config.windows:
                df[f"reversal_rate_{window}"] = df["reversals"].rolling(window).mean()
        
        # Amihud illiquidity
        if "return_1" in df.columns and "dollar_volume" in df.columns:
            amihud = df["return_1"].abs() / df["dollar_volume"].replace(0, np.nan)
            for window in self.config.windows:
                df[f"amihud_{window}"] = amihud.rolling(window).mean()
        
        # Path Efficiency: rolling mean detects trending vs choppy regimes
        if "path_efficiency" in df.columns:
            for window in self.config.windows:
                df[f"path_efficiency_{window}"] = df["path_efficiency"].rolling(window).mean()

        # Signed Path Efficiency: PE × sign(return), so high = strong uptrend,
        # low = strong downtrend.  This makes it usable as a momentum-like
        # ranking factor (long top = long trending-up, short bottom = short
        # trending-down).
        if "path_efficiency" in df.columns and "close" in df.columns:
            bar_return_sign = np.sign(df["close"] - df["open"]) if "open" in df.columns else np.sign(df["close"].diff())
            df["signed_path_efficiency"] = df["path_efficiency"] * bar_return_sign
            for window in self.config.windows:
                df[f"signed_pe_{window}"] = df["signed_path_efficiency"].rolling(window).mean()

        # Intrabar Impact Density: rolling mean detects liquidity regimes
        if "impact_density" in df.columns:
            for window in self.config.windows:
                df[f"impact_density_{window}"] = df["impact_density"].rolling(window).mean()
        
        # VWAP deviation
        if "vwap" in df.columns and "close" in df.columns:
            df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"]
            
            if "medium_price" in df.columns:
                df["vwap_median_dev"] = (df["vwap"] - df["medium_price"]) / df["close"]
        
        return df
    
    def _calc_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate alpha factors."""
        # Alpha 1: Up move ratio based
        if "up_move_ratio" in df.columns:
            for window in [60, 120]:
                df[f"alpha_up_ratio_{window}"] = df["up_move_ratio"].rolling(window).sum()
        
        # Alpha 2: Amihud-like factor
        if "close" in df.columns and "volume" in df.columns and "vwap" in df.columns:
            for window in [60]:
                ret = df["close"].pct_change(window)
                dollar = df["volume"] * df["vwap"]
                amount = np.log10(dollar.rolling(window).sum().replace(0, 1))
                df[f"alpha_amihud_{window}"] = -ret * 1e3 / amount
        
        # Alpha 3: Mean reversion (deviation from MA)
        if "close" in df.columns:
            for window in [60]:
                ma = df["close"].rolling(window).mean()
                std_ratio = df["close"].rolling(window).std() / df["close"].rolling(window * 5).std()
                std_ratio = std_ratio.fillna(1)
                df[f"alpha_deviation_{window}"] = -((df["close"] - ma) / ma) * std_ratio
        
        # Alpha 4: Momentum with volume confirmation
        if "close" in df.columns and "buy_volume" in df.columns and "volume" in df.columns:
            for window in [120]:
                ret = (df["close"] - df["close"].shift(window)) / df["close"]
                buyer = df["buy_volume"]
                seller = df["volume"] - df["buy_volume"]
                imbalance = (buyer - seller) / df["volume"]
                imb_sum = imbalance.rolling(window).sum()
                signed_ret = np.sign(ret) * np.log1p(np.abs(ret)) ** 4
                df[f"alpha_vol_conf_{window}"] = 100 * imb_sum * signed_ret
        
        # Alpha 5: VWAP deviation rolling
        if "vwap" in df.columns and "close" in df.columns:
            for window in [30]:
                vals = (df["vwap"] - df["close"]) / df["close"]
                df[f"alpha_vwap_{window}"] = vals.rolling(window).mean()
        
        # Alpha 6: Trend strength
        if "close" in df.columns:
            for window in [120]:
                price = df["close"]
                net_change = price.diff(window * 10)
                total_movement = price.diff().abs().rolling(window * 10).sum()
                trend_strength = net_change / total_movement.replace(0, np.nan)
                volume_score = df["volume"].rolling(window).mean() / df["volume"] if "volume" in df.columns else 1
                df[f"alpha_trend_{window}"] = trend_strength * volume_score
        
        return df
    
    def _calc_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using talib."""
        if not TALIB_AVAILABLE:
            return df
        
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
        
        try:
            # ADX
            df["adx_14"] = talib.ADX(high, low, close, timeperiod=14)
            
            # MACD
            macd, signal, hist = talib.MACD(close)
            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = hist
            
            # ATR
            df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
            
            # OBV
            df["obv"] = talib.OBV(close, volume.astype(float))
            
            # CCI
            df["cci_14"] = talib.CCI(high, low, close, timeperiod=14)
            
        except Exception as error:
            import logging
            logging.getLogger(__name__).warning(
                f"talib technical indicator calculation failed: {error}"
            )
        
        return df
    
    def _calc_cross_sectional(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-sectional features."""
        if "timestamp" not in df.columns and "start_time" not in df.columns:
            return df
        
        time_col = "timestamp" if "timestamp" in df.columns else "start_time"
        
        for col in self.config.rank_columns:
            if col in df.columns:
                # Rank (0-1)
                df[f"{col}_rank"] = df.groupby(time_col)[col].transform(
                    lambda x: x.rank(pct=True)
                )
                
                # Z-score
                df[f"{col}_zscore"] = df.groupby(time_col)[col].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                )
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize feature columns."""
        feature_cols = [c for c in df.columns if c not in [
            "timestamp", "start_time", "end_time", "asset",
            "open", "high", "low", "close", "volume",
            "buy_volume", "sell_volume", "vwap",
        ]]
        
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    df[col] = (df[col] - mean) / std
                    
                    # Winsorize
                    if self.config.winsorize_std:
                        df[col] = df[col].clip(
                            lower=-self.config.winsorize_std,
                            upper=self.config.winsorize_std,
                        )
        
        return df


def calculate_features(
    data: pd.DataFrame,
    windows: Optional[List[int]] = None,
    include_alphas: bool = True,
    include_microstructure: bool = True,
    include_technical: bool = True,
    normalize: bool = False,
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Convenience function for calculating features.
    
    Args:
        data: Bar data with OHLCV columns
        windows: List of rolling windows (default: [5, 10, 20, 60, 120])
        include_alphas: Include alpha factors
        include_microstructure: Include microstructure features
        include_technical: Include technical indicators (requires talib)
        normalize: Normalize features
        drop_na: Drop rows with NaN values
        
    Returns:
        DataFrame with calculated features
    """
    config = UnifiedFeatureConfig(
        windows=windows or [5, 10, 20, 60, 120],
        include_alphas=include_alphas,
        include_microstructure=include_microstructure,
        include_technical=include_technical,
        normalize=normalize,
        drop_na=drop_na,
    )
    
    calculator = UnifiedFeatureCalculator(config)
    return calculator.calculate(data)


def calculate_features_multi_asset(
    data: pd.DataFrame,
    windows: Optional[List[int]] = None,
    include_cross_sectional: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate features for multi-asset data.
    
    Args:
        data: Bar data with 'asset' column or MultiIndex
        windows: List of rolling windows
        include_cross_sectional: Include cross-sectional ranks and z-scores
        **kwargs: Additional arguments for UnifiedFeatureConfig
        
    Returns:
        DataFrame with features calculated per asset
    """
    config = UnifiedFeatureConfig(
        windows=windows or [5, 10, 20, 60, 120],
        include_cross_sectional=include_cross_sectional,
        **kwargs,
    )
    
    calculator = UnifiedFeatureCalculator(config)
    return calculator.calculate_multi_asset(data)


# Feature selection utilities
def select_features_by_correlation(
    features: pd.DataFrame,
    target: pd.Series,
    top_n: int = 50,
    min_correlation: float = 0.05,
) -> List[str]:
    """
    Select top features by correlation with target.
    
    Args:
        features: Feature DataFrame
        target: Target variable
        top_n: Number of features to select
        min_correlation: Minimum absolute correlation
        
    Returns:
        List of selected feature names
    """
    correlations = {}
    
    for col in features.columns:
        if features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            try:
                corr = features[col].corr(target)
                if not np.isnan(corr) and abs(corr) >= min_correlation:
                    correlations[col] = abs(corr)
            except Exception:
                continue
    
    # Sort by correlation
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    return [f[0] for f in sorted_features[:top_n]]


def get_feature_importance(
    features: pd.DataFrame,
    target: pd.Series,
    method: str = "mutual_info",
) -> pd.Series:
    """
    Calculate feature importance scores.
    
    Args:
        features: Feature DataFrame
        target: Target variable
        method: Importance method ('mutual_info', 'correlation', 'variance')
        
    Returns:
        Series of importance scores
    """
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    
    if method == "correlation":
        scores = features[numeric_cols].corrwith(target).abs()
    
    elif method == "variance":
        scores = features[numeric_cols].var()
    
    elif method == "mutual_info":
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            X = features[numeric_cols].fillna(0)
            y = target.fillna(0)
            
            mi_scores = mutual_info_regression(X, y)
            scores = pd.Series(mi_scores, index=numeric_cols)
        except ImportError:
            # Fall back to correlation
            scores = features[numeric_cols].corrwith(target).abs()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return scores.sort_values(ascending=False)
