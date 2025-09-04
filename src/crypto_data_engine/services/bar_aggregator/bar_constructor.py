import pandas as pd
import numpy as np
from numba import njit
from datetime import datetime, timezone
from scipy.stats import skew, kurtosis


@njit
def get_volume_bar_indices(volume_arr, threshold):
    indices = []
    cum_volume = 0
    for i in range(len(volume_arr)):
        cum_volume += volume_arr[i]
        if cum_volume >= threshold:
            indices.append(i)
            cum_volume = 0
    return indices


class BarConstructor:
    """构建不同类型的 Bar 数据."""

    def __init__(self, folder_path, threshold=100000, bar_type="dollar_bar"):
        self.bar_type = bar_type
        self.folder_path = folder_path
        self.threshold = threshold
        self.col_names = [
            "aggTradeId",
            "price",
            "quantity",
            "firstTradeId",
            "lastTradeId",
            "timestamp",
            "isBuyerMaker",
            "isBestMatch",
        ]

    def _construct_dollar_bar(self, data: pd.DataFrame) -> pd.DataFrame:
        dollar_volume = data["price"] * data["quantity"]
        crosses = get_volume_bar_indices(dollar_volume.values, self.threshold)
        start_idx = 0
        bars = []
        for end_idx in crosses:
            segment = data.iloc[start_idx : end_idx + 1]
            bars.append(self.build_bar(segment, bar_type="dollar_bar"))
            start_idx = end_idx + 1
        if start_idx < len(data):
            segment = data.iloc[start_idx:]
            bars.append(self.build_bar(segment, bar_type="dollar_bar"))
        bars = pd.DataFrame(bars)
        bars.columns = [str(col) for col in bars.columns]
        return bars

    def _construct_volume_bar(self, data: pd.DataFrame) -> pd.DataFrame:
        crosses = get_volume_bar_indices(data["quantity"].values, self.threshold)
        start_idx = 0
        bars = []
        for end_idx in crosses:
            segment = data.iloc[start_idx : end_idx + 1]
            bars.append(self.build_bar(segment, bar_type="volume_bar"))
            start_idx = end_idx + 1
        if start_idx < len(data):
            segment = data.iloc[start_idx:]
            bars.append(self.build_bar(segment, bar_type="volume_bar"))
        bars = pd.DataFrame(bars)
        bars.columns = [str(col) for col in bars.columns]
        return bars

    def _construct_tick_bar(self, data: pd.DataFrame) -> pd.DataFrame:
        segments = [data.iloc[i : i + self.threshold] for i in range(0, len(data), self.threshold)]
        bars = []
        for seg in segments:
            bars.append(self.build_bar(seg, bar_type="tick_bar"))
        bars = pd.DataFrame(bars)
        bars.columns = [str(col) for col in bars.columns]
        return bars

    def build_bar(self, segment: pd.DataFrame, bar_type: str = "tick_bar") -> dict:
        if bar_type == "tick_bar":
            price = segment["price"]
            qty = segment["quantity"]
            direction = segment["isBuyerMaker"].astype(int)
            idx_max = qty.idxmax()
            max_vol = qty.loc[idx_max]
            max_dir = direction.loc[idx_max]
            max_px = price.loc[idx_max]
            if idx_max != price.index[0]:
                prev_px = price.shift().loc[idx_max]
                max_px_imp = np.log(max_px / prev_px)
            else:
                max_px_imp = np.nan
            vol_ratio_bar = max_vol / qty.sum()
            vol_ratio_median = max_vol / qty.median()
            px_vs_vwap = max_px / ((price * qty).sum() / qty.sum()) - 1
            px_pos_in_range = (max_px - price.min()) / (price.max() - price.min() + 1e-12)
            path_length = price.diff().abs().sum()
            streaks = (direction != direction.shift()).cumsum()
            ret = np.log(price / price.shift(1)).dropna()
            return {
                "start_time": self.convert_timestamp(segment["timestamp"].iloc[0]),
                "open": price.iloc[0],
                "high": price.max(),
                "low": price.min(),
                "close": price.iloc[-1],
                "volume": qty.sum(),
                "sell_volume": qty[segment["isBuyerMaker"]].sum(),
                "buy_volume": qty[~segment["isBuyerMaker"]].sum(),
                "buy_ticks": segment["isBuyerMaker"].sum(),
                "tick_nums": len(segment),
                "vwap": (price * qty).sum() / qty.sum(),
                "medium_price": price.median(),
                "price_std": price.std(),
                "volume_std": qty.std(),
                "tick_interval_mean": segment["timestamp"].diff().mean(),
                "reversals": np.count_nonzero(direction != direction.shift()),
                "cumulative_buyer": streaks.value_counts().max(),
                "skewness": skew(ret),
                "kurtosis": kurtosis(ret),
                "up_move_ratio": (price.diff() > 0).mean(),
                "max_trade_volume": max_vol,
                "max_trade_direction": max_dir,
                "max_trade_impact": max_px_imp,
                "max_trade_vol_ratio": vol_ratio_bar,
                "max_trade_vs_median": vol_ratio_median,
                "max_trade_px_vs_vwap": px_vs_vwap,
                "max_trade_px_position": px_pos_in_range,
            }
        elif bar_type in ["dollar_bar", "volume_bar"]:
            return {
                "start_time": self.convert_timestamp(segment["timestamp"].iloc[0]),
                "open": segment["price"].iloc[0],
                "high": segment["price"].max(),
                "low": segment["price"].min(),
                "close": segment["price"].iloc[-1],
                "volume": segment["quantity"].sum(),
                "sell_volume": segment[segment["isBuyerMaker"]]["quantity"].sum(),
                "buy_volume": segment[~segment["isBuyerMaker"]]["quantity"].sum(),
                "vwap": (segment["price"] * segment["quantity"]).sum() / segment["quantity"].sum(),
                "best_match_ratio": segment["isBestMatch"].sum() / len(segment),
            }
        else:
            raise ValueError(f"Unsupported bar type: {bar_type}")

    def convert_timestamp(self, ts):
        length = len(str(ts))
        if length == 13:
            ts = ts / 1000
        elif length >= 16:
            ts = ts / 1000000
        else:
            raise ValueError(f"Unsupported timestamp length: {length}")
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def process_asset_data(self) -> pd.DataFrame:
        dataframes = []
        for file in self.folder_path:
            df = pd.read_parquet(file)
            dataframes.append(df)
        data = pd.concat(dataframes, ignore_index=True)
        data.columns = self.col_names
        if self.bar_type == "dollar_bar":
            bars_df = self._construct_dollar_bar(data)
        elif self.bar_type == "tick_bar":
            bars_df = self._construct_tick_bar(data)
        elif self.bar_type == "volume_bar":
            bars_df = self._construct_volume_bar(data)
        else:
            bars_df = None
        return bars_df