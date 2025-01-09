import sys
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, select_assets
import pandas as pd
import numpy as np


def future_performance(data, n_days):
    """
    计算 signal = 1 的情况下，未来 n 天的平均涨跌幅和涨跌概率。

    参数:
        data: DataFrame, 包含 'signal' 列和 'close' 列的数据集（MultiIndex）
        n_days: int, 未来天数窗口

    返回:
        avg_return: float, 平均涨跌幅
        prob_gain: float, 涨幅概率
        signal_count: int, signal = 1 的数量
    """
    # 初始化列表存储未来的涨跌幅
    future_returns = []

    # 初始化 signal = 1 的数量
    signal_count = 0

    # 按资产分组计算
    for asset, group in data.groupby("asset"):
        group = group.reset_index()  # 重置索引，方便按行号处理

        # 遍历 signal = 1 的行
        for i in group[group["signal"] == 1].index:
            signal_count += 1  # 累加 signal = 1 的数量

            # 获取当前的 close 值
            current_close = group.loc[i, "close"]

            # 检查未来 n 天是否有足够的数据
            if i + n_days >= len(group):  # 如果未来 n 天不足，跳过
                continue

            # 计算未来 n 天的平均 close
            future_close = group.loc[i + 1:i + n_days, "close"].mean()

            # 计算涨跌幅
            return_pct = (future_close - current_close) / current_close
            future_returns.append(return_pct)

    # 计算平均涨跌幅
    avg_return = np.mean(future_returns) if future_returns else 0

    # 计算涨跌概率
    prob_gain = np.mean(np.array(future_returns) > 0) if future_returns else 0

    return avg_return, prob_gain, signal_count


# 策略函数
def generate_signal(data, window, threshold, std_threshold=None):
    """
    生成交易信号列，基于过去 window 天内的最高价和当前 close 价格的比较。

    参数:
        data: DataFrame, 数据集
        window: int, 回看窗口天数
        threshold: float, 阈值（百分比，如 0.1 表示 10%）

    返回:
        带有 signal 列的 DataFrame
    """
    # 添加 signal 列，初始化为 0
    data["signal"] = 0

    # 对每个资产进行操作
    for asset, group in data.groupby("asset"):
        group = group.copy()  # 防止修改原始数据

        for i in range(len(group)):
            if i < window:
                continue  # 跳过窗口不足的前几天

            # 当前 close 和过去 window 天的最高价
            current_close = group.iloc[i]["high"]
            past_max_high_idx = group.iloc[i - window:i]["high"].idxmax()  # 找到最高价所在的索引
            past_max_high = group.loc[past_max_high_idx, "high"]  # 获取过去最高价
            past_low = group.loc[past_max_high_idx, "low"]  # 获取过去最高价对应的最低价

            # 跳过最高价属于最近 window / 10 根 K 线的情况
            recent_kline_limit = max(i - int(window / 10), i - window)
            if group.index.get_loc(past_max_high_idx) >= recent_kline_limit:
                continue  # 如果最高价的索引在最近 window/10 根 K 线内，跳过

            condition_std_low = None
            if std_threshold is not None:
                past_close_std = group.iloc[i - window:i]["close"].std()
                condition_std_low = past_close_std > std_threshold

            # 检查是否满足条件 1
            condition_close_to_low = current_close * (1 - threshold) <= past_max_high <= current_close * (
                    1 + threshold
            )

            # 如果所有条件满足，则标记信号
            if std_threshold is not None:
                if condition_close_to_low and condition_std_low:
                    group.iloc[i, group.columns.get_loc("signal")] = 1
            else:
                if condition_close_to_low:
                    group.iloc[i, group.columns.get_loc("signal")] = 1

        # 更新原始数据
        data.loc[group.index, "signal"] = group["signal"]

    return data


def generate_signal_short(data, window, threshold, std_threshold=None):
    """
    生成交易信号列，基于过去 window 天内的最高价和当前 close 价格的比较，
    同时考虑过去 window 根 K 线的标准差是否低于指定阈值。

    参数:
        data: DataFrame, 数据集
        window: int, 回看窗口天数
        threshold: float, 阈值（百分比，如 0.1 表示 10%）
        std_threshold: float, 标准差阈值

    返回:
        带有 signal 列的 DataFrame
    """
    # 添加 signal 列，初始化为 0
    data["signal"] = 0

    # 对每个资产进行操作
    for asset, group in data.groupby("asset"):
        group = group.copy()  # 防止修改原始数据

        for i in range(len(group)):
            if i < window:
                continue  # 跳过窗口不足的前几天

            # 当前 close 和过去 window 天的最低价
            current_close = group.iloc[i]["close"]
            past_min_low_idx = group.iloc[i - window:i]["close"].idxmin()  # 找到最低价所在的索引
            past_min_low = group.loc[past_min_low_idx, "close"]  # 获取过去最低价

            # 跳过最低价属于最近 window / 10 根 K 线的情况
            recent_kline_limit = min(i - int(window / 10), i - window)
            if group.index.get_loc(past_min_low_idx) <= recent_kline_limit:
                continue  # 如果最低价的索引在最近 window/10 根 K 线内，跳过

            # 检查过去 window 根 K 线的标准差是否低于阈值

            # print(past_close_std)
            condition_std_low = None
            if std_threshold is not None:
                past_close_std = group.iloc[i - window:i]["close"].std()
                condition_std_low = past_close_std <= std_threshold

            # 检查是否满足条件 1
            condition_close_to_low = current_close * (1 - threshold) <= past_min_low <= current_close * (
                    1 + threshold
            )

            # 如果所有条件满足，则标记信号
            if std_threshold is not None:
                if condition_close_to_low and condition_std_low:
                    group.iloc[i, group.columns.get_loc("signal")] = 1
            else:
                if condition_close_to_low:
                    group.iloc[i, group.columns.get_loc("signal")] = 1

        # 更新原始数据
        data.loc[group.index, "signal"] = group["signal"]

    return data


def visualize_signals(data, asset):
    """
    可视化指定资产的价格与 signal=1 的点。

    参数:
        data: DataFrame, 包含信号的完整数据集
        asset: str, 指定要可视化的资产名称（例如 'FIL-USDT_spot'）
    """
    # 筛选指定资产的数据
    asset_data = data.loc[(slice(None), asset), :]

    # 提取时间、收盘价和信号
    times = asset_data.index.get_level_values('time')
    close_prices = asset_data['close']
    signals = asset_data['signal']

    # 找到 signal=1 的点
    signal_times = times[signals == 1]
    signal_prices = close_prices[signals == 1]

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(times, close_prices, label='Close Price', marker='o', linestyle='-', linewidth=2)
    plt.scatter(signal_times, signal_prices, color='red', label='Signal = 1', zorder=5)

    # 添加图例和标题
    plt.title(f"Signals for {asset}", fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()

    # 显示图形
    plt.tight_layout()
    plt.show()


# start = "2021-1-1"
# end = "2021-6-30"

# start = "2022-1-1"
# end = "2022-12-30"

# start = "2023-1-1"
# end = "2023-12-30"

start = "2024-1-1"
end = "2024-11-30"

assets = select_assets(spot=True, n=300)

# assets = []
data = load_filtered_data_as_list(start, end, assets, level="1d")

data = pd.concat(data, ignore_index=True)

data = data.set_index(["time", "asset"])

result = generate_signal(data.copy(), window=30, threshold=0.01, std_threshold=0.02)

avg_return, prob_gain, count = future_performance(result, n_days=3)

print(f"未来 3 天的平均涨跌幅: {avg_return:.4f}")
print(f"未来 3 天的涨幅概率: {prob_gain:.2%}")

avg_return, prob_gain, _ = future_performance(result, n_days=5)
print(f"未来 5 天的平均涨跌幅: {avg_return:.4f}")
print(f"未来 5 天的涨幅概率: {prob_gain:.2%}")

avg_return, prob_gain,_= future_performance(result, n_days=10)
print(f"未来 10 天的平均涨跌幅: {avg_return:.4f}")
print(f"未来 10 天的涨幅概率: {prob_gain:.2%}")

avg_return, prob_gain, _ = future_performance(result, n_days=20)
print(f"未来 20 天的平均涨跌幅: {avg_return:.4f}")
print(f"未来 20 天的涨幅概率: {prob_gain:.2%}")
print(f"数量 = {count}")
# 调用函数生成信号
# result = generate_signal_short(data.copy(), window=60, threshold=0.02, std_threshold=1)
#
# avg_return, prob_gain = future_performance(result, n_days=3)
# print(f"未来 3 天的平均涨跌幅: {avg_return:.4f}")
# print(f"未来 3 天的涨幅概率: {prob_gain:.2%}")
#
# avg_return, prob_gain = future_performance(result, n_days=5)
# print(f"未来 5 天的平均涨跌幅: {avg_return:.4f}")
# print(f"未来 5 天的涨幅概率: {prob_gain:.2%}")
#
# avg_return, prob_gain = future_performance(result, n_days=10)
# print(f"未来 10 天的平均涨跌幅: {avg_return:.4f}")
# print(f"未来 10 天的涨幅概率: {prob_gain:.2%}")
#
# avg_return, prob_gain = future_performance(result, n_days=20)
# print(f"未来 20 天的平均涨跌幅: {avg_return:.4f}")
# print(f"未来 20 天的涨幅概率: {prob_gain:.2%}")
#
for asset in assets:
    if asset in result.index.get_level_values("asset"):
        visualize_signals(result, asset)
    else:
        print(asset)
# # 示例数据
