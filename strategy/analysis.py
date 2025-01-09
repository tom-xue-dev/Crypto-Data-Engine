import sys
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, select_assets
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from strategy import DualMAStrategy


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


def process_asset_signals_wrapper(args):
    return process_asset_signals(*args)


def process_asset_signals(group, window, threshold, std_threshold):
    """
    处理单个资产的数据，生成信号列，并返回处理后的 DataFrame。
    """
    group = group.copy()  # 防止修改原始数据

    # 初始化 signal 列
    group["signal"] = 0
    # 计算基于百分比变化的收益率序列，用于波动率计算
    group["pct_change_high"] = group["close"].pct_change()

    for i in range(len(group)):
        if i < window:
            continue  # 窗口不足时跳过

        # 波动率筛选
        if std_threshold is not None:
            recent_period_length = int(window / 10)
            if i - recent_period_length >= 0:
                recent_slice = group.iloc[i - recent_period_length:i]
                recent_vol = recent_slice["pct_change_high"].std()
                # 如果近期波动率未超过阈值，则跳过此次循环
                if recent_vol > std_threshold:
                    continue

        current_close = group.iloc[i]["close"]
        past_window = group.iloc[i - window:i]
        past_max_high_idx = past_window["high"].idxmax()
        past_max_high = group.loc[past_max_high_idx, "close"]

        # 跳过最高价属于最近 window/10 根 K 线的情况
        recent_kline_limit = max(i - int(window / 10), i - window)
        if group.index.get_loc(past_max_high_idx) >= recent_kline_limit:
            continue

        # 检查是否满足条件：当前价格与过去最高价在指定阈值范围内
        condition_close_to_high = (current_close * (1 - threshold)
                                   <= past_max_high
                                   <= current_close * (1 + threshold))

        if condition_close_to_high:
            if group.iloc[i]["MA30"] < group.iloc[i]["MA5"]:
                continue
            group.iloc[i, group.columns.get_loc("signal")] = 1
            continue

        # --------------------------------------------
        past_window = group.iloc[i - window:i]
        past_min_low_idx = past_window["low"].idxmin()
        past_min_low = group.loc[past_min_low_idx, "close"]

        # 跳过最高价属于最近 window/10 根 K 线的情况
        recent_kline_limit = min(i - int(window / 10), i - window)
        if group.index.get_loc(past_min_low_idx) >= recent_kline_limit:
            continue

        # 检查是否满足条件：当前价格与过去最高价在指定阈值范围内
        condition_close_to_low = (current_close * (1 - threshold)
                                  <= past_min_low
                                  <= current_close * (1 + threshold))

        if condition_close_to_low:
            if group.iloc[i]["MA30"] < group.iloc[i]["MA5"]:
                continue
            group.iloc[i, group.columns.get_loc("signal")] = -1

    return group


def generate_signal(data, window, threshold, std_threshold=None):
    """
    使用多进程并行化生成交易信号，每个资产组独立一个进程计算 signal。
    """
    # 将数据按资产分组
    asset_groups = [group for _, group in data.groupby("asset")]

    # 为每个组准备参数
    params = [(group, window, threshold, std_threshold) for group in asset_groups]

    with ProcessPoolExecutor() as executor:
        # 使用顶层包装函数并行处理每个资产组
        results = executor.map(process_asset_signals_wrapper, params)

    # 合并所有结果
    result_df = pd.concat(results)
    # 可选：按索引排序以恢复原始顺序
    result_df = result_df.sort_index()

    return result_df


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


if __name__ == "__main__":
    # start = "2021-1-1"
    # end = "2021-6-30"

    # start = "2022-1-1"
    # end = "2022-12-30"

    # start = "2023-1-1"
    # end = "2023-12-30"

    start = "2023-1-1"
    end = "2023-11-30"

    assets = select_assets(spot=True, n=360)

    # assets = []
    data = load_filtered_data_as_list(start, end, assets, level="1d")

    strategy = DualMAStrategy(dataset=data, asset=assets, short=5, long=30)

    strategy.generate_signal()

    data = pd.concat(strategy.dataset, ignore_index=True)

    data = data.set_index(["time", "asset"])

    result = generate_signal(data.copy(), window=30, threshold=0.01, std_threshold=0.02)

    avg_return, prob_gain, count = future_performance(result, n_days=3)

    print(f"未来 3 天的平均涨跌幅: {avg_return:.4f}")
    print(f"未来 3 天的涨幅概率: {prob_gain:.2%}")

    avg_return, prob_gain, _ = future_performance(result, n_days=5)
    print(f"未来 5 天的平均涨跌幅: {avg_return:.4f}")
    print(f"未来 5 天的涨幅概率: {prob_gain:.2%}")

    avg_return, prob_gain, _ = future_performance(result, n_days=10)
    print(f"未来 10 天的平均涨跌幅: {avg_return:.4f}")
    print(f"未来 10 天的涨幅概率: {prob_gain:.2%}")

    avg_return, prob_gain, _ = future_performance(result, n_days=20)
    print(f"未来 20 天的平均涨跌幅: {avg_return:.4f}")
    print(f"未来 20 天的涨幅概率: {prob_gain:.2%}")
    print(f"数量 = {count}")

    #
    for asset in assets:
        if asset in result.index.get_level_values("asset"):
            visualize_signals(result, asset)
        else:
            print(asset)
    # # 示例数据
