import pickle
import sys
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, select_assets, map_and_load_pkl_files
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from strategy import DualMAStrategy


def build_segments_n(dataset: pd.DataFrame, n: int) -> list[tuple[float, float]]:
    """
    将 dataset 按 time 维度切分成 2^n 个小块，
    每块计算 (min_low, max_high) 并返回列表。
    假设:
      - MultiIndex: (time, asset)
      - 列包含 ['low', 'high']
      - len(unique_times) >= 2^n
    """
    times = dataset.index.get_level_values('time').unique().sort_values()
    total_times = len(times)
    num_segments = 2 ** n  # 要切分的段数

    if total_times < num_segments:
        raise ValueError(f"time 数量({total_times})不足以切分成 {num_segments} 份。")

    step = total_times // num_segments  # 每段大小 (可能有余数)

    segments = []
    start = 0
    for i in range(num_segments):
        # 最后一段取到结尾，处理可能的余数
        if i == num_segments - 1:
            end = total_times
        else:
            end = start + step

        # 切分 times
        seg_times = times[start:end]
        sub_df = dataset.loc[seg_times]

        min_low = sub_df['low'].min()
        max_high = sub_df['high'].max()
        segments.append((min_low, max_high))

        start = end

    return segments


def compare_and_merge(segA: tuple[float, float], segB: tuple[float, float]) -> tuple[
    tuple[bool, bool], tuple[bool, bool], tuple[float, float]]:
    minA, maxA = segA
    minB, maxB = segB
    is_min_lower = (minA < minB)
    is_max_lower = (maxA < maxB)

    is_min_higher = (minA > minB)
    is_max_higher = (maxA > maxB)

    # 合并后新的 segment
    merged_min = min(minA, minB)
    merged_max = max(maxA, maxB)
    return (is_min_lower, is_max_lower), (is_min_higher, is_max_higher), (merged_min, merged_max)


def compare_segments_n(segments: list[tuple[float, float]]) -> list[tuple[bool, bool, bool, bool]]:
    """
    给定最底层 (2^n 个) segments，使用自底向上的两两合并，
    每合并一次就记录一个 (is_min_lower, is_max_lower)。
    最终返回所有比较结果的列表，总长度为 2^n - 1。
    """
    results = []
    layer = segments[:]

    # 不断两两合并，直到只剩一个
    while len(layer) > 1:
        weight = 0
        for i in range(1, 5):
            if len(layer) == pow(2, i):
                weight = 4 - i
        next_layer = []
        for i in range(0, len(layer), 2):
            segA = layer[i]
            segB = layer[i + 1]
            (is_min_lower, is_max_lower), (is_min_higher, is_max_higher), merged_seg = compare_and_merge(segA, segB)
            results.append(
                (is_min_lower * weight, is_max_lower * weight, is_min_higher * weight, is_max_higher * weight))
            next_layer.append(merged_seg)
        layer = next_layer

    return results


def check_halves_n(dataset: pd.DataFrame, n: int = 3):
    """
    综合入口:
    1) 先按 time 切成 2^n 份，得到每份 (min_low, max_high)
    2) 再自底向上合并并比较，返回 (is_min_lower, is_max_lower) 的列表
       共会返回 2^n - 1 次比较结果
    """
    # 1) 建立 2^n 份 segments
    segments = build_segments_n(dataset, n=n)
    # 2) 自底向上合并并比较
    results = compare_segments_n(segments)
    return segments, results


def future_performance_task(data, n_days, signal):
    """
    独立任务函数，用于计算未来 n 天的平均涨跌幅和涨跌概率。
    """
    avg_return, prob_gain, count = future_performance(data, n_days, signal)
    return n_days, avg_return, prob_gain, count


# 使用进程池实现的代码
def process_future_performance_in_pool(data, n_days_list, signal):
    """
    并行计算多个 n_days 对应的未来表现。

    参数:
        data: DataFrame, 包含 'signal' 和 'close' 列。
        n_days_list: list, 包含多个 n_days 窗口。

    返回:
        results: list, 每个元素是 (n_days, avg_return, prob_gain, count)。
    """
    results = []
    with ProcessPoolExecutor() as executor:
        # 提交每个 n_days 的任务
        futures = {executor.submit(future_performance_task, data, n_days, signal): n_days for n_days in n_days_list}

        # 收集结果
        for future in futures:
            n_days, avg_return, prob_gain, count = future.result()
            results.append((n_days, avg_return, prob_gain, count))
    return results


def future_performance(data, n_days, signal):
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
        for i in group[group["signal"] == signal].index:
            signal_count += 1  # 累加 signal = 1 的数量

            # 获取当前的 close 值
            current_close = group.loc[i, "close"]

            # 检查未来 n 天是否有足够的数据
            if i + n_days >= len(group):  # 如果未来 n 天不足，跳过
                continue

            # 计算未来 n 天的close
            future_close = group.loc[i + n_days, "close"]

            # 计算涨跌幅
            return_pct = (future_close - current_close) / current_close
            future_returns.append(return_pct)

    # 计算平均涨跌幅
    avg_return = np.mean(future_returns) if future_returns else 0

    # 计算涨跌概率
    prob_gain = np.mean(np.array(future_returns) > 0) if future_returns else 0

    return avg_return, prob_gain, signal_count


# 策略函数


def calculate_garman_klass_volatility(group, window):
    """
    在 DataFrame 中添加 Garman-Klass 波动率列。
    """
    group['GK_vol'] = (
            0.5 * (np.log(group['high'] / group['low'])) ** 2 -
            (2 * np.log(2) - 1) / window * (np.log(group['close'] / group['open'])) ** 2
    )
    group['GK_vol_rolling'] = group['GK_vol'].rolling(window=window).mean()
    return group


def process_asset_signals_wrapper(params):
    group, window = params
    return process_asset_group(group, window)


def process_asset_group(group, window):
    """
    对单个资产组的数据进行信号生成。
    """

    group["signal"] = group["count_first"] = group["count_sec"] = 0  # 初始化信号列
    # group["high_low_array"] = "0"
    for i in range(len(group)):
        if i - window <= 0:
            continue  # 跳过窗口不足的前几天
        condition_long = False
        condition_short = False
        sub_df = group.iloc[i - window:i]
        current_close = group.iloc[i]["close"]
        prev_close = group.iloc[i - window]["close"]
        segments, ans = check_halves_n(sub_df)
        count_first_true = sum(x[0] for x in ans)
        count_sec_true = sum(x[1] for x in ans)

        count_third_true = sum(x[2] for x in ans)  # 前段区间最低值是否更高
        count_forth_true = sum(x[3] for x in ans)  # 前段区间最高值是否更高
        # print(f"第1个元素为 True 的元组数量: {count_first_true}")
        # print(f"第2个元素为 True 的元组数量: {count_sec_true}")
        # print(count_third_true,count_forth_true,current_close)
        if count_first_true > 6 or count_sec_true > 6:
            condition_long = True
        if count_first_true < 3 or count_sec_true < 3:
            condition_short = True
        # if count_third_true > 8 and count_forth_true > 8:
        #     condition_short = True
        # if count_third_true < 3 and count_third_true < 3:
        #     condition_long = True
        group.iloc[i, group.columns.get_loc("count_first")] = count_first_true
        group.iloc[i, group.columns.get_loc("count_sec")] = count_sec_true
        # print(str(segments))
        if condition_long:
            group.iloc[i, group.columns.get_loc("signal")] = 1
        if condition_short:
            group.iloc[i, group.columns.get_loc("signal")] = -2
    return group


def generate_signal_short(data, window, threshold, volatility_threshold=None):
    """
    生成交易信号列，基于多线程处理。
    """
    # 按资产分组
    grouped_data = [group for _, group in data.groupby("asset")]

    params = [(group, window) for group in grouped_data]
    # 使用线程池并行处理每个资产组

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_asset_signals_wrapper, params)

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
    close_prices = asset_data['high']
    signals = asset_data['signal']

    # 找到 signal=1 的点
    signal_times = times[signals == -2]
    signal_prices = close_prices[signals == -2]

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

    start = "2018-11-1"
    end = "2019-11-30"

    assets = select_assets(spot=True, n=5)

    # print(assets)
    #assets = ['OCEAN-USDT_spot']
    data = map_and_load_pkl_files(start_time=start, asset_list=assets, end_time=end, level='15min')
    # strategy.calculate_MA(period=300)
    # strategy.calculate_MA(period=1000)
    # print(data)
    result = generate_signal_short(data, window=32, threshold=1)
    print(result)
    with open("data.pkl", "wb") as file:
        pickle.dump(result, file)
    n_days_list = list(range(5, 600, 20))
    results = process_future_performance_in_pool(result, n_days_list, signal=1)

    # 输出结果
    # for n_days, avg_return, prob_gain, count in results:
    #     print(f"未来{n_days}根k线上涨概率为{prob_gain},上涨幅度{avg_return},总数{count}")
    #
    # results = process_future_performance_in_pool(result, n_days_list, signal=-1)
    # for n_days, avg_return, prob_gain, count in results:
    #     print(f"未来{n_days}根k线上涨概率为{prob_gain},上涨幅度{avg_return},总数{count}")
    #
    # #
    for asset in assets:
        if asset in result.index.get_level_values("asset"):
            visualize_signals(result, asset)
        else:
            print(asset)
    # # 示例数据
