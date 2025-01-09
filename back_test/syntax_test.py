import pandas as pd
from read_large_files import load_filtered_data_as_list, select_assets
import time
from concurrent.futures import ProcessPoolExecutor
import numba as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_first_signal_line_for_asset(asset_df, window, tolerance, min_slope=0.1, seq=0):
    """
    针对单个资产，找到第一个信号并绘制对应的直线。
    参数:
      - asset_df: 单个资产的 DataFrame，索引包含 'time' 和 'asset' 层级，包含 'high' 列。
      - window: 用于信号计算的窗口大小。
      - tolerance: 容许的相对差异容忍度。
      - min_slope: 最小斜率要求。
    """
    # 计算信号序列
    signals = calculate_signals_for_asset_low_wrapper(asset_df, window, tolerance, min_slope)

    # 提取时间序列和价格数据
    times = asset_df.index.get_level_values('time')
    closes = asset_df['low'].values
    log_closes = closes  # 假设这里不需要对数转换，如果需要可替换为 np.log(closes)

    # 找到第一个信号的索引
    signal_indices = np.where(signals == 1)[0]
    if len(signal_indices) == 0:
        print("未找到任何信号。")
        return

    first_signal_idx = signal_indices[seq]

    # 根据第一个信号位置计算对应窗口的直线参数
    if first_signal_idx - window + 1 < 0:
        print("信号位置不足以形成完整窗口。")
        return

    # 取出第一个信号对应的窗口数据
    arr = log_closes[first_signal_idx - window + 1: first_signal_idx + 1].copy()
    min_val = np.min(arr)
    arr = arr - min_val  # 平移数据

    half_window = len(arr) // 2
    valid = False
    slope = 0.0
    intercept = 0.0

    # 按照给定逻辑查找符合条件的直线
    for j in range(0, half_window):
        if valid:
            break
        for k in range(half_window, len(arr)):
            if arr[k] == arr[j]:
                continue
            slope_candidate = (arr[k] - arr[j]) / (k - j)
            # 检查最小斜率要求
            if not (slope_candidate > min_slope):
                continue
            # 找到符合条件的直线
            slope = slope_candidate
            intercept = arr[j] - slope * j
            fits_line = True
            for t in range(len(arr)):
                # 如果有点在直线之下，则不符合要求
                if arr[t] < slope * t + intercept:
                    fits_line = False
                    break

            # 如果所有点都在直线下方，则认为找到有效直线
            if not fits_line:
                continue
            valid = True
            for t in range(len(arr)):
                print(arr[t], slope, t, intercept, slope * t + intercept)

    if not valid:
        print("未在第一个信号窗口内找到有效直线。")
        return

    # 使用找到的直线参数绘图
    plt.figure(figsize=(12, 6))
    plt.plot(times, closes, label='High Price', color='blue')

    # 计算窗口内每个点的直线预测值，并还原平移
    window_length = len(arr)
    t_window = np.arange(window_length)
    line_y_transformed = slope * t_window + intercept
    line_y = line_y_transformed + min_val  # 还原平移

    # 获取对应的时间轴
    start_idx = first_signal_idx - window + 1
    end_idx = first_signal_idx + 1
    time_window = times[start_idx:end_idx]

    # 绘制直线和信号垂直线
    plt.plot(time_window, line_y, label='Detected Line', color='red', linestyle='--')
    plt.axvline(x=times[first_signal_idx], color='green', linestyle='--', label='First Signal')

    plt.title('Asset High Prices with Detected Line at First Signal')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


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
            current_close = group.loc[i, "high"]

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


@nb.njit
def calculate_signals_for_asset_low_numba(closes, window, tolerance, min_slope):
    n = len(closes)
    signals = np.zeros(n, dtype=np.int32)
    # 计算对数价格
    log_closes = closes

    for i in range(window - 1, n, window / 20):
        # 取出当前窗口内的对数价格序列
        slope = 0
        intercept = 0
        arr = log_closes[i - window + 1: i + 1]
        if len(arr) < 2:
            continue

        # 对窗口内的价格进行变换：
        # 1. 复制数组，确保不会修改原始数据视图
        arr = arr.copy()
        # 2. 找到窗口内的最小值
        min_val = np.min(arr)
        # 3. 生成对应的 x 坐标序列，从 1 开始以避免除以零
        indices = np.arange(1, len(arr) + 1)
        # 4. 对每个 y 值执行 (y - min) / x 转换
        arr = (arr - min_val)

        half_window = len(arr) // 2
        valid = False
        for j in range(0, half_window):
            if valid is True:
                break
            for k in range(half_window, len(arr)):
                if arr[k] == arr[j]:
                    continue
                slope = (arr[k] - arr[j]) / (k - j)
                # 检查最小斜率要求
                if not (slope > min_slope):
                    continue
                else:
                    # 存在斜率符合要求的直线
                    intercept = arr[j] - slope * j
                    fits_line = True
                    for t in range(len(arr)):
                        # 如果有点在直线之上，则不符合要求
                        if arr[t] < slope * t + intercept:
                            fits_line = False
                            break

                    # 如果所有点都在直线下方，则认为找到有效直线
                    if not fits_line:
                        continue

                    valid = True
                    break

        if valid:
            t_current = len(arr) - 1
            if arr[t_current] != 0:
                relative_difference = abs(arr[t_current] - (slope * t_current + intercept)) / abs(arr[t_current])
                if relative_difference < tolerance * arr[t_current]:
                    signals[i] = 1

    return signals


def calculate_signals_for_asset_low_wrapper(asset_df, window, tolerance, min_slope=0.3):
    closes = asset_df['low'].values
    return calculate_signals_for_asset_low_numba(closes, window, tolerance, min_slope)


def process_asset(asset, df, window, tolerance):
    df_asset = df.xs(asset, level='asset', drop_level=False).sort_index(level='time')
    signals = calculate_signals_for_asset_low_wrapper(df_asset, window, tolerance)
    return asset, signals


def add_signal_column(df, window, tolerance):
    df = df.sort_index(level='time')
    df['signal'] = 0
    assets = df.index.get_level_values('asset').unique()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_asset, asset, df, window, tolerance): asset for asset in assets}
        for future in futures:
            asset, signals = future.result()
            asset_mask = df.index.get_level_values('asset') == asset
            df.loc[asset_mask, 'signal'] = signals

    return df


if __name__ == "__main__":
    start = "2023-3-1"
    end = "2023-9-5"

    # asset = select_assets(spot=True, n=100)
    asset = ["BTC-USDT_spot"]
    data = load_filtered_data_as_list(start_time=start, end_time=end, asset_list=asset, level="15min")

    data = pd.concat(data, ignore_index=True)
    data = data.set_index(["time", "asset"])

    window = 1440
    for seq in range(1, 100,5):
        plot_first_signal_line_for_asset(data, window, tolerance=0.01, min_slope=0.3, seq=seq)
    strategy_results = add_signal_column(data, window, tolerance=0.01)
    for i in range(1,200,10):
        print(i,future_performance(strategy_results,i))
