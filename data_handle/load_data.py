import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go


def load_1m_data(symbol: str, base_data_dir: str = None) -> pd.DataFrame:
    """
    读取指定币种的1分钟K线CSV数据

    参数:
        symbol: 币种名称 (例如 "1INCH-USDT")
        base_data_dir: 数据根目录 (默认为项目路径下的data目录)

    返回:
        pd.DataFrame: 合并后的数据，按时间排序
    """
    # 自动构建路径
    if base_data_dir is None:
        # 默认路径：假设脚本在 data_handle 目录，数据在上级的 data 目录
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        base_data_dir = os.path.join(current_script_path, "..", "data")

    # 目标目录路径
    target_dir = os.path.join(
        base_data_dir,
        "spot",
        symbol,
        "1m"
    )

    # 检查目录是否存在
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"目录不存在: {target_dir}")

    # 获取所有数字开头的CSV文件
    csv_files = [
        f for f in os.listdir(target_dir)
        if re.match(r'^\d+\.csv$', f)  # 匹配纯数字文件名
    ]

    # 按数字顺序排序（避免字符串排序问题）
    csv_files.sort(key=lambda x: int(x.split('.')[0]))
    # 读取并合并数据
    dfs = []
    for f in csv_files:
        file_path = os.path.join(target_dir, f)
        df = pd.read_csv(file_path)
        # 统一时间列处理（假设列名为timestamp）
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # 假设时间戳为毫秒
            df.set_index('timestamp', inplace=True)

        dfs.append(df)
        print(f"已加载: {f} | 数据量: {len(df)} 条")

    if not dfs:
        raise ValueError("未找到有效的CSV文件")

    merged_df = pd.concat(dfs)
    print(merged_df)
    print(f"\n合并完成! 总数据量: {len(merged_df)} 条")
    return merged_df


def generate_volume_bars(df, volume_threshold=1000, include_remainder=True):
    """
    根据成交量阈值生成Volume Bars

    参数:
        df: 输入的DataFrame（需包含时间索引和ohlcv列）
        volume_threshold: 每个Bar的成交量阈值（默认1000）
        include_remainder: 是否包含不足阈值的剩余数据（默认True）

    返回:
        pd.DataFrame: Volume Bars数据（包含时间范围）
    """
    # 确保数据按时间排序
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    bars = []
    current_volume = 0
    current_open = None
    current_high = float('-inf')
    current_low = float('inf')
    current_close = None
    start_time = None
    last_index = df.index[0]  # 初始化最后时间戳

    for index, row in df.iterrows():
        # 初始化新Bar
        if current_volume == 0:
            start_time = index
            current_open = row['open']
            current_high = row['high']
            current_low = row['low']
            current_close = row['close']
            current_volume = row['volume']
        else:
            # 更新价格范围
            current_high = max(current_high, row['high'])
            current_low = min(current_low, row['low'])
            current_close = row['close']
            current_volume += row['volume']

        last_index = index  # 记录最后时间戳

        # 达到成交量阈值时生成Bar
        if current_volume >= volume_threshold:
            bars.append({
                'start_time': start_time,
                'end_time': index,
                'open': current_open,
                'high': current_high,
                'low': current_low,
                'close': current_close,
                'volume': current_volume
            })
            # 重置计数器
            current_volume = 0
            current_open = None
            current_high = float('-inf')
            current_low = float('inf')
            current_close = None

    # 处理剩余未达到阈值的数据
    if include_remainder and current_volume > 0:
        bars.append({
            'start_time': start_time,
            'end_time': last_index,
            'open': current_open,
            'high': current_high,
            'low': current_low,
            'close': current_close,
            'volume': current_volume
        })

    # 转换为DataFrame
    volume_bars = pd.DataFrame(bars)

    # 设置时间范围索引
    if not volume_bars.empty:
        volume_bars['duration'] = volume_bars['end_time'] - volume_bars['start_time']
        volume_bars.set_index('start_time', inplace=True)

    return volume_bars


def heikin_ashi_transform(df):
    """
    输入:
        df: 包含至少 ['time', 'open', 'high', 'low', 'close'] 列的 DataFrame
            时间列可以命名为 time，也可以是索引，用于排序。
    输出:
        DataFrame: 在原始列基础上新增 ['ha_open', 'ha_high', 'ha_low', 'ha_close'] 列。
    """
    # 首先，按时间排序（如果没按时间排序就先排序
    index = df.index
    df = df.sort_values('time').reset_index(drop=True).copy()
    # 准备存放 Heikin Ashi 的各列
    ha_open = []
    ha_close = []
    ha_high = []
    ha_low = []

    for i in range(len(df)):
        O = df.loc[i, 'open']
        H = df.loc[i, 'high']
        L = df.loc[i, 'low']
        C = df.loc[i, 'close']

        # 计算 ha_close(t)
        this_ha_close = (O + H + L + C) / 4

        if i == 0:
            # 第一个 bar 的 ha_open 可以用原始 open (或者 (O + C)/2)
            this_ha_open = O
        else:
            # 按公式:
            # HA_Open(t) = ( HA_Open(t-1) + HA_Close(t-1) ) / 2
            this_ha_open = (ha_open[i - 1] + ha_close[i - 1]) / 2

        # 计算 ha_high, ha_low
        this_ha_high = max(H, this_ha_open, this_ha_close)
        this_ha_low = min(L, this_ha_open, this_ha_close)

        ha_open.append(this_ha_open)
        ha_close.append(this_ha_close)
        ha_high.append(this_ha_high)
        ha_low.append(this_ha_low)

    # 将结果添加到 DataFrame
    df['open'] = ha_open
    df['high'] = ha_high
    df['low'] = ha_low
    df['close'] = ha_close
    df["time"] = index
    df.set_index("time", inplace=True)
    return df


def generate_renko(df, box_size=100):
    """
    根据OHLC数据生成Renko图表的DataFrame
    :param df: DataFrame, 需包含 'open', 'high', 'low', 'close' 列，索引应为 DatetimeIndex
    :param box_size: 砖块大小，可以是固定值或根据ATR计算
    :return: Renko DataFrame，包含 ['time', 'renko_price', 'renko_direction']
    """
    renko = []  # 存储 Renko 砖块数据
    direction = 0  # 方向: 1 (上涨), -1 (下跌)

    # 初始化第一个砖块
    initial_price = df.iloc[0]['close']
    last_renko_price = initial_price - (initial_price % box_size)  # 归一化到最近的砖块价格

    for time, row in df.iterrows():
        close_price = row['close']

        # 计算价格相对最后砖块位置的移动
        price_change = close_price - last_renko_price
        num_boxes = int(price_change // box_size)  # 计算能产生多少个砖块

        # 如果价格突破了砖块边界，生成新砖块
        if abs(num_boxes) > 0:
            for i in range(abs(num_boxes)):
                # 确定砖块方向
                if num_boxes > 0:
                    last_renko_price += box_size  # 上涨砖块
                    new_direction = 1
                else:
                    last_renko_price -= box_size  # 下跌砖块
                    new_direction = -1

                # 避免反向过多的噪声波动
                if new_direction != direction:
                    if len(renko) >= 2 and renko[-1]['renko_direction'] == -new_direction:
                        renko.pop()  # 删除最后一根砖块，避免假突破
                    else:
                        direction = new_direction  # 更新方向

                # 记录 Renko 砖块
                renko.append({'time': time, 'renko_price': last_renko_price, 'renko_direction': direction})

    # 转换为 DataFrame
    renko_df = pd.DataFrame(renko)
    return renko_df



def plot_renko(renko_df):
    """
    绘制 Renko 图表
    :param renko_df: DataFrame, 需包含 'time', 'renko_price', 'renko_direction' 列
    """
    plt.figure(figsize=(12, 6))

    # 上涨砖块（绿色）和下跌砖块（红色）
    for i in range(1, len(renko_df)):
        color = 'green' if renko_df.iloc[i]['renko_direction'] == 1 else 'red'
        plt.plot([renko_df.iloc[i - 1]['time'], renko_df.iloc[i]['time']],
                 [renko_df.iloc[i - 1]['renko_price'], renko_df.iloc[i]['renko_price']],
                 color=color, linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Renko Price")
    plt.title("Renko Chart")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 示例：读取1INCH-USDT的数据
    df = load_1m_data("BTC-USDT")

    df['time'] = pd.to_datetime(df['time'])  # 确保是 datetime 类型
    df.set_index("time", inplace=True)  # 设定为索引
    # 例如对1分钟数据进行重采样为15分钟
    df = df.resample('15min').agg({'open': 'first',
                                   'high': 'max',
                                   'low': 'min',
                                   'close': 'last'})
    df = df[:len(df) // 10]
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['open'], high=df['high'],
                                         low=df['low'], close=df['close'])])
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()
    df = generate_renko(df)
    print(df)
    # mpf.plot(df, type='candle',
    #          volume=False,
    #          title='Original OHLC')
    #
    # # Heikin Ashi
    # mpf.plot(ha_df, type='candle', style='classic', title='Heiken-Ashi Chart', ylabel='Price')

    plot_renko(df)

