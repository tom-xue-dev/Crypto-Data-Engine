import glob
import matplotlib.pyplot as plt
import os
import pandas as pd


def calculate_price_change_mean(short: int, long: int, dataset: pd.DataFrame, n: int):
    """
    记录双均线交叉点，并计算每次交叉后 T+1 到 T+n 日价格变化百分比的均值。

    参数:
        short (int): 短期均线的窗口大小。
        long (int): 长期均线的窗口大小。
        dataset (pd.DataFrame): 时间序列数据，要求包含 'close' 和 'time' 列。
        n (int): 计算价格变化的天数范围。

    返回:
        pd.DataFrame: 包含每次交叉类型、时间、价格变化均值的表。
    """
    if 'close' not in dataset.columns or 'time' not in dataset.columns:
        raise ValueError("输入的 dataset 必须包含 'close' 和 'time' 列。")

    # 确保 'time' 列为 datetime 格式
    dataset['time'] = pd.to_datetime(dataset['time'])
    dataset = dataset.set_index('time')

    # 计算短期和长期均线
    dataset['short_ma'] = dataset['close'].rolling(window=short).mean()
    dataset['long_ma'] = dataset['close'].rolling(window=long).mean()

    # 去除 NaN 值（由于均线计算需要窗口大小）
    dataset = dataset.dropna()

    # 检测交叉点
    cross_points = []
    for i in range(1, len(dataset)):
        prev_short_ma = dataset.iloc[i - 1]['short_ma']
        prev_long_ma = dataset.iloc[i - 1]['long_ma']
        curr_short_ma = dataset.iloc[i]['short_ma']
        curr_long_ma = dataset.iloc[i]['long_ma']

        # 黄金交叉: 短期均线从下向上穿过长期均线
        if prev_short_ma < prev_long_ma and curr_short_ma > curr_long_ma:
            cross_points.append({
                "timestamp": dataset.index[i],
                "price": dataset.iloc[i]['close'],
                "type": "golden"
            })

        # 死亡交叉: 短期均线从上向下穿过长期均线
        elif prev_short_ma > prev_long_ma and curr_short_ma < curr_long_ma:
            cross_points.append({
                "timestamp": dataset.index[i],
                "price": dataset.iloc[i]['close'],
                "type": "death"
            })

    # 转换交叉点为 DataFrame
    cross_points_df = pd.DataFrame(cross_points)

    # # 计算 T+1 到 T+n 日价格变化百分比的均值
    # price_changes = []
    # for index, row in cross_points_df.iterrows():
    #     start_time = row['timestamp']
    #     end_time = start_time + pd.Timedelta(minutes=15 * n)
    #     # 提取 T+1 到 T+n 的数据
    #     future_data = dataset.loc[start_time + pd.Timedelta(minutes=15 * n):end_time]
    #     if not future_data.empty:
    #         # 计算涨跌百分比
    #         percentage_changes = ((future_data['close'] - row['price']) / row['price']) * 100
    #         mean_change = percentage_changes.mean()
    #     else:
    #         mean_change = None

    price_changes = []
    for index, row in cross_points_df.iterrows():
        start_time = row['timestamp']
        end_time = start_time + pd.Timedelta(minutes=15 * n)
        if end_time not in dataset.index:
            print("err")
            break
        future_data = dataset.loc[end_time]
        if not future_data.empty:
            percentage_changes = ((future_data['close'] - row['price']) / row['price']) * 100
        else:
            percentage_changes = None

        price_changes.append({
            "timestamp": row['timestamp'],
            "type": row['type'],
            "mean_change": percentage_changes
        })

    # 转换为 DataFrame
    result_df = pd.DataFrame(price_changes)
    return result_df


def load_dataset_from_folder(folder_path):
    """
    从指定文件夹加载所有数据文件并合并为一个 DataFrame

    :param folder_path: str, 数据文件所在文件夹路径
    :return: pd.DataFrame, 合并后的数据集
    """
    # 获取文件夹中所有 CSV 文件的路径
    file_pattern = os.path.join(folder_path, "*.csv")  # 如果文件格式不是 CSV，请修改为对应格式
    file_list = glob.glob(file_pattern)

    if not file_list:
        raise FileNotFoundError(f"No files found in folder: {folder_path}")

    # 读取所有文件并合并
    dataframes = []
    for file in file_list:
        print(f"Loading file: {file}")
        df = pd.read_csv(file)  # 根据文件格式调整读取方法
        dataframes.append(df)

    # 合并所有 DataFrame
    combined_dataset = pd.concat(dataframes, ignore_index=True)
    return combined_dataset


def split_by_cross_type(result_df: pd.DataFrame):
    """
    根据金叉和死叉的类型，将 result_df 分为两个子集。

    参数:
        result_df (pd.DataFrame): 包含交叉点信息的 DataFrame，必须包含 'type' 列。

    返回:
        pd.DataFrame: 金叉数据集（golden_df）
        pd.DataFrame: 死叉数据集（death_df）
    """
    if 'type' not in result_df.columns:
        raise ValueError("输入的 result_df 必须包含 'type' 列。")

    # 分割为金叉和死叉数据集
    golden_df = result_df[result_df['type'] == 'golden'].reset_index(drop=True)
    death_df = result_df[result_df['type'] == 'death'].reset_index(drop=True)

    return golden_df, death_df


def plot_price_change_histogram(cross_df: pd.DataFrame, title: str):
    """
    绘制涨跌幅直方图，横坐标为时间，纵坐标为涨跌幅。

    参数:
        cross_df (pd.DataFrame): 包含涨跌幅和时间的 DataFrame，要求包含 'timestamp' 和 'mean_change' 列。
        title (str): 图表标题，用于区分金叉和死叉。

    返回:
        None: 直接绘制图表。
    """
    if 'timestamp' not in cross_df.columns or 'mean_change' not in cross_df.columns:
        raise ValueError("输入的 cross_df 必须包含 'timestamp' 和 'mean_change' 列。")

    # 转换时间戳为 datetime 格式，确保一致性
    cross_df['timestamp'] = pd.to_datetime(cross_df['timestamp'])

    # 排序数据以便横坐标按时间显示
    cross_df = cross_df.sort_values(by='timestamp')

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(cross_df['timestamp'], cross_df['mean_change'], width=0.8, color='skyblue', edgecolor='black')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)  # 添加水平基线
    plt.xlabel('Time')
    plt.ylabel('Percentage Change (%)')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def calculate_win_rate(cross_df: pd.DataFrame):
    """
    计算涨跌幅的胜率。

    参数:
        cross_df (pd.DataFrame): 包含涨跌幅的 DataFrame，要求包含 'mean_change' 列。

    返回:
        float: 胜率（介于 0 和 1 之间）。
    """
    if 'mean_change' not in cross_df.columns:
        raise ValueError("输入的 cross_df 必须包含 'mean_change' 列。")

    # 计算胜率（mean_change > 0 的比例）
    win_rate = (cross_df['mean_change'] > 0).mean()
    return win_rate





# 调用示例
data_folder = os.path.join("..", "data", "binance", "BTCUSDT", "15m")
dataset = load_dataset_from_folder(data_folder)
for short in range(3,8):
    for long in range(10,20):
        result_df = calculate_price_change_mean(short=short, long=long, dataset=dataset, n=3)
        # print(result_df.tail(20)
        print(f"short = {short},long = {long},n = 3")
        golden_df, death_df = split_by_cross_type(result_df)

        mean_change = golden_df['mean_change'].mean()
        print(mean_change)
        mean_change = death_df['mean_change'].mean()
        print(mean_change)


        # 输出结果
        # plot_price_change_histogram(golden_df, title="Golden Cross Price Change")
        # plot_price_change_histogram(death_df, title="Death Cross Price Change")
        # 计算金叉的胜率
        golden_win_rate = calculate_win_rate(golden_df)
        print(f"金叉胜率: {golden_win_rate:.2%}")

        # 计算死叉的胜率
        death_win_rate = calculate_win_rate(death_df)
        print(f"死叉胜率: {death_win_rate:.2%}")




