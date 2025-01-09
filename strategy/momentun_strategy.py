import numpy as np
import pandas as pd
from read_large_files import load_filtered_data_as_list, select_assets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

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
                condition_std_low = past_close_std <= std_threshold

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

def make_features_and_labels(data: pd.DataFrame,
                             future_n: int = 5) -> pd.DataFrame:
    """
    基于多重索引 (time, asset) 的行情数据，生成随机森林所需的特征和标签。

    参数：
    -------
    data : pd.DataFrame
        MultiIndex = (time, asset)，含列 [open, high, low, close]。
    future_n : int
        未来多少根K线后判断涨跌（默认5）。

    返回：
    -------
    df_features : pd.DataFrame
        新增了以下列：
        - 'ma_5': 过去5日均线
        - 'ma_10': 过去10日均线
        - 'ma_20': 过去20日均线
        - 'ma_30': 过去30日均线
        - 'ma_90': 过去90日均线
        - 'feature_range80': p90 和 p10 的均值
        - 'std_5': 过去5日标准差
        - 'std_10': 过去10日标准差
        - 'std_20': 过去20日标准差
        - 'label': (0/1)，0表示下跌/持平，1表示上涨
    """

    def p10(s):
        return s.quantile(0.10)

    def p90(s):
        return s.quantile(0.90)

    # 确保按 time 排序
    data = data.sort_index(level='time').copy()

    # 对每个资产进行一次 groupby
    def calculate_features(group):
        # 计算 p10 和 p90
        group['p10'] = group['close'].rolling(30, min_periods=30).apply(p10)
        group['p90'] = group['close'].rolling(30, min_periods=30).apply(p90)
        group['feature_range80'] = (group['p90'] + group['p10']) / 2

        # 计算不同窗口的移动均线
        for window in [5, 30, 90]:
            group[f'ma_{window}'] = group['close'].rolling(window=window, min_periods=window).mean()

        # 计算标准差
        group['std_5'] = group['close'].rolling(window=5, min_periods=5).std()
        group['std_10'] = group['close'].rolling(window=10, min_periods=10).std()
        group['std_20'] = group['close'].rolling(window=20, min_periods=20).std()

        # 计算未来收盘价
        group['future_close'] = group['close'].shift(-future_n)

        # 定义标签
        group['label'] = (group['future_close'] > group['close']).astype(int)

        return group

    # 应用计算
    data = data.groupby(level='asset').apply(calculate_features)

    # 清理无效行：rolling或shift导致的 NaN
    required_columns = [f'ma_{window}' for window in [5, 30, 90]] + ['std_5', 'std_10', 'std_20', 'label']
    data.dropna(subset=required_columns, inplace=True)

    return data


def train_random_forest_and_predict(df_train: pd.DataFrame,
                                    df_test: pd.DataFrame):
    """
    在训练集上训练随机森林，并在测试集上做预测，输出分类结果。
    """
    # 特征列 & 标签列
    features = ['ma_5', 'ma_30', 'ma_90', 'std_5', 'std_10','std_20', 'feature_range80', 'close','signal']
    target = 'label'

    X_train = df_train[features].values
    y_train = df_train[target].values

    X_test = df_test[features].values
    y_test = df_test[target].values

    # 初始化随机森林
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')

    rf.fit(X_train, y_train)

    # 预测
    y_pred = rf.predict(X_test)

    # 打印评估
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 将结果存储到 df_test 中
    df_test['pred_label'] = y_pred
    feature_importance = rf.feature_importances_
    plt.barh(features, feature_importance)
    plt.show()
    # 返回模型 & 测试集(含预测结果)
    return rf, df_test


def train_test_split_by_time(df_features: pd.DataFrame, train_ratio=0.8):
    """
    按时间顺序进行切分，将前 train_ratio 部分数据作为训练集，后面部分作为测试集。
    """
    df_features = df_features.copy()

    # 获取所有 time 索引，并按顺序去重
    # multiIndex 的 level=0 是 time，unique后再sort
    all_times = df_features.index.get_level_values('time').unique()
    all_times = sorted(all_times)

    split_point = int(len(all_times) * train_ratio)
    train_time = all_times[:split_point]
    test_time = all_times[split_point:]

    df_train = df_features.loc[df_features.index.get_level_values('time').isin(train_time)]
    df_test = df_features.loc[df_features.index.get_level_values('time').isin(test_time)]

    return df_train, df_test


def main_flow(data, window_size=100, future_n=5, train_ratio=0.8):
    """
    整合整个流程：
    1) 生成特征和label
    2) 按时间切分train/test
    3) 训练随机森林并预测
    """
    # 1) 生成特征 & label
    df_features = make_features_and_labels(data, future_n=future_n)

    # 2) 划分训练和测试集
    df_train, df_test = train_test_split_by_time(df_features, train_ratio=train_ratio)

    # 3) 训练模型并预测
    model, df_test_pred = train_random_forest_and_predict(df_train, df_test)

    return model, df_train, df_test_pred


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


start = "2023-1-1"
end = "2024-11-30"

assets = select_assets(spot=True, n=300)

# assets = []
data = load_filtered_data_as_list(start, end, assets, level="1d")

data = pd.concat(data, ignore_index=True)

data = data.set_index(["time", "asset"])

data = generate_signal(data.copy(), window=30, threshold=0.01, std_threshold=1)
# 假设 data 是你的 MultiIndex DataFrame: index=(time, asset), columns=[open, high, low, close]
model, df_train, df_test_pred = main_flow(
    data,
    window_size=100,  # 回看100根K线做分位计算
    future_n=5,  # 预测5根K线后涨跌
    train_ratio=0.8
)

# 查看测试集的一部分预测结果
