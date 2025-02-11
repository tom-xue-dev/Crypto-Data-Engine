import pandas as pd
import numpy as np


def compute_ic(df, feature_column, return_column, groupby_column='asset', method='spearman'):
    group_ic = df.groupby(groupby_column).apply(lambda x: x[feature_column].corr(x[return_column], method=method))
    return group_ic

def compute_prediction_metrics(df: pd.DataFrame, prediction_column: str, return_column: str) -> dict:
    """
    计算预测准确率指标，返回预测的 precision 和 recall。

    参数：
        df : pd.DataFrame
            包含预测列和回报列的数据框。
        prediction_column : str
            预测列名称，取值仅限于 1、0、-1，其中 1 和 -1 表示给出预测，0 表示未给出预测。
        return_column : str
            回报列名称，取值仅为 1 和 -1。

    返回：
        dict: 包含两个键 'precision' 和 'recall'，对应的值分别为：
            - precision: 在所有预测不为 0 的样本中，预测正确的比例。
            - recall: 在所有样本中，预测正确的样本比例。
    """
    # 检查必要列是否存在
    for col in [prediction_column, return_column]:
        if col not in df.columns:
            raise ValueError(f"DataFrame 缺少必要的列: {col}")
    mask = df[prediction_column] != 0
    total_predicted = mask.sum()  # 给出预测的样本数量
    correct_predictions = (df.loc[mask, prediction_column] == df.loc[mask, return_column]).sum()

    precision = correct_predictions / total_predicted if total_predicted > 0 else np.nan
    # recall: 在所有样本中预测正确的比例
    recall = correct_predictions / len(df) if len(df) > 0 else np.nan

    return {'precision': precision, 'recall': recall}




def compute_zscore(group: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """
    对指定的 DataFrame 分组数据，计算指定列的滚动 z-score。

    参数：
        group : pd.DataFrame
            单个资产或单个组的数据，要求包含需要计算 z-score 的列。
        column : str
            要计算 z-score 的列名称，例如 'ADX'。
        window : int
            滚动窗口大小，即用来计算均值和标准差的周期数。

    返回：
        pd.DataFrame
            增加了 z-score 列后的 DataFrame，新增的列名格式为 'zscore_{column}'。
    """
    group = group.copy()
    # 计算滚动均值和标准差，min_periods 设为 window 保证只有足够数据时才计算，否则为 NaN
    rolling_mean = group[column].rolling(window=window, min_periods=window).mean()
    rolling_std = group[column].rolling(window=window, min_periods=window).std()
    # 计算 z-score
    group[f"zscore_{column}"] = (group[column] - rolling_mean) / rolling_std
    return group




