import numpy as np
import pandas as pd
import pickle
import sys
import ast

from pandas import DataFrame
from scipy.stats import skew, stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from strategy import DualMAStrategy
from read_large_files import map_and_load_pkl_files, select_assets


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


def make_future_returns(data, return_col_name="fwd_ret_5"):
    """
    基于 MultiIndex [time, asset] 的收盘价，手动生成下一期(下一天)的收益率列。
    收益率计算公式：close(t+1)/close(t) - 1。

    参数：
    -------
    data : pd.DataFrame
        - MultiIndex = [time, asset]。
        - 列包含 "close"。
    return_col_name : str
        - 生成的未来收益率列名，比如 "fwd_ret_1"。

    返回：
    -------
    data_with_ret : pd.DataFrame
        在原 dataframe 上新增一列 return_col_name 表示下一期收益率。
    """

    # 1) 检查是否是 MultiIndex
    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError("数据索引必须是 MultiIndex 格式 [time, asset].")

    # 2) 检查是否有 close 列
    if "close" not in data.columns:
        raise ValueError("'close' 列在DataFrame中不存在！")

    # 3) 复制一份，避免对原数据进行不可控的改动
    data = data.copy()

    # 4) 对每个asset做 shift(-1) 计算未来1期收益
    computed_series = data.groupby(level="asset")["close"].apply(lambda x: x.shift(-5) / x - 1)

    # 重置索引，使其与原 DataFrame 对齐
    computed_series = computed_series.reset_index(level=0, drop=True)

    # 将结果写回到 DataFrame
    data[return_col_name] = computed_series

    return data


def calc_daily_ic(df, date_col="time", asset_col="asset", factor_col="my_factor", ret_col="fwd_ret_5"):
    """
    对每个日期，计算因子和未来收益率的秩相关系数(IC)。
    返回包含 [time, ic] 的 DataFrame。
    """
    # 如果 df 是多重索引 [time, asset]，可以先 .reset_index()
    df = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
    df = df.dropna(subset=[factor_col, ret_col])  # 去掉因子或收益率为空的行

    ics = []
    for current_date, group_data in df.groupby(date_col):
        if len(group_data) < 2:
            continue  # 没有足够资产就跳过

        # 计算斯皮尔曼秩相关
        ic = group_data[[factor_col, ret_col]].corr(method="spearman").iloc[0, 1]
        ics.append((current_date, ic))

    ic_df = pd.DataFrame(ics, columns=[date_col, "IC"])
    return ic_df


def calculate_market_trend(data, n_days=20, bins=5, lookback_days=5):
    """
    根据横截面 "count_first" 和 "count_sec" 的过去 n 天均值，计算未来市场整体上涨或下跌幅度，并对结果进行分箱。

    参数：
    - data: pd.DataFrame
        MultiIndex DataFrame，索引为 [time, asset]，列至少包含 ["count_first", "count_sec", "close"]。
    - n_days: int
        表示未来收益的计算窗口大小。
    - bins: int
        分箱的数量，用于对横截面总值进行分箱。
    - lookback_days: int
        计算横截面总值的过去 n 天均值窗口。

    返回：
    - result: pd.DataFrame
        包含时间索引的 DataFrame，记录横截面过去 n 天均值、对应未来市场整体涨跌幅度及分箱结果。
    """
    # 确保 data 是 MultiIndex 且包含必要列
    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError("数据索引必须是 MultiIndex 格式 [time, asset]")
    required_cols = {"count_first", "count_sec", "close"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"DataFrame 必须包含列: {required_cols}")

    # 创建副本避免修改原数据
    data = data.copy()

    # 计算横截面总值
    data["cross_sectional_sum"] = data["count_first"] + data["count_sec"]

    # 按时间聚合横截面总值
    total_sum_by_time = data.groupby(level="time")["cross_sectional_sum"].sum()

    # 计算过去 n 天均值
    rolling_mean_total_sum = total_sum_by_time.rolling(window=lookback_days).mean()

    # 按资产分组计算未来 n 天的收益
    def calculate_future_ret(group):
        return group["close"].shift(-n_days) / group["close"] - 1

    data["future_ret"] = (
        data.groupby(level="asset").apply(calculate_future_ret).reset_index(level=0, drop=True)
    )

    # 按时间聚合未来收益的均值
    future_ret_by_time = data.groupby(level="time")["future_ret"].mean()

    # 合并横截面过去 n 天均值和未来市场表现
    result = pd.DataFrame({
        "rolling_mean_total_sum": rolling_mean_total_sum,
        "future_mean_ret": future_ret_by_time
    })

    # 删除包含 NaN 值的行
    result.dropna(inplace=True)

    # 对横截面总值的均值进行分箱
    result["rolling_sum_bins"] = pd.qcut(result["rolling_mean_total_sum"], bins, labels=False, duplicates='drop')

    return result


def calculate_momentum_groups_return(df,
                                     n_groups=5,
                                     future_days=(3, 5, 10, 20),
                                     date_col='time',
                                     asset_col='asset',
                                     close_col='close',
                                     factor_col='momentum_factor'):
    """
    将DataFrame按照 `momentum_factor` 分成 n_groups 组，计算未来若干天的收益率。
    参数：
    -------
    df           : 包含至少以下列的DataFrame [time, asset, close, momentum_factor]
    n_groups     : 分组数，例如5组、10组
    future_days  : 要计算的未来天数列表，比如(3, 5, 10, 20)
    date_col     : 日期列名称
    asset_col    : 资产标识列名称
    close_col    : 收盘价列名称
    factor_col   : 动量因子列名称

    返回：
    -------
    result_df    : 各日期下，不同分组在未来 3、5、10、20 天的平均收益率（或其他统计信息）
    """

    df = df.copy()

    # 如果 time, asset 是索引，可以先 reset_index()
    if df.index.names != [None, None]:
        df = df.reset_index()

    # 按 (time, asset) 排序，保证时间序列的正确性
    df.sort_values(by=[date_col, asset_col], inplace=True)

    # 1) 计算未来收益率（对每个 asset 分别 shift）
    #    例如未来 3 天的收益率：close(t+3)/close(t) - 1
    for d in future_days:
        # groupby每个asset，用 shift(-d) 来取未来第d天的close
        df[f'fwd_ret_{d}'] = (
            df.groupby(asset_col)[close_col]
            .transform(lambda x: x.shift(-d) / x - 1)
        )

    # 2) 在同一天内，根据 momentum_factor 分组 (n_groups 个分位数)
    #    新增一列 'factor_quantile'
    #    为了避免极端情况，也可考虑用 qcut(x, n_groups, duplicates='drop')
    def quantile_cut(x, q=n_groups):
        return pd.qcut(x, q, labels=False, duplicates='drop')

    df['factor_quantile'] = df.groupby(date_col)[factor_col].transform(quantile_cut)
    # 3) 按 [time, factor_quantile] 聚合，计算平均未来收益率
    #    你也可以改成中位数、加权平均等
    agg_dict = {f'fwd_ret_{d}': 'mean' for d in future_days}
    result = (
        df.groupby([date_col, 'factor_quantile'])
        .agg(agg_dict)
        .reset_index()
    )
    # print(result[:20])
    # 4) 如果需要，可以再对不同日期做进一步聚合，得到在整个样本期内各分组的平均未来收益率
    #    例如：
    result_overall = result.groupby('factor_quantile').mean()
    # for g,group in result.groupby('factor_quantile'):
    #     print(group['fwd_ret_3'].mean())
    pd.set_option('display.max_columns', None)
    # print(result_overall)
    # 这会得到一个每个分组在不同天数下平均收益率的概览

    # print("res:",result_overall)
    return result


def filter_top_20pct_by_price_volume(data):
    """
    针对 MultiIndex=[time, asset] 的 DataFrame，按每个 time 的横截面，
    只返回 'close * volume' 数值较大的前20%行。
    """
    # 1) 检查索引和所需列
    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError("数据索引必须是MultiIndex格式 [time, asset].")
    required_cols = {"close", "volume"}
    if not required_cols.issubset(data.columns):
        raise ValueError("DataFrame必须至少包含: 'close'和'volume'.")

    data = data.copy()
    # 2) 生成临时列 cv = close * volume
    data["cv"] = data["close"].rolling(10).mean() * data["volume"].rolling(10).mean()

    # 3) 按 (time) 分组，筛选出 cv 排名前 20%的部分
    def top_20pct_in_group(group):
        # 当日 0.8 分位数
        cutoff1 = group["cv"].quantile(0.2)
        cutoff2 = group["cv"].quantile(0.95)
        # 只保留 >= cutoff2 且 <= cutoff1 的行
        return group[(group["cv"] >= cutoff2)]

    # group_keys=False 可以让返回的结果保持原本结构，而不是额外多一级索引
    data_top20 = data.groupby(level="time", group_keys=False).apply(top_20pct_in_group)
    print(data_top20)
    # 4) 可选：删除临时列
    data_top20.drop(columns="cv", inplace=True)

    return data_top20


def generate_cross_sectional_momentum_factor(data, lookback_period=20):
    """
    生成截面“close * volume过去N天均值”动量因子列。

    参数：
    - data: pd.DataFrame
        MultiIndex DataFrame，索引为 [time, asset]，列至少包含 ["close", "volume"]。
    - lookback_period: int
        计算均值的窗口大小。

    返回：
    - data_with_factor: pd.DataFrame
        原始数据框，新增一列 "momentum_factor" 表示上述因子值。
    """
    # 确保 data 是 MultiIndex 且包含 close, volume
    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError("数据索引必须是 MultiIndex 格式 [time, asset]")
    required_cols = {"close", "volume"}
    if not required_cols.issubset(data.columns):
        raise ValueError("DataFrame 必须包含列: 'close' 和 'volume'")

    data = data.copy()
    # 若你的 time 索引不是按顺序排好，建议按 (time, asset) 排序
    data.sort_values(by=["time", "asset"], inplace=True)

    # 1) 生成临时列 cv = close * volume
    data["prev_close"] = data.groupby("asset")["close"].shift(1)

    # 2. 计算单日振幅
    data["AMP"] = (data["high"] - data["low"]) / data["prev_close"]
    # data["momentum_factor2"] = data["volume"] / data["volume"].rolling(5).mean()
    #    data["momentum_factor3"] = data["count_first"].rolling(5).mean() / data["count_first"].rolling(5).std()
    # data["momentum_factor4"] = data["GK_vol"]
    # data["momentum_factor"] = data["momentum_factor4"]
    # factor_cols = ["momentum_factor1", "momentum_factor2", "momentum_factor3"]
    # for col in factor_cols:
    #     data[f"{col}_z"] = (data[col] - data[col].mean()) / data[col].std()
    #
    # # 2. 等权组合因子
    # data["combined_factor_eq"] = data[[f"{col}_z" for col in factor_cols]].mean(axis=1)
    #
    # # 3. PCA 组合因子
    # factor_data = data[[f"{col}_z" for col in factor_cols]]
    # factor_data = factor_data.fillna(factor_data.mean())
    # pca = PCA(n_components=1)
    # data["momentum_factor"] = pca.fit_transform(factor_data)

    # 3. 按 asset 分组，计算滚动均值振幅（如需）
    # data["momentum_factor"] = (
    #     data.groupby("asset")["amplitude"]
    #     .transform(lambda x: x.rolling(window=lookback_period).mean())
    # )
    # (可选) 如果不再需要 "cv" 列，可删除
    # data.drop(columns="cv", inplace=True)

    return data


def assign_signal(df, factor_col="my_factor", signal_col="signal", top_percent=0.1):
    """
    根据因子值，标记因子值最高的前 top_percent 的资产的信号为 -1。

    参数:
        df (pd.DataFrame): 输入的 DataFrame。
        factor_col (str): 因子值列名。
        signal_col (str): 输出的信号列名。
        top_percent (float): 因子值最高的百分比（默认 10%）。

    返回:
        pd.DataFrame: 添加了 signal 列的 DataFrame。
    """
    # 确保因子列中没有 NaN 值
    df = df.dropna(subset=[factor_col]).copy()

    # 计算因子值的分位点
    threshold = df[factor_col].quantile(1 - top_percent)

    # 初始化 signal 列为 0
    df[signal_col] = 0

    # 将因子值高于或等于分位点的部分标记为 -1
    df.loc[(df[factor_col] >= threshold) & (df["volume"] > df["volume"].rolling(5).mean()), signal_col] = -1

    return df


start = "2020-1-1"
end = "2024-11-30"

#assets = select_assets(spot=True, n=200)
# print(assets)
assets = ['BNB-USDT_spot', 'BTC-USDT_spot', 'ETH-USDT_spot', 'ADA-USDT_spot', 'XRP-USDT_spot', 'DOGE-USDT_spot','SOL-USDT_spot', 'PEPE-USDT_spot']

data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="1d")

# with open("data_atr.pkl", "rb") as f:
#     data = pickle.load(f)
# data = calculate_momentum_groups_return(data,n_groups=5)
# print(data)
data = make_future_returns(data)
data = calculate_garman_klass_volatility(data, 5)
data = generate_cross_sectional_momentum_factor(data, lookback_period=12)

# 1. 计算滚动 80% 分位数
data["rolling_80_quantile"] = data["AMP"].rolling(window=20, min_periods=1).quantile(0.8)
data["rolling_20_quantile"] = data["AMP"].rolling(window=20, min_periods=1).quantile(0.2)

condition2 = data["AMP"] < data["rolling_20_quantile"]
condition = data["AMP"] > data["rolling_80_quantile"]

# 然后再滚动求过去 20 天里有多少 True（也就是 sum）
data["momentum_factor"] = condition.rolling(window=20, min_periods=1).sum() - condition2.rolling(window=20,
                                                                                                 min_periods=1).sum()

# 假设data里含有多个股票，用groupby('date')对同一天的所有股票做相关性
# ic_series = data.groupby('time').apply(
#     lambda x: x['factor_count'].corr(x['fwd_ret_5'], method='pearson')
#     # 或者 method='spearman'
# )

# ic_series 就是一个以日期为index的时间序列，每个值是那一天截面上的相关系数
# 然后取平均
# ic_mean = ic_series.mean()
#
# print("IC均值 =", ic_mean)
# print("ICIR =", ic_mean/ic_series.std())
# print(data["IC"].count())
# print(data["open"].count())
# # data = filter_top_20pct_by_price_volume(data)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns',None)
# print(data)


ic_df = calc_daily_ic(data, date_col="time", factor_col="momentum_factor", ret_col="fwd_ret_5")
print(ic_df.head())
print("Mean IC =", ic_df["IC"].mean())
print("ICIR =", ic_df["IC"].mean() / ic_df["IC"].std())
data.to_csv('data_append.csv')
# data = assign_signal(data, factor_col="momentum_factor", signal_col="signal", top_percent=0.05)
# print(data)


with open(f"data_filter.pkl", 'wb') as f:
    pickle.dump(data, f)
# result = calculate_market_trend(data)
# # 筛选 total_sum > 1000 且 future_mean_ret 非 NaN 的行
#
# filtered_result = result[(result["rolling_mean_total_sum"]>80) &(result["rolling_mean_total_sum"]>20) & (result["future_mean_ret"].notna())]
#
# print(filtered_result['future_mean_ret'].median())
# print(filtered_result['future_mean_ret'].mean())
# data = generate_cross_sectional_momentum_factor(data,lookback_period=16)
# df = calculate_momentum_groups_return(data,n_groups=5)
#
# df = df.dropna(subset=["fwd_ret_20"])  # 去掉 fwd_ret_20 列中含 NaN 的行
# high_group = df[df["factor_quantile"] == 4.0]["fwd_ret_5"]
# low_group = df[df["factor_quantile"] == 0.0]["fwd_ret_5"]
# from scipy.stats import mannwhitneyu
# u_stat, p_value = mannwhitneyu(low_group, high_group)
# print(f"U-statistic: {u_stat}, P-value: {p_value}")
