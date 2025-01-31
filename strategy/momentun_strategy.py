import pickle
import sys
import ast
import statsmodels.api as sm
import numpy as np
import pandas as pd
from strategy import DualMAStrategy
from read_large_files import map_and_load_pkl_files, select_assets


def calc_beta(df: pd.DataFrame, window_size: int):
    df["beta"] = np.nan
    for asset, group in df.groupby('asset'):
        for i in range(window_size, len(group)):
            x_window = group.iloc[i - window_size:i]["low"].values
            y_window = group.iloc[i - window_size:i]["high"].values
            X = sm.add_constant(x_window)
            model = sm.OLS(y_window, X)
            results = model.fit()
            beta = results.params[1]
            df.loc[group.index[i], "beta"] = beta
    pd.set_option('display.max_columns', 20)
    return df


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


def calculate_atr(df, time_period=14):
    """
    计算ATR（平均真实波动幅度）并将其插入到原始DataFrame中。

    参数：
        df (pd.DataFrame): MultiIndex DataFrame，索引为time和asset，列包含open, high, low, close。
        time_period (int): ATR的计算周期，默认值为14。

    返回：
        pd.DataFrame: 带有新增ATR列的DataFrame。
    """
    # 验证列是否存在
    required_columns = {'high', 'low', 'close'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame 必须包含以下列：{required_columns}")

    # 计算真实波幅（True Range, TR）
    df['prev_close'] = df.groupby('asset')['close'].shift(1)
    df['tr'] = df[['high', 'low', 'prev_close']].apply(
        lambda row: max(row['high'] - row['low'],
                        abs(row['high'] - row['prev_close']),
                        abs(row['low'] - row['prev_close'])), axis=1
    )

    # 计算ATR（使用简单移动平均）
    df['atr'] = df.groupby('asset')['tr'].transform(lambda x: x.rolling(window=time_period, min_periods=1).mean())

    # 删除中间列
    df.drop(columns=['prev_close', 'tr'], inplace=True)

    return df


def filter_data(data_set):
    for t, group in data.groupby('time'):
        for index, row in group.iterrows():
            if not row['beta'] > 1:  # 判断条件
                if data_set.loc[index, 'signal'] == 1:
                    data_set.loc[index, 'signal'] = 0  # 满足条件更新 'signal' 列
            if row['beta'] < 0.7:
                if data_set.loc[index, 'signal'] == -1:
                    data_set.loc[index, 'signal'] = -2
            if not row['beta'] < 0.7:
                if data_set.loc[index, 'signal'] == -1:
                    data_set.loc[index, 'signal'] = 0
        first_sum = group['count_first'].sum()
        sec_sum = group['count_sec'].sum()
        if first_sum + sec_sum < 30:
            data_set.loc[group.index, 'signal'] = -1
            # if row['beta'] > 1:  # 判断条件
            #     if data_set.loc[index, 'signal'] == -1:
            #         data_set.loc[index, 'signal'] = 0
            # if row['beta'] < 0.6:
            #     if data_set.loc[index, 'signal'] == -1:
            #         data_set.loc[index, 'signal'] = -2

    return data_set

i = 0
while i < 10:
    start = "2019-1-1"
    end = "2022-12-30"
    assets = select_assets(spot=True, n=1)
    # assets = ['BTC-USDT_spot']
    data = map_and_load_pkl_files(start_time=start, end_time=end, asset_list=assets, level='15min')
    if data.empty:
        i -= 1
        continue
    #data = calculate_atr(df=data)
    data = calc_beta(data, window_size=32)
    print(data)
    with open(f"data{i}.pkl", "wb") as f:
        pickle.dump(data, f)
    i+=1
# print(assets)
# with open("data.pkl", "rb") as f:
#     data = pickle.load(f)
#
# strategy = DualMAStrategy(dataset=data, long_period=150, short_period=5)
# data = strategy.dataset


# for i, current_index in enumerate(data.index):
#     current_close = data.loc[current_index]['MA150']
#     prev_index = data.index[i - 20]
#     pre_close = data.loc[prev_index, 'MA150']
#     if current_close < pre_close:
#         if data.loc[current_index]['signal'] == 1:
#             data.loc[current_index,'signal'] = 0

# for t, df in data.groupby('time'):
#     for i in range(len(df)):
#         if df.loc[df.index[i], "signal"] == 1:
#             if df.loc[df.index[i], "GK_vol"] > df.loc[df.index[i], "GK_vol_rolling"] * 3:
#                 data.loc[df.index[i], "signal"] = -1
#
# with open("data_atr.pkl", "wb") as f:
#     pickle.dump(data, f)

# 计算上影线长度
# start = "2020-11-1"
# end = "2022-11-30"
#
# assets = select_assets(spot=True, n=1)
#
# data = map_and_load_pkl_files(asset_list=assets,start_time=start,end_time=end,level="1d")


# data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
# data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
#
# # # 计算过去 5 天成交量的均值
# data['volume_mean_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
# #
# # # 筛选条件
# # # 1. 上影线长度显著（可定义一个阈值，比如上影线占总波动的比例超过一定值）
# # # 2. 当前成交量大于过去 5 天均值
# threshold = 0.5  # 定义上影线的阈值（如占总波动幅度的 50%）
# # data['is_long_upper_shadow'] = (
# #         (data['upper_shadow'] > threshold * (data['high'] - data['low']))
# # )
# data['is_long_upper_shadow'] = (
#         (data['close'] - data['open']) / data['open'] < -0.05
# )
# data['is_long_lower_shadow'] = (
#         (data['lower_shadow'] > threshold * (data['high'] - data['low'])) &  # 下影线很长
#         (data['volume'] > data['volume_mean_5'])  # 成交量大于过去 5 天均值
# )
#
# data.loc[data['is_long_upper_shadow'] & (data['signal'] == 1), 'signal'] = -2
# data.loc[data['is_long_lower_shadow'], 'signal'] = 1

# 示例

# result.index = result.index.droplevel(0)  # 移除多余的 'asset' 层级
# data['RSRI'] = result


# 上一步 result 会是一个 DataFrame，列和原来一样 ['high', 'low']，
# 但它们其实都一样，因为我们返回的是同一个标量beta
# 可以取其中一列赋给 data['RSRI']：
