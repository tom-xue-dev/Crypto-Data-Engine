import pickle
import sys
import numpy as np
import pandas as pd
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets



def calculate_alpha(data):
    """
    alpha99
    ((rank{correlation[      sum{[(high + low) / 2], 19.8975}, sum(adv60, 19.8975), 8.8136] } <
    rank(correlation(low, volume, 6.28259))) * -1)
    """

    data['high_low_mean'] = (data['high'] + data['low']) / 2
    data['high_low_sum'] = (data.groupby('asset')['high_low_mean'].rolling(20).sum().reset_index(level=0, drop=True))
    data['volume_dollar'] = data['close'] * data['volume']
    data['adv60'] = (data.groupby('asset')['volume_dollar'].rolling(60).mean().reset_index(level=0, drop=True))

    data['adv_60_sum'] = data.groupby('asset')['adv60'].rolling(20).sum().reset_index(level=0, drop=True)
    # 计算两个滚动窗口的相关性
    # 第一个相关性：high_low_sum与adv_60_sum的8.8136天（取整为9天）滚动相关系数
    data['corr1'] = data.groupby('asset').apply(
        lambda x: x['high_low_sum'].rolling(window=9).corr(x['adv_60_sum'])
    ).reset_index(level=0, drop=True)

    # 第二个相关性：low与volume的6.28259天（取整为6天）滚动相关系数
    data['corr2'] = data.groupby('asset').apply(
        lambda x: x['low'].rolling(window=6).corr(x['volume'])
    ).reset_index(level=0, drop=True)

    # 计算横截面rank
    data['rank_corr1'] = data.groupby('time')['corr1'].rank()
    data['rank_corr2'] = data.groupby('time')['corr2'].rank()

    # 生成alpha99因子
    data['alpha99'] = (data['rank_corr1'] < data['rank_corr2']).astype(int) * -1
    return data


if __name__ == '__main__':
    start = "2019-1-1"
    end = "2022-12-31"
    while True:
        assets = select_assets(spot=True, n=100)
        #assets = ['BTC-USDT_spot', 'ETH-USDT_spot']
        data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="1d")
        if not data.empty:
            break
    pd.set_option('display.max_columns', None)
    data = calculate_alpha(data)
    # 计算未来收益
    data['forward_return'] = data.groupby('asset')['close'].pct_change(10).shift(-7)

    # 计算选中 vs. 未选中的平均收益
    grouped_return = data.groupby('alpha99')['forward_return'].mean()

    print(f"被选中（alpha99=-1）的资产未来收益: {grouped_return.get(-1, np.nan):.6f}")
    print(f"未选中（alpha99=0）的资产未来收益: {grouped_return.get(0, np.nan):.6f}")

