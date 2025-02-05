import sys

import pandas as pd
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets



def calculate_alpha(data):
    """
    alpha101 本质上是开盘收盘和振幅的比值
    如果我没猜错的话 比值越小（振幅相对实体越大）代表分歧越大，未来下跌可能性会变高
    """
    return (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-3)


if __name__ == '__main__':
    start = "2019-1-1"
    end = "2020-12-31"
    assets =['BSV-USDT_spot','VET-USDT_spot','XLM-USDT_spot']
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="1d")

    print(data)
    ans = calculate_alpha(data)
    pd.set_option('display.max_columns', None)
    data['alpha'] = ans
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.rolling(3).mean()).droplevel(0)
    data.dropna()
    condition_data = data.loc[data['close'] > data['open']]
    condition_data_2 = data.loc[data['close'] < data['open']]
    print(len(data),len(condition_data),len(condition_data_2))
    daily_ic = data.groupby('asset').apply(lambda x: x['alpha'].corr(x['future_return'], method='spearman'))
    condition_ic = condition_data.groupby('asset').apply(lambda x: x['alpha'].corr(x['future_return'], method='spearman'))
    condition_ic_2 = condition_data_2.groupby('asset').apply(
        lambda x: x['alpha'].corr(x['future_return'], method='spearman'))
    ir = daily_ic.mean() / daily_ic.std()
    print("orignal_data ic :",daily_ic.mean(), ir)
    # print("conditional_data ic:",condition_ic.mean())
    # print("conditional_data 2 ic:", condition_ic_2.mean())