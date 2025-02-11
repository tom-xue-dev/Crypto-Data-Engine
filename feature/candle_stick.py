import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from technical_analysis import trend_analysis
from CUSUM_filter import generate_filter_df,triple_barrier_labeling
from labeling import parallel_apply_triple_barrier
from feature_generation import alpha102
import utils as u



if __name__ == '__main__':
    start = "2024-1-1"
    end = "2024-12-31"
    assets = select_assets(start_time=start, spot=True, n=100)
    # print(assets)
    # assets = ['BTC-USDT_spot']
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
    print("start labeling")
    #data['label'] = data.groupby(level='asset', group_keys=False)['close'].apply(triple_barrier_labeling)
    data['label'] = parallel_apply_triple_barrier(data)
    print('label finish')
    data['vwap'] = u.vwap(data)
    # 计算每个资产的对数收益率（当前与前一时点比较）
    #print(len(data))
    data = generate_filter_df(data, sample_column='close', threshold=5 * 0.007)
    # 计算每个 asset 对应的列数（特征数量）
    # 计算每个 asset 在 MultiIndex DataFrame 中的行数
    # print(data['label'].value_counts())




    data['alpha'] = trend_analysis(data)
    #data = data.dropna()
    data = data.dropna()
    # print(data[['label','alpha']].head(500))
    df_nonzero = data

    # hit_rate_1 = ((df_nonzero['label'] == 1) & (df_nonzero['alpha'] == 1))
    # count = hit_rate_1.sum()  # 计算满足条件的行数
    # count_1 = count
    # precision = count / (df_nonzero['alpha'] == 1).sum() # 计算精度（比例）
    # print("precision for class 1:", precision)
    # recall =  count / (df_nonzero['label'] == 1).sum()
    # print("recall for class 1",recall)

    hit_rate_1 = ((df_nonzero['label'] == -1) & (df_nonzero['alpha'] == -1))
    count = hit_rate_1.sum()  # 计算满足条件的行数
    count_2 = count
    precision = count / (df_nonzero['alpha'] == -1).sum() # 计算精度（比例）
    print("precision for class -1:", precision)
    recall =  count / (df_nonzero['label'] == -1).sum()
    print("recall for class -1",recall)

    hit_rate_1 = ((df_nonzero['label'] == 0) & (df_nonzero['alpha'] == 0))
    count = hit_rate_1.sum()  # 计算满足条件的行数
    count_0 = count
    precision = count / (df_nonzero['alpha'] == 0).sum()  # 计算精度（比例）
    print("precision for class 0:", precision)
    recall = count / (df_nonzero['label'] == 0).sum()
    print("recall for class 0", recall)

    print(count_2,count_0)