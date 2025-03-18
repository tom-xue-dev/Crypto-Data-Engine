import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from technical_analysis import trend_analysis
from CUSUM_filter import generate_filter_df
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
    data = generate_filter_df(data, sample_column='close')
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


 def getBins(events,close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    —events.index is event's starttime
    —events[’t1’] is event's endtime
    —events[’trgt’] is event's target
    —events[’side’] (optional) implies the algo's position side
    Case 1: (’side’ not in events): bin in (-1,1) <—label by price action
    Case 2: (’side’ in events): bin in (0,1) <—label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out