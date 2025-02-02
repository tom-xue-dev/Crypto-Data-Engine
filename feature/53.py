import pickle
import sys
import numpy as np
import pandas as pd
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
import statsmodels.api as sm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier


def calculate_alpha(data):
    """
    alpha53是上下影线长度之和比下影线长度
    """
    data['delta'] = ((data['close'] - data['low'])- (data['high']-data['close'])) / (data['close'] - data['low'])
    data['alpha53'] = -1 * (data['delta'] - data['delta'].shift(9))
    data = data.dropna()
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
    data = calculate_alpha(data)
    pd.set_option('display.max_columns', None)
    data['future_20'] = data['close'].rolling(20).mean() / data['close']
    data['future_10'] = data['close'].rolling(10).mean() / data['close']
    data.dropna()
    daily_ic = data.groupby('asset').apply(lambda x: x['alpha53'].corr(x['future_20'], method='spearman'))
    ir = daily_ic.mean() / daily_ic.std()
    print(daily_ic.mean(), ir)
