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
    因子100略微有些复杂
    (0 -
    (1 * (((1.5 * scale(indneutralize(indneutralize(
    rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -  scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
    """
    return (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)


if __name__ == '__main__':
    start = "2019-1-1"
    end = "2022-12-31"
    while True:
        assets = select_assets(spot=True, n=10)
        #assets = ['BTC-USDT_spot', 'ETH-USDT_spot']
        data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
        if not data.empty:
            break
    print(data)
    ans = calculate_alpha(data)
    pd.set_option('display.max_columns', None)
    data['alpha'] = ans
    data['future_20'] = data['close'].rolling(20).mean() / data['close']
    data['future_10'] = data['close'].rolling(10).mean() / data['close']
    data.dropna()
    daily_ic = data.groupby('asset').apply(lambda x: x['alpha'].corr(x['future_20'], method='spearman'))
    ir = daily_ic.mean() / daily_ic.std()
    print(daily_ic.mean(), ir)
