import pickle
import pandas as pd
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from technical_indicator import compute_apo, compute_mom, compute_cci, compute_mfi, compute_beta_regression
from IC_calculator import compute_zscore
from  CUSUM_filter import generate_filter_df

start = "2021-1-1"
end = "2021-12-31"
assets = select_assets(start_time=start, spot=True, n=50)
data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")

# data['returns'] = data.groupby('asset')['close'].pct_change()
# data['vwap'] = u.vwap(data)
# data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
fast_period = 12  # 快速EMA的周期
slow_period = 26  # 慢速EMA的周期
time_period = 12
data = data.groupby('asset', group_keys=False).apply(lambda x: compute_apo(x, fast_period=12, slow_period=26))
data = data.groupby('asset', group_keys=False).apply(lambda x: compute_zscore(x, column='APO', window=1200))
data = data.groupby('asset', group_keys=False).apply(lambda x: compute_cci(x, time_period=12))
data = data.groupby('asset', group_keys=False).apply(lambda x: compute_mfi(x, time_period=12))
data = data.groupby('asset', group_keys=False).apply(lambda x: compute_beta_regression(x, time_period=12))

data = generate_filter_df(data)

data = data.dropna()
print(data)
train_dict = {}
for column in data.columns:
    if column not in ['open', 'close', 'high', 'low','APO','RSI','returns','volumes','MOM']:
        train_dict[column] = data[column].values
with open('data.pkl', 'wb') as f:
        pickle.dump(train_dict, f)
