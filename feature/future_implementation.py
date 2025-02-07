import pandas as pd
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from feature_generation import alpha54 as alpha
import utils as u

start = "2021-1-1"
end = "2021-12-31"
assets = select_assets(start_time=start, spot=True, m=10)
# print(assets)
# assets = ['BTC-USDT_spot','ETH-USDT_spot']
data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="1d")

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

data['returns'] = data.groupby('asset')['close'].pct_change()
data['vwap'] = u.vwap(data)

data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-2) / x - 1).droplevel(0)

data['alpha'] = alpha(data)
print(len(data))
data = data.dropna()
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
print(data.head(50))
print(len(data))
daily_ic = data.groupby('asset').apply(lambda x: x['alpha'].corr(x['future_return'], method='spearman'))
print("IC:", daily_ic.mean())
print("IR", daily_ic.mean() / daily_ic.std())
