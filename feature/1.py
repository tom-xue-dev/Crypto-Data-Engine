import pandas as pd
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from feature_generation import alpha2
from utils import returns, rank

start = "2019-1-1"
end = "2021-12-31"
assets = select_assets(start_time=start,spot=True,n = 10)
print(assets)
#assets = ['XRP-USDT_spot','BTC-USDT_spot']
data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="1d")

data['returns'] = data.groupby('asset')['close'].pct_change()

data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.rolling(5).mean()).droplevel(0)
data['alpha'] = alpha2(data)
daily_ic = data.groupby('asset').apply(lambda x: x['alpha'].corr(x['future_return'], method='spearman'))

print(daily_ic.mean())


