import pandas as pd
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from feature_generation import alpha9 as alpha
from utils import returns, rank
import utils as u

start = "2022-1-1"
end = "2022-12-31"
assets = select_assets(start_time=start, spot=True, n=30)
print(assets)
# assets = ['XRP-USDT_spot','BTC-USDT_spot']
data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="1d")
data['vwap'] = u.vwap(data)
data['returns'] = data.groupby('asset')['close'].pct_change()

data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-5) / x - 1).droplevel(0)
data['alpha'] = alpha(data)
daily_ic = data.groupby('asset').apply(lambda x: x['alpha'].corr(x['future_return'], method='spearman'))

print(daily_ic.mean())
