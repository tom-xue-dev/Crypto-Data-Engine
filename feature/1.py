import pandas as pd

import utils
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
import feature_generation
from utils import returns, rank

start = "2023-1-1"
end = "2023-12-31"
assets = select_assets(start_time=start,spot=True,m=50)
print(assets)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)  # 显示所有行
#assets = ['XRP-USDT_spot','BTC-USDT_spot']
data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="1d")

data['returns'] = returns(data)

data['1D'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-1) / x - 1).droplevel(0)
data['5D'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-5) / x - 1).droplevel(0)
data['10D'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
data['factor'] = feature_generation.alpha1(data)
data = data.drop(columns=['returns'])
print(data.head(300))
data.to_pickle("alpha1.pkl")




