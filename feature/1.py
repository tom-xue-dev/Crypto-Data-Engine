import pandas as pd
from CUSUM_filter import generate_filter_df
import utils
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
import feature_generation
from utils import returns, rank
import numpy as np
import matplotlib.pyplot as plt

start = "2023-1-1"
end = "2023-12-31"
assets = select_assets(start_time=start,spot=True,m=50)
print(assets)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)  # 显示所有行
#assets = ['XRP-USDT_spot','BTC-USDT_spot']
data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
# data['open'] = np.log1p(data['open']).diff()
# data['close'] = np.log1p(data['close']).diff()
# data['high'] = np.log1p(data['high']).diff()
# data['low'] = np.log1p(data['low']).diff()
# data['volume'] = np.log1p(data['volume']).diff()
# data = data.dropna()
data['returns'] = returns(data)
data['vwap'] = utils.vwap(data)
data['10D'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
data['30D'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-30) / x - 1).droplevel(0)
data['50D'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-50) / x - 1).droplevel(0)
data['factor'] = feature_generation.alpha25(data)
data['factor'] = data['factor'].ewm(span=3, adjust=False).mean()
print(data[['factor']].head(10))
df_reset = data.reset_index()  # columns: ['date', 'asset', 'open', 'high', ...]
print(df_reset.columns)
# 2. 将 'date' 设置为单级 DatetimeIndex 后，就可以使用 between_time
df_reset['time'] = pd.to_datetime(df_reset['time'])  # 确保是 datetime 类型
df_time_filtered = (
    df_reset
    .set_index('time')  # 暂时只把 'date' 这一列作为索引
    .between_time('12:00:00', '20:00:00')  # 筛选每天 12:00:00 至 20:00:00
)

# 3. 将 'asset' 重新设为第二层索引，保持原先的 MultiIndex 结构
data = (
    df_time_filtered
    .set_index('asset', append=True)   # append=True 表示在已有索引后面附加
    .sort_index(level=['time', 'asset'])
)
data = data.drop(columns=['returns'])
data = data.drop(columns=['vwap'])
print(data.tail(300))
data.to_pickle("alpha25.pkl")
