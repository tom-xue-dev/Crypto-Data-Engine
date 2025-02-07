import numpy as np
import pandas as pd
from fractional_diff import fracdiff
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
import matplotlib.pyplot as plt
start = "2022-1-1"
end = "2022-12-31"
#assets = select_assets(start_time=start, spot=True, n=10)
# print(assets)
assets = ['BTC-USDT_spot','ETH-USDT_spot']
data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")

data['log_price'] = np.log(data['close'])
data['frac_diff'] = data.groupby('asset')['log_price'].transform(lambda s: fracdiff(s, d=0.6))
pd.set_option('display.max_columns', 100)
data = data.dropna()
# 假设你的 DataFrame 名称为 data，且索引为 MultiIndex (time, asset)
asset = 'BTC-USDT_spot'
# 利用 xs 方法提取指定资产的数据（以资产为 level 提取）
df_asset = data.xs(asset, level='asset')

fig, ax1 = plt.subplots(figsize=(14, 6))

# 绘制对数价格（log_price），使用左侧坐标轴
ax1.plot(df_asset.index, df_asset['log_price'], color='blue', label='Log Price')
ax1.set_xlabel('Time')
ax1.set_ylabel('Log Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 创建共享 x 轴的第二个坐标轴，绘制 frac_diff
ax2 = ax1.twinx()
ax2.plot(df_asset.index, df_asset['frac_diff'], color='red', label='Frac Diff')
ax2.set_ylabel('Frac Diff', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title(f'Asset: {asset} - Log Price and Frac Diff')
plt.show()