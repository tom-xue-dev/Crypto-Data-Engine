import pandas as pd
from read_large_files import load_filtered_data_as_list, select_assets
import time

start = "2022-1-5"
end = "2022-3-1"

asset = []

data = load_filtered_data_as_list(start_time=start, end_time=end, asset_list=asset, level="15min")

data = pd.concat(data, ignore_index=True)
data = data.set_index(["time", "asset"])

s = time.time()
for t, group in data.groupby("time"):
    indices = group.index
    data.loc[indices, "close"] = data.loc[indices, "close"].values + 1


e = time.time()

print(e - s)
