import datetime
import sys

from read_large_files import map_and_load_pkl_files
import pandas as pd
import pickle

with open("data_delayed_ETH.pkl", 'rb') as f:
    data_day = pickle.load(f)
with open("data_15min_ETH.pkl", 'rb') as f:
    data_15m = pickle.load(f)
print(data_15m,data_day)

day_index = [group.index for t, group in data_day.groupby("time")]
# print(min_index)
i = 0
for t, group in data_15m.groupby("time"):
    day_time = day_index[i][0][0]
    min_time = group.index[0][0]
    if day_time == min_time - datetime.timedelta(days=1):
        if data_day.loc[day_index[i][0], "signal"] == -1:
            if data_15m.loc[group.index[0], "signal"] == 1:
                data_15m.loc[group.index[0], "signal"] = -1
        elif data_day.loc[day_index[i][0], "signal"] == 1:
            if data_15m.loc[group.index[0], "signal"] == -1:
                data_15m.loc[group.index[0], "signal"] = 1
        else:
            if data_15m.loc[group.index[0], "signal"] == 1:
                data_15m.loc[group.index[0], "signal"] = -1
            if data_15m.loc[group.index[0], "signal"] == -1:
                data_15m.loc[group.index[0], "signal"] = 1

        i += 1
    else:
        if data_day.loc[day_index[i][0], "signal"] == -1:
            if data_15m.loc[group.index[0], "signal"] == 1:
                data_15m.loc[group.index[0], "signal"] = -1
        elif data_day.loc[day_index[i][0], "signal"] == 1:
            if data_15m.loc[group.index[0], "signal"] == -1:
                data_15m.loc[group.index[0], "signal"] = 1
        else:
            if data_15m.loc[group.index[0], "signal"] == 1:
                data_15m.loc[group.index[0], "signal"] = -1
            if data_15m.loc[group.index[0], "signal"] == -1:
                data_15m.loc[group.index[0], "signal"] = 1
print(data_15m)
with open("data_merge.pkl", 'wb') as f:
    pickle.dump(data_15m, f)
