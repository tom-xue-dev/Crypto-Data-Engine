import pandas as pd
from read_large_files import map_and_load_pkl_files, select_assets
from matplotlib import pyplot as plt


def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)

        elif sPos > h:
            sPos = 0
            tEvents.append(i)
    return tEvents


if __name__ == '__main__':
    start = "2019-1-1"
    end = "2019-6-30"
    asset = ['BTC-USDT_spot']

    data = map_and_load_pkl_files(asset_list=asset, start_time=start, end_time=end, level='15min')
    time_index = data.index.get_level_values('time')

    idx = getTEvents(data['close'], 10)

    filter_data = data.loc[idx]
    filter_idx = filter_data.index.get_level_values('time')
    print(len(idx),len(time_index))
    plt.plot(filter_idx, filter_data['close'], color='red', alpha=0.5)
    plt.plot(time_index, data['close'], color='yellow', alpha=0.5)
    plt.show()
