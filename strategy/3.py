import numpy as np
import pickle
from analysis import process_future_performance_in_pool

def calculate_garman_klass_volatility(group, window):
    """
    在 DataFrame 中添加 Garman-Klass 波动率列。
    """
    group['GK_vol'] = (
            0.5 * (np.log(group['high'] / group['low'])) ** 2 -
            (2 * np.log(2) - 1) / window * (np.log(group['close'] / group['open'])) ** 2
    )
    group['GK_vol_rolling'] = group['GK_vol'].rolling(window=window).mean()
    return group


if __name__ == "__main__":
    with open("data_filter_2.pkl", "rb") as file:
        data = pickle.load(file)
    data = data.set_index(['time', 'asset'])

    window = 50
    result = data.groupby(level='asset', group_keys=False).apply(calculate_garman_klass_volatility, window=window)

    for t, group in result.groupby('asset'):
        condition = (group["GK_vol_rolling"] > 1e-4) & (group["signal"] == 1)
        result.loc[group.index[condition], 'signal'] = 0

    n_days_list = list(range(5, 600, 20))

    n_splits = 24
    split_size = len(data) // n_splits  # 每等分的大小
    # 循环处理每个部分
    with open("data_filter_3.pkl", "wb") as file:
        pickle.dump(result, file)