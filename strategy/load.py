import pickle

import numpy as np

from strategy import DualMAStrategy


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


with open("x_test.pkl", "rb") as f:
    data = pickle.load(f)
with open("predictions.pkl", "rb") as f:
    prediction = pickle.load(f)
print(len(data),len(prediction))
mask = prediction == 0
prediction[mask] = -1

mask = prediction == -3
prediction[mask] = 0

data['signal'] = prediction
print(data['signal'])
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)
