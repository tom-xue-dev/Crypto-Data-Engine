import pandas as pd
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets


def calculate_vwap(df):
    """
    计算 VWAP (成交量加权平均价格)
    :param df: 包含 open, high, low, volume 列的 DataFrame
    :return: VWAP 值
    """
    df['typical_price'] = (df['open'] + df['high'] + df['low']) / 3
    df['price_volume'] = df['typical_price'] * df['volume']

    vwap = df['price_volume'].sum() / df['volume'].sum()
    return vwap


# 示例数据
data = {
    'open': [100, 102, 101],
    'high': [105, 104, 103],
    'low': [98, 100, 99],
    'volume': [200, 150, 250]
}

df = pd.DataFrame(data)
result = calculate_vwap(df)
print("VWAP:", result)
