import pandas as pd
from pathlib import Path


def get_target_file_path(level):
    """
    get the target btc file which stores the specific level
    param:
    level(str): represent the specific level that we want,e.g.15m
    return:
    (str)target file os path if file exists,None otherwise.
    """
    path = ("1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h",
            "6h", "8h", "12h", "1d", "3d", "1mon")
    if level not in path:
        return None
    current_path = Path.cwd()
    target_path = current_path.parent / 'binance_btc' / (level + '.csv')
    if target_path.exists():
        return target_path
    else:
        return None


def get_btc_data(level):
    """
    get btc data in the specific level
    :param level(str):the specific level that we want, e.g.15m
    :return: btc_data(dataframe)
    """
    filepath = get_target_file_path(level)
    btc_data = pd.read_csv(filepath)
    return btc_data
