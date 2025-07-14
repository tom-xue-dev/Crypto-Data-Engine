import pandas as pd

path = r"D:\github\BTC-trading\tick_data\data\ZRXUSDT\ZRXUSDT-aggTrades-2019-02.parquet"
data = pd.read_parquet(path)
print(data)