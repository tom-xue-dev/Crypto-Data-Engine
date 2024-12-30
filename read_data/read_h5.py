import pandas as pd
import time

# 设置 HDF5 文件路径和 key
h5_file_path = "ADA-USDT_spot_ACH-USDT_spot_15min.h5"  # 替换为实际 HDF5 文件路径
dataset_key = "dataset"         # 替换为实际数据集 key

# 记录时间的辅助函数
df = pd.read_hdf(h5_file_path, key=dataset_key)

print(df.tail(10))




