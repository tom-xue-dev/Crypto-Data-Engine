from read_large_files import load_filtered_data_as_list
import time
start_time = "2021-12-01"
end_time = "2022-12-31"
asset_list = []  # 替换为您需要的资产

start = time.time()
# 调用函数读取并过滤数据
filtered_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "15min")

end = time.time()

print(filtered_data_list[0:2])
