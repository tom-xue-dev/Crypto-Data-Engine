import os
import csv
import pandas as pd
from datetime import datetime
import sys
from concurrent.futures import ThreadPoolExecutor
def parse_datetime(date_str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Time data '{date_str}' does not match any known format")

def get_parent_dir():
    """获取当前脚本所在目录的上一级目录"""
    return os.path.abspath(os.path.join(os.getcwd(), os.pardir))

def load_fees_data(fees_path, crypto_name):
    """加载资金费率 CSV 文件并返回 DataFrame"""
    index = crypto_name.find("-")
    for csv_file in os.listdir(fees_path):
        if csv_file.endswith(".csv") and csv_file.startswith(crypto_name[:index] + "_" + crypto_name[index + 1:]):
            file_path = os.path.join(fees_path, csv_file)
            try:
                df = pd.read_csv(file_path)
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                return df
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                return None
    return None

def is_fully_filled(rows, new_column_name):
    """检查新列是否已存在且所有行的该列都已填充"""
    header = rows[0]
    if new_column_name not in header:
        return False

    new_column_index = header.index(new_column_name)
    for row in rows[1:]:
        if len(row) <= new_column_index or row[new_column_index] == "":
            return False
    return True

def merge_fees(crypto_name, level):
    """
    合并永续合约资金费率
    :param crypto_name: 合约名称，如 "BTC-USDT"
    :param level: 时间级别，如 "1m"
    :return: None
    """
    future_path = os.path.join("binance", "futures", crypto_name, level)
    future_files = [f for f in os.listdir(future_path) if f.endswith(".csv")]
    parent_dir = get_parent_dir()
    fees_path = os.path.join(parent_dir, "data_fees")

    # 定义新列名
    new_column_name = "Funding_Rate"

    # 加载资金费率数据
    df = load_fees_data(fees_path, crypto_name)
    insert_position = None
    if df is None or not future_files:
        print(f"No fees data or future files found for {crypto_name}")
        return

    for csv_file in future_files:
        file_path = os.path.join(future_path, csv_file)

        # 读取文件内容
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # 检查是否需要处理该文件
        if is_fully_filled(rows, new_column_name):
            print(f"File {csv_file} already fully filled. Skipping.")
            continue

        if rows:
            # 添加新列名（如果不存在）
            if new_column_name not in rows[0]:
                try:
                    close_index = rows[0].index("close")
                except ValueError:
                    print(f"'close' column not found in {csv_file}")
                    sys.exit(-1)
                    # continue

                insert_position = close_index + 1
                rows[0].insert(insert_position, new_column_name)
            for row in rows[1:]:
                while len(row) <= insert_position:
                    row.append("")  # 确保行的长度足够
            # 双指针法同步遍历时间序列
            future_pointer = 1  # 指向 rows 的数据行（跳过表头）
            fees_pointer = 0  # 指向反转后的 df 的第一行（最早的时间）

            # 反转 fees_data，以便时间按从旧到新排序
            fees_datetimes = df['Datetime'].tolist()[::-1]
            fees_rates = df['Funding Rate'].tolist()[::-1]

            while future_pointer < len(rows) and fees_pointer < len(fees_datetimes) - 1:
                try:
                    future_time = datetime.strptime(rows[future_pointer][0], "%Y-%m-%d %H:%M:%S")
                except ValueError as e:
                    print(f"Error parsing date in {future_path,csv_file}: {e}")
                    sys.exit(-1)

                # 检查 future_time 是否在 fees_datetimes[fees_pointer + 1] 和 fees_datetimes[fees_pointer] 之间
                if fees_datetimes[fees_pointer] <= future_time < fees_datetimes[fees_pointer + 1]:
                    rows[future_pointer][insert_position] = fees_rates[fees_pointer]
                    future_pointer += 1
                elif future_time < fees_datetimes[fees_pointer]:
                    if future_pointer == 1 or rows[future_pointer - 1][insert_position] == 'N/A':
                        rows[future_pointer][insert_position] = 'N/A'
                    future_pointer += 1
                elif future_time >= fees_datetimes[fees_pointer + 1]:
                    fees_pointer += 1
                else:
                    # print(future_time,fees_datetimes[fees_pointer],fees_datetimes[fees_pointer + 1])
                    print("err detected")
                    sys.exit(0)
            # 如果 future_data 剩余部分无法匹配，填充 "N/A"
            while future_pointer < len(rows):
                rows[future_pointer][insert_position] = "N/A"
                future_pointer += 1
        # print(rows)
        # 将更新后的数据写回文件
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"Updated {csv_file} successfully.")
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")

    print(f"Funding rates merged successfully for {crypto_name}")



# 获取当前目录下所有以 .csv 结尾的文件列表
files = os.listdir("binance/futures")
# 打印文件列表
for file in files:

    print("start merge file", file)
    merge_fees(file, "15m")


