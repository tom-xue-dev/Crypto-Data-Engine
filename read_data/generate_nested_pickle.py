import os
import pandas as pd
import pickle


def process_folder(input_folder, output_folder):
    """
    读取一个文件夹中的所有 Parquet 文件，生成对应的 pkl 文件并保存到另一个文件夹。

    参数:
    - input_folder: 输入文件夹路径，包含 Parquet 文件
    - output_folder: 输出文件夹路径，保存生成的 pkl 文件
    """
    # 检查输出文件夹是否存在，如果不存在则创建
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".parquet"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".parquet", ".pkl"))

            # 读取 Parquet 文件
            df = pd.read_parquet(input_path)

            # 确保按时间戳排序
            df = df.sort_values(by="time")

            # 生成嵌套列表 [time, [open, high, low, close, asset]]
            nested_list = df.apply(
                lambda row: [row["time"], [row["open"], row["high"], row["low"], row["close"], row["asset"]]],
                axis=1).tolist()

            # 保存为 pkl 文件
            with open(output_path, 'wb') as f:
                pickle.dump(nested_list, f)

            print(f"成功生成文件: {output_path}")


# 示例使用
input_folder_path = "data_parquet/15min"  # 替换为你的输入文件夹路径
output_folder_path = "nested_pickle/15min"  # 替换为你的输出文件夹路径

process_folder(input_folder_path, output_folder_path)

