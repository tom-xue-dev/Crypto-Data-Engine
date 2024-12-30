import pickle

import pandas as pd


def read_list_pickle(pickle_file):
    """
    读取列表形式的 Pickle 文件。

    参数:
        pickle_file (str): Pickle 文件路径。
    返回:
        list: 反序列化后的列表对象。
    """
    try:
        with open(pickle_file, 'rb') as f:
            data_list = pickle.load(f)  # 加载 Pickle 文件
        print(f"成功读取 Pickle 文件: {pickle_file}")
        return data_list
    except FileNotFoundError:
        print(f"文件未找到: {pickle_file}")
        return None
    except pickle.UnpicklingError as e:
        print(f"Pickle 文件解析失败: {e}")
        return None

# 示例调用
pickle_file = 'nested_pickle/1d/1INCH-USDT_spot_1d.pkl'  # 替换为实际 Pickle 文件路径
data_list = read_list_pickle(pickle_file)
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# 检查读取内容
if isinstance(data, pd.DataFrame):
    print("成功读取 DataFrame 类型的 Pickle 文件！")
    print(f"DataFrame 大小: {data.shape}")  # 输出行列数
    print("前几行数据:")
    print(data.head())  # 打印前几行数据
else:
    print("文件内容不是 DataFrame 类型！")

num_first_layer_elements = len(data_list)
print(f"第一层元素数量: {num_first_layer_elements}")
print(data_list[1100])  # 打印第一个 DataFrame 或其他对象
print(data_list[1110])
print(data_list[1120])

