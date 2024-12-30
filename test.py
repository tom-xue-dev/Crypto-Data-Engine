import pandas as pd
import pickle
import time

def test_dataframe_read_speed():
    """
    测试读取 DataFrame 格式的 Pickle 文件速度。
    """
    pickle_file = "binance/spot/1INCH-USDT/15m/1INCH-USDT_15m_dataframe.pkl"  # 替换为实际文件路径
    start_time = time.time()
    df = pd.read_pickle(pickle_file)  # 读取整个 DataFrame
    end_time = time.time()
    print(f"读取单一 DataFrame 的时间: {end_time - start_time:.6f} 秒")
    assert df is not None  # 添加断言以确保数据被成功读取

def test_list_read_speed():
    """
    测试读取列表形式的 Pickle 文件速度。
    """
    pickle_file = "binance/spot/1INCH-USDT/15m/1INCH-USDT_15m.pkl"  # 替换为实际文件路径
    start_time = time.time()
    with open(pickle_file, 'rb') as f:
        data_list = pickle.load(f)  # 读取整个列表
    end_time = time.time()
    print(f"读取列表形式的时间: {end_time - start_time:.6f} 秒")
    assert data_list is not None and len(data_list) > 0  # 添加断言确保列表不为空




# 测试 DataFrame 格式读取速度
df = test_dataframe_read_speed()

# 测试列表形式读取速度
data_list = test_list_read_speed()