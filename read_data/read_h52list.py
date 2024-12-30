import pandas as pd
import tables  # 需要安装 PyTables: pip install tables
import gc

def read_h5_as_generator(h5_file_path, key="dataset", chunk_size=100000):
    """
    从大文件中分块读取 HDF5 数据，按时间戳分组，逐步返回分组的 DataFrame（生成器方式）。

    参数:
        h5_file_path (str): HDF5 文件路径。
        key (str): 数据集名称，默认为 "dataset"。
        chunk_size (int): 每次读取的行数。

    返回:
        generator: 每次返回一个按时间戳分组的 DataFrame。
    """
    if not tables.is_hdf5_file(h5_file_path):
        raise ValueError(f"文件 {h5_file_path} 不是有效的 HDF5 文件。")

    with pd.HDFStore(h5_file_path, mode="r") as store:
        total_rows = store.get_storer(key).nrows
        start = 0
        time_groups = {}  # 临时存储未完成的时间戳分组

        while start < total_rows:
            # 分块读取数据
            chunk = store.select(key, start=start, stop=start + chunk_size)

            # 按时间戳分组
            for time, group in chunk.groupby("time"):
                if time in time_groups:
                    # 如果时间戳已存在，合并到对应分组
                    time_groups[time] = pd.concat([time_groups[time], group], ignore_index=True)
                else:
                    # 新时间戳直接存储
                    time_groups[time] = group

            # 返回完整的时间戳分组，并从临时分组中移除
            last_chunk_time = chunk["time"].iloc[-1]
            completed_times = [time for time in time_groups if time != last_chunk_time]
            for time in completed_times:
                yield time_groups.pop(time)

            # 更新读取位置
            start += chunk_size

            # 手动释放内存
            del chunk
            gc.collect()

        # 返回最后剩余的未完成时间戳
        for remaining_group in time_groups.values():
            yield remaining_group

        # 清理剩余变量
        time_groups.clear()
        gc.collect()


def process_time_group(df_group):
    """
    对单个时间戳分组 DataFrame 进行处理的示例函数。
    在这里可以执行任意你想要的操作，比如计算指标、做分析或可视化等。
    """
    # 比如你想打印该时间戳以及数据行数
    timestamp = df_group['time'].iloc[0]
    print(f"Processing time group: {timestamp}, rows = {len(df_group)}")

    # 这里可以对 df_group 做一些自定义的统计或转换
    # ...
    # 返回处理后的结果（也可以不返回，在函数里直接写文件等）
    return df_group

def main():
    h5_file_path = "ADA-USDT_spot_ACH-USDT_spot_15min.h5"

    # 1. 初始化计数器，或其他辅助变量
    processed_count = 0

    # 2. 逐个获取“按时间戳分组”的 DataFrame
    for df_group in read_h5_as_generator(h5_file_path, key="dataset", chunk_size=100_000):
        # 3. 对分组做处理
        result_df = process_time_group(df_group)

        # 4.（可选）将处理后的结果写入 CSV（或写回数据库，或进行其他持久化存储）
        #    这里演示“追加模式”写出 CSV，避免加载到内存一次性处理所有数据

        # 5. 释放内存
        del result_df
        gc.collect()

        # 6. 打印进度信息
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"已处理 {processed_count} 个分组...")

    print("处理全部完成！")

if __name__ == "__main__":
    main()


