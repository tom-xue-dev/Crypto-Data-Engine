import pyarrow.parquet as pq


def read_nested_parquet_with_pyarrow(parquet_file_path):
    """
    使用 PyArrow 读取嵌套 Parquet 文件，并检查内容。

    参数:
        parquet_file_path (str): Parquet 文件路径
    """
    try:
        # 读取 Parquet 文件
        table = pq.read_table(parquet_file_path)

        # 转换为 PyArrow 的 Pandas DataFrame
        df = table.to_pandas()

        # 打印嵌套内容的示例
        print(f"读取文件: {parquet_file_path}")
        print(df.head())  # 显示前几行

        # 检查嵌套列
        for index, row in df.iterrows():
            print(f"Time: {row['time']}, Data: {row['data']}")
            if index >= 4:  # 只展示前5行
                break
    except Exception as e:
        print(f"读取文件 {parquet_file_path} 时发生错误: {e}")


read_nested_parquet_with_pyarrow("nested_parquet/1d/1INCH-USDT_future_1d.parquet")