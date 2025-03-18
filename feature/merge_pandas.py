import sys

import numpy as np
import pandas as pd
import pickle

import pandas as pd


def merge_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    合并两个 DataFrame（其中一个的行索引为另一个的子集），
    保留行索引的并集，并填充缺失值为 0。
    若存在重复列名，则使用“行索引较多”的那个 DataFrame 中的列，
    丢弃另一个 DF 中重名的列。

    参数：
        df1, df2: 待合并的两个 DataFrame

    返回：
        merged_df: 合并后的 DataFrame
    """
    # 1. 判断哪个 DataFrame 的行索引更多（或行数更多）
    if len(df1.index) >= len(df2.index):
        bigger = df1
        smaller = df2
    else:
        bigger = df2
        smaller = df1

    # 2. 对两个 DF 做“行索引并集”，并将缺失位置填 0
    #    （如果你只想填充行索引，可以先 reindex 行，然后再做列上的处理）
    bigger_aligned = bigger.reindex(bigger.index.union(smaller.index), fill_value=0)
    smaller_aligned = smaller.reindex(bigger.index.union(smaller.index), fill_value=0)

    # 3. 丢弃 smaller 中和 bigger 重名的列
    #    这样最终合并后就只会保留 bigger 中的重复列，
    #    避免出现列名冲突（诸如 A、A.1）
    overlap_cols = bigger_aligned.columns.intersection(smaller_aligned.columns)
    smaller_aligned = smaller_aligned.drop(columns=overlap_cols, errors='ignore')

    # 4. 拼接列（outer 方式，保留所有行索引），对 NaN 补 0
    merged_df = pd.concat([bigger_aligned, smaller_aligned], axis=1, join='outer').fillna(0)

    return merged_df


with open("data_signal.pkl", "rb") as f:
    signal_data = pickle.load(f)

with open("origin_data.pkl", "rb") as f:
    origin_data = pickle.load(f)
# pd.set_option('display.max_rows', 100)  # 最多显示 100 行
# pd.set_option('display.max_columns', 50)  # 最多显示 50 列
#
#
# data = merge_df(signal_data, origin_data)
#
# res = data[["close", "signal"]]
# for asset,group in res.groupby('asset'):
#     print(asset)
#     print(group['signal'].value_counts().get(2, 0))
# with open("final_data.pkl", "wb") as f:
#     pickle.dump(res, f)


# 假设 df 是你的多级索引 DataFrame，索引 levels=[time, asset]
# df.index.names == ['time', 'asset']
# df['label'] 为上涨概率

# ------- 1. 提取 time（日期）的 dayofweek，用于识别周日（dayofweek == 6） -------
df = signal_data.copy()
print(df.columns)
print(df['label'].values)
df['decile'] = pd.qcut(df['signal'], q=5, labels=False, duplicates='drop')
#print(group['alpha106'])
# 2) 分组后计算统计指标（如均值、数量等）
group_stats = df.groupby('decile')['future_return'].agg(['mean', 'count', 'std'])
print(group_stats)
print(df[['signal', 'future_return']].corr().iloc[0, 1])

bin_edges = pd.qcut(df['signal'], q=5, retbins=True, duplicates='drop')[1]

# 创建 DataFrame 展示分箱边界
bin_edges_df = pd.DataFrame({
    'Decile': range(1, len(bin_edges)),  # 1~10
    'Lower Bound': bin_edges[:-1],  # 每个区间的下界
    'Upper Bound': bin_edges[1:]   # 每个区间的上界
})

# 输出分箱边界值
print(bin_edges_df)

