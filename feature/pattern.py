import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
from technical_analysis import trend_analysis
from CUSUM_filter import generate_filter_df, triple_barrier_labeling
from labeling import parallel_apply_triple_barrier, triple_barrier_labeling
from feature_generation import alpha102
import utils as u
import talib

from feature_generation import alpha1, alpha6, alpha8, alpha10, alpha19, alpha20, alpha24, alpha25, alpha26, alpha32, \
    alpha35, alpha44, alpha46, alpha49, alpha51, alpha68, alpha84, alpha94, alpha95, alpha2, alpha102


if __name__ == '__main__':
    pattern_functions = talib.get_function_groups()["Pattern Recognition"]
    select_columns = ['CDLHANGINGMAN', 'CDLSHOOTINGSTAR', 'CDLSTALLEDPATTERN', 'CDLTHRUSTING', 'CDLADVANCEBLOCK',
                      'CDLDOJISTAR', 'CDLINVERTEDHAMMER', 'CDLTASUKIGAP', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
                      'CDLXSIDEGAP3METHODS', 'CDLHAMMER', 'CDLHARAMICROSS', 'CDLTASUKIGAP']
    print(pattern_functions)
    start = "2020-1-1"
    end = "2024-12-31"
    assets = select_assets(start_time=start, spot=True, n=50)
    # print(assets)
    # assets = ['BTC-USDT_spot']
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
    data['vwap'] = u.vwap(data)
    print("start label")
    data['label'] = parallel_apply_triple_barrier(data)
    # 计算各个 label 的数量



    # data['label'] = np.where(data['future_return'] > 0, 1, 0)


    # 计算每个 asset 对应的列数（特征数量）
    # 计算每个 asset 在 MultiIndex DataFrame 中的行数
    # print(data['label'].value_counts())

    def compute_patterns(df):
        """对每个 `asset` 组计算所有形态"""
        result = {}
        for pattern in pattern_functions:
            result[pattern] = getattr(talib, pattern)(df["open"], df["high"], df["low"], df["close"]) / 100
        return pd.DataFrame(result, index=df.index)


    data = generate_filter_df(data, threshold=5 * 0.007)
    label_counts = data['label'].value_counts()

    # 输出每个 label 的数量，如果某个 label 不存在则返回 0
    print("label==0:", label_counts.get(0, 0))
    print("label==1:", label_counts.get(1, 0))
    print("label==2:", label_counts.get(2, 0))
    print("start func")
    data[pattern_functions] = data.groupby(level="asset", group_keys=False).apply(compute_patterns)
    print(data[select_columns].columns)
    alpha_funcs = [
        ('alpha1', alpha1),
        ('alpha6', alpha6),
        ('alpha8', alpha8),
        ('alpha10', alpha10),
        ('alpha19', alpha19),
        ('alpha24', alpha24),
        ('alpha26', alpha26),
        ('alpha32', alpha32),
        ('alpha35', alpha35),
        ('alpha46', alpha46),
    ]
    for col_name, func in alpha_funcs:
        data[col_name] = func(data)
    data = data.dropna()

    train_dict1 = {col: data[col].values for col in select_columns}

    train_dict2 = {
        'alpha1': data['alpha1'].values,
        'alpha6': data['alpha6'].values,
        'alpha8': data['alpha8'].values,
        'alpha10': data['alpha10'].values,
        'alpha19': data['alpha19'].values,
        'alpha24': data['alpha24'].values,
        'alpha26': data['alpha26'].values,
        'alpha32': data['alpha32'].values,
        'alpha35': data['alpha35'].values,
        'alpha46': data['alpha46'].values,
        'label': data['label'].values
    }

    # 方法1：使用 update 方法
    train_dict = train_dict1.copy()  # 先复制一个字典
    train_dict.update(train_dict2)

    with open('data.pkl', 'wb') as f:
        pickle.dump(train_dict, f)


    def compute_accuracy(df):
        """计算每个形态对 label=1（上涨）和 label=-1（下跌）的预测准确率"""
        accuracy_up = {}  # 预测上涨的准确率
        accuracy_down = {}  # 预测下跌的准确率

        for pattern in pattern_functions:  # 遍历所有 K 线形态
            signal = df[pattern]

            # 预测上涨（100）且实际也上涨（label=1）
            correct_up = (signal == 100) & (df["label"] == 1)
            # 预测下跌（-100）且实际也下跌（label=-1）
            correct_down = (signal == -100) & (df["label"] == -1)

            # 计算准确率
            accuracy_up[pattern] = correct_up.mean()  # 计算看涨信号的正确率
            accuracy_down[pattern] = correct_down.mean()  # 计算看跌信号的正确率

        return pd.DataFrame({"Up_Accuracy": accuracy_up, "Down_Accuracy": accuracy_down})


    def compute_signal_accuracy(df):
        """
        对 DataFrame df 中每个形态（pattern）下的各个信号值计算预测准确率。

        假设：
          - df 中包含各个形态的列，其列名由 pattern_functions 列表给出
          - df 中有一列 "label"，其中 1 表示上涨，-1 表示下跌
        返回：
          - 一个字典，形如 {pattern: {signal_value: accuracy, ...}, ...}
        """
        accuracy = {}
        for pattern in pattern_functions:
            pattern_signals = df[pattern]
            # 获取该形态中所有出现的信号值
            unique_signals = pattern_signals.unique()
            signal_accuracy = {}
            for sig in unique_signals:
                # 对于信号值为 0（或中性信号），可以选择跳过（或单独统计）
                if sig == 0:
                    continue
                # 筛选出该信号值对应的样本
                mask = (pattern_signals == sig)
                n_samples = mask.sum()
                # 如果该信号出现次数太少（例如少于10次），可以考虑忽略
                if n_samples < 10:
                    continue
                # 根据信号判断预测方向：
                # 若 sig > 0，则认为预测上涨；若 sig < 0，则认为预测下跌
                if sig > 0:
                    correct = (df.loc[mask, "label"] == 1)
                else:  # sig < 0
                    correct = (df.loc[mask, "label"] == -1)
                accuracy_val = correct.mean()  # 正确率
                signal_accuracy[sig] = accuracy_val
            accuracy[pattern] = signal_accuracy
        return accuracy

    # 示例 1：对整个数据集计算各形态下每个信号的预测准确率
    # signal_accuracy_all = compute_signal_accuracy(data)
    # print("各形态下各信号的预测准确率：")
    # for pattern, signal_dict in signal_accuracy_all.items():
    #     print(f"\nPattern: {pattern}")
    #     for sig, acc in signal_dict.items():
    #         print(f"  Signal {sig}: Accuracy = {acc:.2%}")
    #
    # # 示例 2：如果数据中包含多个资产，并且希望按资产分组统计，再聚合得到总体情况
    # # 假设 data 的索引中有一个 level 是 "asset"
    # signal_accuracy_by_asset = data.groupby(level="asset").apply(compute_signal_accuracy)

    # signal_accuracy_by_asset 的结构为：
    # asset1 -> {pattern1: {sig: acc, ...}, pattern2: {...}, ...}
    # asset2 -> {pattern1: {sig: acc, ...}, pattern2: {...}, ...}
    #
    # 如果希望对所有资产的结果进行聚合，可以自行实现数据合并，
    # 例如计算每个形态下、每个信号值在所有资产上的平均准确率：
    # from collections import defaultdict
    #
    # # 建立一个嵌套字典存储聚合结果：{pattern: {signal_value: [acc1, acc2, ...], ...}, ...}
    # aggregate_accuracy = defaultdict(lambda: defaultdict(list))
    #
    # for asset, asset_result in signal_accuracy_by_asset.items():
    #     for pattern, signal_dict in asset_result.items():
    #         for sig, acc in signal_dict.items():
    #             aggregate_accuracy[pattern][sig].append(acc)
    #
    # # 计算平均准确率
    # average_accuracy = {}
    # for pattern, sig_dict in aggregate_accuracy.items():
    #     average_accuracy[pattern] = {sig: np.mean(acc_list) for sig, acc_list in sig_dict.items()}
    #
    # print("\n各形态下各信号在所有资产上的平均预测准确率：")
    # for pattern, signal_dict in average_accuracy.items():
    #     print(f"\nPattern: {pattern}")
    #     for sig, acc in signal_dict.items():
    #         print(f"  Signal {sig}: Average Accuracy = {acc:.2%}")
