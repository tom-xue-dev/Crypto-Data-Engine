import gc
import pickle
from strategy import DualMAStrategy
import pandas as pd

from analysis import process_future_performance_in_pool

def filter_data(dataset):
    dataset["MA50_shifted"] = dataset.groupby("asset")["MA50"].shift(5)
    dataset.loc[
        (dataset["signal"] == 1) & ( dataset["MA50"] > dataset["MA50_shifted"]),
        "signal"
    ] = 0

    # dataset.loc[
    #     (dataset["signal"] == -1) & ( dataset["MA50"] > dataset["MA50_shifted"]),
    #     "signal"
    # ] = 0
    return dataset
if __name__ == "__main__":
    with open("data.pkl", "rb") as file:
        data = pickle.load(file)
    strategy = DualMAStrategy(dataset=data,long_period=50,short_period=5)
    data = filter_data(strategy.dataset)
    with open("filter_data.pkl", "wb") as file:
        pickle.dump(data,file)
    # n_days_list = list(range(5, 600, 20))
    # # results = process_future_performance_in_pool(data1, n_days_list, signal=1)
    # # # 输出结果
    # # for n_days, avg_return, prob_gain, count in results:
    # #     print(f"未来{n_days}根k线上涨概率为{prob_gain},上涨幅度{avg_return},总数{count}")
    # # del data1
    # # gc.collect()
    # with open("data_filter_3.pkl", "rb") as file:
    #     data = pickle.load(file)
    # results = process_future_performance_in_pool(data, n_days_list, signal=1)
    # for n_days, avg_return, prob_gain, count in results:
    #     print(f"未来{n_days}根k线上涨概率为{prob_gain},上涨幅度{avg_return},总数{count}")
    # print()
