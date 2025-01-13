import gc
import pickle

import pandas as pd

from analysis import process_future_performance_in_pool

if __name__ == "__main__":
    # with open("data_filter.pkl", "rb") as file:
    #     data1 = pickle.load(file)
    #
    n_days_list = list(range(5, 600, 20))
    # results = process_future_performance_in_pool(data1, n_days_list, signal=1)
    # # 输出结果
    # for n_days, avg_return, prob_gain, count in results:
    #     print(f"未来{n_days}根k线上涨概率为{prob_gain},上涨幅度{avg_return},总数{count}")
    # del data1
    # gc.collect()
    with open("data_filter_3.pkl", "rb") as file:
        data = pickle.load(file)
    results = process_future_performance_in_pool(data, n_days_list, signal=1)
    for n_days, avg_return, prob_gain, count in results:
        print(f"未来{n_days}根k线上涨概率为{prob_gain},上涨幅度{avg_return},总数{count}")
    print()
