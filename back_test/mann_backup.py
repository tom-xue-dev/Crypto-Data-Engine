import os  # 导入 os 模块以使用 os.fsync
import time
import itertools

from read_large_files import load_filtered_data_as_list, select_assets
from strategy import DualMAStrategy
from backtest_simulation import Backtest, Broker
from Account import Account, PositionManager, DefaultStopLossLogic
from mann import MannKendallTrendByRow, filter_signals_by_daily_vectorized
from back_test_evaluation import PerformanceAnalyzer

# ============== 可根据实际情况修改的配置 ==============
start_time = "2021-01-01"
end_time = "2023-12-31"
initial_cash = 10000
threshold_for_position = 0.1
max_drawdown_for_stoploss = 0.1
spot_asset_number = 30

# 需要调参的 grid (示例)
param_grid_min = {
    "window_size": [48, 96],  # 15分钟级别的窗口大小
    "z_crit": [1.2,1.4,1.6]  # 15分钟级别的 z_crit 阈值
}
param_grid_day = {
    "window_size": [7, 14],  # 日级别的窗口大小
    "z_crit": [1.5, 1.8,2.0,2.2]  # 日级别的 z_crit 阈值
}

# 定义要存储回测结果的文件
results_file = "grid_search_results.txt"

# ============== 1. 准备数据 ==============
asset_list = select_assets(spot=True, n=spot_asset_number)
min_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "15min")
day_data_list = load_filtered_data_as_list(start_time, end_time, asset_list, "1d")

# ============== 2. 定义网格搜索并进行回测 ==============
best_final_value = -float("inf")  # 用于记录当前搜索中最优的净值
best_params = None  # 记录最优参数组合

# 存储所有回测结果(可选)
search_records = []

# 获取所有可能的 (min_strategy, day_strategy) 参数组合的笛卡尔积
all_min_param_combinations = list(itertools.product(
    param_grid_min["window_size"], param_grid_min["z_crit"]
))
all_day_param_combinations = list(itertools.product(
    param_grid_day["window_size"], param_grid_day["z_crit"]
))

total_combinations = len(all_min_param_combinations) * len(all_day_param_combinations)
print("开始网格搜索，共需测试的参数组合数：", total_combinations)

# 使用 with 语句在循环外部打开文件，提高效率
with open(results_file, "a", encoding="utf-8") as f:
    for min_win_size, min_z_crit in all_min_param_combinations:
        for day_win_size, day_z_crit in all_day_param_combinations:
            # ------------ 2.1 设置策略参数 -----------
            min_strategy = MannKendallTrendByRow(
                min_data_list.copy(),  # 使用浅拷贝，若需要深拷贝请使用 copy.deepcopy()
                window_size=min_win_size,
                asset=asset_list,
                z_crit=min_z_crit
            )
            day_strategy = MannKendallTrendByRow(
                day_data_list.copy(),
                window_size=day_win_size,
                asset=asset_list,
                z_crit=day_z_crit
            )

            # ------------ 2.2 生成信号 -----------
            min_strategy.generate_signal()
            day_strategy.generate_signal()

            strategy_results = filter_signals_by_daily_vectorized(
                min_strategy.dataset,
                day_strategy.dataset
            )

            # ------------ 2.3 设置回测环境并运行 -----------
            account = Account(initial_cash=initial_cash)
            stop_logic = DefaultStopLossLogic(max_drawdown=max_drawdown_for_stoploss)
            broker = Broker(account, stop_loss_logic=stop_logic)

            pos_manager = PositionManager(threshold=threshold_for_position)
            backtester = Backtest(broker, strategy_results, pos_manager)

            result = backtester.run()

            # ------------ 2.4 分析回测结果 -----------
            analyser = PerformanceAnalyzer(result["net_value_history"])
            summary = analyser.summary()  # 假设返回一个字符串或可序列化的内容

            # 获取一个简单的选优指标：最终净值(可替换为想要的指标)
            final_net_value = analyser.calculate_annual_return()

            # ------------ 2.5 记录回测信息到文件 -----------
            f.write(f"参数组合：min_window_size={min_win_size}, "
                    f"min_z_crit={min_z_crit}, day_window_size={day_win_size}, "
                    f"day_z_crit={day_z_crit}\n")
            f.write(f"Final Net Value: {final_net_value}\n")
            f.write(f"Summary: {summary}\n")
            f.write("-" * 50 + "\n")

            # 刷新缓冲区并强制写入磁盘
            f.flush()
            os.fsync(f.fileno())

            # ------------ 2.6 判断是否最优 -----------
            if final_net_value > best_final_value:
                best_final_value = final_net_value
                best_params = {
                    "min_window_size": min_win_size,
                    "min_z_crit": min_z_crit,
                    "day_window_size": day_win_size,
                    "day_z_crit": day_z_crit
                }

            print(f"完成参数组合: (min_win={min_win_size}, min_z={min_z_crit}, "
                  f"day_win={day_win_size}, day_z={day_z_crit}) => FinalValue={final_net_value}")

# ============== 3. 输出最优结果 ==============
print("\n网格搜索结束！")
print("最优参数组合及结果：", best_params)
print("对应的最终净值：", best_final_value)
