import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
import utils as u
import gplearn.functions as gf
from gplearn.fitness import make_fitness

def information_coefficient(y, y_pred, w):
    return np.corrcoef(y, y_pred)[0, 1]  # 计算 RankIC

ic_fitness = make_fitness(function=information_coefficient, greater_is_better=True)

start = "2023-1-1"
end = "2023-12-31"
assets = select_assets(start_time=start,spot=True,m=50)
data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="1d")
data['1D'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
data['5D'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-30) / x - 1).droplevel(0)
data['10D'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-50) / x - 1).droplevel(0)
X = data[["open", "high", "low", "close", "volume"]].values  # 变成 NumPy
y = data["5D"].values  # 目标值也转换成 NumPy
print(X)
print(y)
function_set = [
    gf.add2, gf.sub2, gf.mul2, gf.div2, gf.log1
]
valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)  # 找到没有 NaN 的行
X, y = X[valid_idx], y[valid_idx]
model = SymbolicRegressor(
    function_set=["add", "sub", "mul", "div"],  # 允许的数学运算
    generations=10,  # 进化轮数
    population_size=1000,  # 种群大小
    parsimony_coefficient=0.0001,  # 复杂度惩罚，避免过拟合
    random_state=42
)

model.fit(X, y)

print("最优表达式:", model._program)
