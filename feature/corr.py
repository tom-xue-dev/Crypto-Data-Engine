import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd
df1 = pd.read_pickle("alpha19.pkl")
df2 = pd.read_pickle("alpha19_1.pkl")
print(df1)
print(df2)
df1 = df1.dropna(subset=['factor', '1D', '5D', '10D'])
df2 = df2.dropna(subset=['factor', '1D', '5D', '10D'])
factor1 = df1["factor"]
factor2 = df2["factor"]
min_len = min(len(factor1), len(factor2))
factor1 = factor1[:min_len]
factor2 = factor2[:min_len]
pearson_corr, _ = pearsonr(factor1, factor2)
spearman_corr, _ = spearmanr(factor1, factor2)
kendall_corr, _ = kendalltau(factor1, factor2)

# 输出结果
print(f"皮尔逊相关系数: {pearson_corr:.4f}")
print(f"斯皮尔曼秩相关系数: {spearman_corr:.4f}")
print(f"肯德尔秩相关系数: {kendall_corr:.4f}")

plt.scatter(factor1, factor2, alpha=0.6)
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Scatter Plot of Factor1 vs Factor2")
plt.show()