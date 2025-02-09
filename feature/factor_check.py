import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

for key,item in data.items():
    # plt.plot(item)
    # plt.show()
    print(key,np.max(item))
# features = [
#     "alpha1", "alpha6", "alpha8", "alpha10", "alpha19", "alpha20","alpha24","alpha25","alpha26",
#     "alpha32", "alpha35", "alpha44","alpha46","alpha49", "alpha51", "alpha68", "alpha84",
#     "alpha94", "alpha95"
# ]

del data['label']
# del data['alpha84']
# del data['alpha6']
# del data ['alpha10']
del data['alpha49']

# del data ['alpha20']


features = list(data.keys())
print(features)
alpha_values = np.array(list(data.values()))

# 如果每一行代表一个特征，可以直接计算相关系数矩阵
corr_matrix = np.corrcoef(alpha_values)

print("特征之间的相关性矩阵：")
print(corr_matrix)

threshold = 0.3
n = corr_matrix.shape[0]

print("以下配对的相关系数大于 %.2f:" % threshold)
for i in range(n):
    for j in range(i + 1, n):
        # 如果你只关注正相关（即数值>0.03），直接用下面这一行：
        if abs(corr_matrix[i, j]) > threshold:
        # 如果你想按绝对值筛选（|r|>0.03），则改为：
        # if abs(corr_matrix[i, j]) > threshold:
            print(f"({features[i]}, {features[j]}): {corr_matrix[i, j]:.4f}")
