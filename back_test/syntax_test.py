import pandas as pd

# 笛卡尔积生成多级索引
index = pd.MultiIndex.from_product([['A', 'B'], [1, 2, 3]], names=['Level1', 'Level2'])
df = pd.DataFrame({'Value': range(6)}, index=index)

print(df)
