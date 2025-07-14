# 回测相关API文档

## backtest_simulation.py

---

### backtest类:

def __init__(self, broker: Broker, strategy_results: pd.DataFrame, pos_manager=PositionManager())

初始化函数，传入撮合器，策略生成后的dataframe以及position manager

