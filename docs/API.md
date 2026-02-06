# Crypto Data Engine — API Reference

> 本文档覆盖所有核心 Python 模块的公开 API 以及 REST 端点。

---

## 目录

- [Bar Aggregation](#bar-aggregation)
- [Feature Calculation](#feature-calculation)
- [Signal Generation](#signal-generation)
- [Backtest Engine](#backtest-engine)
- [Asset Pool](#asset-pool)
- [Data Download](#data-download)
- [Task Manager](#task-manager)
- [REST API Endpoints](#rest-api-endpoints)

---

## Bar Aggregation

**模块路径**: `crypto_data_engine.services.bar_aggregator`

将 tick 数据聚合为不同类型的 bar（时间 bar、tick bar、volume bar、dollar bar）。

### `aggregate_bars()`

通用入口，自动选择最优聚合实现（Numba / 多进程）。

```python
from crypto_data_engine.services.bar_aggregator import aggregate_bars

bars = aggregate_bars(
    data,                          # DataFrame | str | Path | List[str]
    bar_type="dollar_bar",         # BarType | str
    threshold=1_000_000,           # int | float | str
    use_numba=True,                # 启用 Numba 加速
    use_multiprocess=True,         # 启用多进程
    n_workers=4,                   # 进程数
    include_advanced=True,         # 含 VWAP/volatility 等高级字段
    symbol="BTCUSDT",              # 标识符
    progress_callback=None,        # 进度回调 (current, total)
)
```

### 便捷函数

| 函数 | 用途 | 关键参数 |
|------|------|----------|
| `build_time_bars(data, interval="5min")` | 固定时间区间 | `interval`: 时间字符串 |
| `build_tick_bars(data, n_ticks=1000)` | 固定 tick 数 | `n_ticks`: 每 bar tick 数 |
| `build_volume_bars(data, volume_threshold=100_000)` | 固定成交量 | `volume_threshold` |
| `build_dollar_bars(data, dollar_threshold=1_000_000)` | 固定美元成交额 | `dollar_threshold` |
| `benchmark_aggregation(data, bar_type, threshold)` | 性能基准测试 | `n_iterations=3` |

### `FastBarAggregator`

高性能聚合器，支持 Numba JIT 加速。

```python
from crypto_data_engine.services.bar_aggregator import FastBarAggregator

aggregator = FastBarAggregator(bar_type="dollar_bar", threshold=1_000_000)
result: AggregationResult = aggregator.aggregate(tick_data)
```

### `StreamingAggregator`

内存高效的流式聚合器，适用于超大数据集。

```python
from crypto_data_engine.services.bar_aggregator import create_streaming_aggregator

streamer = create_streaming_aggregator("dollar_bar", 1_000_000, chunk_size=1_000_000)
for chunk in data_chunks:
    bars = streamer.process(chunk)
```

---

## Feature Calculation

**模块路径**: `crypto_data_engine.services.feature`

从 bar 数据计算因子特征（收益率、波动率、动量、微观结构、alpha 因子等）。

### `calculate_features()`

最常用的便捷函数。

```python
from crypto_data_engine.services.feature import calculate_features

features = calculate_features(
    data,                           # 含 OHLCV 列的 DataFrame
    windows=[5, 10, 20, 60, 120],   # 滚动窗口
    include_alphas=True,            # 含 alpha 因子
    include_microstructure=True,    # 含微观结构特征
    include_technical=True,         # 含技术指标（需 talib）
    normalize=False,                # 标准化
    drop_na=False,                  # 丢弃 NaN 行
)
```

### `calculate_features_multi_asset()`

多资产截面特征计算（含排名、z-score 等截面特征）。

```python
from crypto_data_engine.services.feature import calculate_features_multi_asset

features = calculate_features_multi_asset(
    data,                            # MultiIndex (timestamp, asset) DataFrame
    windows=[5, 10, 20],
    include_cross_sectional=True,    # 截面特征
)
```

### `UnifiedFeatureCalculator`

更精细控制的面向对象接口。

```python
from crypto_data_engine.services.feature import UnifiedFeatureConfig, UnifiedFeatureCalculator

config = UnifiedFeatureConfig(
    windows=[5, 10, 20, 60, 120],
    include_returns=True,
    include_volatility=True,
    include_momentum=True,
    include_volume=True,
    include_microstructure=True,
    include_alphas=True,
    include_technical=True,
    normalize=False,
    winsorize_std=3.0,
)
calculator = UnifiedFeatureCalculator(config)
features = calculator.calculate(bar_data, asset="BTCUSDT")
```

### 特征选择工具

```python
from crypto_data_engine.services.feature import select_features_by_correlation, get_feature_importance

# 按相关性选择 Top-N 特征
selected = select_features_by_correlation(features, target, top_n=50, min_correlation=0.05)

# 特征重要性排序
importance = get_feature_importance(features, target, method="mutual_info")
```

---

## Signal Generation

**模块路径**: `crypto_data_engine.services.signal_generation`

从因子/规则/集成方法生成交易信号。

### `FactorSignalGenerator`

基于因子加权组合的信号生成器。

```python
from crypto_data_engine.services.signal_generation.factor_signal import (
    FactorConfig, FactorSignalGenerator,
)

generator = FactorSignalGenerator(
    factors=[
        FactorConfig(column="momentum_20", weight=0.6),
        FactorConfig(column="volatility_20", weight=-0.4),
    ],
    long_threshold=0.5,
    short_threshold=-0.5,
)
signal = generator.generate(cross_section_data)
```

### `RankSignalGenerator`

基于因子排名的信号生成器（适用于截面策略）。

```python
from crypto_data_engine.services.signal_generation.factor_signal import RankSignalGenerator

generator = RankSignalGenerator(
    factor_col="return_20",
    top_n_long=10,
    top_n_short=10,
    ascending=False,
)
signal = generator.generate(cross_section_data)
```

### `RuleSignalGenerator`

基于规则条件的信号生成器。

```python
from crypto_data_engine.services.signal_generation.rule_signal import (
    RuleCondition, ComparisonOperator, RuleSignalGenerator,
)

generator = RuleSignalGenerator(
    long_conditions=[
        RuleCondition(column="rsi", operator=ComparisonOperator.LESS_THAN, value=30),
    ],
    short_conditions=[
        RuleCondition(column="rsi", operator=ComparisonOperator.GREATER_THAN, value=70),
    ],
)
```

### `TechnicalRuleGenerator` — 预置技术策略

```python
from crypto_data_engine.services.signal_generation.rule_signal import TechnicalRuleGenerator

rsi_gen     = TechnicalRuleGenerator.rsi_strategy(oversold=30, overbought=70)
ma_gen      = TechnicalRuleGenerator.ma_crossover(fast_ma="sma_10", slow_ma="sma_50")
bb_gen      = TechnicalRuleGenerator.bollinger_bands()
mom_gen     = TechnicalRuleGenerator.momentum(threshold=0.05)
```

### `EnsembleSignalGenerator`

组合多个信号生成器。

```python
from crypto_data_engine.services.signal_generation.ensemble import (
    EnsembleMethod, GeneratorConfig, EnsembleSignalGenerator,
)

ensemble = EnsembleSignalGenerator(
    generators=[
        GeneratorConfig(generator=factor_gen, weight=0.6),
        GeneratorConfig(generator=rule_gen, weight=0.4),
    ],
    method=EnsembleMethod.WEIGHTED_AVERAGE,
)
```

---

## Backtest Engine

**模块路径**: `crypto_data_engine.services.back_test`

支持截面（cross-sectional）和时间序列（time-series）两种回测模式。

### 快速开始

```python
from crypto_data_engine.services.back_test import (
    BacktestConfig, BacktestMode,
    RiskConfigModel, CostConfigModel,
    create_backtest_engine, create_strategy,
)

config = BacktestConfig(
    mode=BacktestMode.CROSS_SECTIONAL,
    initial_capital=1_000_000,
    rebalance_frequency="W-MON",
    risk_config=RiskConfigModel(max_position_size=0.2, max_leverage=1.0),
    cost_config=CostConfigModel(commission_rate=0.001, slippage_rate=0.0005),
)

strategy = create_strategy("momentum", lookback_col="return_20", top_n_long=5, top_n_short=5)
engine = create_backtest_engine(config, strategy)
result = engine.run(feature_data)
```

### `BacktestConfig` — 配置项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | `BacktestMode` | `CROSS_SECTIONAL` | 回测模式 |
| `initial_capital` | `float` | `1_000_000` | 初始资金 |
| `rebalance_frequency` | `str` | `"W-MON"` | 再平衡频率 |
| `top_n_long` | `int` | `10` | 做多资产数 |
| `top_n_short` | `int` | `10` | 做空资产数 |
| `ranking_factor` | `str` | `"return_1w"` | 排序因子列 |
| `risk_config` | `RiskConfigModel` | `None` | 风控配置 |
| `cost_config` | `CostConfigModel` | `None` | 成本模型 |
| `allow_short` | `bool` | `True` | 允许做空 |
| `warmup_periods` | `int` | `0` | 预热期数 |
| `log_trades` | `bool` | `True` | 记录交易 |

### 工厂函数

```python
# 通用工厂
engine = create_backtest_engine(config, strategy, logger=None)

# 便捷策略创建
strategy = create_strategy("momentum", lookback_col="return_20", top_n_long=10)
strategy = create_strategy("mean_reversion", lookback_col="return_5", top_n_long=10)
strategy = create_strategy("equal_weight", max_assets=20)
strategy = create_strategy("long_short", long_col="momentum_20", short_col="volatility_20")

# 预配置回测
engine = create_momentum_backtest(config, lookback_col="return_20", top_n_long=10)
engine = create_mean_reversion_backtest(config, lookback_col="return_5")
```

### 引擎类

- **`CrossSectionalEngine`** — 固定周期再平衡，截面排名选股
- **`TimeSeriesEngine`** — 逐 bar 决策，适用于单资产/时间序列策略

### Portfolio & Trade Records

```python
from crypto_data_engine.services.back_test import Portfolio, Position

# 回测后获取结果
nav_history = engine.get_nav_history()       # Dict[datetime, float]
trades = engine.get_trades()                 # List[TradeRecord]
snapshots = engine.get_snapshots()           # List[PortfolioSnapshot]
```

---

## Asset Pool

**模块路径**: `crypto_data_engine.services.asset_pool`

动态资产池选择器，查询 Binance Futures API 获取 Top-N 成交额交易对。

```python
from crypto_data_engine.services.asset_pool.asset_selector import (
    AssetPoolSelector, AssetPoolConfig,
)

selector = AssetPoolSelector(config=AssetPoolConfig(top_n=100))
symbols = selector.get_current_pool(top_n=30, force_refresh=False)
pool_info = selector.get_pool_info()
```

| 方法 | 说明 |
|------|------|
| `get_current_pool(top_n, force_refresh)` | 获取当前资产池（带缓存） |
| `update_pool()` | 强制刷新 |
| `get_pool_info()` | 返回池的元数据 |

---

## Data Download

**模块路径**: `crypto_data_engine.services.tick_data_scraper`

支持 Binance (Spot/Futures)、OKX 等交易所的 tick 数据下载。

### `tick_worker.run_download()`

```python
from crypto_data_engine.services.tick_data_scraper.tick_worker import run_download

run_download(
    exchange_name="binance_futures",
    symbols=["BTCUSDT", "ETHUSDT"],       # None = 全部
    start_date="2026-01",
    end_date="2026-01",
    max_threads=8,
)
```

### `ExchangeFactory`

```python
from crypto_data_engine.services.tick_data_scraper.downloader.exchange_factory import ExchangeFactory

adapter = ExchangeFactory.create_adapter("binance_futures")
exchanges = ExchangeFactory.get_supported_exchanges()
```

---

## Task Manager

**模块路径**: `crypto_data_engine.common.task_manager`

轻量级任务管理器，支持 IO / CPU 密集型任务调度，Redis 或内存存储。

```python
from crypto_data_engine.common.task_manager import create_task_manager, get_task_manager

# 创建（通常在启动时）
manager = create_task_manager(store="memory", max_io_threads=16, max_compute_processes=8)

# 获取全局实例
manager = get_task_manager()
```

| 方法 | 说明 |
|------|------|
| `create_task(metadata)` | 创建任务记录 |
| `get_task(task_id)` | 查询任务状态 |
| `update_task(task_id, status, progress, ...)` | 更新任务 |
| `list_tasks(status, limit, offset)` | 列出任务 |
| `submit_io_task(task_id, func, *args)` | 提交 IO 型任务（线程池） |
| `submit_compute_task(task_id, func, *args)` | 提交 CPU 型任务（进程池） |
| `cancel_task(task_id)` | 取消任务 |
| `cleanup_expired(ttl_seconds)` | 清理过期任务 |
| `shutdown(wait)` | 关闭执行器 |

---

## REST API Endpoints

启动服务：

```bash
# CLI
crypto-engine serve --host 0.0.0.0 --port 8000

# 或直接
uvicorn crypto_data_engine.api.main:app --reload --port 8000
```

文档地址：`http://localhost:8000/docs`（Swagger UI）

### Backtest — `/api/backtest`

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/run` | 提交回测任务 |
| `GET` | `/status/{task_id}` | 查询回测状态 |
| `GET` | `/result/{task_id}` | 获取回测结果 |
| `GET` | `/trades/{task_id}` | 分页查询交易记录 |
| `DELETE` | `/{task_id}` | 取消/删除回测 |
| `GET` | `/list` | 列出所有回测任务 |
| `GET` | `/logs/{task_id}` | 获取交易日志 |
| `GET` | `/logs/{task_id}/trades` | 交易执行日志 |
| `GET` | `/logs/{task_id}/signals` | 策略信号日志 |
| `GET` | `/logs/{task_id}/snapshots` | 组合快照 |
| `GET` | `/logs/{task_id}/rebalances` | 再平衡事件 |
| `GET` | `/logs/{task_id}/export` | 导出日志 |

### Strategy — `/api/strategy`

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/list` | 列出可用策略 |
| `GET` | `/{name}` | 策略详情 |
| `GET` | `/{name}/params` | 策略参数定义 |
| `POST` | `/validate` | 验证策略参数 |
| `GET` | `/presets/list` | 预置配置列表 |
| `GET` | `/presets/{name}` | 预置配置详情 |

### Visualization — `/api/viz`

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/nav/{task_id}` | NAV 时间序列 |
| `GET` | `/drawdown/{task_id}` | 回撤时间序列 |
| `GET` | `/returns/{task_id}` | 收益分布 |
| `GET` | `/heatmap/{task_id}` | 月度收益热力图 |
| `GET` | `/trades/{task_id}/summary` | 交易分析摘要 |
| `GET` | `/performance/{task_id}` | 综合绩效报告 |

### Data Download — `/api/v1/download`

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/exchanges` | 支持的交易所列表 |
| `GET` | `/{source}/symbols` | 交易对列表 |
| `POST` | `/downloads/jobs` | 创建批量下载任务 |
| `GET` | `/status/{task_id}` | 下载任务状态 |

### Bar Aggregation — `/api/v1/aggregate`

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/bars` | 提交聚合任务 |
| `GET` | `/status/{task_id}` | 聚合任务状态 |
