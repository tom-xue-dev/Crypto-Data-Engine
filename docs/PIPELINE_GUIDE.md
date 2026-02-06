# 后端全流程指南：从 CLI 启动到 Backtest 出结果

> 本文档描述从 CLI 命令触发，经过 **数据下载 → Bar 聚合 → 特征计算 → 回测** 的完整函数调用链。

---

## 目录

- [项目结构概览](#项目结构概览)
- [CLI 命令一览](#cli-命令一览)
- [流程一：数据下载 (data download)](#流程一数据下载-data-download)
- [流程二：全量 Pipeline (pipeline run)](#流程二全量-pipeline-pipeline-run)
  - [Step 1: 资产池选择](#step-1-资产池选择)
  - [Step 2: Bar 聚合](#step-2-bar-聚合)
  - [Step 3: 特征计算](#step-3-特征计算)
  - [Step 4: 回测执行](#step-4-回测执行)
  - [Step 5: 验证 & 报告](#step-5-验证--报告)
- [动态资产池机制详解](#动态资产池机制详解)
- [配置系统](#配置系统)
- [数据目录结构](#数据目录结构)

---

## 项目结构概览

```
src/crypto_data_engine/
├── main.py                          # CLI 入口，注册所有子命令
├── app/                             # CLI 命令定义层
│   ├── data_cmd.py                  #   data download / list / info
│   ├── aggregate_cmd.py             #   aggregate <symbol>
│   ├── feature_cmd.py               #   features <file>
│   ├── backtest_cmd.py              #   backtest
│   ├── pipeline_cmd.py              #   pipeline run（全流程编排）
│   └── server.py                    #   serve / dev（API 服务）
├── common/
│   ├── config/
│   │   ├── config_settings.py       # Settings 单例，懒加载各模块配置
│   │   ├── downloader_config.py     # 多交易所下载配置
│   │   └── paths.py                 # 路径常量
│   └── logger/logger.py             # 日志系统
├── services/
│   ├── asset_pool/
│   │   └── asset_selector.py        # Binance API 资产池选择器
│   ├── tick_data_scraper/
│   │   ├── tick_worker.py           # 下载入口函数
│   │   └── downloader/
│   │       ├── downloader.py        # FileDownloader（下载+解压+转换）
│   │       ├── exchange_factory.py  # 交易所适配器工厂
│   │       ├── binance_futures.py   # Binance USDT-M 合约适配器
│   │       ├── binance.py           # Binance 现货适配器
│   │       └── okx.py              # OKX 适配器
│   ├── bar_aggregator/
│   │   ├── unified.py               # aggregate_bars() 统一入口
│   │   ├── fast_aggregator.py       # Numba 加速聚合
│   │   ├── bar_types.py             # Bar 类型定义（time/tick/volume/dollar）
│   │   └── feature_calculator.py    # Bar 级别特征
│   ├── feature/
│   │   └── unified_features.py      # UnifiedFeatureCalculator
│   └── back_test/
│       ├── config.py                # BacktestConfig, AssetPoolConfig
│       ├── factory.py               # create_backtest_engine(), create_strategy()
│       ├── engine/
│       │   ├── cross_sectional.py   # CrossSectionalEngine（含动态资产池）
│       │   └── time_series.py       # TimeSeriesEngine
│       ├── strategies/
│       │   └── base_strategies.py   # 内置策略
│       ├── portfolio/               # 组合管理、仓位、订单执行
│       ├── asset_selector.py        # 回测内资产筛选器
│       ├── trading_log.py           # 交易日志
│       └── visualization.py         # 可视化报告
```

---

## CLI 命令一览

入口点：`main.py` → `app = typer.Typer()`

```bash
# 初始化 & 测试
main init                 # 生成 YAML 配置模板
main test                 # 运行 pytest

# 数据管理
main data download        # 下载 tick 数据
main data list            # 列出已有数据
main data info <SYMBOL>   # 查看某交易对数据详情

# 单步操作
main aggregate <SYMBOL>   # 单个交易对 tick → bar
main features <FILE>      # 单个文件计算特征
main backtest             # 独立回测

# 全流程 Pipeline（推荐）
main pipeline run         # 资产池 → 聚合 → 特征 → 回测 → 报告

# API 服务
main serve                # 启动生产 API 服务
main dev                  # 启动开发模式 API
```

---

## 流程一：数据下载 (data download)

### 命令

```bash
main data download --exchange binance_futures --start-date 2020-01 --end-date 2025-12 --threads 8
```

### 完整调用链

```
main data download
│
├─ app/data_cmd.py :: download()
│   接收 CLI 参数: exchange, symbols, start_date, end_date, threads
│
├─ services/tick_data_scraper/tick_worker.py :: run_download()
│   │  加载配置: settings.downloader_cfg.get_merged_config(exchange_name)
│   │  ┌──────────────────────────────────────────┐
│   │  │ common/config/config_settings.py          │
│   │  │   settings.downloader_cfg                 │
│   │  │     → MultiExchangeDownloadConfig         │
│   │  │     → get_merged_config("binance_futures") │
│   │  │       合并 base config + exchange config   │
│   │  └──────────────────────────────────────────┘
│   │
│   ├─ downloader.py :: DownloadContext.__init__(config, start_date, end_date, symbols)
│   │   创建交易所适配器:
│   │   └─ exchange_factory.py :: ExchangeFactory.create_adapter("binance_futures", config)
│   │       └─ binance_futures.py :: BinanceFuturesAdapter(config)
│   │
│   └─ downloader.py :: FileDownloader.run_download_pipeline()
│       │
│       ├─ [获取交易对列表]
│       │   adapter.get_all_symbols()
│       │   → 调用 Binance API: GET https://fapi.binance.com/fapi/v1/exchangeInfo
│       │   → 过滤 USDT-M 永续合约，返回 ~540 个交易对
│       │
│       ├─ [生成下载任务] _generate_download_tasks(symbols, start_date, end_date)
│       │   遍历 symbols × [start_date ... end_date] 月份
│       │   ├─ 文件已存在 → _is_valid_zip() 校验 ZIP 完整性
│       │   │   ├─ 完整 → 跳过
│       │   │   └─ 损坏/截断 → 删除并重新加入队列（断点恢复保护）
│       │   └─ 文件不存在 → 加入下载队列
│       │
│       ├─ [Phase 1: 并行下载] ThreadPoolExecutor(max_workers)
│       │   每个任务调用 download_file(symbol, year, month):
│       │   ├─ HEAD 请求检查 URL 可用性
│       │   ├─ GET 流式下载 ZIP 文件
│       │   ├─ _verify_download() 校验文件（size + checksum）
│       │   └─ 失败则删除 + 重试（最多 3 次，指数退避）
│       │
│       └─ [Phase 2: 解压转换] ProcessPoolExecutor
│           _extract_and_convert_batch(zip_files):
│           ├─ extract_archive() → 解压 ZIP 到同目录
│           └─ convert_dir_to_parquet() → CSV → Parquet
```

### 产出

```
E:/data/binance_futures/
├── BTCUSDT/
│   ├── BTCUSDT-aggTrades-2020-01.zip       # 原始 ZIP
│   ├── BTCUSDT-aggTrades-2020-01/          # 解压目录
│   │   └── BTCUSDT-aggTrades-2020-01.csv
│   └── *.parquet                            # 转换后的 Parquet
├── ETHUSDT/
│   └── ...
└── ...  (540+ 个交易对)
```

---

## 流程二：全量 Pipeline (pipeline run)

### 命令

```bash
# 静态资产池模式（默认，先选 top100 再处理）
main pipeline run --top-n 100 --threshold 1h --workers 4

# 动态资产池模式（全量资产，回测时按月筛选）
main pipeline run --dynamic-pool --pool-top-n 50 --pool-reselect MS --pool-lookback 30D
```

### 全局调用链概览

```
main pipeline run
│
└─ app/pipeline_cmd.py :: run()       # @pipeline_app.command()
    │
    ├─ Step 1: 资产池选择 / 全量扫描
    ├─ Step 2: Bar 聚合（tick → OHLCV bars）
    ├─ Step 3: 特征计算（技术指标、alpha 因子等）
    ├─ Step 4: 回测执行（CrossSectionalEngine）
    └─ Step 5: 验证 & 报告输出
```

---

### Step 1: 资产池选择

#### 静态模式（`--dynamic-pool` 关闭，默认）

```
pipeline_cmd.py :: step_select_asset_pool(top_n, cache_dir)
│
├─ services/asset_pool/asset_selector.py :: AssetPoolSelector.__init__(config)
│
└─ AssetPoolSelector.get_current_pool(force_refresh=True)
    │
    ├─ _fetch_top_symbols_by_volume(top_n)
    │   GET https://fapi.binance.com/fapi/v1/ticker/24hr
    │   → 过滤 USDT 交易对，排除稳定币
    │   → 按 24h quoteVolume 降序排列
    │   → 取 top N（如 100 个）
    │
    └─ _save_to_cache() → E:/data/asset_pool.json
```

**输出**: `List[str]` 如 `["BTCUSDT", "ETHUSDT", "SOLUSDT", ...]`

#### 动态模式（`--dynamic-pool` 启用）

不调用 API，直接扫描磁盘获取所有有 tick 数据的交易对：

```python
symbols = [d.name for d in tick_dir.iterdir()
           if d.is_dir() and any(d.rglob("*.parquet"))]
```

**输出**: 全量 symbols（如 540 个）。后续由回测引擎在每个月/季度动态筛选。

---

### Step 2: Bar 聚合

```
pipeline_cmd.py :: step_aggregate_bars(symbols, tick_data_dir, bar_output_dir, ...)
│  使用 ProcessPoolExecutor 并行处理多个交易对
│
├─ [per symbol] _aggregate_single_asset(asset, tick_data_dir, bar_output_dir, ...)
│   │
│   ├─ 读取该交易对所有 Parquet tick 文件
│   │   tick_dir/{symbol}/*.parquet → pd.concat → tick_data DataFrame
│   │   列: agg_trade_id, price, quantity, transact_time, is_buyer_maker
│   │
│   ├─ 列重命名: transact_time → timestamp, agg_trade_id → agg_trade_id
│   │
│   └─ services/bar_aggregator/unified.py :: aggregate_bars(
│       │   tick_data,
│       │   bar_type="time_bar",
│       │   threshold="1h",        # 1 小时 K 线
│       │   use_numba=True,        # Numba 加速
│       │   include_advanced=True,  # 包含高级特征列
│       │   symbol="BTCUSDT"
│       │ )
│       │
│       ├─ [Numba 路径] fast_aggregator.py :: FastBarAggregator.aggregate()
│       │   高性能 Numba JIT 编译聚合
│       │
│       └─ [Fallback] bar_types.py :: TimeBarBuilder.build()
│           Pandas 实现的时间 bar 聚合
│
└─ 输出 Parquet: bar_output_dir/{symbol}/{symbol}_time_bar_1h.parquet
```

**Bar 数据列（include_advanced=True）**:

| 列名 | 说明 |
|------|------|
| `start_time`, `end_time` | Bar 起止时间 |
| `open`, `high`, `low`, `close` | OHLC 价格 |
| `volume` | 成交量 |
| `buy_volume`, `sell_volume` | 主买/主卖量 |
| `vwap` | 成交量加权均价 |
| `dollar_volume` | 成交额 (price × quantity) |
| `tick_count` | 成交笔数 |
| `up_move_ratio` | 上涨比例 |
| `medium_price` | 中位价 |
| `reversals` | 价格反转次数 |

---

### Step 3: 特征计算

```
pipeline_cmd.py :: step_calculate_features(bar_paths, feature_windows=[5,10,20,60])
│
├─ 创建 UnifiedFeatureConfig:
│   windows=[5,10,20,60], include_returns=True, include_volatility=True,
│   include_momentum=True, include_volume=True, include_microstructure=True,
│   include_alphas=False, include_technical=False
│
├─ 创建 UnifiedFeatureCalculator(config)
│
├─ [per asset] calculator.calculate(bars_df, asset="BTCUSDT")
│   │
│   │  services/feature/unified_features.py :: UnifiedFeatureCalculator.calculate()
│   │
│   ├─ _calc_returns()
│   │   return_1, log_return_1, return_{w}, log_return_{w}  (w ∈ windows)
│   │
│   ├─ _calc_volatility()
│   │   volatility_{w}, volatility_ann_{w}, parkinson_{w}
│   │
│   ├─ _calc_momentum()
│   │   momentum_{w}, sma_{w}, ema_{w}, price_sma_{w}, bb_pos_{w}, rsi_{w}
│   │
│   ├─ _calc_volume_features()
│   │   volume_sma_{w}, volume_ratio_{w}, dollar_vol_sma_{w},
│   │   buy_ratio, imbalance
│   │
│   └─ _calc_microstructure()
│       up_ratio_sma_{w}, reversal_rate_{w}, amihud_{w},
│       vwap_deviation, vwap_median_dev
│
├─ 为每个 asset 的 DataFrame 添加 asset 列，concat 合并
│
└─ 设置 MultiIndex: (start_time, asset)
   排序后返回
```

**输出**: `pd.DataFrame` with `MultiIndex(timestamp, asset)`

```
                              open    high   ...  return_20  dollar_volume  volatility_20
(2023-01-01 00:00, BTCUSDT)  16500   16550  ...    0.032     1.2e9          0.018
(2023-01-01 00:00, ETHUSDT)  1200    1215   ...    0.045     5.6e8          0.025
(2023-01-01 01:00, BTCUSDT)  16520   16580  ...    0.028     1.1e9          0.017
...
```

---

### Step 4: 回测执行

```
pipeline_cmd.py :: step_run_backtest(
│   data=featured_data,
│   initial_capital=1_000_000,
│   top_n_long=10, top_n_short=10,
│   rebalance_frequency="W-MON",
│   ranking_factor="return_20",
│   dynamic_pool=True,              # 动态资产池开关
│   pool_top_n=50,
│   pool_reselect_frequency="MS",   # 月初刷新
│   pool_lookback_period="30D"      # 回看 30 天
│ )
│
├─ 创建 AssetPoolConfig(enabled=True, reselect_frequency="MS", ...)
│
├─ 创建 BacktestConfig(
│       mode=CROSS_SECTIONAL,
│       rebalance_frequency="W-MON",
│       asset_pool_config=AssetPoolConfig(...),
│       risk_config=RiskConfigModel(max_position_size=0.1, max_drawdown=0.3),
│       cost_config=CostConfigModel(taker_rate=0.0005, slippage_rate=0.0003),
│   )
│
├─ factory.py :: create_strategy("momentum", lookback_col="return_20",
│                                 top_n_long=10, top_n_short=10)
│   → MomentumStrategy 实例
│
├─ factory.py :: create_backtest_engine(config, strategy, logger)
│   → CrossSectionalEngine 实例
│
└─ CrossSectionalEngine.run(featured_data)
    │
    ├─ _prepare_data(data)
    │   确保 MultiIndex (timestamp, asset) 格式
    │
    ├─ self._full_data = data   # 保存完整数据，供资产池回看使用
    │
    ├─ _generate_rebalance_dates(timestamps)
    │   按 "W-MON" 生成 rebalance 日期序列
    │
    └─ [主循环] for timestamp in timestamps:
        │
        ├─ cross_section = data.loc[timestamp]
        │   获取当前时间截面（所有资产该时刻的数据行）
        │
        ├─ portfolio.update_prices(prices)
        ├─ _record_nav(timestamp)
        │
        └─ if _should_rebalance(timestamp):
            │
            └─ _rebalance(timestamp, cross_section, prices)
                │
                ├─ _select_assets(cross_section, timestamp)
                │   │
                │   ├─ 应用 exclude/include 列表
                │   │
                │   └─ [if asset_pool_config.enabled]
                │       │
                │       └─ _get_or_refresh_pool(candidates, timestamp, pool_config)
                │           │
                │           ├─ 检查是否需要刷新（上次刷新距今 >= reselect_frequency）
                │           │   例: "MS"=月初刷新, "QS"=季初刷新
                │           │
                │           ├─ [需要刷新]
                │           │   _select_by_rolling_volume(candidates, timestamp, pool_config)
                │           │   │
                │           │   ├─ 计算回看窗口: [timestamp - 30D, timestamp]
                │           │   ├─ 从 self._full_data 切出该窗口的数据
                │           │   ├─ 按 asset 分组，计算 mean(dollar_volume)
                │           │   ├─ 降序排列，取 top N (如 50)
                │           │   └─ 返回 List[str]，缓存结果
                │           │
                │           └─ [无需刷新]
                │               返回缓存的 _cached_pool_assets
                │
                ├─ active_data = cross_section.loc[active_assets]
                │   只保留当前资产池内的资产
                │
                ├─ MomentumStrategy.generate_weights(active_data, current_weights)
                │   │
                │   ├─ 按 ranking_factor（如 return_20）降序排列
                │   ├─ top 10 → long（每个权重 = 0.5 / 10 = 5%）
                │   ├─ bottom 10 → short（每个权重 = -5%）
                │   └─ 返回 Dict[str, float] 目标权重
                │
                ├─ _apply_risk_limits(target_weights)
                │   限制单个仓位 ≤ max_position_size, 总多头/空头敞口
                │
                └─ portfolio.rebalance_to_weights(target_weights, prices, timestamp)
                    执行调仓，记录交易
```

---

### Step 5: 验证 & 报告

```
pipeline_cmd.py :: step_validate_and_report(backtest_output, output_dir)
│
├─ 提取 nav_history, trades
│
├─ 计算关键指标:
│   ├─ Total Return
│   ├─ Max Drawdown
│   ├─ Annualized Volatility
│   ├─ Annualized Return
│   ├─ Sharpe Ratio
│   ├─ Win Rate, Profit Factor
│   └─ 逐笔交易统计
│
├─ 打印结果到终端
│
└─ 保存文件:
    ├─ output_dir/nav_history.csv      # NAV 时间序列
    ├─ output_dir/trade_log.json       # 所有交易记录
    └─ output_dir/backtest_summary.json # 汇总指标
```

---

## 动态资产池机制详解

### 问题背景

静态模式下，资产池在 pipeline 开始时用 Binance 当前 24h 成交量快照选定，之后不再变化。这意味着：
- 2020 年的回测仍然使用 2026 年的热门币种（存活偏差）
- 中途退市/新上市的币种无法动态纳入/剔除

### 动态模式工作原理

```
时间轴:
2020-01 ──── 2020-02 ──── 2020-03 ──── ... ──── 2025-12

  ↓ 月初       ↓ 月初       ↓ 月初
  刷新池       刷新池       刷新池       每月重新按 dollar_volume 排名

  ↓W-MON      ↓W-MON      ↓W-MON
  rebalance   rebalance   rebalance    每周 rebalance 只在当前池内选股
```

**关键参数**:

| 参数 | CLI Flag | 默认值 | 说明 |
|------|----------|--------|------|
| `pool_top_n` | `--pool-top-n` | 100 | 每期保留多少个资产 |
| `pool_reselect_frequency` | `--pool-reselect` | `MS` | 池刷新频率（`MS`=月初, `QS`=季初, `D`=每天） |
| `pool_lookback_period` | `--pool-lookback` | `30D` | 计算平均成交额的回看窗口 |

**缓存机制**: 资产池不会在每次 rebalance（如每周一）都重新计算。`_get_or_refresh_pool()` 检查上次刷新时间，只有当距今 >= `reselect_frequency` 时才重新计算，其余 rebalance 复用缓存。

---

## 配置系统

### 配置加载优先级

```
环境变量 (.env)  >  YAML 配置文件  >  代码默认值
```

### 关键配置类

| 类 | 文件 | 说明 |
|----|------|------|
| `Settings` | `common/config/config_settings.py` | 全局单例，懒加载 |
| `MultiExchangeDownloadConfig` | `common/config/downloader_config.py` | 下载配置（网络、重试、多交易所） |
| `BacktestConfig` | `services/back_test/config.py` | 回测配置（模式、资金、调仓频率等） |
| `AssetPoolConfig` | `services/back_test/config.py` | 动态资产池配置 |
| `RiskConfigModel` | `services/back_test/config.py` | 风控配置（最大仓位、最大回撤等） |
| `CostConfigModel` | `services/back_test/config.py` | 交易成本配置（手续费、滑点等） |
| `UnifiedFeatureConfig` | `services/feature/unified_features.py` | 特征计算配置 |

### 支持的交易所

| 名称 | 适配器 | 数据目录（默认） |
|------|--------|-----------------|
| `binance` | `BinanceAdapter` | `{PROJECT_ROOT}/data/binance` |
| `binance_futures` | `BinanceFuturesAdapter` | `E:/data/binance_futures` |
| `okx` | `OKXAdapter` | `{PROJECT_ROOT}/data/okx` |
| `bybit` | （仅配置，无适配器） | `{PROJECT_ROOT}/data/bybit` |

---

## 数据目录结构

```
E:/data/
├── binance_futures/                          # tick 数据根目录
│   ├── BTCUSDT/
│   │   ├── BTCUSDT-aggTrades-2020-01.zip     # 原始下载
│   │   ├── BTCUSDT-aggTrades-2020-01/        # 解压目录
│   │   │   └── *.csv
│   │   └── *.parquet                          # 转换后
│   ├── ETHUSDT/
│   └── ...
│
├── asset_pool.json                           # 资产池缓存
│
├── bar_data/                                 # 聚合后的 bar 数据
│   ├── BTCUSDT/
│   │   └── BTCUSDT_time_bar_1h.parquet
│   ├── ETHUSDT/
│   └── ...
│
└── backtest_results/                         # 回测输出
    ├── nav_history.csv
    ├── trade_log.json
    └── backtest_summary.json
```

---

## 快速上手示例

```bash
# 1. 下载 Binance Futures 2020~2025 的 tick 数据
main data download --start-date 2020-01 --end-date 2025-12 --threads 8

# 2. 运行完整 pipeline（静态资产池，top 30）
main pipeline run --top-n 30 --threshold 1h --workers 4 --long-n 5 --short-n 5

# 3. 运行完整 pipeline（动态月度资产池，top 50）
main pipeline run \
  --dynamic-pool \
  --pool-top-n 50 \
  --pool-reselect MS \
  --pool-lookback 30D \
  --threshold 1h \
  --workers 4 \
  --long-n 10 \
  --short-n 10 \
  --rebalance W-MON \
  --factor return_20

# 4. 跳过聚合（已有 bar 数据时）
main pipeline run --skip-aggregation --dynamic-pool --pool-top-n 50
```
