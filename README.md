# Crypto Data Engine

> 端到端加密货币量化交易研究平台 — 从 Tick 数据采集到多因子策略回测

一个覆盖**数据采集、信息驱动Bar聚合、微结构因子工程、多空组合回测、Walk-Forward验证**全流程的量化研究系统。基于 488 个加密货币交易对、6 年日频数据，系统化验证了反转 + 动量复合策略的有效性。

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![LOC](https://img.shields.io/badge/Python_LOC-69K+-yellow.svg)](./)
[![Research](https://img.shields.io/badge/Research_Experiments-120+-purple.svg)](./scripts/)
[![Tests](https://img.shields.io/badge/Tests-21_files-success.svg)](./tests/)

---

## 项目亮点

- 🏗️ **完整的量化研究流水线**: Tick采集 → Dollar Bar聚合 → 微结构特征 → 因子构建 → 回测 → Walk-Forward验证
- ⚡ **高性能数据处理**: Numba JIT 加速 Bar 聚合、ProcessPool 多进程并行、StreamingAggregator 流式处理
- 🔬 **120+ 策略实验**: 系统化因子测试（动量、反转、微结构、订单流），完整记录探索过程
- 📊 **严格的统计验证**: IS/OOS 样本分割、8折Walk-Forward、费率敏感性分析、换手率优化
- 🎯 **最终策略**: 50/50 反转+动量复合策略，Walk-Forward 8折**100%正折**，OOS Sharpe 3.44

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CRYPTO DATA ENGINE                                │
│                                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ 1. DATA      │───▶│ 2. BAR       │───▶│ 3. FEATURE   │───▶│ 4. BACKTEST   │  │
│  │  COLLECTION  │    │  AGGREGATION │    │  ENGINEERING │    │  ENGINE       │  │
│  └─────────────┘    └──────────────┘    └──────────────┘    └───────────────┘  │
│                                                                                 │
│  Binance/OKX/Bybit   Dollar/Volume/     9 Microstructure    Cross-Sectional    │
│  aggTrades → Parquet  Tick/Time Bar      + Rolling Factors   Time-Series        │
│  488 symbols          Numba JIT          VPIN/Kyle/OFI       Walk-Forward       │
│  Multi-threaded       Streaming Mode     Whale/Burst/Jump    Funding Rate       │
│                                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ 5. SIGNAL    │───▶│ 6. PORTFOLIO │───▶│ 7. API       │───▶│ 8. FRONTEND   │  │
│  │  GENERATION  │    │  MANAGEMENT  │    │  SERVER      │    │  DASHBOARD    │  │
│  └─────────────┘    └──────────────┘    └──────────────┘    └───────────────┘  │
│                                                                                 │
│  Factor/Rule/        Position Sizing    FastAPI + Redis      React + ECharts    │
│  Ensemble/Hybrid     Order Execution    Background Tasks     NAV/DD/Monthly     │
│  Regime Sizing       Risk Management    REST API             Trade Logs         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 代码规模

| 模块 | 文件数 | 代码行数 | 说明 |
|------|:------:|:--------:|------|
| `src/` 核心引擎 | 68 | ~22,000 | 数据采集、聚合、特征、回测、API |
| `scripts/` 策略研究 | 120 | ~41,500 | 因子测试、参数扫描、Walk-Forward |
| `tests/` 测试套件 | 21 | ~5,800 | 单元测试、集成测试、无前视偏差验证 |
| `frontend/` 可视化 | 6 | ~500 | React Dashboard |
| **总计** | **215** | **~69,800** | |

研究产出: 93 份 CSV 数据表 + 124 张分析图表

---

## 核心模块详解

### 1. 多交易所 Tick 数据采集

```
src/crypto_data_engine/services/tick_data_scraper/
├── downloader/
│   ├── binance.py / binance_futures.py    # Binance 现货/合约适配器
│   ├── okx.py                             # OKX 适配器
│   └── exchange_factory.py                # 交易所工厂模式
├── extractor/convert.py                   # ZIP → Parquet 转换
└── tick_worker.py                         # Redis队列驱动的下载流水线
```

- **多线程并行下载** (ThreadPoolExecutor, max_workers=8)
- **Redis 队列** 驱动下载→转换流水线，支持断点续传
- **进度追踪**: `get_pipeline_progress(job_id)` 实时查询
- **数据格式**: aggTrades → Parquet (PyArrow, 支持 mmap)
- **数据量**: 488 个 USDT 永续合约，2020-02 ~ 2025-12

```bash
poetry run main data download --exchange binance_futures --start-date 2024-01 --end-date 2024-06
```

### 2. 信息驱动 Bar 聚合

```
src/crypto_data_engine/services/bar_aggregator/
├── unified.py              # 统一入口: aggregate_bars(), create_streaming_aggregator()
├── fast_aggregator.py      # Numba JIT 加速聚合 (>10K行自动启用)
├── dollar_profile.py       # 自适应 Dollar Bar 阈值计算
├── bar_types.py            # BarType 枚举 + Builder 模式
├── batch_aggregator.py     # ProcessPool 多进程批量聚合
└── tick_feature_enricher.py # 9维 Tick 微结构特征注入
```

**支持 4 种 Bar 类型:**

| Bar 类型 | 采样依据 | 优势 |
|----------|----------|------|
| **Dollar Bar** | 固定美元成交额 | 信息均匀采样，消除流动性偏差 |
| Volume Bar | 固定成交量 | 标准化交易活动 |
| Tick Bar | 固定笔数 | 均匀交易频率 |
| Time Bar | 固定时间间隔 | 传统方法 |

**关键实现:**
- **自适应阈值**: `dollar_profile.py` 根据过去 10 天日均成交额自动计算，目标 ~50 bars/天
- **Numba JIT**: `fast_aggregator.py` 对 volume/dollar bar 核心循环 JIT 编译，10x 加速
- **流式聚合**: `StreamingAggregator.process_chunk()` 支持逐块处理，内存占用恒定
- **9 维微结构特征** 随 Bar 生成同步计算:

```
VPIN (体积同步概率)  │  Kyle's Lambda (市场冲击)  │  Burstiness (到达聚集度)
Toxicity (毒性订单流)  │  Jump Ratio (跳跃频率)     │  Whale Imbalance (大单方向)
Whale Impact (大单冲击) │  Toxicity Run (连续毒性)   │  Toxicity Max (峰值毒性)
```

### 3. 多层因子工程

```
src/crypto_data_engine/services/feature/
├── unified_features.py              # 26+ 统一因子计算器
├── order_flow_factors.py            # 订单流因子 (OFI, Centroid, LargeTrade)
├── tick_microstructure_factors.py   # Tick 微结构因子
└── Factor.py                        # 基础因子框架
```

**因子体系 (3 层):**

| 层级 | 因子类别 | 示例 |
|------|----------|------|
| **L1 原始特征** | 价量、微结构 | ret_2h, vpin_24h, ofi_2h, whale_imbalance |
| **L2 衍生因子** | 滚动统计、交互 | zscore(ret_2h) × inv_vpin, flow_3 = EqW(ofi, centroid, largeTrade) |
| **L3 组合信号** | 多因子融合 | rev_x_inv_vpin + RegimeSizing(confirm_score) |

**Z-score 标准化方法:**
- `persym`: 每标的滚动 30 天 z-score (消除个体均值差异)
- `xsect`: 跨截面 z-score (排名选股)
- `hybrid`: 先 persym 再 xsect (双重标准化)

### 4. 回测引擎

```
src/crypto_data_engine/services/back_test/
├── engine/
│   ├── base_engine.py          # BaseBacktestEngine (NAV追踪, 交易记录, 绩效计算)
│   ├── cross_sectional.py      # 横截面引擎: 固定频率调仓 (日/周/月)
│   └── time_series.py          # 时间序列引擎: 逐Bar决策
├── portfolio/
│   ├── portfolio.py            # 多空组合管理
│   ├── position.py             # 仓位建模
│   └── order_executor.py       # 订单执行 + 滑点模拟
├── strategies/
│   └── base_strategies.py      # 策略基类 (generate_signal / generate_weights)
├── trading_log.py              # 全链路交易日志
├── walk_forward.py             # Walk-Forward 验证框架
└── visualization.py            # 绩效可视化
```

**回测特性:**
- **无前视偏差**: 信号在 T 日 EOD 计算，T+1 日生效，PnL 在 T+1 日结算
- **真实成本模型**: Maker/Taker 费率 + 滑点 + Funding Rate (3次/天结算)
- **换手率追踪**: 按日/按调仓记录 Turnover，精确计算费用消耗
- **Buffer Zone**: 减少不必要换手 — 持仓在 Top(n_ls + buffer) 内则保留
- **Regime Sizing**: 确认分数 < 0 时自动缩放仓位

### 5. 信号生成框架

```
src/crypto_data_engine/services/signal_generation/
├── base.py                    # 信号生成器基类
├── factor_signal.py           # 因子驱动信号 (横截面排名)
├── rule_signal.py             # 规则驱动信号 (阈值/趋势)
├── order_flow_strategy.py     # 订单流策略 (OFI/VPIN)
├── hybrid_strategy.py         # 混合策略 (多信号融合)
└── ensemble.py                # 集成信号 (加权投票)
```

### 6. API 服务 + 前端

```
src/crypto_data_engine/api/
├── main.py          # FastAPI App Factory + Router 注册
├── routers/         # RESTful 端点 (download, backtest, aggregation, feature, strategy, visualization)
├── schemas/         # Pydantic 请求/响应模型
└── storage.py       # 任务状态持久化

frontend/            # React 18 + TypeScript + Ant Design + ECharts
```

- **异步任务执行**: ThreadPoolExecutor (16线程 I/O) + ProcessPoolExecutor (8进程 CPU)
- **Redis 任务状态**: 实时进度追踪、结果缓存
- **可视化**: NAV曲线、回撤图、月度热力图、交易散点图

---

## 策略研究成果

### 研究路径 (120+ 实验)

```
Phase 1: 因子探索
├── 单因子IC测试 (26个因子)
├── 动量因子筛选 (OFI, Flow3, 趋势因子)
├── 反转因子构建 (ret_2h/4h/12h × VPIN交互)
└── 微结构因子评估 (VPIN, Kyle, Whale, Burst)

Phase 2: 参数优化
├── 信号窗口 × 持仓期匹配 (7×4=28组合)
├── Z-score方法对比 (persym vs xsect vs hybrid)
├── 资产池筛选 (Top20/50/100, 流动性过滤)
└── 换手率优化 (Buffer Zone + MinTrade, 15种控制)

Phase 3: 组合构建
├── 双层策略 (信号层 + 确认层, 6个变种)
├── 多因子复合 (反转 × 动量, 4种配比)
├── 费率压力测试 (5/10/15/20/30 bps)
└── Regime Sizing (市场状态自适应)

Phase 4: 验证与报告
├── IS/OOS 样本分割 (2020-2023 / 2024-2025)
├── Walk-Forward 8折验证 (180d训练 / 90d测试)
├── 过拟合诊断 (Decay Ratio, 正折比例)
└── 实盘可行性评估 (容量、滑点、Maker执行)
```

### 最终策略: 50/50 Reversal + Momentum

| 组件 | 配置 |
|------|------|
| **反转腿 (50%)** | 2h收益率反转 × VPIN交互, persym zscore, R1日频, 10LS, Buffer=10, MinTrade=2%, RegimeSz |
| **动量腿 (50%)** | EqW(ofi_14d R14 + flow_3 R7), Top50池, xsect zscore, 含Funding Rate |

**回测结果 (@10bps, 488标的, 2020-2025):**

| 指标 | 全周期 | IS (2020-23) | OOS (2024-25) |
|------|:------:|:------------:|:-------------:|
| Sharpe | 2.76 | 2.41 | 3.44 |
| 年化收益 | 58.4% | 47.7% | 80.7% |
| 最大回撤 | -17.2% | -17.2% | -12.9% |
| 月胜率 | 77.8% | 70.2% | 92.0% |
| Calmar | 3.40 | 2.77 | 6.26 |

**Walk-Forward 验证 (8折):**
- 测试 Sharpe 均值: **+5.08** | 中位数: +4.90
- **100% 正折** (8/8 折测试期均盈利)
- 腿间相关性: ρ ≈ 0.05 (几乎不相关 → 有效分散)

> ⚠️ 保守预期: 扣除多重测试偏差和市场regime因素后，实盘预期 Sharpe **1.0-1.5**, 年化 **15-30%**

---

## 快速开始

### 后端

```bash
# 安装依赖
poetry install

# 启动 API 服务 (开发模式, 自动重载)
poetry run main dev

# 或生产模式
poetry run main serve --host 127.0.0.1 --port 8000

# 下载 Tick 数据
poetry run main data download --start-date 2025-01 --end-date 2025-06

# 聚合 Dollar Bar
poetry run main aggregate BTCUSDT --bar-type dollar_bar

# 运行回测
poetry run main backtest --strategy momentum --mode cross_sectional

# 运行测试
poetry run main test
poetry run main test --coverage
```

### 前端

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
npm run build     # 生产构建
```

### Docker 部署

```bash
cd deploy
docker-compose up -d   # 启动 Redis + API Server
```

---

## 目录结构

```
crypto-data-engine/
│
├── src/crypto_data_engine/           # 核心引擎 (68 files, ~22K LOC)
│   ├── main.py                       # Typer CLI 入口
│   ├── app/                          # CLI 命令模块
│   │   ├── server.py                 #   API 服务启动
│   │   ├── data_cmd.py               #   数据下载命令
│   │   ├── aggregate_cmd.py          #   Bar 聚合命令
│   │   ├── backtest_cmd.py           #   回测命令
│   │   └── pipeline_cmd.py           #   全流程编排
│   ├── api/                          # FastAPI 应用
│   │   ├── main.py                   #   App Factory + 路由注册
│   │   ├── routers/                  #   6 个路由模块
│   │   └── schemas/                  #   Pydantic 模型
│   ├── core/                         # 基类与协议
│   │   ├── base.py                   #   TradeRecord, BacktestResult, BaseStrategy
│   │   └── interfaces.py             #   IBacktestEngine 协议
│   ├── services/
│   │   ├── tick_data_scraper/        #   Tick 数据采集 (多交易所)
│   │   ├── bar_aggregator/           #   Bar 聚合 (Dollar/Volume/Tick/Time)
│   │   ├── feature/                  #   因子工程 (26+ 因子)
│   │   ├── signal_generation/        #   信号生成 (Factor/Rule/Ensemble/Hybrid)
│   │   ├── back_test/                #   回测引擎 (XS/TS + Portfolio)
│   │   ├── asset_pool/               #   动态资产池管理
│   │   └── funding_rate/             #   Funding Rate 加载
│   └── common/
│       ├── config/                   #   配置管理 (Pydantic Settings + YAML)
│       ├── logger/                   #   Loguru 日志
│       └── task_manager.py           #   后台任务管理 (Redis-backed)
│
├── scripts/                          # 策略研究实验 (120 files, ~41K LOC)
│   ├── run_exp{1-8}_*.py             #   Phase 1: 基础因子实验
│   ├── run_expM{1-6}_*.py            #   Phase 2: 动量因子深度研究
│   ├── run_expT{1-9}_*.py            #   Phase 3: Tick微结构因子
│   ├── run_phase{2,3}_*.py           #   Phase 4: 多层策略
│   ├── run_turnover_optimization.py  #   换手率优化
│   ├── run_composite_optimized.py    #   最终复合策略回测
│   └── run_walk_forward.py           #   Walk-Forward 验证
│
├── tests/                            # 测试套件 (21 files, ~5.8K LOC)
│   ├── test_cross_sectional_engine.py
│   ├── test_time_series_engine.py
│   ├── test_portfolio.py
│   ├── test_bar_aggregator.py
│   ├── test_tick_features.py
│   ├── test_e2e_backtest.py
│   └── test_cross_sectional_no_lookahead.py   # 前视偏差专项检测
│
├── frontend/                         # React 前端
│   └── src/                          #   Dashboard + 可视化
│
├── data/
│   └── backtest_reports/             # 研究产出 (93 CSV + 124 PNG)
│       ├── composite_optimized/      #   最终策略报告
│       ├── walk_forward/             #   Walk-Forward 验证
│       ├── turnover_optimization/    #   换手率分析
│       └── long_horizon_reversal/    #   长周期反转实验
│
├── deploy/                           # Docker 部署
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── pyproject.toml                    # Poetry 项目配置
```

---

## 技术栈

### 后端

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.12 | 核心开发 |
| Web | FastAPI + Uvicorn | REST API 服务 |
| 数据 | Pandas, NumPy, PyArrow | 数据处理与分析 |
| 加速 | Numba JIT, SciPy | Bar聚合性能优化 |
| 交易所 | ccxt | 多交易所统一接口 |
| 任务 | Redis + 自研 TaskManager | 后台任务调度 |
| 日志 | Loguru | 结构化日志 |
| 配置 | Pydantic Settings | 类型安全配置 |
| CLI | Typer | 命令行工具 |
| 测试 | Pytest + pytest-asyncio | 异步测试支持 |
| 可视化 | Matplotlib | 回测图表 |

### 前端

| 类别 | 技术 |
|------|------|
| 框架 | React 18 + TypeScript |
| UI | Ant Design 5 |
| 图表 | ECharts |
| 构建 | Vite |

---

## 测试

```bash
# 运行全部测试
poetry run main test

# 运行特定测试
poetry run main test --file test_bar_aggregator.py

# 覆盖率报告
poetry run main test --coverage
```

**测试覆盖范围:**
- ✅ Bar 聚合正确性 (Dollar/Volume/Tick Bar)
- ✅ Tick 微结构特征计算
- ✅ 横截面 / 时间序列引擎
- ✅ Portfolio 仓位与订单执行
- ✅ 无前视偏差验证 (专项测试)
- ✅ E2E 集成测试 (数据→策略→回测→结果)
- ✅ API 端点测试
- ✅ 信号生成逻辑
- ✅ Funding Rate 加载

---

## 许可证

MIT License

---

**最后更新**: 2026-02-16
