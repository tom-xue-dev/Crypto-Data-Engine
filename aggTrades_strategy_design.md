# aggTrades 量化交易策略设计方案

> 标的：BTCUSDT · 数据源：Binance aggTrades · 频率：中高频（持仓分钟～小时级）

---

## 1. 数据处理层

### 1.1 聚合方式选择

本策略采用 **三轨并行聚合**，不同聚合方式服务于不同目的：

| 聚合方式 | 参数建议 | 用途 |
|---------|---------|------|
| **Time-based** | 1s / 5s / 1min | 构建标准 OHLCV K线，用于趋势判断和回测对齐 |
| **Volume-based** | 每 N BTC 成交量切一根bar（N 可取 5-20 BTC，需根据近期日均成交量动态校准） | 核心信号bar：在高波动期自动加密采样、低波动期自动降频，天然适配市场节奏 |
| **Tick-based** | 每 200-500 笔 aggTrades 切一根bar | 辅助信号：用于捕捉极端订单流事件 |

**推荐主时间轴：Volume-based bar**，原因如下：
- aggTrades 的逐笔粒度在 time-based 聚合中会被大量浪费（1分钟内可能有几千笔成交被压成一根K线）
- Volume bar 让每根 bar 承载相同的"市场活动量"，统计特征更稳定，信号信噪比更高
- 在剧烈波动时自动产生更多 bar（更高采样率），在横盘时减少噪声

**动态校准方法**：每日收盘后，取过去 7 日日均 BTC 成交量，除以目标 bar 数（如 1500 根/天），得到单根 bar 的 volume 阈值。

### 1.2 特征工程

以下所有特征均可从 aggTrades 的 6 个原始字段直接计算，无需 orderbook 数据。

#### A. 基础 OHLCV（每根 volume bar 内计算）

- `open` / `high` / `low` / `close`：该 bar 内首笔、最高、最低、末笔成交价
- `volume`：总成交量（BTC），volume bar 下该值恒定，但 time bar 下有意义
- `quote_volume`：总成交额（USDT）= Σ(price × quantity)
- `trade_count`：该 bar 内 aggTrade 笔数
- `duration`：该 bar 的时间跨度（毫秒），volume bar 下此值反映市场活跃度——duration 越短说明成交越密集

#### B. 买卖力量分解（核心优势特征，直接利用 is_buyer_maker）

```
buy_volume  = Σ quantity  where is_buyer_maker == False  （主动买入量）
sell_volume = Σ quantity  where is_buyer_maker == True   （主动卖出量）
```

由此衍生：

| 特征名 | 计算公式 | 含义 |
|--------|---------|------|
| **OFI**（Order Flow Imbalance） | (buy_volume - sell_volume) / (buy_volume + sell_volume) | 订单流不平衡度，范围 [-1, 1]，正值表示买方主导 |
| **delta** | buy_volume - sell_volume | 绝对净买入量 |
| **cumulative_delta** | 对 delta 做累积求和（跨 bar 滚动） | 累积净买入，类似 OBV 但基于真实方向 |
| **buy_trade_count / sell_trade_count** | 按 is_buyer_maker 分组计数 | 买卖笔数，大单少笔 vs 小单多笔可区分机构与散户 |
| **avg_buy_size / avg_sell_size** | buy_volume / buy_trade_count | 平均单笔买入/卖出规模 |
| **large_trade_ratio** | Σ quantity(where quantity > Q90) / total_volume | 大单占比（Q90 为近期成交量 90 分位数） |

#### C. VWAP 及其衍生

| 特征名 | 计算公式 | 用途 |
|--------|---------|------|
| **rolling_vwap** | Σ(price × quantity) / Σ(quantity)，滚动窗口 N 根 bar | 动态均价锚点 |
| **vwap_deviation** | (close - rolling_vwap) / rolling_vwap | 价格偏离 VWAP 的程度，用于均值回归信号 |
| **buy_vwap / sell_vwap** | 分别对主动买入/卖出计算 VWAP | 若 buy_vwap > sell_vwap，说明买方愿意在更高价成交——多头信号 |

#### D. 微观结构特征

| 特征名 | 计算公式 | 含义 |
|--------|---------|------|
| **trade_intensity** | trade_count / duration（毫秒） | 成交密度，突然飙升往往预示趋势启动 |
| **price_impact** | (close - open) / total_volume | 单位成交量引起的价格变动，衡量市场流动性/冲击 |
| **tick_direction** | 对每笔 aggTrade，与前一笔比较价格方向（+1/0/-1） | 用于计算 tick rule 统计特征 |
| **uptick_ratio** | bar 内 uptick 笔数 / 总笔数 | 微观价格趋势强度 |
| **realized_volatility** | bar 内逐笔 log return 的标准差 | 微观波动率 |
| **kyle_lambda** | 对滚动窗口做回归 ΔP = λ × signed_volume + ε 中的 λ | Kyle 模型的价格冲击系数，衡量信息不对称程度 |

#### E. 跨周期特征

- 对上述特征在多个滚动窗口（短 = 20 bar，中 = 100 bar，长 = 500 bar）计算 z-score
- 例如：`OFI_zscore_20` = (OFI - mean_20) / std_20
- z-score 标准化让不同市况下的信号具有可比性

---

## 2. 信号生成层

### 信号 A：订单流动量信号（Order Flow Momentum）

**逻辑原理**：当大量资金持续单方向主动成交时（连续多根 bar 的 OFI 偏向同一方向），价格往往会继续沿该方向移动。这本质上是在检测"知情交易者"的持续建仓行为。

**参数**：
- `ofi_lookback`：OFI 的指数移动平均窗口，建议 10-30 根 volume bar
- `delta_lookback`：cumulative_delta 的短期斜率计算窗口，建议 5-15 根 bar
- `intensity_threshold`：trade_intensity z-score 阈值，建议 1.5
- `large_trade_threshold`：large_trade_ratio z-score 阈值，建议 1.0

**触发条件（做多）**：
1. `EMA(OFI, ofi_lookback) > 0.15`（持续买方主导）
2. `cumulative_delta` 在 `delta_lookback` 窗口内斜率 > 0（净买入加速）
3. `trade_intensity_zscore > intensity_threshold`（成交密度异常升高）
4. `large_trade_ratio_zscore > large_trade_threshold`（大单占比上升，机构参与）
5. `avg_buy_size > avg_sell_size × 1.2`（买方平均单量显著大于卖方）

以上条件 1-2 为必要条件，3-5 中满足任意 2 条即触发。做空信号完全镜像。

**预期适用市场**：趋势启动阶段和趋势延续阶段。在窄幅震荡期间会产生较多假信号，需要配合信号 C 过滤。

---

### 信号 B：VWAP 均值回归信号（VWAP Mean Reversion）

**逻辑原理**：价格短期大幅偏离 VWAP 后，若订单流动量减弱（OFI 回归中性），大概率会向 VWAP 回归。这捕捉的是过度反应后的修正。

**参数**：
- `vwap_window`：rolling_vwap 计算窗口，建议 100-300 根 volume bar（约对应 1-3 小时）
- `deviation_entry`：vwap_deviation 绝对值的入场阈值，建议 0.15%-0.35%（需根据波动率动态调整）
- `deviation_exit`：vwap_deviation 绝对值的出场阈值，建议 0.03%-0.05%
- `ofi_cooldown`：OFI 回归中性的阈值，`|EMA(OFI, 10)| < 0.08`

**触发条件（做多 = 价格低于 VWAP 后回归）**：
1. `vwap_deviation < -deviation_entry`（价格显著低于 VWAP）
2. `|EMA(OFI, 10)| < 0.08`（订单流不平衡消退，抛压衰竭）
3. `realized_volatility` 在下降（波动率从高点回落，市场从恐慌中平静）
4. `sell_volume` 在最近 5 根 bar 逐步递减（卖方力量耗尽）

出场：`vwap_deviation` 回归到 `deviation_exit` 以内，或触发止损。

**动态阈值调整**：`deviation_entry` 应设为近期（500 bar）`|vwap_deviation|` 的 90 分位数，确保只在真正的极端偏离时触发。

**预期适用市场**：震荡市和趋势末端的回调。在单边暴涨/暴跌中会被"碾压"，此时需要信号 A 的趋势方向作为过滤——若信号 A 处于强趋势状态，信号 B 不应逆势开仓。

---

### 信号 C：波动率体制过滤器（Volatility Regime Filter）

**逻辑原理**：这不是一个直接的交易信号，而是一个 **体制识别器**，决定当前应偏向动量策略（信号 A）还是均值回归策略（信号 B），或者完全空仓。

**参数**：
- `vol_short`：短期波动率窗口，20 根 bar
- `vol_long`：长期波动率窗口，200 根 bar
- `vol_ratio_high`：高波动阈值，建议 1.8
- `vol_ratio_low`：低波动阈值，建议 0.6
- `duration_ma`：volume bar duration 的移动平均窗口，50 根 bar

**体制划分**：

```
vol_ratio = realized_vol_short / realized_vol_long
duration_ratio = current_duration / MA(duration, duration_ma)
```

| 体制 | 条件 | 策略偏好 |
|------|------|---------|
| **高波动扩张期** | vol_ratio > vol_ratio_high 且 duration_ratio < 0.5（bar 产生极快） | 仅允许信号 A（趋势跟随），信号 B 禁止开仓 |
| **正常波动期** | vol_ratio_low ≤ vol_ratio ≤ vol_ratio_high | 信号 A 和信号 B 均可触发，信号 B 权重更高 |
| **低波动收缩期** | vol_ratio < vol_ratio_low 且 duration_ratio > 2.0（bar 产生极慢） | 减半仓位，仅允许信号 B，且放宽 deviation_entry 阈值 |
| **极端波动/流动性危机** | vol_ratio > 3.0 或 price_impact_zscore > 3.0 | 全部空仓，暂停交易 |

**信号组合权重**：

信号 A 和信号 B 可能同时触发（方向一致时叠加信心，方向冲突时取消）：
- 同向触发：仓位可叠加至最大允许值
- 反向触发：不开新仓，已有持仓保持不变
- 信号 C 判定为极端体制：强制平仓或禁止新开仓

---

## 3. 风控与执行层

### 3.1 仓位管理

**基础仓位计算（基于 ATR 的风险预算法）**：

```
atr = 近 100 根 volume bar 的 true range 均值
risk_per_trade = 账户净值 × 1%       （单笔最大风险）
base_position = risk_per_trade / (atr × stop_multiplier)
```

**动态调整系数**：

```
final_position = base_position × regime_factor × signal_strength × kelly_factor
```

| 调整因子 | 说明 |
|---------|------|
| `regime_factor` | 高波动期 = 0.5，正常期 = 1.0，低波动期 = 0.7 |
| `signal_strength` | 基于触发条件满足数量的连续评分 [0.5, 1.5] |
| `kelly_factor` | 基于近期该信号胜率和盈亏比的 half-Kelly 系数，上限 cap 在 1.0 |

**硬性上限**：
- 单笔仓位不超过账户净值的 5%（按当前价格计）
- 同方向总持仓不超过账户净值的 10%
- 新仓位开立时，当日已实现亏损若 > 账户净值 3%，停止交易至次日

### 3.2 止损逻辑

**三层止损机制**（触发任一即平仓）：

| 层级 | 类型 | 规则 | 备注 |
|------|------|------|------|
| L1 | **固定止损** | 入场价 ± ATR × 1.5 | 防范黑天鹅，永远不撤销 |
| L2 | **订单流止损** | 持多仓时：EMA(OFI, 5) < -0.25 持续 3 根 bar | 利用 aggTrades 的方向信息做"智能"止损，比纯价格止损更早发现趋势反转 |
| L3 | **时间止损** | 持仓超过 `max_holding_bars`（建议 300-800 根 volume bar，约 2-6 小时）且未达到止盈 | 避免在震荡中长期消耗 |

**移动止盈**（Trailing Take-Profit）：
- 当浮盈达到 ATR × 1.0 时，启动追踪止盈
- 追踪距离 = ATR × 0.8，从最高浮盈点回撤超过该距离即平仓
- 若 cumulative_delta 持续加速且方向与持仓一致，追踪距离可放宽至 ATR × 1.2（让利润奔跑）

### 3.3 全局风控

| 规则 | 参数 |
|------|------|
| 单日最大亏损 | 账户净值的 3%，触发后当日停止交易 |
| 连续亏损次数 | 连续 5 笔亏损后，仓位减半，连续 8 笔亏损后暂停 4 小时 |
| 滚动最大回撤 | 近 7 日净值回撤 > 8% 时，仓位降至 25%，直到新高或回撤收窄至 4% |
| 流动性保护 | 若 trade_intensity 低于近期均值的 30%（可能是交易所维护或极端行情），暂停交易 |
| 关联市场检查 | 如有条件，监控 ETH/USDT 的 aggTrades OFI 作为辅助验证（可选扩展） |

### 3.4 执行细节

- **下单方式**：优先使用限价单（Limit Order），挂在 best bid/ask 附近（根据 aggTrades 最新价推算）
- **滑点预估**：回测时需模拟滑点。基于历史数据中 `price_impact` 的分布，取 75 分位数作为滑点估计
- **最小利润阈值**：预期盈利若 < 手续费 × 3，则不开仓（Maker 0.02% + Taker 0.04% = 单边 ~0.03%，双边 ~0.06%）

---

## 4. 回测框架设计

### 4.1 整体流程

```
原始数据加载
    │
    ▼
数据清洗（去重、时间排序、异常价格过滤）
    │
    ▼
Volume Bar 聚合
    │
    ▼
特征计算（全部特征向量化批量计算，不可未来函数）
    │
    ▼
信号生成（逐 bar 遍历，模拟实时状态机）
    │
    ▼
模拟执行（含滑点、手续费、资金占用）
    │
    ▼
绩效评估 + 归因分析
    │
    ▼
参数敏感性分析 + 分段稳定性检验
```

### 4.2 关键实现原则

**严格禁止未来信息泄露**：
- 所有特征必须只使用当前 bar 及之前的数据
- 动态阈值（如 deviation_entry 取 90 分位数）只能用过去 N bar 计算，不可全局计算
- volume bar 的 volume 阈值校准，只能用截至前一日的数据

**事件驱动回测引擎**（非向量化回测）：
- 由于信号涉及仓位状态（有仓/无仓）、订单流止损等状态逻辑，建议使用 **事件驱动** 架构
- 每根 bar 到达时：更新特征 → 检查止损/止盈 → 检查开仓信号 → 记录状态

**滑点与手续费模型**：
- 手续费：Maker 0.02%，Taker 0.04%。假设信号 A 用 Taker 入场 Maker 出场，信号 B 全 Limit（Maker + Maker）
- 滑点：固定 0.01% + 变动部分（基于该 bar 的 price_impact × position_size）
- 资金利率（如使用永续合约）：每 8 小时按历史 funding rate 扣减

### 4.3 评估指标体系

**核心指标**（必须报告）：

| 指标 | 说明 | 目标参考值 |
|------|------|-----------|
| 年化收益率 | - | > 30%（中高频预期） |
| 夏普比率 (Sharpe) | 用逐日收益计算，无风险利率取 0 | > 2.0 |
| Sortino 比率 | 只惩罚下行波动 | > 2.5 |
| 最大回撤 (MDD) | 净值曲线的峰到谷最大降幅 | < 10% |
| 最大回撤持续天数 | 从回撤开始到恢复新高的最长天数 | < 15 天 |
| Calmar 比率 | 年化收益 / 最大回撤 | > 3.0 |
| 胜率 | 盈利交易笔数 / 总交易笔数 | > 45% |
| 盈亏比 | 平均盈利 / 平均亏损 | > 1.5 |
| 交易频率 | 日均交易笔数 | 5-30 笔/天 |

**辅助指标**：

| 指标 | 说明 |
|------|------|
| 信号 A 单独绩效 | 分拆评估，确认每个信号独立有效 |
| 信号 B 单独绩效 | 同上 |
| 滑点敏感性 | 将滑点放大 2x / 3x 后策略是否仍盈利 |
| 分时段绩效 | 按 UTC 小时分组，观察是否某些时段贡献全部利润（若是，则策略可能只需在特定时段运行） |
| 月度收益分布 | 确认盈利不集中在极少数月份 |
| Profit Factor | 总盈利 / 总亏损 | 

### 4.4 防过拟合措施

**训练/验证/测试集划分**：
- 建议数据量：至少 6 个月 aggTrades（BTCUSDT 日均约 200-400 万笔）
- 划分：60% 训练 / 20% 验证 / 20% 测试，**按时间顺序**，绝不随机打乱
- 测试集只在最终评估时使用一次

**Walk-Forward 滚动优化**：
- 训练窗口：30 天，验证窗口：7 天，步进：7 天
- 每步在训练窗口上选参数，在验证窗口上记录绩效
- 最终报告所有验证窗口拼接后的综合绩效

**参数稳定性检验**：
- 对每个参数在 ±30% 范围内扰动，观察夏普比率变化
- 若夏普比率在扰动范围内波动 > 50%，该参数可能过拟合
- 优先选择"平台型"参数（在一个较宽范围内绩效稳定的参数值）

**多市场/多币对验证**（可选但强烈推荐）：
- 将策略无修改地应用到 ETHUSDT、SOLUSDT 等其他高流动性币对
- 若策略在多个币对上均盈利（即使绩效不同），过拟合风险大大降低

---

## 5. 项目文件结构

```
aggtrades_strategy/
│
├── config/
│   ├── settings.yaml              # 全局参数配置（交易对、API密钥路径、运行模式）
│   └── strategy_params.yaml       # 策略参数（信号阈值、窗口长度、风控参数）
│
├── data/
│   ├── collector.py               # 通过 Binance REST/WebSocket 采集 aggTrades 原始数据
│   ├── storage.py                 # 原始数据落盘（Parquet 格式），按日期分片存储
│   └── cleaner.py                 # 数据清洗：去重、排序、异常值过滤、缺失时段标记
│
├── features/
│   ├── bar_builder.py             # 将 aggTrades 聚合为 volume bar / time bar / tick bar
│   ├── ofi_features.py            # 买卖力量分解：OFI、delta、cumulative delta、大单占比
│   ├── vwap_features.py           # VWAP、vwap_deviation、buy_vwap vs sell_vwap
│   ├── microstructure.py          # trade_intensity、price_impact、kyle_lambda、realized_vol
│   └── feature_pipeline.py        # 编排所有特征计算流程，输出完整特征 DataFrame
│
├── signals/
│   ├── signal_a_orderflow.py      # 信号A：订单流动量信号的生成逻辑
│   ├── signal_b_vwap_revert.py    # 信号B：VWAP均值回归信号的生成逻辑
│   ├── signal_c_regime.py         # 信号C：波动率体制过滤器
│   └── signal_combiner.py         # 信号组合、冲突消解、最终交易决策输出
│
├── execution/
│   ├── position_sizer.py          # 仓位计算：ATR风险预算、动态调整系数、Kelly系数
│   ├── risk_manager.py            # 三层止损、追踪止盈、全局风控（日亏损限制、回撤控制）
│   └── order_executor.py          # 下单执行：限价单管理、超时撤单重挂、实盘下单接口
│
├── backtest/
│   ├── engine.py                  # 事件驱动回测引擎核心：逐bar推进、状态机管理
│   ├── simulator.py               # 成交模拟器：滑点模型、手续费模型、资金利率
│   ├── metrics.py                 # 绩效指标计算：Sharpe、MDD、胜率、盈亏比、Calmar等
│   ├── analyzer.py                # 归因分析：分信号绩效、分时段绩效、参数敏感性报告
│   └── walk_forward.py            # Walk-Forward 滚动优化与防过拟合检验
│
├── live/
│   ├── stream_handler.py          # 实时 WebSocket 数据流接入与心跳管理
│   ├── live_engine.py             # 实盘运行主循环：数据→特征→信号→执行的实时编排
│   └── monitor.py                 # 运行监控：仓位状态、PnL推送、异常报警（如断线、延迟）
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # 数据探索：aggTrades 分布特征、日内模式可视化
│   ├── 02_feature_analysis.ipynb  # 特征有效性验证：IC值、分组回测
│   └── 03_backtest_report.ipynb   # 回测结果可视化与最终报告生成
│
├── tests/
│   ├── test_bar_builder.py        # bar 聚合逻辑的单元测试
│   ├── test_features.py           # 特征计算正确性测试（用已知数据验证）
│   ├── test_signals.py            # 信号触发条件的边界测试
│   └── test_risk_manager.py       # 风控规则的单元测试（止损触发、仓位限制）
│
├── requirements.txt               # 依赖清单：pandas, numpy, pyarrow, websockets, pyyaml, etc.
└── README.md                      # 项目说明、快速启动指南、回测运行命令
```

### 各模块依赖关系

```
collector → storage → cleaner → bar_builder → feature_pipeline
                                                     │
                                         ┌───────────┼───────────┐
                                         ▼           ▼           ▼
                                    signal_a    signal_b    signal_c
                                         │           │           │
                                         └─────┬─────┘───────────┘
                                               ▼
                                        signal_combiner
                                               │
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                             position_sizer  risk_mgr  order_executor
                                    │          │          │
                                    └──────────┼──────────┘
                                               ▼
                                  engine.py (回测) / live_engine.py (实盘)
```

---

## 附录：实施优先级建议

| 阶段 | 内容 | 预计工时 |
|------|------|---------|
| **Phase 1** | 数据采集 + 存储 + Volume Bar 构建 + 基础特征 | 3-5 天 |
| **Phase 2** | 信号 A（订单流动量）+ 简单回测引擎 + 基础风控 | 5-7 天 |
| **Phase 3** | 信号 B（VWAP 回归）+ 信号 C（体制过滤）+ 完整回测 | 5-7 天 |
| **Phase 4** | Walk-Forward 验证 + 参数优化 + 多币对测试 | 3-5 天 |
| **Phase 5** | 实盘接入 + 监控系统 + 小资金试运行 | 3-5 天 |

**建议**：Phase 1-2 完成后即可获得初步回测结果，据此判断是否值得继续投入。不要等全部完成后才第一次回测。
