# 动量策略系统性回测大纲

## 目标

在 dollar bar 横截面框架下，系统性地发掘和验证**中期动量因子**（持仓 7-28 天），
与已有的日内反转策略（持仓 1 天）形成低相关组合，提升整体投资组合的 Sharpe 和稳健性。

---

## 已有结论回顾

### 已验证有效的动量因子（Session 53c091d2, M1-M6）

| 因子 | 配置 | OOS Sharpe | 持仓周期 | 备注 |
|------|------|-----------|---------|------|
| OFI_14d | Top20, 3L3S, R7 | +1.96 | ~1周 | IS 不稳定 (+0.45) |
| path_eff_21d | Top50, 10L10S, R14 | +3.29 | ~2周 | 最稳定单因子 |
| flow_3 (OFI+centroid+大单) | Top50, 15L15S, R14 | +2.39 | ~2周 | 复合因子 |
| relstr_7d | Top20, 5L5S, R21 | +2.21 | ~3周 | 最正交 (ρ<0.3) |
| EqW(asym+ofi) | Top50, 10L10S, R14 | +2.79 | ~2周 | MDD最低 -11% |

### 已验证失败

- 纯价格动量 (ret_7d, ret_14d, ret_21d, ret_28d) → OOS 全部负 Sharpe
- 分布形态因子 (skewness, kurtosis) → regime 不稳定
- 时序模式 (ret_autocorr, ofi_autocorr) → 信号太弱
- buy_ratio → 与 OFI ρ=1.0 完全冗余

### ⚠️ 已知问题

1. **M 系列使用同日信号，未做 T-1 lag** — 对 R7/R14 影响小但非零
2. **Tick regime 因子（VPIN/jump）从未测过 7d+ 的预测能力** — 只测过 1d fwd（IC≈0）和作为反转 filter
3. **enriched bars 现在有 488 symbols**（M系列时部分因子只有 124 symbols）

---

## 数据基础

### Dollar Bar 原始列 (23列, 488 symbols)
```
价格: open, high, low, close, vwap
量: volume, buy_volume, sell_volume, dollar_volume
微观: tick_count, price_std, volume_std, tick_interval_mean
结构: up_move_ratio, down_move_ratio, reversals, buy_sell_imbalance
大单: max_trade_volume, max_trade_ratio
效率: path_efficiency, impact_density
```

### Enriched 额外列 (9列, 488 symbols)
```
tick_vpin, tick_toxicity_run_mean, tick_toxicity_run_max, tick_toxicity_ratio,
tick_kyle_lambda, tick_burstiness, tick_jump_ratio,
tick_whale_imbalance, tick_whale_impact
```

### 日频聚合方式
```
每个交易日对所有 bar 做 aggregation:
  close → last (收盘价)
  ofi → mean (buy_sell_imbalance 日均值)
  其他列 → mean / sum / last 视具体因子而定
```

---

## Phase M-NEW: 动量因子系统性研究

### Step 1: 因子构造 (Factor Construction)

从现有 32 列原始数据出发，构造多维度的动量因子候选池。

#### A. 微观结构动量 (已部分验证)
```
A1. ofi_{W}d         = rolling_mean(buy_sell_imbalance, W)     W ∈ {7,14,21,28}
A2. path_eff_{W}d    = rolling_mean(path_efficiency, W)        W ∈ {14,21,28}
A3. centroid_{W}d    = rolling_mean((vwap-low)/(high-low), W)  W ∈ {7,14,21}
A4. large_trade_{W}d = rolling_mean(max_trade_ratio, W)        W ∈ {7,14,21}
A5. impact_{W}d      = rolling_mean(impact_density, W)         W ∈ {14,21}
```

#### B. Tick 微观动量 (全新 — 从未测过中期预测力)
```
B1. vpin_{W}d           = rolling_mean(tick_vpin, W)              W ∈ {7,14,21}
B2. kyle_lambda_{W}d    = rolling_mean(tick_kyle_lambda, W)       W ∈ {14,21}
B3. whale_imbalance_{W}d = rolling_mean(tick_whale_imbalance, W)  W ∈ {7,14,21}
B4. toxicity_{W}d       = rolling_mean(tick_toxicity_ratio, W)    W ∈ {14,21}
B5. burstiness_{W}d     = rolling_mean(tick_burstiness, W)        W ∈ {14,21}
B6. jump_ratio_{W}d     = rolling_mean(tick_jump_ratio, W)        W ∈ {14,21}
B7. whale_impact_{W}d   = rolling_mean(tick_whale_impact, W)      W ∈ {14,21}
```

#### C. 量价复合因子 (新构造)
```
C1. vol_adj_mom_{W}d     = ret_{W}d / realized_vol_{W}d          Sharpe-like 动量
C2. ofi_acceleration     = ofi_7d - ofi_21d                      订单流加速度
C3. flow_persistence     = rolling_autocorr(ofi, 14d)            资金流持续性
C4. volume_price_div     = sign(ret_{W}d) × sign(Δvol_{W}d)      量价背离
C5. smart_money_flow     = whale_imbalance × abs(ret)             大单×价格冲击
C6. info_asymmetry       = vpin × kyle_lambda                     信息不对称综合
C7. liquidity_momentum   = -rolling_mean(impact_density_change, W) 流动性改善动量
```

#### D. 横截面相对因子
```
D1. relative_strength_{W}d = coin_ret_{W}d - BTC_ret_{W}d        相对强度
D2. idiosyncratic_vol      = std(residual after BTC regression, 20d)
D3. beta_change            = beta_20d - beta_60d                  Beta变化
```

#### E. 多频率混合
```
E1. flow_3           = mean(ofi_14d, centroid_14d, large_trade_14d)
E2. tick_flow_3      = mean(whale_imbalance_14d, vpin_14d, kyle_lambda_14d)
E3. multi_freq_ofi   = 0.5×ofi_7d + 0.3×ofi_14d + 0.2×ofi_21d
```

> **总计: ~50+ 因子变体**

---

### Step 2: 多维度 IC 分析 (Information Coefficient)

对每个因子，计算其对不同 forward return horizon 的横截面 Rank IC。

#### 2a. IC × Forward Horizon 矩阵
```
Forward Horizons: 1d, 3d, 5d, 7d, 14d, 21d, 28d
对每个因子 f, 每个交易日 d:
  IC(f, d, h) = Spearman_rank_corr(signal[d], ret[d+1 : d+h])

输出:
  IC_mean(f, h)  = mean over all dates
  ICIR(f, h)     = IC_mean / IC_std
  IC_decay(f)    = IC 随 horizon 的衰减速率
```

关键判断标准:
- IC > 0.03 且 ICIR > 0.3 才值得继续
- IC 在 7-14d 达峰的因子适合动量 (vs 1d 达峰 → 反转)
- IC 缓慢衰减的因子 → 信号持续性好，适合低频 rebal

#### 2b. IS/OOS 分期 IC
```
IS:  2020-01 ~ 2023-12
OOS: 2024-01 ~ 2026-01
要求 IC_IS 和 IC_OOS 同号且量级相近 (regime stability)
```

#### 2c. 因子间相关性矩阵
```
对通过 IC 筛选的因子，计算 OOS 期间的 pairwise rank correlation
识别冗余因子组 (ρ > 0.7) 和正交因子对 (ρ < 0.3)
```

---

### Step 3: 持仓周期 × 因子 收益扫描

对通过 Step 2 筛选的因子，做**持仓周期敏感性分析**。

#### 3a. 参数网格
```
因子:        Step 2 筛选出的 Top K (K ≈ 10-15)
Rebal:       R3, R5, R7, R14, R21, R28
Pool:        ALL, Top100, Top50, Top20
N_LS:        3, 5, 10, 15, 30
Weight:      equal, signal_prop
Fee:         10bps per turnover
信号延迟:    same_day (R≤3), prev_day (R≥5)
```

#### 3b. 输出
```
每个 (因子, rebal, pool, N_LS, weight) 组合:
  - IS Sharpe, OOS Sharpe, Full Sharpe
  - OOS MDD, OOS Calmar
  - 年化换手率
  - OOS 月度胜率

关键分析:
  1. 最优 rebal 是否与 IC 达峰 horizon 一致
  2. 短 rebal (R3-R5) vs 长 rebal (R14-R28) 的 fee-adjusted 差异
  3. Pool 大小对 alpha 稀释/集中效果
```

#### 3c. 费率敏感性
```
对最优配置，测试 fee = 5, 8, 10, 15, 20 bps
确定 breakeven fee level
动量策略换手率低，fee 敏感性应远低于反转策略
```

---

### Step 4: 分层回测 (Quantile Analysis)

对 Step 3 确认有效的因子，做**严格的分层回测**验证因子单调性。

#### 4a. 分层方法
```
每个 rebal 日:
  1. 对 pool 内所有 symbol 按因子值排序
  2. 等分为 5 层 (Q1=最低 ... Q5=最高)
  3. 每层等权做多，持有至下次 rebal
  4. 记录每层的累积收益

判断标准:
  - Q5-Q1 spread 持续为正
  - 中间层 (Q2-Q4) 呈单调递增
  - 非仅 Q5 有正收益 (avoid concentration risk)
```

#### 4b. 分层收益图
```
5 条累积收益曲线 + L/S spread 曲线
分 IS/OOS 两段展示
```

#### 4c. 子周期分析
```
按年度/半年度切分，检查因子在各子周期的:
  - Q5-Q1 spread 是否持续为正
  - 排名是否稳定 (Kendall tau)
  - 识别 regime flip 风险
```

---

### Step 5: Regime 过滤 (Regime Conditioning)

在 Step 4 确认因子有效后，叠加 regime filter 进一步提升风险调整收益。

#### 5a. VPIN Regime Filter (已在反转策略上验证有效)
```
思路: VPIN 高 → 知情交易 → 动量信号可能更可靠（信息驱动的趋势延续）
      VPIN 低 → 噪音交易 → 动量信号可能是假突破

注意: 这里的逻辑与反转策略**相反**!
  反转: VPIN高 → 抑制反转 (信息驱动不会反转)
  动量: VPIN高 → 增强动量 (信息驱动趋势延续)

实验:
  - vpin_boost:   signal × (1 + α × vpin_zscore)  当 vpin_z > threshold
  - vpin_filter:  suppress signal when vpin_z < threshold (过滤噪音动量)
  - threshold grid: 0.5, 1.0, 1.5, 2.0
```

#### 5b. Jump Ratio Regime Filter
```
思路: 跳跃事件后的动量更持续 (信息冲击)
  - jump_boost: 高 jump_ratio 时放大动量信号
  - jump_filter: 无跳跃时降低仓位
```

#### 5c. 波动率 Regime
```
  - vol_condition: 高波动率时缩小仓位 (vol targeting)
  - vol_breakout:  波动率扩张 + OFI 同向 → 增强信号
```

#### 5d. BTC Beta Regime
```
  - beta_neutral: 回归 BTC 后取残差动量 → 去市场因子
  - regime_switch: BTC 上涨时做多 high-beta, BTC 下跌时做多 low-beta
```

---

### Step 6: 多因子组合与优化

#### 6a. 双因子等权
```
对 Step 4-5 中存活的因子两两组合:
  signal_combo = 0.5 × zscore(f1) + 0.5 × zscore(f2)
  要求 ρ(f1, f2) < 0.5 才值得组合
```

#### 6b. IC 加权
```
  rolling_ic_weight: 过去 60d 的 IC 均值作为权重
  更新频率: 每月
```

#### 6c. 动量 + 反转 Portfolio
```
  Rev leg:  VPIN-filtered 反转 (R1, 30L30S, same_day) → 已有 OOS +3.06
  Mom leg:  最优动量因子 (R7-R14, best config from Step 3-5)
  
  组合方式:
    - EqW (50/50)
    - Risk Parity (按波动率反比配权)
    - Dynamic (根据 regime 调整 Rev/Mom 比例)
  
  期望: Rev 和 Mom 相关性 ρ ≈ 0 → 组合 Sharpe ≈ √(S_rev² + S_mom²)
```

---

### Step 7: Walk-Forward 验证

#### 7a. 滚动窗口验证
```
Training: 365d
Testing:  180d
Step:     90d
~19-20 个窗口

指标:
  - WF Mean Sharpe
  - WF Std
  - % Positive windows
  - Max degradation (OOS vs IS)
```

#### 7b. 稳健性检查
```
  - 随机删除 20% 的日期 → 结果变化幅度
  - 随机删除 30% 的 symbol → 结果变化幅度
  - 延迟 1 天 / 2 天交易 → decay rate
```

---

### Step 8: 补充分析

#### 8a. 容量分析
```
  - 换手率 × 平均持仓 dollar volume
  - 估算可容纳资金量 ($1M, $5M, $10M)
  - 冲击成本模型: slippage = k × √(order_size / ADV)
```

#### 8b. 因子拥挤度
```
  - 因子收益与市场波动的相关性
  - 因子收益自相关 → 拥挤信号
  - drawdown clustering → 系统性风险
```

#### 8c. 实盘可行性
```
  - 动量策略 rebal 周期长 (R7-R14) → 执行窗口宽松
  - 不需要 EOD 精确执行 → 可以 TWAP 分散下单
  - fee budget: 动量换手 ~0.15/日 vs 反转 ~2.4/日 → fee 几乎可忽略
```

---

## 执行计划 (Workplan)

- [x] **Step 1+2**: 因子构造 + IC分析 — 50因子, 7 horizon IC, 相关性矩阵 ✅
  - smart_money_flow IC=0.092最高, VPIN中期IC=0.043确认
  - 所有因子IC随horizon单调递增, 峰值在28d
- [x] **Step 3**: 持仓周期扫描 — 2040 configs参数网格 ✅
  - ofi_14d R14 T50 15LS OOS+2.40, AnnRet 58.8%, Calmar 4.84🏆
  - R7-R14最优区间, T50最优pool, equal weight最优
- [x] **Step 4**: 分层回测 — Quintile分析 ✅
  - flow_3唯一单调因子(Q1→Q5 Sharpe递增), L/S +2.47
  - ofi_14d L/S +2.20, 非严格单调但极端分层有效
- [x] **Step 5**: Regime过滤 — 13种regime × 3因子 ✅
  - 动量regime增量有限 (max +0.45 centroid kyle_boost)
  - flow_3 vpin_suppress +0.27 (与初始假设相反!)
  - 结论: 动量因子直接使用, 不需要regime层
- [x] **Step 6**: 多因子组合 — 15种组合方案 ✅
  - EqW(ofi+flow3+cen) OOS+2.94, AnnRet 92.7%, Calmar 4.98🏆
  - EqW(ofi+cen) OOS+2.78, Calmar 6.41 (最高风险调整)
  - Rev×Mom ρ=0.03 → 组合潜力巨大
- [x] **Step 7**: Walk-Forward — 22个3月窗口 ✅
  - ofi_14d: 77%正窗口, mean Sharpe +1.25
  - EqW(3F): 77%正窗口, mean Sharpe +1.35
- [ ] **Step 8**: 补充分析 — 容量/拥挤度/实盘可行性 (待做)

---

## 技术约束备忘

1. **信号延迟**: R≤3 用 same_day, R≥5 用 prev_day (T7 发现)
2. **PnL 时序**: 先用旧权重算 PnL, 再 rebalance (无前视偏差)
3. **时间戳**: datetime64[ms].astype(int64) 是 ms 不是 ns, 要转 datetime64[ns] 再转 int64
4. **数据路径**: bars → `E:\data\dollar_bar\bars\`, enriched → `E:\data\dollar_bar\bars_enriched\`
5. **IS/OOS 切分**: 2024-01-01
6. **Fee**: 10bps per turnover (Binance futures taker ~5bps + slippage ~3-5bps)
7. **Windows 环境**: PowerShell 中不能用 f-string with braces, 用脚本文件代替 -c 模式
