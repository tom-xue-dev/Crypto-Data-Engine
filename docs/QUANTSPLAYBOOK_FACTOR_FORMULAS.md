# QuantsPlaybook 择时与动量因子公式汇总

> 本文将你刚才要的“择时 + 动量”核心因子，统一整理为一个可直接查阅的公式手册。  
> 来源：`hugo2046/QuantsPlaybook`（本地快照：`master@d97ea1e`）。  
> 说明：标注“**按 Notebook 描述推导**”的条目，是按文字步骤还原为数学表达。

## 0. 记号约定

- \(P_t\)：收盘价，\(O_t\)：开盘价，\(H_t/L_t\)：最高/最低价
- \(r_t\)：收益率（默认日频）
- \(\sigma_t\)：波动率（滚动标准差）
- \(\text{rank}(\cdot)\)：截面排序（文中会注明升序/降序）

---

## 1. 择时类因子

### 1.1 RSRS / QRS（支撑阻力相对强度）

1) 回归斜率（窗口 \(N\)）：
\[
H = \alpha + \beta L + \varepsilon,\quad \beta_t
\]

2) 代码中的快速实现（等价写法）：
\[
\beta_t=\frac{\text{std}(H)}{\text{std}(L)}\cdot \text{corr}(H,L)
\]

3) 标准分（窗口 \(M\)）：
\[
z_t = \frac{\beta_t-\mu_{t,M}}{\sigma_{t,M}}
\]

4) 惩罚项：
\[
\text{Reg}_t = \text{corr}(H,L)^n
\]

5) QRS 信号：
\[
\text{QRS}_t = z_t \cdot \text{Reg}_t
\]
（可选：\(\text{Reg}_t\) 再除其滚动均值做归一化）

---

### 1.2 ICU 均线（稳健回归均线）

在窗口 \(N\) 内对价格做 Siegel Repeated Median 回归，得到截距 \(\hat\alpha_t\)、斜率 \(\hat\beta_t\)，ICU 均线为窗口末端预测值：
\[
\text{ICU}_t = \hat\alpha_t + \hat\beta_t (N-1)
\]

---

### 1.3 鳄鱼线 / AO / 分型 / MACD / 北向分位

1) 鳄鱼线（Jaw/Teeth/Lips）：
\[
\text{Alligator}^{(i)}_t = \text{SMA}_{p_i}(P)_t \text{ shifted by } l_i,\;
(p,l)=(13,8),(8,5),(5,3)
\]

2) AO（该仓库版本）：
\[
\text{AO}_t=\text{SMA}_5\!\left(\frac{H_t-L_t}{2}\right)-\text{SMA}_{34}\!\left(\frac{H_t-L_t}{2}\right)
\]

3) 分型判定（3 根）：
- 顶分型：\(H_{t-1}<H_t>H_{t+1}\) 且 \(L_{t-1}<L_t>L_{t+1}\)
- 底分型：\(H_{t-1}>H_t<H_{t+1}\) 且 \(L_{t-1}>L_t<L_{t+1}\)

4) MACD 信号核心：DIF 上/下穿 DEA + 柱体翻红/翻绿 + 零轴约束。

5) 北向分位信号（60 日）：
- \(>80\%\) 分位：+1
- \(<20\%\) 分位：-1

---

### 1.4 日内动量（噪声区间法）

1) 日内 VWAP：
\[
\text{VWAP}_{t,\tau}=\frac{\sum_{s\le\tau} P_{t,s}V_{t,s}}{\sum_{s\le\tau}V_{t,s}}
\]

2) 开盘位移绝对值：
\[
d_{t,\tau}=\left|\frac{P_{t,\tau}}{O_t}-1\right|
\]

3) 同时刻滚动噪声（窗口 \(W\)，默认 14）：
\[
\sigma_{t,\tau}=\text{MA}_W(d_{\cdot,\tau})
\]

4) 上下边界：
\[
\text{UB}_{t,\tau}=\max(O_t,P_{t-1,\text{close}})\cdot(1+\sigma_{t,\tau})
\]
\[
\text{LB}_{t,\tau}=\min(O_t,P_{t-1,\text{close}})\cdot(1-\sigma_{t,\tau})
\]

---

### 1.5 行业 NHNL（净新高占比）

行业 \(i\) 在时点 \(t\) 的指标：
\[
\text{NHNL}_{i,t}=\frac{\text{NH}_{i,t}-\text{NL}_{i,t}}{N_i}
\]
- \(\text{NH}\)：窗口内创新高数量
- \(\text{NL}\)：窗口内创新低数量
- \(N_i\)：行业股票数

---

### 1.6 C-VIX / C-SKEW

1) 远期价格：
\[
F = K + e^{RT}(C-P)
\]

2) 方差项：
\[
\sigma^2=\frac{2}{T}\sum_K \frac{\Delta K}{K^2}e^{RT}Q(K)-\frac{1}{T}\left(\frac{F}{K_0}-1\right)^2
\]

3) 30 天 VIX 插值：
\[
w=\frac{T_2-\frac{30}{365}}{T_2-T_1},\quad
\text{VIX}=\sqrt{\left(T_1\sigma_1w+T_2\sigma_2(1-w)\right)\frac{365}{30}}
\]

4) 偏度指数：
\[
\text{SKEW}=100-10\cdot\left(w\cdot s_1+(1-w)\cdot s_2\right)
\]

---

## 2. 动量类因子

### 2.1 基础动量与高质量动量

1) 基础动量（60 日风险惩罚）：
\[
\text{MOM}_{\text{basic}}=r_{60}-3000\cdot \sigma_{60}^2
\]

2) 改进项：
\[
\text{MAX}=\max(r_{t-19:t}),\quad
\text{ID}=\text{std}(\text{turnover}_{t-59:t})
\]

3) 综合高质量动量（等权排序融合）：
\[
\text{Score}=\text{mean}\left(
\text{rank}_{desc}(\text{MAX}),
\text{rank}_{desc}(\text{ID}),
\text{rank}_{asc}(\text{MOM}_{basic})
\right)
\]

---

### 2.2 A 股“振幅切分”动量（按 Notebook 描述推导）

在 \(N\) 日窗口内，按日振幅 \(a_d\) 对收益 \(r_d\) 排序并切分：
- 低振幅集合 \(L_\lambda\)：振幅最低 \(\lambda\) 比例交易日
- 高振幅集合 \(H_\lambda\)：振幅最高 \((1-\lambda)\) 或对称比例交易日

\[
A_\lambda=\sum_{d\in L_\lambda} r_d,\quad
B_\lambda=\sum_{d\in H_\lambda} r_d
\]

文中结论：低振幅部分更偏动量，高振幅部分更偏反转。

---

### 2.3 球队硬币（Coin-Team）与波动/换手翻转

给定基准矩阵 \(b_{i,t}\) 与截面均值 \(\bar b_t\)：
\[
\text{coin}_{i,t}=\text{sign}\big((b_{i,t}-\bar b_t)\cdot s\big),\; s\in\{1,-1\}
\]
\[
f^{flip}_{i,t}=f^{raw}_{i,t}\cdot \text{coin}_{i,t}
\]

1) 波动翻转：
\[
f^{vol}_{i,t}=\text{flip}\!\left(\text{std}_W(r_{i,t}),\; \text{mean}_W(r_{i,t})\right)
\]

2) 换手翻转：
\[
\Delta \text{turn}_{i,t}=\text{turn}_{i,t}-\text{turn}_{i,t-1}
\]
\[
f^{turn}_{i,t}=\text{mean}_W\left(\text{flip}(\Delta\text{turn}_{i,t}, r_{i,t})\right)
\]

3) 修正版：
\[
f^{revise}=\frac{f^{vol}+f^{turn}}{2}
\]

4) coin\_team 聚合：
\[
f^{coin\_team}=f^{revise}_{interday}+f^{revise}_{intraday}+f^{revise}_{overnight}
\]

---

### 2.4 传统时序动量（TSMOM）

\[
r^{TSMOM,i}_{t,t+1}
=
\text{sgn}\!\left(r^i_{t-252,t}\right)
\cdot
\frac{\sigma_{tgt}}{\sigma^i_t}
\cdot
r^i_{t,t+1}
\]

其中 \(\sigma_{tgt}\) 为目标波动率，\(\sigma^i_t\) 为资产 \(i\) 的当期波动率估计。

---

## 3. 代码来源索引（便于回溯）

- `SignalMaker/qrs.py`
- `C-择时类/RSRS择时指标/py/RSRS.ipynb`
- `C-择时类/ICU均线/src/icu_ma.py`
- `C-择时类/基于鳄鱼线的指数择时及轮动策略/src/SignalMaker.py`
- `C-择时类/另类ETF交易策略：日内动量/src/SignalMaker.py`
- `C-择时类/行业指数顶部和底部信号/scr/core.py`
- `C-择时类/C-VIX中国版VIX编制手册/scr/calc_func.py`
- `B-因子构建类/高质量动量因子选股/高质量动量选股.ipynb`
- `B-因子构建类/A股市场中如何构造动量因子？/notebook/A股市场中如何构造动量因子.ipynb`
- `B-因子构建类/个股动量效应的识别及球队硬币因子/FactorZoo/SportBetting.py`
- `D-组合优化/MLT_TSMOM/mlt_tsmom.ipynb`
