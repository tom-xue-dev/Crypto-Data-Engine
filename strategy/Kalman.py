import numpy as np
import pandas as pd
from read_large_files import load_filtered_data_as_list, select_assets, map_and_load_pkl_files
import matplotlib.pyplot as plt


# ======================
# 1. 卡尔曼滤波类（保持不变）
# ======================
def calculate_ic(factor, future_ret, method='spearman'):
    """
    计算 RSI(因子) 与 未来收益 的相关系数(IC)
    :param factor: pd.Series, RSI等因子值
    :param future_ret: pd.Series, 对应时点的未来收益
    :param method: 'pearson' or 'spearman'
    :return: float, 整段样本期的相关系数
    """
    # 对齐
    valid = factor.notna() & future_ret.notna()
    if method == 'pearson':
        corr = np.corrcoef(factor[valid], future_ret[valid])[0, 1]
    else:
        corr = factor[valid].corr(future_ret[valid], method='spearman')
    return corr
class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, F, H, Q, R):
        self.state = initial_state
        self.covariance = initial_covariance
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

    def predict(self):
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.state

    def update(self, z):
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.state
        self.state += K @ y
        self.covariance = (np.eye(2) - K @ self.H) @ self.covariance
        return self.state


def compute_rsi(series, n=14):
    """
    计算RSI，返回与series同长度的序列
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # 这里用rolling的mean简化RSI
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()

    # 避免除零
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)

    return rsi


# ======================
# 2. 参数初始化（需历史数据训练）
# ======================
def initialize_parameters(historical_prices):
    """基于历史数据计算初始Q和R"""
    returns = np.diff(historical_prices)
    R = np.var(returns)  # 观测噪声 = 收益方差
    Q_scale = 0.1 * R  # 过程噪声为观测噪声的10%
    Q = np.eye(2) * Q_scale
    return Q, np.array([[R]])


# ======================
# 3. 自适应逻辑增强
# ======================
def adaptive_Q(velocities, current_volatility):
    """根据市场波动率动态调整过程噪声"""
    Q_base = np.eye(2) * 0.1
    return Q_base * (1 + current_volatility)


def adaptive_R(prices, volatility_window=20):
    """基于波动率调整观测噪声，并过滤异常值"""
    recent_prices = prices[-volatility_window:]
    volatility = np.std(recent_prices)

    # 过滤异常值（如涨跌停）
    last_price = prices[-1]
    if abs(last_price - np.mean(recent_prices)) > 3 * volatility:
        return np.array([[volatility * 10]])  # 临时增大噪声协方差
    return np.array([[volatility]])

while True:
    start = "2019-1-1"
    end = "2022-12-30"
    assets = select_assets(spot=True,n=1)
    #assets = ['BTC-USDT_spot']
    data = map_and_load_pkl_files(start_time=start, end_time=end, asset_list=assets, level='15min')
    if data.empty:
        continue
    btc_data = data.xs(assets[0], level='asset').sort_index()
    historical_prices = btc_data['close'].values

    # 初始化 Q, R
    Q_init, R_init = initialize_parameters(historical_prices)

    # 状态转移矩阵 F、观测矩阵 H 等
    F = np.array([[1, 1],
                  [0, 1]])
    H = np.array([[1, 0]])

    initial_state = np.array([[historical_prices[0]],  # 初始价格
                              [0]])  # 初始速度
    initial_covariance = np.eye(2) * 0.1

    kf = KalmanFilter(initial_state, initial_covariance, F, H, Q_init, R_init)

    filtered_prices = []
    velocities = []

    window_size = 20

    for i in range(len(btc_data)):
        # 1) 预测
        kf.predict()

        # 2) 自适应噪声（可选）
        if i >= window_size:
            recent_prices = btc_data['close'].values[i - window_size:i]
            current_volatility = np.std(recent_prices)

            # 自适应Q
            new_Q = adaptive_Q(velocities, current_volatility) if velocities else kf.Q
            kf.Q = new_Q

            # 自适应R
            new_R = adaptive_R(recent_prices, volatility_window=window_size)
            kf.R = new_R

        # 3) 更新
        current_price = btc_data['close'].values[i]
        kf.update(np.array([current_price]))

        # 4) 记录结果
        filtered_prices.append(kf.state[0, 0])  # state[0,0]表示价格
        velocities.append(kf.state[1, 0])  # state[1,0]表示速度

    # ============================
    # 2. 绘制可视化
    # ============================
    time_index = btc_data.index  # 取出时间轴

    # plt.figure(figsize=(12, 6))
    # plt.plot(time_index, btc_data['close'], label='Raw Price', color='red', alpha=0.6)
    # plt.plot(time_index, filtered_prices, label='Filtered Price (Kalman)', color='blue')
    #
    # plt.title('BTC-USDT Spot Price: Raw vs. Kalman Filtered')
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()
    data['rsi_raw'] = compute_rsi(data['close'], n=40)

    # 2) 卡尔曼滤波后收盘价
    data['close_kalman'] = filtered_prices

    # 3) 过滤后的RSI
    data['rsi_kalman'] = compute_rsi(data['close_kalman'], n=40)





    # 定义未来1期的简单收益
    data['fwd_ret_1'] = data['close'].shift(-20) / data['close'] - 1

    # 原始RSI的IC
    ic_raw = calculate_ic(data['rsi_raw'], data['fwd_ret_1'])

    # 滤波后RSI的IC
    ic_kalman = calculate_ic(data['rsi_kalman'], data['fwd_ret_1'], method='pearson')

    print(f"IC(RSI_raw) = {ic_raw:.4f}")
    print(f"IC(RSI_kalman) = {ic_kalman:.4f}")
