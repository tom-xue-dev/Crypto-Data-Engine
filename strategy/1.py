import pickle
import sys
import numpy as np
import pandas as pd
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets
import statsmodels.api as sm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier


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


def calculate_garman_klass_volatility(group, window):
    """
    在 DataFrame 中添加 Garman-Klass 波动率列。
    """
    group['GK_vol'] = (
            0.5 * (np.log(group['high'] / group['low'])) ** 2 -
            (2 * np.log(2) - 1) / window * (np.log(group['close'] / group['open'])) ** 2
    )
    group['GK_vol_rolling'] = group['GK_vol'].rolling(window=window).mean()
    return group


def llt_filter(price, alpha):
    """
    使用 LLT 滤波器计算低延迟趋势线。
    :param price: 输入的价格序列 (list, np.array, 或 pd.Series)
    :param alpha: 平滑系数，范围 (0, 1)
    :return: 滤波后的 LLT 序列 (np.array)
    """
    # 强制转换为 NumPy 数组（兼容 Pandas Series）
    price = np.asarray(price)

    if len(price) < 3:
        raise ValueError("数据长度必须至少为 3")

    llt = np.zeros_like(price)
    llt[0] = price[0]
    llt[1] = price[1]

    c0 = alpha - (alpha ** 2) / 4
    c1 = (alpha ** 2) / 2
    c2 = alpha - (3 * alpha ** 2) / 4
    d1 = 2 * (1 - alpha)
    d2 = (1 - alpha) ** 2

    for t in range(2, len(price)):
        llt[t] = (
                c0 * price[t] +
                c1 * price[t - 1] -
                c2 * price[t - 2] +
                d1 * llt[t - 1] -
                d2 * llt[t - 2]
        )

    return llt


def calc_beta(df: pd.DataFrame, window_size: int):
    df["beta"] = np.nan
    for asset, group in df.groupby('asset'):
        for i in range(window_size, len(group)):
            x_window = group.iloc[i - window_size:i]["low"].values
            y_window = group.iloc[i - window_size:i]["high"].values
            X = sm.add_constant(x_window)
            model = sm.OLS(y_window, X)
            results = model.fit()
            beta = results.params[1]
            df.loc[group.index[i], "beta"] = beta
    pd.set_option('display.max_columns', 20)
    return df


def calculate_obv(df):
    df['price_change'] = df['close'].diff()
    df['obv'] = 0  # 初始值为 0
    df['obv_delta'] = df['volume'] * np.sign(df['price_change'])  # np.sign: 价格变动正负号
    df['obv'] = df['obv_delta'].cumsum()

    return df


def triple_barrier_labeling(prices, upper_pct=0.03, lower_pct=0.03, max_time=75):
    labels = []
    for t in range(len(prices)):
        start_price = prices.iloc[t]
        upper_barrier = start_price * (1 + upper_pct)
        lower_barrier = start_price * (1 - lower_pct)
        # 遍历时间段内的价格
        for forward in range(1, max_time + 1):
            if t + forward >= len(prices):  # 超出时间序列范围
                break
            current_price = prices.iloc[t + forward]

            if current_price >= upper_barrier:
                labels.append(1)  # 触发上限屏障
                break
            elif current_price <= lower_barrier:
                labels.append(0)  # 触发下限屏障
                break
        else:
            # 时间屏障触发（未触发上下屏障）
            future_price = prices.iloc[min(t + max_time, len(prices) - 1)]
            labels.append(2)

    # 填充缺失的标签，确保返回长度与 prices 一致
    while len(labels) < len(prices):
        labels.append(np.nan)  # 或其他默认值（如 0）

    return pd.Series(labels, index=prices.index)


results = {}
cnt = 0
while True:

    start = "2019-1-1"
    end = "2022-12-30"
    assets = select_assets(spot=True, n=1)
    # assets = ['BTC-USDT_spot']
    data = map_and_load_pkl_files(start_time=start, end_time=end, asset_list=assets, level='15min')
    if data.empty:
        continue
    print("cnt =", cnt)
    if cnt >= 10:
        break
    cnt += 1
    # print(data)
    # 1. 遍历每个资产
    for asset, group in data.groupby('asset'):
        # 2. 计算对数收益率 (log_return)
        group['log_return'] = np.log(group['close'] / group['close'].shift(1))
        group['mean_20'] = group['close'].ewm(span=20, adjust=False).mean()
        group['std_20'] = group['close'].ewm(span=20, adjust=False).std()
        group['theta_20'] = (group['close'] - group['mean_20']) / group['std_20']
        group['mean_50'] = group['close'].ewm(span=50, adjust=False).mean()
        group['std_50'] = group['close'].ewm(span=50, adjust=False).std()
        group['theta_50'] = (group['close'] - group['mean_50']) / group['std_50']

        # 计算未来 5、10、20 根 K 线的收益
        group['return_5'] = group['close'].shift(-5).rolling(window=5).mean() / group['close']
        group['return_10'] = group['close'].shift(-10).rolling(window=10).mean() / group['close']
        group['return_20'] = group['close'].shift(-20).rolling(window=20).mean() / group['close']
        group['return_75'] = group['close'].shift(-75).rolling(window=75).mean() / group['close']
        n = 30  # n日均线窗口
        k = 1200  # 未来收益天数
        group["n_day_ma"] = group['close'].ewm(span=n, adjust=False).mean()
        # df["n_day_ma"] = df['close'].rolling(n).mean()
        # 计算均线斜率（简单使用两点斜率）
        group["slope"] = group["n_day_ma"].diff() / 1  # 计算相邻点间的斜率
        group["slope_normalized"] = (group["slope"] - group["slope"].rolling(k).mean()) / group["slope"].rolling(
            k).std()

        group['rsi_10'] = compute_rsi(data['close'], n=10)
        group['rsi_50'] = compute_rsi(data['close'], n=50)
        group['rsi_100'] = compute_rsi(data['close'], n=100)

        group['upper_shadow'] = group['high'] - group[['open', 'close']].max(axis=1)
        group = calculate_garman_klass_volatility(group, window=30)
        # Z-Score 标准化

        # group['price_high'] = group['close'].rolling(20).max()
        # group['obv_high'] = group['obv'].rolling(20).max()
        # group['top_divergence'] = (group['close'] == group['price_high']) & (group['obv'] < group['obv_high'])
        column_names = ['theta_20', 'theta_50', 'rsi_10', 'rsi_50', 'rsi_100']
        for i in range(len(column_names)):
            group[f'feature_{i}'] = group[column_names[i]]
        # for i in range(4, 5):
        #     group[f'feature_{i}'] = group['upper_shadow']
        for i in range(5, 6):
            group[f'feature_{i}'] = group['GK_vol_rolling']
        group['label'] = triple_barrier_labeling(data['close'])
        #group['label'] = (group['return_75'] > 1.02).astype(int)
        # 5. 丢弃无效数据（由于 shift 产生的 NaN）
        group.dropna(inplace=True)

        # 6. 准备训练数据和标签
        #    将过去 50 根收盘价作为特征列
        feature_cols = [f'feature_{i}' for i in range(2, 6)]
        X = group[feature_cols]

        y = group['label']

        # 7. 划分训练集和测试集
        #    对于时间序列，可以考虑使用基于时间的切分方法
        #    这里简单演示用 train_test_split 做随机划分。
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # shuffle=False 保持时间顺序
        )

        # 8. XGBoost 模型训练
        model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
        print(X_train)
        print(y_train)
        model.fit(X_train, y_train)

        threshold = 0.5
        probs = model.predict_proba(X_test)  # 形状 (n_samples, n_classes)

        max_probs = np.max(probs, axis=1)  # 每个样本的最高投票比例
        predictions = np.argmax(probs, axis=1)  # 每个样本最可能的类别

        predictions[max_probs < threshold] = -3  # -1 代表不确定类别
        # X_test_with_extra = X_test.copy()
        # X_test_with_extra['close'] = group.loc[X_test.index, 'close']
        # X_test_with_extra['open'] = group.loc[X_test.index, 'open']
        # with open("x_test.pkl", "wb") as file:
        #     pickle.dump(X_test_with_extra, file)
        # with open("predictions.pkl", "wb") as file:
        #     pickle.dump(predictions, file)
        mask = predictions != -3  # 仅保留非 -1 样本
        y_test = y_test[mask]  # 筛选对应的真实值
        y_pred = predictions[mask]  # 筛选对应的预测值
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)

        print(f"Asset: {asset}")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(report)

        # 将模型和结果存下来
        results[asset] = {
            'accuracy': acc,
            'classification_report': report
        }


acc_list = [v['accuracy'] for v in results.values()]

# 计算均值
if len(acc_list) == 0:
    mean_acc = 0.0  # 默认值
else:
    mean_acc = sum(acc_list) / len(acc_list)

# 计算方差（总体方差）
if len(acc_list) == 0:
    variance_acc = 0.0  # 默认值
else:
    squared_diffs = [(x - mean_acc) ** 2 for x in acc_list]
    variance_acc = sum(squared_diffs) / len(acc_list)

# 输出结果（保留4位小数）
print(f"所有资产的准确率均值: {mean_acc:.4f}")
print(f"所有资产的准确率方差: {variance_acc:.4f}")
