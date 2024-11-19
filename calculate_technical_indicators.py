import pandas as pd
import back_test


def calculate_sma_in_memory(data, period):
    """
    Calculate Moving Average (MA) and return the updated dataset with the MA column.

    Args:
        data : Path to the input CSV file.
        period (int): Period for the moving average.

    Returns:
        pd.DataFrame: A DataFrame with the original data and the MA column added.
    """

    # Check if the 'close' column exists
    if 'close' not in data.columns:
        raise ValueError("The CSV file does not contain the 'close' column required for MA calculation.")

    # Ensure the data is sorted by 'time' column

    # Calculate the moving average (MA)
    column_name = f'MA_{period}'
    data[column_name] = data['close'].rolling(window=period).mean()

    # Return the updated dataset
    return data


def calculate_rsi_in_memory(data, period=14):
    """
    计算数据集中 'close' 列的RSI值，使用简单移动平均（SMA）方法，并将结果添加到数据集的新列。

    参数:
        data (pd.DataFrame): 包含 'close' 列的DataFrame。
        period (int): RSI计算周期，默认14。

    返回:
        pd.DataFrame: 更新后的DataFrame，包含新增的 'RSI' 列。
    """
    # 定位 'close' 列
    close = data["close"]

    # 计算价格变化
    change = close.diff()

    # 分别计算涨幅和跌幅
    gain = change.apply(lambda x: x if x > 0 else 0)
    loss = change.apply(lambda x: -x if x < 0 else 0)

    # 计算简单移动平均
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # 计算RS值
    rs = avg_gain / avg_loss

    # 计算RSI值
    data["RSI"] = 100 - (100 / (1 + rs))

    # 返回更新后的数据集
    return data


def calculate_macd_in_memory(data, short_period, long_period, signal_period):
    """
    Calculate Moving Average Convergence Divergence (MACD) and add related columns.

    Args:
        data (pd.DataFrame): The input dataset.
        short_period (int): Short period for EMA.
        long_period (int): Long period for EMA.
        signal_period (int): Signal period for MACD.

    Returns:
        pd.DataFrame: Updated dataset with MACD columns.
    """
    data['EMA_Short'] = data['close'].ewm(span=short_period, adjust=False).mean()
    data['EMA_Long'] = data['close'].ewm(span=long_period, adjust=False).mean()

    data['MACD'] = data['EMA_Short'] - data['EMA_Long']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']

    return data


def calculate_kdj_in_memory(data, period=9, k_smooth=3, d_smooth=3):
    """
    Calculate KDJ (9,3,3) for the given dataset.

    Args:
        data (pd.DataFrame): Input dataset with 'high', 'low', and 'close' columns.
        period (int): Period for RSV calculation (default 9).
        k_smooth (int): Smoothing period for K (default 3).
        d_smooth (int): Smoothing period for D (default 3).

    Returns:
        pd.DataFrame: The input dataset with added 'K', 'D', 'J' columns.
    """
    # Validate input data
    required_columns = {'high', 'low', 'close'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Calculate rolling high and low for the given period
    data['low_period'] = data['low'].rolling(window=period, min_periods=1).min()
    data['high_period'] = data['high'].rolling(window=period, min_periods=1).max()

    # Calculate RSV (Raw Stochastic Value)
    data['RSV'] = (data['close'] - data['low_period']) / (data['high_period'] - data['low_period']) * 100

    # Smooth K value using an exponential moving average
    data['K'] = data['RSV'].ewm(alpha=1 / k_smooth, adjust=False).mean()

    # Smooth D value using an exponential moving average
    data['D'] = data['K'].ewm(alpha=1 / d_smooth, adjust=False).mean()

    # Calculate J value
    data['J'] = 3 * data['K'] - 2 * data['D']

    # Drop intermediate columns if not needed
    data.drop(columns=['low_period', 'high_period', 'RSV'], inplace=True)

    return data



def calculate_boll_in_memory(data, period=20, bandwidth=2):
    """
    Calculate Bollinger Bands using SMA as the middle band.

    Args:
        data (pd.DataFrame): Input data with a 'close' column.
        period (int): The period for the SMA and standard deviation.
        bandwidth (float): The multiplier for standard deviation.

    Returns:
        pd.DataFrame: A DataFrame with added Bollinger Bands columns.
    """
    if 'close' not in data.columns:
        raise ValueError("Input data must contain a 'close' column.")

    # Calculate SMA (middle band)
    data['BOLL_Middle'] = data['close'].rolling(window=period).mean()

    # Calculate standard deviation for the same period
    data['BOLL_Std'] = data['close'].rolling(window=period).apply(lambda x: x.std(ddof=0), raw=True)

    # Calculate upper and lower bands
    data['BOLL_Upper'] = data['BOLL_Middle'] + bandwidth * data['BOLL_Std']
    data['BOLL_Lower'] = data['BOLL_Middle'] - bandwidth * data['BOLL_Std']

    # Drop intermediate standard deviation column if not needed
    data.drop(columns=['BOLL_Std'], inplace=True)

    return data




def calculate_vwap_in_memory(data, period=14):
    """
    Calculate Volume Weighted Average Price (VWAP) for a given period.

    Args:
        data (pd.DataFrame): The input data with 'high', 'low', 'close', and 'volume' columns.
        period (int): The time window for VWAP calculation (default is 14).

    Returns:
        pd.DataFrame: The input data with an additional 'VWAP' column.
    """
    # Validate input data
    required_columns = {'high', 'low', 'close', 'volume'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Input data must contain the following columns: {required_columns}")

    # Calculate Typical Price
    data['Typical_Price'] = (data['high'] + data['low'] + data['close']) / 3

    # Calculate Weighted Price and Cumulative Volume
    data['Weighted_Price'] = data['Typical_Price'] * data['volume']
    data['Cumulative_Weighted_Price'] = data['Weighted_Price'].rolling(window=period).sum()
    data['Cumulative_Volume'] = data['volume'].rolling(window=period).sum()

    # Calculate VWAP
    data['VWAP'] = data['Cumulative_Weighted_Price'] / data['Cumulative_Volume']

    # Drop intermediate columns if not needed
    data.drop(columns=['Typical_Price', 'Weighted_Price', 'Cumulative_Weighted_Price', 'Cumulative_Volume'], inplace=True)

    return data



# Example usage
input_csv = "binance_btc/15m.csv"  # Path to the input CSV file
period = 7  # Moving average period
data = pd.read_csv(input_csv)
# Call the function
#updated_dataset = calculate_rsi_in_memory(data,14)
updated_dataset = calculate_boll_in_memory(data)
# Display the updated dataset
#print(updated_dataset[['time','K','D','J']].tail(20))

print(updated_dataset[updated_dataset["time"] == '2024-09-17 01:15:00'])


