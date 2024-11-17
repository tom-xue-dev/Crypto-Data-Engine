import pandas as pd


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
    data = data.sort_values('time')

    # Calculate the moving average (MA)
    column_name = f'MA_{period}'
    data[column_name] = data['close'].rolling(window=period).mean()

    # Return the updated dataset
    return data

def calculate_rsi_in_memory(data, period):
    """
    Calculate RSI for a dataset with descending timestamp order.

    Args:
        data (pd.DataFrame): Dataset in descending timestamp order with 'time' and 'close' columns.
        period (int): RSI calculation period (default is 14).

    Returns:
        pd.DataFrame: Dataset with an added 'RSI' column in descending timestamp order.
    """
    # Ensure the timestamps are sorted in ascending order for correct RSI calculation
    data = data.sort_values(by='time', ascending=True)

    # Calculate price changes
    delta = data['close'].diff()

    # Separate upward and downward changes
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss using rolling mean
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Calculate RS (Relative Strength) and RSI
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Restore the original descending timestamp order
    data = data.sort_values(by='time', ascending=False)

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

def calculate_obv_sma_in_memory(data, sma_period=9):
    """
    向量化计算时间戳降序数据集的 OBV 和 OBV_SMA。

    参数：
        data (pd.DataFrame): 时间戳降序排列的数据集，需包含 'time', 'close', 'volume' 列。
        sma_period (int): OBV 的简单移动平均周期，默认为 9。

    返回：
        pd.DataFrame: 包含 OBV 和 OBV_SMA 列的更新数据集，恢复为降序排列。
    """
    # 将数据按时间戳升序排列
    data = data.sort_values(by='time', ascending=True).reset_index(drop=True)

    # 计算 OBV 的方向：1 表示上涨，-1 表示下跌，0 表示持平
    obv_direction = data['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # 计算 OBV（累积成交量变化）
    data['OBV'] = (obv_direction * data['volume']).cumsum()

    # 计算 OBV 的简单移动平均
    data['OBV_SMA'] = data['OBV'].rolling(window=sma_period).mean()

    # 恢复为时间戳降序排列
    data = data.sort_values(by='time', ascending=False).reset_index(drop=True)

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
    data = data.sort_values('time')
    if 'close' not in data.columns:
        raise ValueError("Input data must contain a 'close' column.")

    # Calculate SMA (middle band)
    data['BOLL_Middle'] = data['close'].rolling(window=period).mean()

    # Calculate standard deviation for the same period
    data['BOLL_Std'] = data['close'].rolling(window=period).std()

    # Calculate upper and lower bands
    data['BOLL_Upper'] = data['BOLL_Middle'] + bandwidth * data['BOLL_Std']
    data['BOLL_Lower'] = data['BOLL_Middle'] - bandwidth * data['BOLL_Std']

    # Drop intermediate standard deviation column if not needed
    data.drop(columns=['BOLL_Std'], inplace=True)

    return data




def calculate_vwap_in_memory(data):
    """
    Calculate Volume Weighted Average Price (VWAP) and add it to the dataset.

    Args:
        data (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: Updated dataset with VWAP column.
    """
    data['Typical_Price'] = (data['high'] + data['low'] + data['close']) / 3
    data['VWAP'] = (data['Typical_Price'] * data['volume']).cumsum() / data['volume'].cumsum()

    return data


def calculate_sma(series, window):
    """
    Calculate Simple Moving Average (SMA) for a given Pandas Series.

    Args:
        series (pd.Series): The input data series (e.g., 'close' prices).
        window (int): The period for the SMA calculation.

    Returns:
        pd.Series: A Pandas Series containing the SMA values.
    """
    if not isinstance(series, pd.Series):
        raise ValueError("Input data must be a Pandas Series.")

    if window <= 0:
        raise ValueError("Window size must be greater than 0.")

    # Calculate the rolling mean (SMA)
    return series.rolling(window=window, min_periods=1).mean()



# Example usage
input_csv = "binance_btc/15m.csv"  # Path to the input CSV file
period = 7  # Moving average period
data = pd.read_csv(input_csv)
# Call the function
updated_dataset = calculate_boll_in_memory(data)

# Display the updated dataset
print(updated_dataset[['time','close','BOLL_Middle','BOLL_Upper','BOLL_Lower']].tail(20))
#print(updated_dataset[['close']].tail(20))


