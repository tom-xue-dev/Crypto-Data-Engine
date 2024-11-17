import pandas as pd


def calculate_ma_in_memory(data, period):
    """
    Calculate Moving Average (MA) and return the updated dataset with the MA column.

    Args:
        data : Path to the input CSV file.
        period (int): Period for the moving average.

    Returns:
        pd.DataFrame: A DataFrame with the original data and the MA column added.
    """

    # Check if the 'close' column exists
    if "close" not in data.columns:
        raise ValueError(
            "The CSV file does not contain the 'close' column required for MA calculation."
        )

    # Ensure the data is sorted by 'time' column
    data = data.sort_values("time")

    # Calculate the moving average (MA)
    column_name = f"MA_{period}"
    data[column_name] = data["close"].rolling(window=period).mean()

    # Return the updated dataset
    return data


import pandas as pd


def calculate_rsi_in_memory(data, period):
    """
    Calculate Relative Strength Index (RSI) and add it to the dataset.

    Args:
        data (pd.DataFrame): The input dataset containing the 'close' column.
        period (int): The period for RSI calculation (default is 14).

    Returns:
        pd.DataFrame: The dataset with an added 'RSI' column.
    """
    # Calculate price changes
    delta = data["close"].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gains and losses using a rolling window
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss

    # Calculate RSI
    data["RSI"] = 100 - (100 / (1 + rs))

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
    data["EMA_Short"] = data["close"].ewm(span=short_period, adjust=False).mean()
    data["EMA_Long"] = data["close"].ewm(span=long_period, adjust=False).mean()

    data["MACD"] = data["EMA_Short"] - data["EMA_Long"]
    data["Signal_Line"] = data["MACD"].ewm(span=signal_period, adjust=False).mean()
    data["MACD_Histogram"] = data["MACD"] - data["Signal_Line"]

    return data


def calculate_obv_in_memory(data):
    """
    Calculate On-Balance Volume (OBV) and add it to the dataset.

    Args:
        data (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: Updated dataset with OBV column.
    """
    data["OBV"] = 0
    data["OBV"] = (
        data["volume"] * ((data["close"] > data["close"].shift(1)) * 2 - 1)
    ).cumsum()
    return data


def calculate_kdj_in_memory(data, period):
    """
    Calculate Stochastic Oscillator (KDJ) and add K, D, J columns to the dataset.

    Args:
        data (pd.DataFrame): The input dataset.
        period (int): Period for KDJ calculation.

    Returns:
        pd.DataFrame: Updated dataset with K, D, and J columns.
    """
    data["Low_Min"] = data["low"].rolling(window=period).min()
    data["High_Max"] = data["high"].rolling(window=period).max()

    data["RSV"] = (
        (data["close"] - data["Low_Min"]) / (data["High_Max"] - data["Low_Min"]) * 100
    )
    data["K"] = data["RSV"].ewm(alpha=1 / 3, adjust=False).mean()
    data["D"] = data["K"].ewm(alpha=1 / 3, adjust=False).mean()
    data["J"] = 3 * data["K"] - 2 * data["D"]

    return data


def calculate_boll_in_memory(data, period):
    """
    Calculate Bollinger Bands (BOLL) and add upper, middle, and lower bands to the dataset.

    Args:
        data (pd.DataFrame): The input dataset.
        period (int): Period for BOLL calculation.

    Returns:
        pd.DataFrame: Updated dataset with BOLL columns.
    """
    data["BOLL_Middle"] = data["close"].rolling(window=period).mean()
    data["BOLL_Std"] = data["close"].rolling(window=period).std()

    data["BOLL_Upper"] = data["BOLL_Middle"] + 2 * data["BOLL_Std"]
    data["BOLL_Lower"] = data["BOLL_Middle"] - 2 * data["BOLL_Std"]

    return data


def calculate_vwap_in_memory(data):
    """
    Calculate Volume Weighted Average Price (VWAP) and add it to the dataset.

    Args:
        data (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: Updated dataset with VWAP column.
    """
    data["Typical_Price"] = (data["high"] + data["low"] + data["close"]) / 3
    data["VWAP"] = (data["Typical_Price"] * data["volume"]).cumsum() / data[
        "volume"
    ].cumsum()

    return data
