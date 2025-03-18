import yfinance as yf
import pandas as pd
import pickle

tickers = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOG", "META", "TSLA", "AVGO", "TSM",
    "WMT", "LLY", "JPM", "V", "MA", "UNH", "JNJ", "COST",
    "XOM", "NFLX", "HD", "PG", "BAC", "NVO", "SAP", "ABBV", "VZ", "DIS",
    "CRM", "PFE", "INTC", "CSCO", "IBM", "NKE", "MCD", "ADBE", "CMCSA", "PEP",
    "KO", "MRK", "CVX", "AMGN", "QCOM", "TXN", "SBUX", "PYPL", "GS", "MS",
    "UPS", "RTX", "BMY", "NEE", "LMT", "CAT", "MMM", "DHR", "HON", "GE",
    "CVS", "GILD", "PM", "GM", "F", "DAL", "AAL", "DOW", "FCX", "KMI",
    "OXY", "MPC", "TJX", "LOW", "TGT", "GD", "NOC", "DE", "CMI", "PH",
    "ROK", "COP", "HAL", "SLB", "EOG", "TSN", "ADM", "KHC", "MDLZ", "CL"
]

data = yf.download(tickers, period='max')
# 假设 df 为你的原始 DataFrame
df_stacked = data.stack(level='Ticker')
df_stacked = df_stacked.rename_axis(index={'Ticker': "asset", 'Date': "time"})
# df_stacked = df_stacked.reset_index(level='asset')
df_reset = df_stacked.reset_index()

# 对 time 列进行转换，比如只保留日期部分
df_reset['time'] = pd.to_datetime(df_reset['time']).dt.strftime('%Y-%m-%d')

# 重新将转换后的列设为 MultiIndex
df_stacked = df_reset.set_index(['time', 'asset'])
# print(df_stacked)
df_stacked.columns = df_stacked.columns.str.lower()
df = df_stacked.drop(columns='close')
df = df.rename(columns={'adj close': 'close'})
print(df)
with open('us_share.pkl', 'wb') as f:
    pickle.dump(df_stacked, f)