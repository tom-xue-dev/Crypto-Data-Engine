import sys

import ccxt
import csv
from datetime import datetime,timedelta
import time

# 初始化交易所（以 Binance 为例）
exchange = ccxt.binance({
    'options': {
        'defaultType': 'future'  # 确保加载的是期货市场
    }
})

# 加载市场数据
markets = exchange.load_markets()

# 筛选出所有永续合约的交易对
perpetual_symbols = [
    market['symbol']
    for market in markets.values()
    if market.get('contract') and market.get('swap')
]

# 指定要获取历史资金费率的交易对
for symbol in perpetual_symbols:
    limit = 1000  # 每次请求最多获取的记录数
    filename = f"{symbol.replace('/', '_').replace(':', '_')}_funding_rate_history.csv"

    # 初始化 CSV 文件并写入表头
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Datetime', 'Funding Rate'])

    # 获取历史资金费率数据
    all_funding_rates = []
    end_time = int(datetime.now().timestamp() * 1000)  # 当前时间的时间戳（毫秒）
    start_time =int((datetime.now() - timedelta(days=150)).timestamp() * 1000)

    print(f"开始获取 {symbol} 的历史资金费率数据...")

    while True:
        try:
            # 获取历史资金费率数据
            funding_rate_history = exchange.fetch_funding_rate_history(symbol, limit=limit, params={"startTime":start_time,"endTime": end_time})
            end_time_dt = datetime.fromtimestamp(end_time / 1000)
            new_time_dt = end_time_dt - timedelta(days=150)
            end_time = int(new_time_dt.timestamp() * 1000)

            start_time_dt = datetime.fromtimestamp(start_time / 1000)
            new_start_dt = start_time_dt - timedelta(days=150)
            start_time = int(new_start_dt.timestamp() * 1000)
            if not funding_rate_history:
                print("没有更多数据了。")
                break

            # 处理并保存数据
            with open(filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for rate in reversed(funding_rate_history):
                    timestamp = rate['timestamp']
                    datetime_str = datetime.utcfromtimestamp(rate['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    funding_rate = float(rate['fundingRate'])
                    writer.writerow([timestamp, datetime_str, funding_rate])

            # 更新 `end_time` 为获取到的最早时间戳，向更早的数据查询
            # end_time = funding_rate_history[-1]['timestamp'] - 1  # 避免重复数据
            print(f"已获取数据，最新的结束时间: {datetime.utcfromtimestamp(rate['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
            # 避免请求过于频繁，稍作延时
            time.sleep(0.1)

        except Exception as e:
            print(f"获取数据时发生错误: {e}")
            break

    print(f"数据已保存到 {filename}")



