import ccxt

# 交易所列表
exchange_ids = ['binance', 'okx', 'bitget','bybit','huobi']

# 存储每个交易所的永续合约交易对
swap_markets = {}

# 遍历交易所
for exchange_id in exchange_ids:
    try:
        # 创建交易所实例
        exchange = getattr(ccxt, exchange_id)()

        # 加载市场数据
        markets = exchange.load_markets()

        # 获取当前交易所的永续合约交易对
        swap_pairs = {symbol for symbol, market in markets.items() if market.get('type') == 'swap'}

        # 存储到字典
        swap_markets[exchange_id] = swap_pairs
        print(f"{exchange_id} 支持的永续合约数量: {len(swap_pairs)}")

    except Exception as e:
        print(f"交易所 {exchange_id} 发生错误: {e}")

# 计算所有交易所共有的永续合约交易对
common_swap_pairs = set.intersection(*swap_markets.values())

# 输出共有的永续合约
print(f"\n所有交易所都支持的永续合约交易对: {common_swap_pairs}")
