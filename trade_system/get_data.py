import ccxt
import time
import traceback
import concurrent.futures


def create_exchange(exchange_id, api_key=None, secret=None, password=None):
    """
    创建指定交易所实例的工具函数。
    如果只获取行情数据，api_key等可不填或填写假值；若要真实下单，需要正确的apiKey和secret。
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "apiKey": api_key or "",
            "secret": secret or "",
            "password": password or "",
            "enableRateLimit": True,  # 避免访问过快被限流
        })
        return exchange
    except Exception as e:
        print(f"创建交易所实例失败: {exchange_id}, 原因：{str(e)}")
        return None


def get_best_bid_ask(exchange, symbol):
    """
    获取某个交易所该交易对的 best bid (最高买价) 和 best ask (最低卖价)。
    """
    try:
        order_book = exchange.fetch_order_book(symbol)
        best_bid = order_book['bids'][0][0] if order_book['bids'] else None
        best_ask = order_book['asks'][0][0] if order_book['asks'] else None
        return best_bid, best_ask
    except Exception as e:
        # 可能是交易对不存在或网络异常
        print(f"{exchange.id} 获取 {symbol} orderbook 失败, 原因: {str(e)}")
        return None, None


# 仅示例，需结合你的具体需求调整
def fetch_symbol_arbitrage_batch(exchange, symbol_list):
    """
    一次批量获取多个symbol的行情信息，
    并返回 { symbol: (best_bid, best_ask) } 的字典
    """
    try:
        # 如果你的交易所和 ccxt 版本支持 fetch_tickers(symbol_list)
        tickers_data = exchange.fetch_tickers(symbol_list)
        # 你也可以先 exchange.fetch_tickers() 不传参，拿到所有，然后再从里面挑
    except Exception as e:
        print(f"{exchange.id} 批量获取行情失败: {e}")
        return {}

    result = {}
    for sym in symbol_list:
        if sym in tickers_data:
            ticker = tickers_data[sym]
            # ccxt的 ticker 数据结构通常包含 bid / ask / ...
            best_bid = ticker.get('bid')
            best_ask = ticker.get('ask')
            # 如果没有 bid / ask 则说明行情不完整
            if best_bid is not None and best_ask is not None:
                result[sym] = (best_bid, best_ask)

    return result


def cross_exchange_arbitrage_test(exchange_ids, symbols):
    # 创建交易所实例
    exchanges = [create_exchange(eid) for eid in exchange_ids if create_exchange(eid) is not None]
    if not exchanges:
        print("无法创建任何可用交易所实例。")
        return

    while True:
        # 对于每个交易所，用批量接口一次性获取全部符号的行情
        all_quotes = {}  # 形如 { exchange_id: { symbol: (bid, ask), ... }, ... }
        for ex in exchanges:
            symbol_arbitrage_data = fetch_symbol_arbitrage_batch(ex, symbols)
            all_quotes[ex.id] = symbol_arbitrage_data

        # 处理 all_quotes，类似地找到最高买价、最低卖价
        # 然后计算可能的套利机会
        best_arbs = []
        for symbol in symbols:
            # 收集各交易所在这个 symbol 上的 (bid, ask)
            quotes_for_symbol = []
            for ex in exchanges:
                if symbol in all_quotes[ex.id]:
                    bid, ask = all_quotes[ex.id][symbol]
                    quotes_for_symbol.append((ex.id, bid, ask))
            if len(quotes_for_symbol) < 2:
                continue

            # 找最高 bid / 最低 ask
            hbe, highest_bid, _ = max(quotes_for_symbol, key=lambda x: x[1])
            lae, _, lowest_ask = min(quotes_for_symbol, key=lambda x: x[2])
            potential_profit = highest_bid - lowest_ask
            fee_rate = 0.0007
            net_profit_rate = (potential_profit - (lowest_ask * fee_rate + highest_bid * fee_rate)) / (
                        lowest_ask + highest_bid)
            best_arbs.append({
                'symbol': symbol,
                'highest_bid_exchange': hbe,
                'lowest_ask_exchange': lae,
                'highest_bid': highest_bid,
                'lowest_ask': lowest_ask,
                'net_profit_rate': net_profit_rate
            })

        # 在这一轮计算完以后，找 net_profit_rate 最大者
        if not best_arbs:
            print("无足够报价数据，等待下一轮...")
            time.sleep(1)
            continue

        best_opportunity = max(best_arbs, key=lambda x: x['net_profit_rate'])
        if best_opportunity['net_profit_rate'] > 0:
            print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()), end=' ')
            print(f"存在潜在套利机会: {best_opportunity}")
            print("-" * 60)
        else:
            print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()), end=' ')
            print("无套利机会")
        time.sleep(0.05)


if __name__ == "__main__":
    # 示例：要监控的交易所
    exchange_ids = ['binance', 'okx', 'bitget', 'bybit']

    # 示例性的交易对列表，需要根据实际情况验证交易所是否支持
    symbols_top100 = ['POPCAT/USDT:USDT', 'FARTCOIN/USDT:USDT', 'SWELL/USDT:USDT', 'HMSTR/USDT:USDT',
                      'STX/USDT:USDT', 'LRC/USDT:USDT', 'JTO/USDT:USDT', 'PIPPIN/USDT:USDT', 'ALCH/USDT:USDT',
                      'ARC/USDT:USDT', 'KAITO/USDT:USDT', 'APT/USDT:USDT',
                      'POL/USDT:USDT', 'THETA/USDT:USDT', 'ALGO/USDT:USDT', 'BERA/USDT:USDT',
                      'ADA/USDT:USDT', 'W/USDT:USDT', 'OM/USDT:USDT', 'AEVO/USDT:USDT', 'FIL/USDT:USDT',
                      'BNB/USDT:USDT', 'VINE/USDT:USDT', 'ZETA/USDT:USDT', 'UNI/USDT:USDT', 'CFX/USDT:USDT',
                      'BTC/USDT:USDT', 'USTC/USDT:USDT', 'ACT/USDT:USDT', 'DYDX/USDT:USDT', 'UXLINK/USDT:USDT',
                      'ARB/USDT:USDT', 'ATOM/USDT:USDT', 'ONDO/USDT:USDT', 'SUSHI/USDT:USDT', 'INJ/USDT:USDT',
                      'GOAT/USDT:USDT', 'OP/USDT:USDT', 'MKR/USDT:USDT', 'LINK/USDT:USDT', 'PNUT/USDT:USDT',
                      '1INCH/USDT:USDT', 'LPT/USDT:USDT', 'MEME/USDT:USDT', 'AXS/USDT:USDT', 'STRK/USDT:USDT',
                      'BIO/USDT:USDT', 'VIRTUAL/USDT:USDT', 'TNSR/USDT:USDT', 'AAVE/USDT:USDT', 'TRX/USDT:USDT',
                      'BCH/USDT:USDT', 'ETHFI/USDT:USDT', 'SOLV/USDT:USDT', 'WOO/USDT:USDT', 'ME/USDT:USDT',
                      'KSM/USDT:USDT', 'LTC/USDT:USDT', 'ZEREBRO/USDT:USDT', 'MINA/USDT:USDT', 'PEOPLE/USDT:USDT',
                      'S/USDT:USDT', 'BIGTIME/USDT:USDT', 'SHELL/USDT:USDT', 'TAO/USDT:USDT', 'XLM/USDT:USDT',
                      'JUP/USDT:USDT', 'WLD/USDT:USDT', 'SOL/USDT:USDT', 'SONIC/USDT:USDT', 'AI16Z/USDT:USDT',
                      'TRB/USDT:USDT', 'IMX/USDT:USDT', 'EIGEN/USDT:USDT', 'SNX/USDT:USDT', 'IP/USDT:USDT',
                      'LDO/USDT:USDT', 'SSV/USDT:USDT', 'DOT/USDT:USDT', 'VANA/USDT:USDT',
                      'NOT/USDT:USDT', 'EOS/USDT:USDT', 'COMP/USDT:USDT', 'SUI/USDT:USDT', 'DOGS/USDT:USDT',
                      'RDNT/USDT:USDT', 'ENS/USDT:USDT', 'BOME/USDT:USDT', 'HBAR/USDT:USDT', 'PENGU/USDT:USDT',
                      'ZK/USDT:USDT', 'GRASS/USDT:USDT', 'APE/USDT:USDT', 'AVAX/USDT:USDT',
                      'CRV/USDT:USDT', 'SWARMS/USDT:USDT', 'BSV/USDT:USDT', 'ORDI/USDT:USDT', 'ZRO/USDT:USDT',
                      'TIA/USDT:USDT', 'NEAR/USDT:USDT', 'ETC/USDT:USDT', 'MEW/USDT:USDT',
                      'ETH/USDT:USDT', 'TRUMP/USDT:USDT', 'MOODENG/USDT:USDT', 'NEIROETH/USDT:USDT', 'PYTH/USDT:USDT',
                      'CHZ/USDT:USDT', 'SLERF/USDT:USDT', 'MASK/USDT:USDT', 'SAND/USDT:USDT',
                      'DOGE/USDT:USDT', 'XRP/USDT:USDT', 'ICP/USDT:USDT', 'AIXBT/USDT:USDT', 'MOVE/USDT:USDT',
                      'TON/USDT:USDT', 'MANA/USDT:USDT', 'ANIME/USDT:USDT', 'CATI/USDT:USDT', 'GMT/USDT:USDT',
                      'SCR/USDT:USDT', 'GRT/USDT:USDT', 'GALA/USDT:USDT', 'YGG/USDT:USDT', 'WIF/USDT:USDT',
                      'BLUR/USDT:USDT']

    cross_exchange_arbitrage_test(exchange_ids, symbols_top100)
