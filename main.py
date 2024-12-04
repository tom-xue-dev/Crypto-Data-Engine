from processors.bybit_processor import BybitProcessor
from processors.binance_processor import BinanceProcessor


if __name__ == "__main__":
    processor1 = BinanceProcessor('BTCUSDT','15m')
    processor1.make_csv(max_threads=100)
