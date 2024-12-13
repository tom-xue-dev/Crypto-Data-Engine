from helpers.model import BasicInfo, Task
from helpers.proxy import ProxyManager
from core.processor import Processor


def main():
    ProxyManager.start_proxies()
    task = Task(
        BasicInfo("binance", "spot", "BTC/USDT", "15m"), "fetch_all_ohlcv", {}, {}
    )
    processor = Processor(task)
    processor.execute()
    ProxyManager.stop_proxies()


if __name__ == "__main__":
    main()
