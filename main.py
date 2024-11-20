import json
import questionary

from processors.binance_processor import BinanceProcessor
from processors.bybit_processor import BybitProcessor
from processors.okx_processor import OkxProcessor


def main():
    with open('url_config.json', 'r') as f:
        url_config = json.load(f)

    exchanges = list(url_config.keys())
    exchange_answer = questionary.select(
        "选择交易所：",
        choices=exchanges
    ).ask()
    selected_exchange = exchange_answer

    symbols = url_config[selected_exchange]['symbol']
    symbol_answer = questionary.select(
        "选择交易对：",
        choices=symbols
    ).ask()
    selected_symbol = symbol_answer

    intervals = list(url_config[selected_exchange]['interval'].keys())
    interval_answer = questionary.select(
        "选择时间间隔：",
        choices=intervals
    ).ask()
    selected_interval = interval_answer

    max_threads_input = questionary.text(
        "输入最大线程数（可选，默认50）:"
    ).ask()

    max_threads = int(max_threads_input) if max_threads_input and max_threads_input.isdigit() else None

    save_times_input = questionary.text(
        "输入每多少次成功请求后保存一次（可选，默认100）"
    ).ask()
    save_times = int(save_times_input) if save_times_input and save_times_input.isdigit() else None

    exchange_processors = {
        'binance': BinanceProcessor,
        'okx': OkxProcessor,
        'bybit': BybitProcessor,
    }

    ProcessorClass = exchange_processors.get(selected_exchange.lower())

    if not ProcessorClass:
        print(f"Processor for exchange '{selected_exchange}' is not available.")
        return

    processor = ProcessorClass(selected_symbol, selected_interval)

    if max_threads is not None and save_times is not None:
        processor.make_csv(max_threads=max_threads, save_times=save_times)
    else:
        processor.make_csv()

if __name__ == "__main__":
    main()

