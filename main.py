import json
import questionary

from klines_processor import KLinesProcessor

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

    processor = KLinesProcessor(selected_exchange, selected_symbol, selected_interval)

    mode_answer = questionary.select(
        "选择模式",
        choices=["制作历史数据","更新数据"]
    ).ask()

    if mode_answer == "制作历史数据":

        max_threads_input = questionary.text(
            "输入最大线程数:"
        ).ask()

        max_threads = int(max_threads_input) if max_threads_input and max_threads_input.isdigit() else 0


        processor.make_history_data(max_threads=max_threads)
    
    else:
        print("暂不支持更新模式")
    

if __name__ == "__main__":
    main()

