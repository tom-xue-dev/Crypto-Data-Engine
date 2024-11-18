from processors.bybit_processor import BybitProcessor


if __name__ == "__main__":
    processor1 = BybitProcessor("BTCUSDT","1m")
    processor1.set_save_times(10)
    processor1.make_csv()