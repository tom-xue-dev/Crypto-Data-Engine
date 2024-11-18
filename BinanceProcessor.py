import requests
from KLinesProcessor import KLinesProcessor

class BinanceProcessor(KLinesProcessor):
    def __init__(self,symbol,interval) -> None:
        super().__init__("binance",symbol,interval)
    
    def _get_klines_data(self, params):
        response = requests.get(self._base_url, headers=self._get_random_headers(),params=params)
        if response.status_code == 200:
            return response.json()
        else:
            self._logger.error(f"请求失败，状态码: {response.status_code}")
            return []
        
    def _get_data_list(self, new_data):
        return new_data



t = BinanceProcessor("BTCUSDT","15m")
t.set_save_times(10)
t.make_csv()

    