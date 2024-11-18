import requests
from klines_processor import KLinesProcessor

class BybitProcessor(KLinesProcessor):
    def __init__(self,symbol,interval) -> None:
        super().__init__("bybit",symbol,interval)
    
    def _get_klines_data(self, params):
        response = requests.get(self._base_url, headers=self._get_random_headers(),params=params)
        try:
            if response.status_code == 200:
                data = response.json()
                if data.get("ret_code") == 0:
                    return data.get("result", [])
                else:
                    self._logger.error(f"请求时间点为{KLinesProcessor._timestamp_to_datetime(params["to"])}数据时出现错误（时间戳：{params["to"]}），API返回错误代码: {data.get('ret_code')}, 消息: {data.get('ret_msg')}")
            else:
                self._logger.error(f"请求时间点为{KLinesProcessor._timestamp_to_datetime(params["to"])}数据时出现错误（时间戳：{params["to"]}），请求失败，状态码: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self._logger.error(f"请求时间点为{KLinesProcessor._timestamp_to_datetime(params["to"])}数据时出现错误（时间戳：{params["to"]}），请求异常: {e}")
        
    def _get_data_list(self, new_data):
        return [
            {
                'time': item['t'],
                'open': item['o'],
                'high': item['h'],
                'low': item['l'],
                'close': item['c'],
                'volume': item['v']
            }
            for item in new_data
        ]
