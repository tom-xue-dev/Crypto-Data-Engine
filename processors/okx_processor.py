import requests
from klines_processor import KLinesProcessor

class OkxProcessor(KLinesProcessor):
    def __init__(self,symbol,interval) -> None:
        super().__init__("okx",symbol,interval)
    
    def _get_klines_data(self, params):
        try:
            response = requests.get(self._base_url, headers=self._get_random_headers(),params=params)
            time = params["after"]
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == '0':
                    return data.get("data", []), KLinesProcessor.KlinesDataFlag.NORMAL
                self._logger.error(f"请求时间点为{self._timestamp_to_datetime(time)}数据时出现错误（时间戳：{time}），API返回错误代码: {data.get('ret_code')}, 消息: {data.get('ret_msg')}")
                return None, KLinesProcessor.KlinesDataFlag.ERROR
            self._logger.error(f"请求时间点为{self._timestamp_to_datetime(time)}数据时出现错误（时间戳：{time}），请求失败，状态码: {response.status_code}")
            return None, KLinesProcessor.KlinesDataFlag.ERROR
        except requests.exceptions.RequestException as e:
            self._logger.error(f"请求时间点为{self._timestamp_to_datetime(time)}数据时出现错误（时间戳：{time}），请求异常: {e}")
            return None, KLinesProcessor.KlinesDataFlag.ERROR
        
    def _get_data_list(self, new_data):
        return new_data