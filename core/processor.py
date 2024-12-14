from core.analyzer import TaskAnalyzer
from helpers.utils import *
from helpers.model import *
from helpers.logger import Logger
from helpers.proxy import ProxyManager
from core.saver import CSVSaver
from core.factory import Factory


class Processor:

    def __init__(self, task: Task) -> None:
        """
        任务处理类，初始化时分析任务，调用用execute来完成任务。
        """
        self.task = task
        # 初始化任务分析器
        self._analyzer = TaskAnalyzer(task, fetch_data_func=self._fetch_data)
        # 初始化数据保存器
        self._saver = CSVSaver(task.info, task.save_params)
        # 初始化数据工厂
        self._factory = Factory(self._fetch_data)

    def execute(self):
        try:
            Logger.critical(f"任务{self.task.name}开始", prefix=self.task.info)
            sub_task = self._analyzer.initialize_sub_task()
            data = self._factory.complete(sub_task)
            self._saver.save(data)
            Logger.critical(f"任务{self.task.name}完成", prefix=self.task.info)
            return True
        except Exception as e:
            Logger.critical(
                f"任务{self.task.name}失败，失败原因：{e}", prefix=self.task.info
            )
            return False 

    def _fetch_data(self, since_ts):
        try:
            self._analyzer.get_ccxt_exchange().proxies = ProxyManager.get_random_proxy()
            data = self._analyzer.fetch_data(since=since_ts)
            return data, DataFlag.NORMAL
        except Exception:
            return None, DataFlag.ERROR
