from functools import partial
from time import time
import ccxt

from core.factory import Factory
from core.saver import CSVSaver
from helpers.config import Config
from helpers.logger import Logger
from helpers.model import BasicInfo
from helpers.proxy import ProxyManager
from helpers.func_storage import *
from helpers.utils import safe_load


class CCXTExchangeWrapper:

    def __init__(self, exchange_obj: ccxt.Exchange):
        self.exchange_obj = exchange_obj

    def fetch_all_ohlcv(self, symbol: str, timeframe, **kwargs):
        """
        fetch_ohlcv的包装方法

        kwargs：
            - `since` (`int`): 开始时间，默认值存储在配置文件的 `DEFAULT_SINCE_TIME`。
            - `util` (`int`): 结束时间，默认值为当前时间。
            - `thread_count` (`int`): 线程数，默认值存储在配置文件的 `GLOBAL_THREADS`。
            - `drop_last` (`bool`): 是否去除最后一根蜡烛，默认值为 `True`。
            - `fix_integrity` (`bool`): 是否对缺失数据进行补全，默认值为 `True`。
            - `save_missing_times` (`bool`): 是否保存缺失时间点到文本，默认值为 `True`。
        """
        info = BasicInfo(
            self.exchange_obj.id,
            self.exchange_obj.options.get("defaultType"),
            symbol.replace("/", "-"),
            timeframe,
        )
        # 需要since参数
        fetch_func = self._ccxt_fetch_func_wrapper(
            partial(self.exchange_obj.fetch_ohlcv, symbol=symbol, timeframe=timeframe)
        )
        init_func = partial(
            make_sub_task,
            since=safe_load(
                kwargs.get("since"),
                default=Config("DEFAULT_SINCE_TIME"),
            ),
            until=safe_load(kwargs.get("until"), default=int(time() * 1000)),
            thread_count=safe_load(
                kwargs.get("thread_count"), default=Config("GLOABLE_THREADS")
            ),
            fetch_data_func=fetch_func,
        )
        # TODO
        parse_func = parse_ohlcv
        if kwargs.get("columns") is None:
            kwargs["columns"] = ["time", "open", "high", "low", "close"]
        kwargs["timeframe"] = timeframe
        saver = CSVSaver(info, kwargs)
        factory = Factory(fetch_func)
        self._execute(info, init_func, saver, factory, parse_func)

    def fetch_all_funding_rate_history(self, symbol: str, **kwargs):
        """
        fetch_funding_rate_history的包装方法

        kwargs：
            - `thread_count` (`int`): 线程数，默认值存储在配置文件的 `GLOBAL_THREADS`。
            - `drop_last` (`bool`): 是否去除最后一根蜡烛，默认值为 `True`。
        """
        info = BasicInfo(
            self.exchange_obj.id,
            "futures",
            symbol.replace("/", "-"),
            "funding_rate",
        )
        # 需要since参数
        fetch_func = self._ccxt_fetch_func_wrapper(
            partial(self.exchange_obj.fetch_funding_rate_history, symbol=symbol)
        )
        parse_func = parse_funding_rate
        init_func = partial(
            make_sub_task,
            since=Config("DEFAULT_SINCE_TIME"),
            until=int(time() * 1000),
            thread_count=safe_load(
                kwargs.get("thread_count"), default=Config("GLOABLE_THREADS")
            ),
            fetch_data_func=fetch_func,
            parse_func=parse_func,
        )
        kwargs["columns"] = ["time", "rate"]
        kwargs["save_mode"] = "MODE_1"
        saver = CSVSaver(info, kwargs)
        factory = Factory(fetch_func)
        self._execute(info, init_func, saver, factory, parse_func)

    def _ccxt_fetch_func_wrapper(self, ccxt_fetch_func):
        def wrapped(*args, **kwargs):
            try:
                self.exchange_obj.proxies = ProxyManager.get_random_proxy()
                return ccxt_fetch_func(*args, **kwargs), DataFlag.NORMAL
            except Exception:
                return None, DataFlag.ERROR

        return wrapped

    @staticmethod
    def _execute(info, init_func, saver: CSVSaver, factory: Factory, parse_func=None):
        try:
            Logger.info("任务开始", prefix=info)
            sub_task = init_func()
            raw_data = factory.complete(sub_task)
            data = parse_func(raw_data) if callable(parse_func) else raw_data
            saver.save(data)
            Logger.critical("任务完成", prefix=info)
            return True
        except Exception as e:
            Logger.error(f"任务失败，原因：{e}", prefix=info)
            return False
