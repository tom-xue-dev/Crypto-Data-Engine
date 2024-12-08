import json
import logging
import os
import random
import subprocess
import threading
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from time import sleep, time

import pandas as pd
import pytz
from fake_useragent import UserAgent
import requests
import yaml


class KLinesProcessor:

    class KlinesDataFlag(Enum):
        NORMAL = 0
        ERROR = 1

    def __init__(self, exchange, type, symbol, interval) -> None:
        self.exchange = exchange
        self.type = type
        self._url_config, self._config = self._load_config()
        self._validate_input(symbol, interval)
        self.symbol = symbol
        self.interval = interval
        self._base_url = self._url_config["base_url"]
        self._params_template = self._url_config["params"]
        self._timestamp_rate = {"params": 1000, "data": 1000}
        timestamp_type = self._url_config.get("timestamp_type")
        if timestamp_type:
            for k, v in timestamp_type.items():
                if v == "ms":
                    self._timestamp_rate[k] = 1000
                else:
                    self._timestamp_rate[k] = 1
        self._processing_rules = self._url_config.get("processing_rules", {})
        self._columns = list(self._processing_rules.get("field_mappings").keys())

        # 标准间隔：见 url_config.json 中 "binance" 的 "interval" 键名，用于文件夹命名、频率获取
        self._standard_interval = (
            self._url_config["interval"][self.interval]
            if self._url_config["interval"][self.interval] is not None
            else self.interval
        )
        self._end_time = None

        self._work_folder = os.path.join(
            "data", self.exchange, self.type, self.symbol, self._standard_interval
        )
        self._log_folder = os.path.join(self._work_folder, "log")
        self._log_file = os.path.join(self._log_folder, "app.log")
        self._tmp_csv_folder = os.path.join(self._work_folder, "tmp")
        self._tmp_csv_file = os.path.join(
            self._tmp_csv_folder, f"{self._standard_interval}.csv"
        )
        self._failed_timestamps_file = os.path.join(
            self._log_folder, "failed_timestamps.txt"
        )

        self._make_dir()
        self._logger = self._get_logger()

        # 最大尝试次数
        self._max_make_csv_attempt_count = self._config["MAX_MAKE_CSV_ATTEMPT_COUNT"]
        # 制作历史数据中允许最大请求失败的时间戳数量，超过数量后将1次请求次数，并减半最大线程数开始重试
        self._allow_max_failed_timestamps = self._config["ALLOW_MAX_FAILED_TIMESTAMPS"]
        # 完成历史数据制作前，允许对请求失败的时间戳再次请求的次数
        self._allow_max_failed_timestamps_attempt_time = self._config[
            "ALLOW_MAX_FAILED_TIMESTAMPS_ATTEMPT_TIME"
        ]
        # 完成历史数据制作前，允许最大缺失时间点数量
        self._allow_max_missing_timestamps = self._config[
            "ALLOW_MAX_MISSING_TIMESTAMPS"
        ]
        # 分割后的CSV最大行数
        self._splitted_csv_max_rows = self._config["SPLITTED_CSV_MAX_ROWS"]
        # 缓存的最大数据量
        self._cached_data_amount = self._config["CACHED_DATA_AMOUNT"]
        # 默认最大线程数
        self._max_threads = self._config["MAX_THREADS"]
        self._enable_proxy = self._config["ENABLE_PROXY"]
        # 设置代理
        self._proxy_manager = None
        proxies_path = os.path.join("config", "proxies.yml")
        if self._enable_proxy and os.path.exists(proxies_path):
            self._proxy_manager = ProxyManager(proxies_path)

    def make_history_data(
        self,
        max_threads=None,
        start_time=None,
        end_time=None,
        drop_first=True,
        replace_missing_time_with_adjacent=True,
        delete_tmp=True,
    ):
        """生成历史数据"""
        splitted_file = os.path.join(self._work_folder, "0.csv")
        if os.path.exists(splitted_file):
            self._logger.error("历史数据已制作完成，无需制作")
            raise RuntimeError("异常终止，请查看日志")
        if self._proxy_manager:
            proxies_amount = self._proxy_manager.start_proxies()
            self._logger.info(f"代理配置成功，共加载 {proxies_amount} 条代理")
        try:
            if max_threads > 0:
                self._max_threads = max_threads
            self._initialize_delta_time()
            self._end_time = end_time
            while True:
                self._logger.info(f"设置最大线程数为 {self._max_threads}")
                self._initialize_attributes()

                if start_time:
                    self._block_timestamp = int(
                        self._datetime_to_timestamp(
                            start_time, rate=self._timestamp_rate["params"]
                        )
                    )
                else:
                    if os.path.exists(self._tmp_csv_file):
                        df = pd.read_csv(self._tmp_csv_file)
                        self._block_timestamp = self._datetime_to_timestamp(
                            df.iloc[-1]["time"], rate=self._timestamp_rate["params"]
                        )
                    else:
                        self._block_timestamp = int(
                            time() * self._timestamp_rate["params"]
                        )
                self._logger.info(
                    f"从 {self._timestamp_to_datetime(self._block_timestamp,rate=self._timestamp_rate['params'])}（时间戳：{self._block_timestamp}）开始获取历史数据..."
                )
                threads = []
                for _ in range(self._max_threads):
                    t = threading.Thread(target=self._worker)
                    t.start()
                    threads.append(t)

                while True:
                    alive_threads = [t for t in threads if t.is_alive()]
                    if not alive_threads:
                        break
                    try:
                        for t in alive_threads:
                            t.join(timeout=0.1)
                    except KeyboardInterrupt:
                        self._logger.warning("手动停止数据收集")
                        self._stop_event.set()
                        for t in threads:
                            t.join()

                if self._is_abnormal_termination:
                    self._max_make_csv_attempt_count -= 1
                    if os.path.exists(self._tmp_csv_file):
                        os.remove(self._tmp_csv_file)
                    if os.path.exists(self._failed_timestamps_file):
                        os.remove(self._failed_timestamps_file)
                    if self._max_make_csv_attempt_count > 0:
                        self._logger.info(
                            f"剩余尝试次数：{self._max_make_csv_attempt_count}"
                        )
                        self._max_threads = max(int(self._max_threads / 2), 1)
                        self._logger.info("等待10秒后开始重试...")
                        sleep(10)
                        self._logger.info(
                            f"重试开始，最大线程数减半为{self._max_threads}"
                        )
                    else:
                        self._logger.error("尝试次数耗尽，终止此次数据获取")
                        raise RuntimeError("异常终止，请查看日志")
                else:
                    self._save_to_csv(self._new_data, self._tmp_csv_file)
                    self._drop_duplicates(self._tmp_csv_file)
                    self._sort_csv(self._tmp_csv_file, ascending=False)

                    if self._data_collected:
                        thread_list = [self._max_threads]
                        while (
                            len(thread_list)
                            < self._allow_max_failed_timestamps_attempt_time
                        ):
                            if self._enable_proxy:
                                next_threads = thread_list[-1]
                            else:
                                next_threads = max(int(thread_list[-1] / 2), 1)
                            thread_list.append(next_threads)
                        for thread_number in thread_list:
                            if self._retry_failed_timestamps(
                                self._tmp_csv_file, thread_number
                            ):
                                break
                        self._drop_duplicates(self._tmp_csv_file)
                        self._sort_csv(self._tmp_csv_file, ascending=False)
                        if os.path.exists(self._failed_timestamps_file):
                            self._logger.error("仍存在请求失败的时间戳")
                            raise RuntimeError("异常终止，请查看日志")
                        if self._fix_csv_data_integrity(
                            self._tmp_csv_file,
                            replace=replace_missing_time_with_adjacent,
                        ):
                            if drop_first:
                                self._drop_first_data(self._tmp_csv_file)
                            self._split_csv()
                        if delete_tmp and os.path.exists(self._tmp_csv_file):
                            os.remove(self._tmp_csv_file)
                            self._logger.info(f"删除{self._tmp_csv_file}成功")
                        self._logger.critical("历史数据已制作完成！")
                    return
        finally:
            # 关闭代理池
            if self._proxy_manager:
                self._proxy_manager.stop_proxies()
                self._logger.info("代理关闭成功")

    def _load_config(self):
        """加载配置文件"""
        url_config_path = os.path.join("config", "url_config.json")
        config_path = os.path.join("config", "config.yml")
        with open(url_config_path, "r", encoding="utf-8") as file:
            url_config = json.load(file)[self.exchange][self.type]
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return url_config, config

    def _validate_input(self, symbol, interval):
        """验证输入的 symbol 和 interval 是否有效"""
        if (
            symbol in self._url_config["symbol"]
            and interval in self._url_config["interval"]
        ):
            return
        raise ValueError(
            f"{self.exchange}_{self.symbol}_{self.interval}：非法的 symbol 或 interval 输入，请查看 url_config.json"
        )

    def _get_logger(self):
        """获取配置好的日志记录器"""
        logger = logging.getLogger(f"{self.exchange}_{self.symbol}_{self.interval}")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:

            # 创建控制台处理器，设置日志级别为 INFO
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 创建文件处理器，设置日志级别为 DEBUG
            file_handler = RotatingFileHandler(
                self._log_file,
                maxBytes=3 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)

            # 创建日志格式
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # 添加处理器到日志记录器
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger

    def _make_dir(self):
        """创建必要的目录结构"""
        os.makedirs(self._work_folder, exist_ok=True)
        os.makedirs(self._log_folder, exist_ok=True)
        tmp_folder = os.path.join(self._work_folder, "tmp")
        os.makedirs(tmp_folder, exist_ok=True)

    def _get_klines_data(self, params):
        """根据配置文件获取 K 线数据"""
        try:
            timestamp_key = next(
                (key for key, val in self._params_template.items() if val == "!ET!"),
                None,
            )
            timestamp = params.get(timestamp_key)
            datetime = self._timestamp_to_datetime(
                timestamp, rate=self._timestamp_rate["params"]
            )
            chosen_proxy = None
            if self._proxy_manager:
                chosen_proxy = self._proxy_manager.get_random_proxy()
            response = requests.get(
                self._base_url,
                headers=self._get_random_headers(),
                params=params,
                proxies=chosen_proxy,
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                response_code_field = self._processing_rules.get("response_code_field")
                success_code = self._processing_rules.get("success_code")
                result_field = self._processing_rules.get("result_field")
                result_field_2 = self._processing_rules.get("result_field_2")

                if response_code_field is not None:
                    if data.get(response_code_field) == success_code:
                        if result_field_2 is not None:
                            result_data = data.get(result_field, {}).get(
                                result_field_2, []
                            )
                        else:
                            result_data = data.get(result_field, [])
                        return result_data, self.KlinesDataFlag.NORMAL
                    else:
                        self._logger.error(
                            f"请求时间点为{datetime}数据时出现错误（时间戳：{timestamp}），"
                            f"API返回错误代码: {data.get(response_code_field)}"
                        )
                        return None, self.KlinesDataFlag.ERROR
                else:
                    # 不需要额外状态码的 API
                    return data, self.KlinesDataFlag.NORMAL
            else:
                self._logger.error(
                    f"请求时间点为{datetime}数据时出现错误（时间戳：{timestamp}），"
                    f"请求失败，状态码: {response.status_code}"
                )
                return None, self.KlinesDataFlag.ERROR
        except requests.exceptions.RequestException as e:
            self._logger.error(
                f"请求时间点为{datetime}数据时出现错误（时间戳：{timestamp}），请求异常: {e}"
            )
            return None, self.KlinesDataFlag.ERROR

    def _initialize_attributes(self):
        """初始化属性以准备数据收集"""
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._new_data = []
        self._timer = 0
        self._data_collected = False  # 指示是否已收集完所有数据
        self._is_abnormal_termination = False

    def _worker(self):
        """工作线程，负责获取数据并处理"""
        while not self._stop_event.is_set():
            block_timestamp = self._get_next_block_timestamp()
            if block_timestamp is None:
                break
            block_datetime = self._timestamp_to_datetime(
                block_timestamp, rate=self._timestamp_rate["params"]
            )
            name = threading.current_thread().name
            params = self._make_params(block_timestamp)
            data, flag = self._get_klines_data(params)
            if flag == self.KlinesDataFlag.NORMAL:
                if data:
                    self._logger.info(f"线程 {name} 获取到数据，time: {block_datetime}")
                    with self._lock:
                        self._new_data.extend(data)
                        self._timer += 1
                        if self._timer >= self._cached_data_amount:
                            self._save_to_csv(self._new_data, self._tmp_csv_file)
                            self._new_data = []
                            self._timer = 0
                else:
                    self._logger.warning(
                        f"线程 {name} 没有获取到数据，time: {block_datetime}"
                    )
                    if not self._end_time:
                        with self._lock:
                            self._data_collected = True
                        break
            else:
                self._logger.warning(
                    f"线程 {name} 获取数据时发生错误，time: {block_datetime}"
                )
                with self._lock:
                    self._save_failed_timestamp(block_timestamp)

    def _save_failed_timestamp(self, timestamp):
        """保存获取失败的时间戳"""
        with open(self._failed_timestamps_file, "a") as f:
            f.write(f"{timestamp}\n")
        datetime = self._timestamp_to_datetime(
            timestamp, rate=self._timestamp_rate["params"]
        )
        self._logger.info(
            f"保存失败的时间：{datetime}（时间戳：{timestamp}）到 {self._failed_timestamps_file}"
        )
        with open(self._failed_timestamps_file, "r") as f:
            lines = f.readlines()
        if not self._enable_proxy and len(lines) > self._allow_max_failed_timestamps:
            # self._stop_event.set()
            # self._is_abnormal_termination = True
            sleep(10)
            self._allow_max_failed_timestamps *= 2

    def _retry_failed_timestamps(self, file_name, thread_number):
        """重试获取失败的时间戳数据"""
        import queue

        if not os.path.exists(self._failed_timestamps_file):
            self._logger.info("没有失败的时间戳需要重试")
            return True

        with open(self._failed_timestamps_file, "r") as f:
            timestamps = f.read().splitlines()
        timestamps = list(set(int(ts) for ts in timestamps))
        if not timestamps:
            self._logger.info("没有失败的时间戳需要重试")
            if os.path.exists(self._failed_timestamps_file):
                os.remove(self._failed_timestamps_file)
            return True

        self._logger.info("开始重试失败的时间戳请求...")
        timestamp_queue = queue.Queue()
        for ts in timestamps:
            timestamp_queue.put(ts)

        self._logger.info(f"使用线程数: {thread_number} 进行重试")
        file_lock = threading.Lock()

        def retry_timestamp():
            while True:
                try:
                    ts = timestamp_queue.get_nowait()
                except queue.Empty:
                    break
                params = self._make_params(ts)
                data, flag = self._get_klines_data(params)
                dt = self._timestamp_to_datetime(
                    ts, rate=self._timestamp_rate["params"]
                )
                if flag == self.KlinesDataFlag.NORMAL:
                    self._logger.info(f"重试成功，时间: {dt}（时间戳：{ts}）")
                    with file_lock:
                        self._save_to_csv(data, file_name)
                        self._remove_failed_timestamp(ts)
                else:
                    self._logger.info(f"重试失败，时间: {dt}（时间戳：{ts}）")
                timestamp_queue.task_done()

        threads = []
        for _ in range(thread_number):
            t = threading.Thread(target=retry_timestamp)
            threads.append(t)
            t.start()

        timestamp_queue.join()
        for t in threads:
            t.join()

    def _remove_failed_timestamp(self, timestamp):
        """移除已成功获取数据的失败时间戳"""
        if not os.path.exists(self._failed_timestamps_file):
            return
        with open(self._failed_timestamps_file, "r") as f:
            lines = f.readlines()
        with open(self._failed_timestamps_file, "w") as f:
            for line in lines:
                if int(line.strip()) != timestamp:
                    f.write(line)

    def _initialize_delta_time(self, start_time=None):
        mapping = {
            "1s": 1,
            "1m": 60,
            "2m": 120,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400,
            "3d": 259200,
            "1w": 604800,
            "1M": 2592000,
        }
        if start_time:
            start_timestamp = self._datetime_to_timestamp(
                start_time, self._timestamp_rate["params"]
            )
        else:
            start_timestamp = time()
        test_timestamp = int(
            (start_timestamp - mapping[self._standard_interval])
            * self._timestamp_rate["params"]
        )
        params = self._make_params(test_timestamp)
        test_raw_data, flag = self._get_klines_data(params)
        if (
            flag == self.KlinesDataFlag.NORMAL
            and test_raw_data is not None
            and len(test_raw_data) > 0
        ):
            test_data = self._process_data(test_raw_data)
            delta_time = int(
                abs(int(test_data[0].get("time")) - int(test_data[-1].get("time")))
                - mapping[self._standard_interval] * self._timestamp_rate["data"]
            )
            if self._timestamp_rate["data"] > self._timestamp_rate["params"]:
                delta_time = int(delta_time / 1000)
            elif self._timestamp_rate["data"] < self._timestamp_rate["params"]:
                delta_time = int(delta_time * 1000)
            self._logger.info(f"设置delta_time为 {delta_time} ")
            self._delta_timestamp = delta_time
        else:
            self._logger.error("设置delta_time时出现错误")
            raise RuntimeError("异常终止，请查看日志")

    def _get_next_block_timestamp(self):
        """获取下一个数据块的时间戳"""
        with self._lock:
            if (
                self._data_collected
                or self._block_timestamp < self._timestamp_rate["params"] * 1230000000
                or (
                    self._end_time
                    and self._block_timestamp
                    < self._datetime_to_timestamp(
                        self._end_time, self._timestamp_rate["params"]
                    )
                )
            ):
                self._data_collected = True
                return None
            current_block_timestamp = self._block_timestamp
            self._block_timestamp -= self._delta_timestamp
            return current_block_timestamp

    def _make_params(self, end_timestamp):
        """根据模板生成请求参数"""
        template = self._params_template.copy()
        params = {}
        for k, v in template.items():
            if v == "!ET!":
                params[k] = end_timestamp
            elif v == "!S!":
                params[k] = self.symbol
            elif v == "!I!":
                params[k] = self.interval
            else:
                params[k] = v
        return params

    def _split_csv(self):
        """拆分 CSV 文件为多个小文件"""
        try:
            df = pd.read_csv(self._tmp_csv_file)
            df.sort_values(by="time", ascending=True, inplace=True)
        except Exception as e:
            self._logger.error(f"读取文件时发生错误: {e}")
            return

        chunk_size = self._splitted_csv_max_rows
        file_count = 0
        # 按块大小拆分 DataFrame
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            output_file = os.path.join(self._work_folder, f"{file_count}.csv")
            try:
                chunk.to_csv(output_file, index=False, header=self._columns)
                self._logger.info(f"已创建文件: {output_file}，包含 {len(chunk)} 行。")
                file_count += 1
            except Exception as e:
                self._logger.error(f"写入文件 {output_file} 时发生错误: {e}")
                continue
        self._logger.critical(f"拆分完成，总共创建了 {file_count} 个文件。")

    def _save_to_csv(self, new_data, file_name, transfer_time=True, online_data=True):
        """将新数据保存到 CSV 文件"""
        if not new_data:
            return
        if online_data:
            data_list = self._process_data(new_data)
        else:
            data_list = new_data
        df = pd.DataFrame(data_list, columns=self._columns)
        if transfer_time:
            df["time"] = df["time"].apply(
                self._timestamp_to_datetime, args=(self._timestamp_rate["data"],)
            )
        df.to_csv(
            file_name, mode="a", header=not os.path.exists(file_name), index=False
        )
        self._logger.info(f"数据保存到 {file_name}")

    def _process_data(self, new_data):
        """根据配置文件处理新获取的数据"""
        field_mappings = self._processing_rules.get("field_mappings", {})
        processed_data = []
        for item in new_data:
            processed_item = {}
            for csv_field, data_field in field_mappings.items():
                if isinstance(data_field, int):
                    processed_item[csv_field] = item[data_field]
                else:
                    processed_item[csv_field] = item.get(data_field, None)
            processed_data.append(processed_item)
        return processed_data

    def _get_missing_times_list(self, file_name):
        """获取缺失的时间点列表"""
        try:
            self._logger.info(f"检查文件 {file_name} 中的缺失数据...")
            df_time = pd.read_csv(file_name, usecols=["time"])
            df = pd.read_csv(file_name)
            df_time["time"] = pd.to_datetime(df_time["time"], errors="coerce")
            min_time = df_time["time"].min()
            max_time = df_time["time"].max()
            expected_times = pd.date_range(
                start=min_time, end=max_time, freq=self._get_freq()
            )
            existing_times = pd.Series(df_time["time"].unique())
            missing_times = expected_times.difference(existing_times)

            if missing_times.empty:
                self._logger.info("所有预期的时间点数据均存在")
                return None, df, df_time

            self._logger.warning(f"共缺失 {len(missing_times)} 条数据")
            return missing_times, df, df_time
        except Exception as e:
            self._logger.error(f"检查过程中发生错误：{e}")
            return None, None, None

    def _get_freq(self):
        """根据标准间隔获取 pandas 频率字符串"""
        unit = self._standard_interval[-1]
        if unit == "m":
            return f"{self._standard_interval[:-1]}min"  # 分钟
        elif unit == "h":
            return f"{self._standard_interval[:-1]}H"  # 小时
        elif unit == "d":
            return f"{self._standard_interval[:-1]}D"  # 天
        elif unit == "w":
            return f"{self._standard_interval[:-1]}W"  # 周
        elif unit == "M":
            return f"{self._standard_interval[:-1]}M"  # 月
        else:
            raise ValueError("不支持的格式")

    def _fix_csv_data_integrity(self, file_name, replace=True):
        """修复 CSV 数据的完整性，填补缺失的数据点"""
        missing_times, df, df_time = self._get_missing_times_list(file_name)
        new_data = []
        if df is None or df_time is None:
            return False
        if missing_times is None:
            return True
        missing_times_file = os.path.join(self._log_folder, "missing_times.txt")
        if not os.path.exists(missing_times_file):
            for missing_time in missing_times:
                with open(missing_times_file, "a") as f:
                    f.write(f"{missing_time}\n")
        if replace:
            if len(missing_times) > self._allow_max_missing_timestamps:
                self._logger.error("过多的缺失时间点，停止后续操作！")
                raise RuntimeError("异常终止，请查看日志")
            for missing_time in missing_times:
                closest_time_idx = (df_time["time"] - missing_time).abs().idxmin()
                closest_data = list(df.iloc[closest_time_idx])
                origin_time = closest_data[0]
                closest_data[0] = missing_time
                new_data.append(closest_data)
                self._logger.warning(
                    f"使用 {origin_time} 的数据代替缺失的时间 {missing_time} 的数据"
                )

            self._save_to_csv(
                new_data, file_name, transfer_time=False, online_data=False
            )
            self._sort_csv(file_name)
            self._drop_duplicates(file_name)
        return True

    def _sort_csv(self, file_name, ascending=True):
        """对 CSV 文件中的数据进行排序"""
        df = pd.read_csv(file_name)
        df.sort_values(by="time", ascending=ascending, inplace=True)
        df.to_csv(file_name, index=False)
        self._logger.info(f"对 {file_name} 中的数据完成排序")

    def _drop_duplicates(self, file_name):
        """去除 CSV 文件中的重复数据"""
        df = pd.read_csv(file_name)
        df.drop_duplicates(subset=["time"], keep="first", inplace=True)
        df.to_csv(file_name, index=False)
        self._logger.info(f"对 {file_name} 中的数据完成去重")

    def _drop_first_data(self, file_name):
        df = pd.read_csv(file_name)
        df = df.drop(index=0)
        df.to_csv(file_name, index=False)
        self._logger.info(f"删除 {file_name} 中第一条数据")

    def _timestamp_to_datetime(self, timestamp, rate=1):
        """
        将时间戳（毫秒）转换为系统时间
        :param timestamp: 时间戳（毫秒）
        :return: 转换后的系统时间字符串
        """
        timestamp = int(int(timestamp) / rate)
        dt_object = datetime.utcfromtimestamp(timestamp)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S")  # 格式化为“年-月-日 时:分:秒”

    def _datetime_to_timestamp(self, datetime_str, rate=1):
        """
        将格式化后的 datetime 字符串转换为时间戳（毫秒）
        :param datetime_str: 格式化的 datetime 字符串，例如 '2024-11-15 14:30:00'
        :return: 对应的时间戳（毫秒）
        """
        datetime_format = "%Y-%m-%d %H:%M:%S"
        utc_tz = pytz.timezone("UTC")
        dt_object = datetime.strptime(datetime_str, datetime_format)
        dt_object = utc_tz.localize(dt_object)
        timestamp = int(dt_object.timestamp() * rate)
        return timestamp

    @staticmethod
    def _get_random_headers():
        """
        返回一个包含随机 User-Agent 的请求头
        """
        ua = UserAgent()
        headers = {
            "User-Agent": ua.random,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
        return headers


class ProxyManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.processes = []
        self.proxies = []  # 存储代理信息，用于requests使用
        self.bin_path = None

    def _load_config(self):
        with open(self.config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.bin_path = config.get("bin_path")
        nodes = config.get("proxies", [])
        base_port = config.get("base_port")
        commands = []
        for i, node in enumerate(nodes):
            method = node.get("cipher")
            password = node.get("password")
            server = node.get("server")
            port = node.get("port")
            local_port = base_port + i
            # ss-local 启动参数示例：
            # ss-local -s server -p server_port -l local_port -k password -m method
            cmd = [
                self.bin_path,
                "-s",
                str(server),
                "-p",
                str(port),
                "-l",
                str(local_port),
                "-k",
                str(password),
                "-m",
                str(method),
                "--fast-open",  # 可选参数，根据需要添加
            ]
            commands.append((cmd, local_port))
        return commands

    def start_proxies(self):
        """启动所有代理进程"""
        commands = self._load_config()
        for cmd, local_port in commands:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(p)
            # 为requests配置socks5代理
            proxy = {
                "http": f"socks5h://127.0.0.1:{local_port}",
                "https": f"socks5h://127.0.0.1:{local_port}",
            }
            self.proxies.append(proxy)
        return len(commands)

    def stop_proxies(self):
        """停止所有代理进程"""
        for p in self.processes:
            # 优雅结束进程
            p.terminate()

        # 等待进程结束
        for p in self.processes:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

        self.processes.clear()
        self.proxies.clear()

    def get_random_proxy(self):
        """随机获取一个代理"""
        if not self.proxies:
            return None
        return random.choice(self.proxies)
