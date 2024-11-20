from datetime import datetime
import json
from logging.handlers import RotatingFileHandler
from time import sleep, time
import os
import pytz
from enum import Enum
import pandas as pd
from abc import abstractmethod
from fake_useragent import UserAgent
import logging
import threading


class KLinesProcessor:
    CURRENT_TIMESTAMP = int(time() * 1000)
    class DataMode(Enum):
        CREATE = 0
        CONTINUE = 1  # 收集历史数据
        UPDATE = 2  # 更新数据

    class KlinesDataFlag(Enum):
        NORMAL = 0
        ERROR = 1

    def __init__(self, name, symbol, interval) -> None:
        self.name = name
        self.config = self._load_config()
        self._validate_input(symbol, interval)
        self.symbol = symbol
        self.interval = interval
        self._interval_mapping = self.config["interval_mapping"] if "interval_mapping" in self.config else None
        self._base_url = self.config["base_url"]
        self._params_template = self.config["params"]
        self._delta_time = self.config["interval"][self.interval]
        self._columns = self.config["columns"]
        self._timestamp_rate = 1000 if self.config["timestamp_type"] == "ms" else 1

        # 标准间隔：见url_config.json 中 "binance" 的 "interval" 键名，用于文件夹命名、频率获取
        self._standard_interval = self._interval_mapping[self.interval] if self._interval_mapping and self.interval in self._interval_mapping else self.interval
        self._work_folder = os.path.join("data", self.name, self.symbol, self._standard_interval)
        self._log_folder = os.path.join(self._work_folder, "log")
        self._log_file = os.path.join(self._log_folder, "app.log")
        self._tmp_csv_file = os.path.join(self._work_folder, "tmp", self._standard_interval + ".csv")
        self._failed_timestamps_file = os.path.join(self._log_folder, 'failed_timestamps.txt')

        self._make_dir()
        self._logger = self._get_logger()

        # 最多尝试make_csv的次数
        self.max_make_csv_attempt_count = 2
        # 一次make_csv中，允许最多的失败时间戳数量
        self.allow_max_failed_timestamps_number = 100
        # 在完成历史数据制作前，允许对请求失败的时间戳再次请求的次数
        self.allow_max_failed_timestamps_attemp_time = 5
        # 在完成历史数据制作前，允许最多的缺失时间点数量
        self.allow_max_missing_times = 1000

    def _load_config(self):
        with open("url_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)[self.name]
        return config

    def _validate_input(self, symbol, interval):
        if symbol in self.config["symbol"] and interval in list(self.config["interval"].keys()):
            return
        raise ValueError(f"{self.name}_{self.symbol}_{self.interval}：非法的symbol或interval输入，请查看url_config.json")

    def _get_logger(self):
        # 创建日志记录器
        logger = logging.getLogger(self.name + "_" + self.symbol + "_" + self.interval)
        logger.setLevel(logging.DEBUG)
        # 创建控制台处理器（StreamHandler），输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 设置控制台输出的日志级别为INFO
        # 创建文件处理器（RotatingFileHandler），输出到文件
        file_handler = RotatingFileHandler(self._log_file, maxBytes=3 * 1024 * 1024, backupCount=3, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 设置文件输出的日志级别为DEBUG
        # 创建日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)  # 为控制台处理器设置格式
        file_handler.setFormatter(formatter)  # 为文件处理器设置格式
        # 将处理器添加到日志记录器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        return logger

    def _make_dir(self):
        # 创建总文件夹
        if not os.path.exists(self._work_folder):
            os.makedirs(self._work_folder)
        # 创建日志文件夹
        log_foler_name = os.path.join(self._work_folder, "log")
        if not os.path.exists(log_foler_name):
            os.makedirs(log_foler_name)
        # 创建未拆分的文件存储文件夹
        tmp_folder_name = os.path.join(self._work_folder, "tmp")
        if not os.path.exists(tmp_folder_name):
            os.makedirs(tmp_folder_name)

    @abstractmethod
    def _get_data_list(self, new_data):
        return new_data

    @abstractmethod
    def _get_klines_data(self, params):
        pass

    def make_csv(self, max_rows = 100000, save_times = 100, max_threads = 50, mode = None):
        while True:
            self._initialize_attributes()
            self.max_rows = max_rows
            self._logger.info(f"设置分割后的csv最大行数为 {max_rows} ")
            self.save_times = save_times
            self._logger.info(f"设置每 {save_times} 次查询后保存到文件")
            self.max_threads = max_threads
            self._logger.info(f"设置最大线程数为 {max_threads}")
            if mode is None:
                self.mode = self._get_data_mode()
            else:
                self.mode = mode
                self._logger.warning(f"手动设置模式为 {mode}，请检查逻辑是否正确")
            if self.mode in [KLinesProcessor.DataMode.CREATE, KLinesProcessor.DataMode.CONTINUE]:
                if self._make_history_data():
                    break
            elif self.mode == KLinesProcessor.DataMode.UPDATE:
                # self._update_data()
                self._logger.warning(f"暂不支持 {mode} 模式")
                break
            if self._is_abnormal_termination: # 有异常需要重试
                self.max_make_csv_attempt_count -= 1
                if os.path.exists(self._tmp_csv_file):
                    os.remove(self._tmp_csv_file)
                if os.path.exists(self._failed_timestamps_file):
                    os.remove(self._failed_timestamps_file)
                if self.max_make_csv_attempt_count > 0: # 有异常且有剩余重试次数
                    self._logger.info(f"剩余尝试次数：{self.max_make_csv_attempt_count}")
                    max_threads = int(max(max_threads // 2 , 1))
                    self._logger.info("重试开始，最大线程数减半")
                else:
                    self._logger.error("尝试次数耗尽，终止此次数据获取")
                    raise RuntimeError("异常终止，请查看日志")
            
    def _initialize_attributes(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._new_data = []
        self._timer = 0
        self._data_collected = False  # 用于指示是否已收集完所有数据
        self._is_abnormal_termination = False

    def _get_data_mode(self):
        splited_file_name = os.path.join(self._work_folder, '0.csv')
        if os.path.exists(splited_file_name):  # 如果指定文件夹下有一个拆分的csv文件，代表历史数据已收集完成，进行数据更新
            self._logger.info(f"存在已拆分的csv文件{self._tmp_csv_file}，自动设置数据模式为UPDATE")
            return KLinesProcessor.DataMode.UPDATE
        if os.path.exists(self._tmp_csv_file):  # 如果指定文件夹下有未拆分的csv文件，代表历史数据尚未收集完成
            self._logger.info(f"存在未拆分的CSV文件{self._tmp_csv_file}，自动设置数据模式为CONTINUE")
            return KLinesProcessor.DataMode.CONTINUE
        else:
            self._logger.info(f"不存在任何csv文件，自动设置数据模式为CREATE")
            return KLinesProcessor.DataMode.CREATE

    def _make_history_data(self):
        if self.max_threads > 0:
            if self.mode == KLinesProcessor.DataMode.CREATE:
                self._block_time = KLinesProcessor.CURRENT_TIMESTAMP
            else:
                df = pd.read_csv(self._tmp_csv_file)
                self._block_time = self._datetime_to_timestamp(df.iloc[-1]["time"])
            self._logger.info(f"从{self._timestamp_to_datetime(self._block_time)}（时间戳：{self._block_time}）开始获取历史数据...")
            threads = []
            for _ in range(self.max_threads):
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
                return False    
            self._save_to_csv(self._new_data, self._tmp_csv_file)
            self._drop_duplicates(self._tmp_csv_file)
            self._sort_csv(self._tmp_csv_file,False)
            if self._data_collected:
                thread_list = [self.max_threads]
                while len(thread_list) < self.allow_max_failed_timestamps_attemp_time:
                    next_threads = max(int(thread_list[-1] / 2), 1)  
                    thread_list.append(next_threads)
                for thread_number in thread_list:
                    if self._retry_failed_timestamps(self._tmp_csv_file, thread_number):
                        break
                if os.path.exists(self._failed_timestamps_file):
                    self._logger.error("仍存在请求失败的时间戳")
                    raise RuntimeError("异常终止，请查看日志")
                splited_file_name = os.path.join(self._work_folder, '0.csv')
                if self._fix_csv_data_integrity(self._tmp_csv_file) and not os.path.exists(splited_file_name):
                    self._split_csv()
            return True
        else:
            self._logger.error(f"线程数设置错误：{self.max_threads}")
            return False

    def _worker(self):
        while not self._stop_event.is_set():
            block_time = self._get_next_block_time()
            if block_time is None:
                break
            params = self._make_params(block_time)
            data, flag = self._get_klines_data(params)
            if flag == KLinesProcessor.KlinesDataFlag.NORMAL:
                if data:
                    self._logger.info(f"线程 {threading.current_thread().name} 获取到数据，time: {self._timestamp_to_datetime(block_time)}")
                    with self._lock:
                        self._new_data.extend(data)
                        self._timer += 1
                        if self._timer >= self.save_times:
                            self._save_to_csv(self._new_data, self._tmp_csv_file)
                            self._new_data = []
                            self._timer = 0
                else:
                    self._logger.warning(f"线程 {threading.current_thread().name} 没有获取到数据，time: {self._timestamp_to_datetime(block_time)}")
                    with self._lock:
                        self._data_collected = True
                    break
            else:
                self._logger.warning(f"线程 {threading.current_thread().name} 获取数据时发生错误，time: {self._timestamp_to_datetime(block_time)}")
                self._save_failed_timestamp(block_time)

    def _save_failed_timestamp(self, timestamp):
        with self._lock:
            with open(self._failed_timestamps_file, 'a') as f:
                f.write(f"{timestamp}\n")
            self._logger.info(f"保存失败的时间戳 {self._timestamp_to_datetime(timestamp)} 到 {self._failed_timestamps_file}")
            with open(self._failed_timestamps_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > self.allow_max_failed_timestamps_number:
                    self._stop_event.set()
                    self._is_abnormal_termination = True

    def _retry_failed_timestamps(self, file_name, thread_number):
        import queue
        if not os.path.exists(self._failed_timestamps_file):
            self._logger.info("没有失败的时间戳需要重试。")
            return True
        with open(self._failed_timestamps_file, 'r') as f:
            timestamps = f.read().splitlines()
        timestamps = list(set(int(ts) for ts in timestamps))
        if len(timestamps) == 0:
            self._logger.info("没有失败的时间戳需要重试。")
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
                if flag == KLinesProcessor.KlinesDataFlag.NORMAL and data:
                    self._logger.info(f"重试成功，获取到数据，time: {self._timestamp_to_datetime(ts)}")
                    with file_lock:
                        self._save_to_csv(data, file_name)
                        self._remove_failed_timestamp(ts)
                else:
                    self._logger.warning(f"重试失败，time: {self._timestamp_to_datetime(ts)}，将保留时间戳以供下次重试")
                timestamp_queue.task_done()

        threads = []
        for _ in range(thread_number):
            t = threading.Thread(target=retry_timestamp)
            threads.append(t)
            t.start()
        timestamp_queue.join()
        for t in threads:
            t.join()
        self._drop_duplicates(file_name)
        self._sort_csv(file_name, ascending=False)

    def _remove_failed_timestamp(self, timestamp):
        with self._lock:
            if not os.path.exists(self._failed_timestamps_file):
                return
            with open(self._failed_timestamps_file, 'r') as f:
                lines = f.readlines()
            with open(self._failed_timestamps_file, 'w') as f:
                for line in lines:
                    if int(line.strip()) != timestamp:
                        f.write(line)
            self._logger.info(f"从 {self._failed_timestamps_file} 中删除已成功获取数据的时间戳 {self._timestamp_to_datetime(timestamp)}")

    def _get_next_block_time(self):
        with self._lock:
            if self._data_collected or self._block_time < self._timestamp_rate * 946656000:
                return None
            current_block_time = self._block_time
            self._block_time -= self._delta_time
            return current_block_time

    def _make_params(self, end_time):
        template = self._params_template.copy()
        params = {}
        for k, v in template.items():
            if v == "!ET!":
                params[k] = end_time
            elif v == "!S!":
                params[k] = self.symbol
            elif v == "!I!":
                params[k] = self.interval
            else:
                params[k] = v
        return params

    def _split_csv(self):
        try:
            df = pd.read_csv(self._tmp_csv_file)
            df.sort_values(by='time', ascending=True, inplace=True)
        except Exception as e:
            self._logger.error(f"读取文件时发生错误: {e}")
            return
        chunk_size = self.max_rows
        file_count = 0
        # Split the sorted DataFrame into chunks
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            output_file = os.path.join(self._work_folder, f"{file_count}.csv")
            try:
                chunk.to_csv(output_file, index=False, header=self._columns)
                self._logger.info(f"已创建文件: {output_file}，包含 {len(chunk)} 行。")
                file_count += 1
            except Exception as e:
                self._logger.error(f"写入文件 {output_file} 时发生错误: {e}")
                continue
        self._logger.critical(f"拆分完成，总共创建了 {file_count} 个文件。")

    def _save_to_csv(self, new_data, file_name, transfer_time=True, online_data = True):
        if new_data is None or len(new_data) == 0:
            return
        if online_data:
            data_list = self._get_data_list(new_data)
        else:
            data_list = new_data
        df = pd.DataFrame(data_list, columns=self._columns)
        if transfer_time:
            df['time'] = df['time'].apply(lambda x: self._timestamp_to_datetime(x))
        df.to_csv(file_name, mode='a', header=not pd.io.common.file_exists(file_name), index=False)
        self._logger.info(f"数据保存到 {file_name} ")

    def _save_new_data_update_mode(self, new_data):
        csv_files = [f for f in os.listdir(self._work_folder) if f.endswith('.csv')]
        latest_csv = max(csv_files, key=lambda x: int(x.split('.')[0]))
        latest_csv_path = os.path.join(self._work_folder, latest_csv)
        latest_csv_len = len(pd.read_csv(latest_csv_path))
        if len(new_data) + latest_csv_len > self.max_rows:
            self._logger.info(f" {latest_csv_path} 中的数据即将超出 {self.max_rows} 容量限制，对数据进行分割")
            # 将数据前半部分存在原csv中
            new_data_part1 = new_data[:self.max_rows - latest_csv_len]
            self._save_to_csv(new_data_part1, latest_csv_path)
            self._fix_csv_data_integrity(latest_csv_path)
            # 将数据后半部分存在新创建的csv中
            new_data_part2 = new_data[self.max_rows - latest_csv_len + 1:]
            new_csv_index = int(latest_csv.split('.')[0]) + 1
            new_csv_path = os.path.join(self._work_folder, str(new_csv_index) + '.csv')
            self._save_to_csv(new_data_part2, new_csv_path)
            self._fix_csv_data_integrity(new_csv_path)
        else:
            self._save_to_csv(new_data, latest_csv_path)
            self._fix_csv_data_integrity(latest_csv_path)

    def _update_data(self):
        new_data = []
        timer = 0
        csv_files = [f for f in os.listdir(self._work_folder) if f.endswith('.csv')]
        latest_csv = max(csv_files, key=lambda x: int(x.split('.')[0]))
        latest_csv_path = os.path.join(self._work_folder, latest_csv)
        df = pd.read_csv(latest_csv_path)
        end_time = self._datetime_to_timestamp(df.iloc[-1]["time"])
        self._logger.info(f"从{self._timestamp_to_datetime(end_time)}（时间戳：{end_time}）开始获取最新数据...")
        try:
            while True:
                params = self._make_params(end_time)
                data = self._get_klines_data(params)
                self._logger.info(f"获取到数据，time: {self._timestamp_to_datetime(end_time)}")
                if end_time - self._delta_time > KLinesProcessor.CURRENT_TIMESTAMP:
                    self._logger.critical("数据已最新")
                    self._save_new_data_update_mode(new_data)
                    break
                new_data.extend(data)
                timer = timer + 1
                end_time = end_time + self._delta_time
                if timer >= self.save_times:
                    self._save_new_data_update_mode(new_data)
                    new_data = []
                    timer = 0
        except KeyboardInterrupt:
            self._logger.warning("手动停止数据采集。")
            self._save_new_data_update_mode(new_data)

    def get_missing_times_list(self, file_name, output = False):
        try:
            # 读取时间列，假设时间列为字符串格式
            self._logger.info(f"检查文件 {file_name} 中的缺失数据...")
            df_time = pd.read_csv(file_name, usecols=['time'])
            df = pd.read_csv(file_name)
            # 转换为 datetime 类型
            df_time['time'] = pd.to_datetime(df_time['time'], errors='coerce')
            # 确定时间范围
            min_time = df_time['time'].min()
            max_time = df_time['time'].max()
            # 生成预期的时间序列
            expected_times = pd.date_range(start=min_time, end=max_time, freq=self._get_freq())
            # 获取现有的时间点
            existing_times = pd.Series(df_time['time'].unique())
            # 查找缺失的时间点
            missing_times = expected_times.difference(existing_times)
            # 输出结果
            if missing_times.empty:
                self._logger.info("所有预期的时间点数据均存在")
                return None,df,df_time
            self._logger.warning(f"共缺失{len(missing_times)}个数据，开始用表中数据补齐")
            if output:
                self._logger.warning("以下为缺失数据：")
                for missing_time in missing_times:
                    self._logger.info(missing_time)
            return missing_times, df, df_time
        except Exception as e:
            self._logger.error(f"检查过程中发生错误：{e}")
            return None, None, None
        
    def _get_freq(self):
        # 提取时间单位（最后一个字符）
        unit = self._standard_interval[-1]
        # 对不同的时间单位进行映射
        if unit == 'm':  
            return f"{self._standard_interval[:-1]}min"  
        elif unit == 'h':  # 如果是小时
            return f"{self._standard_interval[:-1]}H"  
        elif unit == 'd':  # 如果是天
            return f"{self._standard_interval[:-1]}D" 
        elif unit == 'w':  # 如果是周
            return f"{self._standard_interval[:-1]}W" 
        elif unit == 'M':  # 如果是月
            return f"{self._standard_interval[:-1]}M" 
        else:
            raise ValueError("Unsupported interval format")

    def _fix_csv_data_integrity(self, file_name):
        missing_times, df, df_time = self.get_missing_times_list(file_name)
        new_data = []
        if df is None or df_time is None:
            return False
        if missing_times is None:
            return True
        if len(missing_times) > self.allow_max_missing_times:
            self._logger.error("过多的缺失时间点，停止后续操作！")
            raise RuntimeError()
        for missing_time in missing_times:
            closest_time = df_time['time'].sub(missing_time).abs().idxmin()
            closest_data = list(df.iloc[closest_time])
            origin_time = closest_data[0]
            closest_data[0] = missing_time
            new_data.append(closest_data)
            self._logger.warning(f"使用 {origin_time} 的数据代替缺失的时间 {missing_time} 的数据")
        self._save_to_csv(new_data, file_name, False, False)
        self._sort_csv(file_name)
        self._drop_duplicates(file_name)
        return True
    
    def _sort_csv(self, file_name, ascending=True):
        df = pd.read_csv(file_name)
        df.sort_values(by='time', ascending=ascending, inplace=True)
        df.to_csv(file_name, index=False)
        self._logger.info(f"对{file_name}中的数据完成排序")

    def _drop_duplicates(self, file_name):
        df = pd.read_csv(file_name)
        df.drop_duplicates(subset=['time'], keep='first', inplace=True)
        df.to_csv(file_name, index=False)
        self._logger.info(f"对{file_name}中的数据完成去重")

    def _timestamp_to_datetime(self, timestamp):
        """
        将时间戳（毫秒）转换为系统时间
        :param timestamp: 时间戳（毫秒）
        :return: 转换后的系统时间字符串
        """
        # 将毫秒级时间戳转换为秒级
        timestamp = int(timestamp) / self._timestamp_rate
        # 使用datetime模块转换
        dt_object = datetime.utcfromtimestamp(timestamp)
        return dt_object.strftime('%Y-%m-%d %H:%M:%S')  # 格式化为“年-月-日 时:分:秒”

    def _datetime_to_timestamp(self, datetime_str):
        """
        将格式化后的datetime字符串转换为时间戳（毫秒）
        :param datetime_str: 格式化的datetime字符串，例如 '2024-11-15 14:30:00'
        :return: 对应的时间戳（毫秒）
        """
        # 定义输入的时间格式
        datetime_format = '%Y-%m-%d %H:%M:%S'
        # 将字符串转换为datetime对象（假设是UTC时间）
        utc_tz = pytz.timezone('UTC')
        dt_object = datetime.strptime(datetime_str, datetime_format)
        # 将datetime对象转换为UTC时区
        dt_object = utc_tz.localize(dt_object)
        # 获取时间戳（秒），然后乘以1000转换为毫秒
        timestamp = int(dt_object.timestamp() * self._timestamp_rate)
        return timestamp

    @staticmethod
    def _get_random_headers():
        """
        返回一个包含随机User-Agent的请求头
        """
        ua = UserAgent()
        headers = {
            "User-Agent": ua.random,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
        return headers