from datetime import datetime
import json
from logging.handlers import RotatingFileHandler
from time import time
import os
import pytz
from enum import Enum
import pandas as pd
from abc import abstractmethod
from fake_useragent import UserAgent
import logging


class KLinesProcessor:
    CURRENT_TIMESTAMP = int(time() * 1000)
    class Mode(Enum):
        CREATE = 0
        CONTINUE = 1, # 收集历史数据
        UPDATE = 2 # 更新数据

    def __init__(self, name, symbol, interval) -> None:
        self.name = name
        self.config = self._load_config()
        self._validate_input(symbol, interval)
        self.symbol = symbol
        self.interval = interval
        self.save_times = 100
        self.max_rows = 1000000
        self._base_url = self.config["base_url"]
        self._params_template = self.config["params"]
        self._delta_time = self.config["interval"][self.interval]
        self._columns = self.config["columns"]
        self._timestamp_type = self.config["timestamp_type"]

        self._folder_name = os.path.join("data", name, symbol, interval)
        self._log_file_name = os.path.join(self._folder_name,"log","app.log")
        self._tmp_file_name = os.path.join(self._folder_name, "tmp", interval + ".csv")
        self._make_dir()
        self._logger = self._get_logger()
        self._mode = self._get_data_mode()

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
        logger = logging.getLogger(self.name +"_"+ self.symbol +"_"+ self.interval)
        logger.setLevel(logging.DEBUG)
        # 创建控制台处理器（StreamHandler），输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 设置控制台输出的日志级别为INFO
        # 创建文件处理器（RotatingFileHandler），输出到文件
        file_handler = RotatingFileHandler(self._log_file_name, maxBytes=3*1024*1024, backupCount=3, encoding='utf-8')
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
        if not os.path.exists(self._folder_name):
            os.makedirs(self._folder_name)
        # 创建日志文件夹
        log_foler_name = os.path.join(self._folder_name,"log")
        if not os.path.exists(log_foler_name):
            os.makedirs(log_foler_name)
        # 创建未拆分的文件存储文件夹
        tmp_folder_name = os.path.join(self._folder_name,"tmp")
        if not os.path.exists(tmp_folder_name):
            os.makedirs(tmp_folder_name)
    
    def set_save_times(self, time):
        self.save_times = time
        self._logger.info(f"设置每{time}次循环保存1次")

    def set_max_rows(self, rows):
        self.max_rows = rows
        self._logger.info(f"设置分割后的csv文件最大行数未{rows}")

    @abstractmethod
    def _get_data_list(self, new_data):
        return new_data

    @abstractmethod
    def _get_klines_data(self, params):
        pass
        
    def make_csv(self):
        if self._mode in [KLinesProcessor.Mode.CREATE, KLinesProcessor.Mode.CONTINUE]:
            self._make_history_data()
            return
        if self._mode == KLinesProcessor.Mode.UPDATE:
            self._update_data()
            return
        

    def _get_data_mode(self):
        splited_file_name = os.path.join(self._folder_name, '0.csv')
        if os.path.exists(splited_file_name): # 如果指定文件夹下有一个拆分的csv文件，代表历史数据已收集完成，进行数据更新
            self._logger.info(f"存在已拆分的csv文件{self._tmp_file_name}，自动设置数据模式为UPDATE")
            return KLinesProcessor.Mode.UPDATE
        if os.path.exists(self._tmp_file_name): # 如果指定文件夹下有未拆分的csv文件，代表历史数据尚未收集完成
            self._logger.info(f"存在未拆分的CSV文件{self._tmp_file_name}，自动设置数据模式为CONTINUE")
            return KLinesProcessor.Mode.CONTINUE
        else:
            self._logger.info(f"不存在任何csv文件，自动设置数据模式为CREATE")
            return KLinesProcessor.Mode.CREATE


    def _make_history_data(self):
        new_data = []
        if self._mode == KLinesProcessor.Mode.CREATE:
            end_time = KLinesProcessor.CURRENT_TIMESTAMP
        else:
            df = pd.read_csv(self._tmp_file_name)
            end_time = self._datetime_to_timestamp(df.iloc[-1]["time"])
        self._logger.info(f"从{self._timestamp_to_datetime(end_time)}（时间戳：{end_time}）开始获取历史数据...")
        timer = 0
        try:
            while True:
                params = self._make_params(end_time)
                data = self._get_klines_data(params)
                self._logger.info(f"获取到数据，endTime: {self._timestamp_to_datetime(end_time)}")
                if not data: # 所有数据请求完成
                    self._logger.critical("历史数据已收集完成")
                    try:
                        self._save_new_data(new_data, self._tmp_file_name)
                        self._drop_duplicates(self._tmp_file_name)
                        self._sort_csv(self._tmp_file_name)
                        splited_file_name = os.path.join(self._folder_name, self.interval + '0.csv')
                        if self._satisfy_csv_data_integrity(self._tmp_file_name) and not os.path.exists(splited_file_name):
                            self._split_csv()
                        break
                    except KeyboardInterrupt:
                        break
                new_data.extend(data)
                timer = timer + 1
                end_time = end_time - self._delta_time
                if timer >= self.save_times:
                    self._save_new_data(new_data, self._tmp_file_name)
                    new_data = []
                    timer = 0
        except KeyboardInterrupt:
            self._logger.warning("手动停止数据收集")
            self._save_new_data(new_data, self._tmp_file_name)
            self._drop_duplicates(self._tmp_file_name)
            self._sort_csv(self._tmp_file_name, ascending=False)

    def _update_data(self):
        new_data = []
        timer = 0
        csv_files = [f for f in os.listdir(self._folder_name) if f.endswith('.csv')]
        latest_csv = max(csv_files, key=lambda x: int(x.split('.')[0]))
        latest_csv_path = os.path.join(self._folder_name, latest_csv)
        df = pd.read_csv(latest_csv_path)
        end_time = self._datetime_to_timestamp(df.iloc[-1]["time"])
        self._logger.info(f"从{self._timestamp_to_datetime(end_time)}（时间戳：{end_time}）开始获取最新数据...")
        try:
            while True:
                params = self._make_params(end_time)
                data = self._get_klines_data(params)
                self._logger.info(f"获取到数据，endTime: {self._timestamp_to_datetime(end_time)}")
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
            self._logger.warning("\n手动停止数据采集。")
            self._save_new_data_update_mode(new_data)

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
        chunk_size = self.max_rows
        try:
            reader = pd.read_csv(self._tmp_file_name, chunksize=chunk_size)
        except Exception as e:
            self._logger.error(f"读取文件时发生错误: {e}")
            return
        file_count = 0
        for i, chunk in enumerate(reader):
            output_file = os.path.join(self._folder_name, f"{file_count}.csv")
            file_count += 1
            try:
                chunk.to_csv(output_file, index=False, header=self._columns)
                self._logger.info(f"已创建文件: {output_file}，包含 {len(chunk)} 行。")
            except Exception as e:
                self._logger.error(f"写入文件 {output_file} 时发生错误: {e}")
                continue
        self._logger.critical(f"拆分完成，总共创建了 {file_count} 个文件。")

    def _save_new_data(self, new_data, file_name, transfer_time = True):
        if len(new_data) == 0:
            return
        df = pd.DataFrame(self._get_data_list(new_data), columns=self._columns)
        if transfer_time:
            df['time'] = df['time'].apply(lambda x:self._timestamp_to_datetime(x))
        df.to_csv(file_name, mode='a', header=not pd.io.common.file_exists(file_name), index=False)
        self._logger.info(f"数据保存到 {file_name} ")

    def _save_new_data_update_mode(self,new_data):
        csv_files = [f for f in os.listdir(self._folder_name) if f.endswith('.csv')]
        latest_csv = max(csv_files, key=lambda x: int(x.split('.')[0]))
        latest_csv_path = os.path.join(self._folder_name, latest_csv)
        latest_csv_len = len(pd.read_csv(latest_csv_path))
        if len(new_data) + latest_csv_len > self.max_rows:
            self._logger.info(f" {latest_csv_path} 中的数据即将超出 {self.max_rows} 容量限制，对数据进行分割")
            # 将数据前半部分存在原csv中
            new_data_part1 = new_data[:self.max_rows - latest_csv_len]
            self._save_new_data(new_data_part1, latest_csv_path)
            self._drop_duplicates(latest_csv_path)
            self._sort_csv(latest_csv_path)
            self._satisfy_csv_data_integrity(latest_csv_path)
            # 将数据后半部分存在新创建的csv中
            new_data_part2 = new_data[self.max_rows - latest_csv_len + 1:]
            new_csv_path = os.path.join(self._folder_name, str(latest_csv + 1)+'.csv')
            self._save_new_data(new_data_part2, new_csv_path)
            self._drop_duplicates(new_csv_path)
            self._sort_csv(new_csv_path)
            self._satisfy_csv_data_integrity(new_csv_path)
        else:
            self._save_new_data(new_data, latest_csv_path)
            self._drop_duplicates(latest_csv_path)
            self._sort_csv(latest_csv_path)
            self._satisfy_csv_data_integrity(latest_csv_path)

    # def get_klines_data_by_end_time(self,end_time):
    #     params = self._make_params(end_time)
    #     self.get_klines_data(params)

    def _satisfy_csv_data_integrity(self, file_name):
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
            # TODO:支持所有时间种类的转换
            freq = str(self.interval).replace('m','min')
            # 生成预期的时间序列
            expected_times = pd.date_range(start=min_time, end=max_time, freq=freq)
            # 获取现有的时间点
            existing_times = pd.Series(df_time['time'].unique())
            # 查找缺失的时间点
            missing_times = expected_times.difference(existing_times)
            # 输出结果
            if missing_times.empty:
                self._logger.info("所有预期的时间点数据均存在")
                return True
            self._logger.warning(f"共缺失{len(missing_times)}个数据，开始用表中数据补齐")
            new_data = []
            for missing_time in missing_times:
                closest_time = df_time['time'].sub(missing_time).abs().idxmin()
                closest_data = list(df.iloc[closest_time])
                closest_data[0] = missing_time
                new_data.append(closest_data)
                self._logger.warning(f"使用 {closest_data[0]} 的数据代替缺失的时间 {missing_time} 的数据")
            self._save_new_data(new_data, file_name, False)
            self._sort_csv(file_name)
            self._drop_duplicates(file_name)
            return True    
        except Exception as e:
            self._logger.error(f"检查过程中发生错误：{e}")
            return False

    def _sort_csv(self,file_name, ascending=True):
        df = pd.read_csv(file_name)
        df.sort_values(by='time', ascending=ascending, inplace=True)
        df.to_csv(file_name, index=False)
        self._logger.info(f"对{file_name}中的数据完成排序")

    def _drop_duplicates(self,file_name):
        df = pd.read_csv(file_name)
        df.drop_duplicates(subset=['time'], keep='first',inplace=True)
        df.to_csv(file_name, index=False)
        self._logger.info(f"对{file_name}中的数据完成去重")

    def _timestamp_to_datetime(self,timestamp):
        """
        将时间戳（毫秒）转换为系统时间
        :param timestamp: 时间戳（毫秒）
        :return: 转换后的系统时间字符串
        """
        rate = 1000 if self._timestamp_type == "ms" else 1
        # 将毫秒级时间戳转换为秒级
        timestamp = timestamp / rate
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
        rate = 1000 if self._timestamp_type == "ms" else 1
        timestamp = int(dt_object.timestamp() * rate)
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
    