from enum import Enum
import os

import numpy as np
import pandas as pd

from helpers.config import Config
from helpers.logger import Logger
from helpers.model import BasicInfo
from helpers.utils import interval_to_timestamp, timestamp_to_datetime


def _id(func_id):
    def decorator(func):
        func.id = func_id  # 为函数添加 id 属性
        return func

    return decorator


class CSVSaver:
    _mapping = {
        "DEFAULT": [
            "drop_duplicates",
            "sort",
            "save_missing_times",
            "fix_integrity",
            "drop_last",
            "transfer_time",
        ]
    }

    def __init__(
        self,
        info: BasicInfo,
        params: dict,
    ):
        self.info = info
        self.params = params
        self._work_folder = self._initialize_work_folder()
        self._df = None
        self._missing_times = None
        # 收集所有带有 id 属性的方法，存入 self._actions 字典
        self._actions = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "id"):
                self._actions[attr.id] = attr

    def _initialize_work_folder(self):
        work_folder = os.path.join(
            Config("DATA_PATH"),
            self.info.exchange,
            self.info.type,
            self.info.symbol.replace("/", "-"),
            self.info.interval,
        )
        os.makedirs(work_folder, exist_ok=True)
        return work_folder

    @_id("sort")
    def _sort(self, ascending=True):
        self._df.sort_values(by="time", ascending=ascending, inplace=True)

    @_id("drop_duplicates")
    def _drop_duplicates(self):
        self._df.drop_duplicates(subset=["time"], keep="first", inplace=True)

    @_id("drop_last")
    def _drop_last(self):
        self._df = self._df.iloc[:-1]

    @_id("transfer_time")
    def _transfer_time(self):
        self._df["time"] = pd.to_datetime(self._df["time"], unit="ms")

    def _initialize_missing_times(self):
        """
        使用纯timestamp进行缺失值判断:
        要求: self._df 已排序
        """
        if self._missing_times is not None:
            return
        Logger.info("检查缺失数据...")
        df = self._df
        if df.empty:
            Logger.info("无数据，不存在缺失点")
            return

        df["time"] = df["time"].astype(np.int64)

        min_time = df["time"].min()
        max_time = df["time"].max()

        delta_timestamp = interval_to_timestamp(self.info.interval)
        expected_timestamps = np.arange(
            min_time, max_time + delta_timestamp, delta_timestamp
        )
        existing_timestamps = set(df["time"].values)
        missing_timestamps = set(expected_timestamps) - existing_timestamps

        if not missing_timestamps:
            Logger.info("所有预期的时间点数据均存在")
            self._missing_times = set()
        else:
            Logger.info(f"共缺失 {len(missing_timestamps)} 条数据")
            self._missing_times = sorted(missing_timestamps)

    @_id("fix_integrity")
    def _fix_data_integrity(self):
        """
        使用相邻数据填补缺失的数据点
        要求: self._df已排序
        """
        self._initialize_missing_times()
        if len(self._missing_times) == 0:
            return
        df = self._df
        new_data = []
        arr = df["time"].values
        for missing_ts in self._missing_times:
            insert_pos = np.searchsorted(arr, missing_ts)
            candidates = []
            if insert_pos > 0:
                candidates.append(
                    (abs(arr[insert_pos - 1] - missing_ts), insert_pos - 1)
                )
            if insert_pos < len(arr):
                candidates.append((abs(arr[insert_pos] - missing_ts), insert_pos))

            closest_idx = min(candidates, key=lambda x: x[0])[1]
            closest_data = df.iloc[closest_idx].copy()
            closest_data["time"] = missing_ts
            new_data.append(closest_data)

        if new_data:
            self._append_data([row for row in new_data])
            self._sort()

    def _append_data(self, data_list):
        try:
            new_df = pd.DataFrame(data_list, columns=self.params.get("columns"))
            if self._df is None:
                self._df = new_df
            else:
                self._df = pd.concat([self._df, new_df], ignore_index=True)
        except Exception as e:
            Logger.error(f"追加数据时发生错误: {e}")

    def _load_file(self, file_name):
        """在开始时如果已有临时文件，加载到内存中"""
        if os.path.exists(file_name):
            try:
                df = pd.read_csv(file_name)
                self._df = df
                Logger.info(f"从 {file_name} 加载数据")
            except Exception as e:
                Logger.error(f"加载文件 {file_name} 时出错: {e}")

    def _save_file(self):
        df = self._df
        chunk_size = Config("CSV_CHUNK_SIZE")
        file_count = 0
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i : i + chunk_size]
            output_file = os.path.join(self._work_folder, f"{file_count}.csv")
            try:
                chunk.to_csv(
                    output_file, index=False, header=self.params.get("columns")
                )
                Logger.info(f"已创建文件: {output_file}，包含 {len(chunk)} 行。")
                file_count += 1
            except Exception as e:
                Logger.error(f"写入文件 {output_file} 时发生错误: {e}")
                continue

    @_id("save_missing_times")
    def _save_missing_times(self):
        self._initialize_missing_times()
        if len(self._missing_times) == 0:
            return
        file_name = os.path.join(self._work_folder, "missingtimes.txt")
        with open(file_name, "w") as f:
            for missing_timestamp in self._missing_times:
                missing_time = timestamp_to_datetime(missing_timestamp)
                f.write(f"{missing_time} ({missing_timestamp})\n")

    def history_data_exists(self, work_folder):
        splitted_file = os.path.join(work_folder, "0.csv")
        if os.path.exists(splitted_file):
            return True
        return False

    def save(self, processed_data):
        # 从_mapping中根据传入的mode获取要执行的动作列表
        self._append_data(processed_data)
        mode = self.params.get("mode") if self.params.get("mode") else "DEFAULT"
        actions = self._mapping[mode]
        if self.params.get("drop_last") == False:
            actions.remove("drop_last")
        if self.params.get("fix_integrity") == False:
            actions.remove("fix_integrity")
        if self.params.get("save_missing_times") == False:
            actions.remove("save_missing_times")
        # 按顺序执行动作列表中定义的操作
        for action in actions:
            # 根据动作名称从 self._actions 中找到对应的方法并调用
            func = self._actions.get(action)
            if func:
                func()  # 调用已绑定到实例的方法
            else:
                Logger.warn(f"找不到操作: {action}")
        self._save_file()
