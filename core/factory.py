import queue
import threading

from helpers.config import Config
from helpers.model import SubTask, DataFlag


class Factory:

    def __init__(
        self, fetch_data_callback, max_attempts=Config("MAX_ATTEMPT_TIMES")
    ):
        self.fetch_data_callback = fetch_data_callback
        self.max_attempts = max_attempts
        self._online_queue = queue.Queue()
        self._local_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._cached_raw_data = []
        self._abnormal_termination_info = None

    def complete(self, task: SubTask):
        self._initialize_flow_line(task.timestamp_list)
        self._start_workers(task.online_worker_count, task.local_worker_count)
        self._online_queue.join()
        self._local_queue.join()
        self._stop_workers()
        if self._abnormal_termination_info:
            raise RuntimeError(self._abnormal_termination_info)
        return self._cached_raw_data

    def _initialize_flow_line(self, task_list):
        for ts in task_list:
            self._online_queue.put((ts, 0))  # 时间戳，已尝试次数

    def _start_workers(self, online_worker_count, local_worker_count):
        self.online_threads = self._start_threads(
            online_worker_count, self._online_worker
        )
        self.local_threads = self._start_threads(local_worker_count, self._local_worker)

    def _stop_workers(self):
        self._stop_threads(self.local_threads, self._local_queue)
        self._stop_threads(self.online_threads, self._online_queue)

    def _start_threads(self, thread_count, target):
        threads = []
        for _ in range(thread_count):
            t = threading.Thread(target=target)
            t.start()
            threads.append(t)
        return threads

    def _stop_threads(self, threads, queue):
        for _ in range(len(threads)):
            queue.put(None)
        for t in threads:
            t.join()

    def _online_worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._online_queue.get()
            except queue.Empty:
                continue
            if item is None:
                self._online_queue.task_done()
                break
            ts, attempt_times = item
            if attempt_times > self.max_attempts:
                with self._lock:
                    self._stop_event.set()
                    if self._abnormal_termination_info is None:
                        self._abnormal_termination_info = f"请求（{ts}）数据时超过了允许最多尝试次数：{attempt_times - 1}"
            # TODO since=ts
            data, flag = self.fetch_data_callback(since=ts)
            self._local_queue.put((ts, data, flag, attempt_times + 1))
            self._online_queue.task_done()

    def _local_worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._local_queue.get()
            except queue.Empty:
                continue
            if item is None:
                self._local_queue.task_done()
                break
            ts, data, flag, attempt_times = item
            if flag == DataFlag.NORMAL:
                if data:
                    self._cached_raw_data.extend(data)
            else:
                self._online_queue.put((ts, attempt_times))
            self._local_queue.task_done()
