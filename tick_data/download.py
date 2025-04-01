import os
import requests
import zipfile
import hashlib
import pandas as pd
import concurrent.futures
import multiprocessing
from queue import Queue
from threading import Thread
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import time
from Config import Config
import sys
from pathlib import Path

class DownloadContext:
    def __init__(self, config):
        self.save_dir = config['data_dir']
        self.base_url = config['base_url']
        self.completed_tasks_file = config['completed_tasks_file']

# ------------------- Utils -------------------
def get_all_symbols(suffix_filter=None):
    url = 'https://api.binance.com/api/v3/exchangeInfo'
    response = requests.get(url)
    data = response.json()
    symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
    if suffix_filter:
        symbols = [s for s in symbols if s.endswith(suffix_filter)]
    return symbols

def verify_checksum(file_path, expected_checksum):
    expected_checksum = expected_checksum.split()[0].strip()
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    computed_checksum = sha256.hexdigest()
    return computed_checksum == expected_checksum

def read_task_file(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f.readlines())

def append_task_file(path, task):
    with open(path, "a") as f:
        f.write(task + "\n")

def download_zip(symbol, year, month, context):
    date_str = f"{year}-{month:02d}"
    file_name = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{context.base_url}/{symbol}/{file_name}"
    local_zip_path = os.path.join(context.save_dir, file_name)

    if os.path.exists(local_zip_path.replace(".zip", ".parquet")):
        return None

    response = requests.head(url, timeout=5)
    if response.status_code != 200:
        return None

    try:
        r = requests.get(url, stream=True, timeout=10)
        with open(local_zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
        return local_zip_path

    except Exception:
        return None

def consumer_worker(queue, counter, context):
    while True:
        item = queue.get()
        if item is None:
            break
        symbol, zip_path, year, month = item
        task_id = f"{symbol}/{year}-{month}"
        try:
            process_file(symbol, zip_path, year, month, context)
            with counter.get_lock():
                counter.value += 1
            append_task_file(context.completed_tasks_file, task_id)
        except Exception:
            pass
        queue.task_done()

def process_file(symbol, zip_path, year, month, context):
    file_name = os.path.basename(zip_path)
    checksum_url = f"{context.base_url}/{symbol}/{file_name}.CHECKSUM"

    try:
        checksum_response = requests.get(checksum_url, timeout=10)
        if checksum_response.status_code == 200:
            expected_checksum = checksum_response.text
            if not verify_checksum(zip_path, expected_checksum):
                os.remove(zip_path)
                return

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(context.save_dir)

        csv_path = zip_path.replace(".zip", ".csv")

        symbol_dir = os.path.join(context.save_dir, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        parquet_path = Path(symbol_dir) / f"{year}-{month:02d}.parquet"
        df = pd.read_csv(csv_path, header=None)
        df.to_parquet(parquet_path, compression="brotli")

        os.remove(zip_path)
        os.remove(csv_path)

    except Exception:
        pass

# ------------------- Pipeline -------------------
def run_pipeline(config):
    max_threads = config['max_threads']
    convert_processes = config['convert_processes']
    queue_size = config['queue_size']
    start_date = config['start_date']
    end_date = config['end_date']

    context = DownloadContext(config)

    os.makedirs(context.save_dir, exist_ok=True)
    queue = multiprocessing.JoinableQueue(maxsize=queue_size)
    counter = multiprocessing.Value('i', 0)

    if config['symbols'] == "auto":
        symbols = get_all_symbols(config.get('filter_suffix'))
    else:
        symbols = config['symbols']

    completed_tasks = read_task_file(context.completed_tasks_file)
    task_list = []

    for symbol in symbols:
        current_date = datetime.strptime(start_date, "%Y-%m")
        end_dt = datetime.strptime(end_date, "%Y-%m")
        while current_date <= end_dt:
            task_id = f"{symbol}/{current_date.year}-{current_date.month}"
            if task_id not in completed_tasks:
                task_list.append((symbol, current_date.year, current_date.month))
            current_date += relativedelta(months=1)

    total_tasks = len(task_list)

    consumers = []
    for _ in range(convert_processes):
        p = multiprocessing.Process(target=consumer_worker, args=(queue, counter, context))
        p.start()
        consumers.append(p)

    start_time = time.time()

    def producer(symbol, year, month, pbar):
        zip_path = download_zip(symbol, year, month, context)
        if zip_path:
            queue.put((symbol, zip_path, year, month))
        pbar.update(1)

    with tqdm(total=total_tasks, desc="[Download Progress]") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(producer, symbol, year, month, pbar) for (symbol, year, month) in task_list]
            concurrent.futures.wait(futures)

    queue.join()
    for _ in consumers:
        queue.put(None)
    for p in consumers:
        p.join()

    end_time = time.time()

    print("\n-----------------------------")
    print(f"✅ 已处理: {counter.value} / {total_tasks}")
    print(f"⏰ 总耗时: {end_time - start_time:.2f} 秒")
    print("-----------------------------")

if __name__ == "__main__":
    config = Config(path="downloader.yaml")
    run_pipeline(config)
