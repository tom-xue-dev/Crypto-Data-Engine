import os
import requests
import zipfile
import threading
import hashlib
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import concurrent.futures
import pandas as pd

def get_all_symbols():
    """
    获取所有现货交易对的列表
    """
    url = 'https://api.binance.com/api/v3/exchangeInfo'
    response = requests.get(url)
    data = response.json()
    symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
    return symbols

BASE_URL = "https://data.binance.vision/data/spot/monthly/aggTrades"

def verify_checksum(file_path, expected_checksum):
    """
    计算文件的 SHA256 校验和，并与 expected_checksum 比较
    """
    # 提取 expected_checksum 中的校验码部分（第一个空格前的字符串）
    expected_checksum = expected_checksum.split()[0].strip()

    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    computed_checksum = sha256.hexdigest()
    return computed_checksum == expected_checksum

def download_and_extract(symbol, save_dir, year, month):
    """
    下载并解压指定交易对某年某月的聚合交易数据，
    同时下载对应的校验和文件，验证 ZIP 文件完整性，
    再将解压后的 CSV 转换为 Parquet 格式。
    """
    date_str = f"{year}-{month:02d}"
    file_name = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"{BASE_URL}/{symbol}/{file_name}"
    local_zip_path = os.path.join(save_dir, file_name)
    extracted_csv_path = local_zip_path.replace(".zip", ".csv")
    extracted_parquet_path = local_zip_path.replace(".zip", ".parquet")

    # 如果 Parquet 文件已存在，跳过下载和转换
    if os.path.exists(extracted_parquet_path):
        print(f"✅ {file_name} 已存在，跳过下载")
        return

    # 检查文件是否存在
    response = requests.head(url, timeout=5)
    if response.status_code == 200:
        print(f"⬇️ 开始下载: {url}")
        response = requests.get(url, stream=True, timeout=10)
        with open(local_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"✅ 下载完成: {local_zip_path}")

        # 下载校验和文件
        checksum_file_name = f"{file_name}.CHECKSUM"
        checksum_url = f"{BASE_URL}/{symbol}/{checksum_file_name}"
        checksum_response = requests.get(checksum_url, timeout=10)
        if checksum_response.status_code == 200:
            expected_checksum = checksum_response.text
            print(f"⬇️ 校验和下载完成: {checksum_url}")
        else:
            print(f"❌ 未能下载校验和文件: {checksum_url}")
            expected_checksum = None

        # 如果拿到校验和，则验证文件完整性
        if expected_checksum:
            if verify_checksum(local_zip_path, expected_checksum):
                print(f"✅ 校验成功: {local_zip_path}")
            else:
                print(f"❌ 校验失败: {local_zip_path}，删除文件")
                os.remove(local_zip_path)
                return

        #解压 ZIP 文件
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)
        print(f"✅ 解压完成: {local_zip_path}")

        # 将 CSV 转换为 Parquet 格式
        try:
            df = pd.read_csv(extracted_csv_path, header=None)
            df.to_parquet(extracted_parquet_path,compression="brotli")
            print(f"✅ 转换为 Parquet 完成: {extracted_parquet_path}")
            os.remove(extracted_csv_path)  # 删除 CSV 文件，节省空间
        except Exception as e:
            print(f"❌ 转换为 Parquet 时出错: {e}")

        os.remove(local_zip_path)  # 删除 ZIP 文件
    else:
        print(f"❌ 未找到数据，跳过: {url}")

def download_all_data(symbol, save_dir="data", start_date="2017-01", end_date="2025-03", max_threads=5):
    """
    多线程下载指定交易对的所有可用数据
    """
    os.makedirs(save_dir, exist_ok=True)

    # 解析起止时间
    current_date = datetime.strptime(start_date, "%Y-%m")
    end_date = datetime.strptime(end_date, "%Y-%m")

    # 生成所有需要下载的任务
    tasks = []
    while current_date <= end_date:
        tasks.append((symbol, save_dir, current_date.year, current_date.month))
        current_date += relativedelta(months=1)

    # 线程池并发下载
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(download_and_extract, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 等待所有任务完成

if __name__ == "__main__":
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)
    symbols = get_all_symbols()
    # 如有需要，可过滤部分交易对，例如只下载 USDT 交易对：
    # symbols = [symbol for symbol in symbols if symbol.endswith("USDT")]
    flag = False
    for symbol in symbols:
        if symbol != 'REDUSDT' and flag == False:
            continue
        else:
            flag =True
        download_all_data(symbol, save_dir="data", max_threads=16)
