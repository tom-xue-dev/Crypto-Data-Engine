import json
import threading
import time
from queue import Queue
import websocket
import datetime
import csv

# WebSocket参数
symbol = "btcusdt"
socket = f"wss://stream.binance.com:9443/ws/{symbol}@trade"

# 创建线程安全队列
message_queue = Queue()

# 定义统计数据
stats = {
    "sell_count": 0,
    "sell_usdt": 0.0,
    "sell_btc": 0.0,
    "buy_count": 0,
    "buy_usdt": 0.0,
    "buy_btc": 0.0,
}

lock = threading.Lock()

file_index = 1  # 当前文件编号
record_count = 0  # 当前文件的记录条数
max_records_per_file = 100000  # 每个文件的最大记录数
csv_file = f"trade_list/trade_stats{file_index:05d}.csv"  # 当前文件名


def initialize_csv():
    global csv_file, file_index
    csv_file = f"trade_list/trade_stats{file_index:05d}.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(
            [
                "Time Window Start",
                "Sell Orders",
                "Sell USDT",
                "Sell BTC",
                "Buy Orders",
                "Buy USDT",
                "Buy BTC",
            ]
        )


def write_to_csv(start_time, stats):
    """
    将统计数据写入 CSV 文件，必要时切换到新文件。
    """
    global csv_file, file_index, record_count
    with lock:
        # 检查是否需要切换文件
        if record_count >= max_records_per_file:
            file_index += 1
            record_count = 0
            initialize_csv()  # 创建新文件

        # 写入当前统计数据
        with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    start_time,
                    stats["sell_count"],
                    stats["sell_usdt"],
                    stats["sell_btc"],
                    stats["buy_count"],
                    stats["buy_usdt"],
                    stats["buy_btc"],
                ]
            )
            record_count += 1


def process_message():
    """
    从队列中处理消息并更新统计数据。
    """
    while True:
        try:
            message = message_queue.get()
            if message is None:
                break

            # 解析JSON数据
            data = json.loads(message)
            price = float(data["p"])
            quantity = float(data["q"])
            is_sell_order = data["m"]  # True 表示卖单

            with lock:
                if is_sell_order:
                    stats["sell_count"] += 1
                    stats["sell_usdt"] += price * quantity
                    stats["sell_btc"] += quantity
                else:
                    stats["buy_count"] += 1
                    stats["buy_usdt"] += price * quantity
                    stats["buy_btc"] += quantity
        except Exception as e:
            print(f"Error processing message: {e}")


def reset_and_print_stats():
    """
    每隔1000毫秒打印统计数据并重置计数器。
    """
    global csv_file, file_index, record_count
    while True:
        # 记录当前时间戳
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        time.sleep(1)  # 等待 1000ms

        with lock:
            # 检查是否需要切换文件
            if record_count >= max_records_per_file:
                file_index += 1
                record_count = 0
                initialize_csv()  # 创建新文件

            # 写入当前统计数据
            with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        start_time,
                        stats["sell_count"],
                        stats["sell_usdt"],
                        stats["sell_btc"],
                        stats["buy_count"],
                        stats["buy_usdt"],
                        stats["buy_btc"],
                    ]
                )
                record_count += 1

            # 重置统计数据
            stats["sell_count"] = 0
            stats["sell_usdt"] = 0.0
            stats["sell_btc"] = 0.0
            stats["buy_count"] = 0
            stats["buy_usdt"] = 0.0
            stats["buy_btc"] = 0.0


def on_message(ws, message):
    """
    WebSocket消息处理，将消息加入队列。
    """
    message_queue.put(message)


def on_error(ws, error):
    print(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    print("### WebSocket closed ###")


def on_open(ws):
    print("Opened connection")


initialize_csv()

# 启动WebSocket客户端
ws = websocket.WebSocketApp(
    socket, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close
)

# 启动WebSocket线程
ws_thread = threading.Thread(target=ws.run_forever)
ws_thread.start()

# 启动消息处理线程
processing_thread = threading.Thread(target=process_message)
processing_thread.start()

# 启动统计定时器
stats_thread = threading.Thread(target=reset_and_print_stats)
stats_thread.start()

try:
    ws_thread.join()
    processing_thread.join()
    stats_thread.join()
except KeyboardInterrupt:
    # 优雅关闭
    message_queue.put(None)
    ws.close()
    ws_thread.join()
    processing_thread.join()
    stats_thread.join()
