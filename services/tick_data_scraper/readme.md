# Crypto Quant Data Processing Pipeline

🚀 **数字货币高频数据下载与预处理流水线**

本项目实现了完整的加密货币逐笔成交数据的自动下载、数据清洗与 Bar 预处理流程，支持多进程、多线程、断点续传，并保留良好的可扩展性，适用于后续因子研究与策略开发。

---

## 📂 项目结构
```
project/
├── downloader.yaml               # 下载模块配置文件
├── preprocessor.yaml             # 预处理模块配置文件
├── Config.py                     # 配置读取模块
├── downloader_pipeline.py        # 数据下载主程序
├── pre_process.py                # 数据预处理主程序
├── bar_constructor.py            # Bar构建类
├── path_utils.py                 # 路径辅助函数
├── data/                         # 下载的原始数据
├── data_aggr/                    # 预处理后的聚合数据
└── README.md                     # 项目说明
```

---

## ⚙️ 功能特色
✅ 支持 Binance 月度逐笔成交数据自动下载  
✅ 多线程下载 + 多进程解压转换  
✅ SHA256 校验文件完整性  
✅ **断点续传机制**，失败自动补齐  
✅ 数据存储结构清晰，按交易对分类  
✅ 支持 Tick Bar, Volume Bar, Dollar Bar 聚合  

---

## 📥 下载模块

### 配置文件：`downloader.yaml`
```
data_dir: "./data"
start_date: "2021-01"
end_date: "2021-03"
max_threads: 8
convert_processes: 4
queue_size: 20
symbols: "auto"        # 或 ['BTCUSDT', 'ETHUSDT']
filter_suffix: "USDT"
base_url: "https://data.binance.vision/data/spot/monthly/aggTrades"
completed_tasks_file: "./completed_tasks.txt"
```

### 运行
```bash
python downloader_pipeline.py
```

---

## 🛠️ 预处理模块

### 配置文件：`preprocessor.yaml`
```
data_dir: "./data"
aggregated_data_dir: "./data_aggr"
bar_type: "volume_bar"  # 可选 tick_bar, volume_bar, dollar_bar
threshold: 10000000
process_num_limit: 4
```

### 运行
```bash
python pre_process.py
```

---

## 🎯 运行效果示例
```
[Download Progress] ██████████ 100/100  ETA: 00:00
✅ 已处理: 100 / 100
⏰ 总耗时: 180.42 秒

[Preprocessing Assets] ██████████ 10/10  ETA: 00:00
⏰ 聚合完成，共处理 10 个资产
```

---

## 📌 设计思路
- 下载模块采用 **Producer-Consumer 模型**，避免内存堆积
- 采用 **JoinableQueue** 实现限流与断点续传
- **配置文件全参数管理**，支持灵活调整
- 预处理阶段支持 **多进程并行**
- **存储结构专业**，每个交易对单独文件夹，便于后续因子研究

---

## ⭐️ 后续扩展建议
- 因子生成模块
- 因子收益率测试
- 机器学习标签与训练模块
- 回测模块

该项目已具备 **完整数据处理底座**，可作为量化研究项目基础。

