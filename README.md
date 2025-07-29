# Quantitative Backtesting System

> 🚀 A Modular, Distributed Framework for Quantitative Trading Backtesting

This project implements a scalable quantitative backtesting system for cryptocurrency trading strategies. It processes high-frequency tick data from sources like Binance, aggregates it into bars (e.g., tick bars, volume bars, dollar bars), generates features, produces trading signals, performs backtesting, and visualizes results.

The system is built with a microservices architecture and supports distributed computing via Celery and Ray. Each module communicates via HTTP APIs to ensure loose coupling, scalability, and independent deployment.

---

## 🔧 Key Modules

- **Tick Data Download**: Fetches and processes raw trade data from exchanges like Binance.
- **Bar Aggregation**: Constructs custom bars (tick, volume, dollar) from raw data.
- **Feature Generation**: Computes statistical features and technical indicators.
- **Signal Generation**: Applies rules or ML models to produce trading signals.
- **Backtesting**: Simulates trading strategies with slippage, fees, and risk controls.
- **Visualization**: Creates charts, reports, and dashboards for strategy evaluation.

---

## 📂 Project Structure

```text
project/
├── api_gateway/                  # FastAPI API Gateway for HTTP endpoints
│   ├── main.py                   # FastAPI app entrypoint
│   └── tasks.py                  # Celery task definitions
├── celery_worker/                # Celery workers for task execution
│   └── worker.py                 # Worker configuration and tasks
├── ray_cluster/                  # Ray integration for distributed compute
│   └── ray_tasks.py              # Ray remote functions
├── modules/                      # Microservices for each core function
│   ├── download/                 # Tick data download service
│   ├── bar_aggregation/          # Bar construction service
│   ├── feature_generation/       # Feature engineering service
│   ├── signal_generation/        # Signal creation service
│   ├── backtesting/              # Backtesting engine service
│   └── visualization/            # Reporting and visualization service
├── configs/                      # Configuration files (YAML/.env)
│   ├── celery_config.py          # Celery settings
│   └── ray_config.py             # Ray cluster settings
├── data/                         # Raw and processed data storage
├── docker-compose.yml            # Containerized deployment
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ⚙️ Architecture Overview

```
Client → FastAPI (API Gateway)
         → Celery.delay() Submit Task
         → Redis (Broker/Backend)
         → Celery Worker
         → download_task() → Ray.remote()
         → Ray Cluster (distributed compute)
         → Optional Redis Backend (for results)
```

### Description

- **FastAPI Gateway**: Receives and routes incoming requests.
- **Celery Dispatcher**: Enqueues tasks to Redis and distributes to workers.
- **Ray Cluster**: Handles compute-intensive IO/CPU tasks in parallel.
- **Redis**: Acts as broker/backend for Celery task coordination.

---

## 🛠 Installation & Setup

### Prerequisites

- Python 3.10+
- Redis (broker/backend)
- Ray (`pip install ray[default]`)
- Docker (optional, for full deployment)

### Setup

```bash
git clone https://github.com/your-repo/quant-backtest-system.git
cd quant-backtest-system
pip install -r requirements.txt
```

### Start Services

```bash
# Redis
redis-server         # or: docker run -d -p 6379:6379 redis

# Ray (start head node)
ray start --head

# Celery worker
celery -A api_gateway.tasks worker --loglevel=info

# API Gateway
uvicorn api_gateway.main:app --reload
```

### Docker (optional)

```bash
docker-compose up -d
```

---

## 🚀 Usage Example: Submitting a Download Task

```bash
curl -X POST "http://localhost:8000/submit_download"      -H "Content-Type: application/json"      -d '{
           "symbols": ["BTCUSDT", "ETHUSDT"],
           "start_date": "2022-01",
           "end_date": "2022-03"
         }'
```

You will receive a `task_id`.

Check status:

```bash
curl http://localhost:8000/status/{task_id}
```

---

## 🔍 Monitoring

| Component | Tool                  | URL                         |
|-----------|-----------------------|-----------------------------|
| Celery    | Flower                | http://localhost:5555       |
| Ray       | Ray Dashboard         | http://localhost:8265       |

---

## ➕ Extending the System

To add a new module (e.g., feature generation):

1. Create a FastAPI service under `modules/feature_generation/`
2. Add Celery tasks that call Ray remote functions
3. Define a `POST` endpoint in API Gateway to submit this task
4. Optionally, register this module in your gateway or orchestrator

---

## 📈 Performance & Scaling

- **Horizontal Scaling**: Add more Celery workers or Ray nodes
- **Parquet Format**: Efficient for storing high-frequency data
- **Fault Tolerance**:
  - Celery retries failed tasks
  - Ray recovers failed workers automatically

---

## 🔮 Future Enhancements

- Add ML-based signal generation (e.g., PyTorch integration)
- Use Kubernetes for full orchestration
- Support additional exchanges (e.g., Coinbase, Kraken)

---

## 📧 Contact

For issues, please open a GitHub issue or contact:  
📮 `your-email@example.com`

---

## 🌟 Contributing

Pull requests are welcome! Please focus on:

- Modular, testable design
- Documentation and clarity
- Code quality and extensibility