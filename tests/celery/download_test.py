"""
测试完整的FastAPI → Celery → Worker流程
"""
import requests
import time
import json

# FastAPI服务器地址
API_BASE = "http://localhost:8000"


def test_full_pipeline():
    print("🚀 测试完整流程: FastAPI → Celery → Worker")
    # 1️⃣ 测试Celery连接
    print("\n1️⃣ 测试Celery连接...")
    response = requests.post(f"{API_BASE}/api/v1/download/test/health")
    if response.status_code == 200:
        print("✅ Celery连接正常")
        print(f"Worker信息: {json.dumps(response.json(), indent=2)}")
    else:
        print("❌ Celery连接失败")
        return
    # 2️⃣ 创建单个下载任务
    print("\n2️⃣ 创建单个下载任务...")
    task_data = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "year": 2020,
        "month": 1
    }

    response = requests.post(
        f"{API_BASE}/api/v1/download/downloads/single",
        json=task_data
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ 任务创建成功")
        print(f"DB任务ID: {result['db_task_id']}")
        print(f"Celery任务ID: {result['celery_task_id']}")

        celery_task_id = result['celery_task_id']

        # 3️⃣ 监控任务状态
        print("\n3️⃣ 监控任务执行...")
        for i in range(10):  # 最多监控10次
            response = requests.get(
                f"{API_BASE}/api/v1/download/celery/status/{celery_task_id}"
            )

            if response.status_code == 200:
                status = response.json()
                print(f"状态: {status['status']} | 成功: {status['successful']} | 失败: {status['failed']}")

                if status['successful'] or status['failed']:
                    print(f"最终结果: {json.dumps(status['result'], indent=2)}")
                    break

            time.sleep(5)  # 等待5秒
    else:
        print(f"❌ 任务创建失败: {response.text}")


if __name__ == "__main__":
    test_full_pipeline()