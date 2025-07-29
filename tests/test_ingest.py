import pytest
from httpx import AsyncClient
from fastapi import FastAPI

from server.routers.datascraper import data_scraper_router

# 构造一个最小的 app 实例用于测试
app = FastAPI()
app.include_router(data_scraper_router)

@pytest.mark.asyncio
async def test_ingest_and_status():
    async with AsyncClient(base_url="http://test") as ac:
        # 构造请求体
        payload = {
            "symbol": "BTCUSDT",
            "start": "2023-01-01",
            "end": "2023-01-01",
            "interval": "1s",
            "io_limit": 10
        }

        # 发送 POST /ingest
        resp = await ac.post("/ingest", json=payload)
        assert resp.status_code == 200
        task_id = resp.json()["task_id"]
        assert isinstance(task_id, str)

        # 发送 GET /status/{task_id}
        resp2 = await ac.get(f"/status/{task_id}")
        assert resp2.status_code == 200
        status = resp2.json()
        assert "stage" in status or "progress" in status  # 视你的实现而定
