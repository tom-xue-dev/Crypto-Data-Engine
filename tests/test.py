# python
# tests/test_service_comm.py
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient
from celery import shared_task
from startup_server import create_app
from celery_app import create_celery_app

TASK_NAME = "tests.add_numbers"

def register_test_task(celery_app):
    @celery_app.task(name=TASK_NAME)
    def add_numbers(x, y):
        return x + y
    return add_numbers

def build_temp_router(celery_app):
    router = APIRouter()

    @router.post("/test/add")
    def add_endpoint(payload: dict):
        """
        模拟 API 调用 Celery 任务。eager 模式下会同步返回结果。
        """
        x = payload.get("x", 0)
        y = payload.get("y", 0)
        # send_task 通过任务名调用，更贴近跨服务通信（解耦模块导入）
        res = celery_app.send_task(TASK_NAME, args=[x, y])
        return {"result": res.get(timeout=5)}  # eager 下立即可 get
    return router

def test_api_to_celery_roundtrip(api_client, celery_eager_app, injected_settings):
    # 1) 注册测试任务
    register_test_task(celery_eager_app)

    # 2) 动态挂载测试路由
    app = api_client.app
    app.include_router(build_temp_router(celery_eager_app))

    # 3) 发起请求，验证 Celery 执行与返回
    r = api_client.post("/test/add", json={"x": 7, "y": 5})
    assert r.status_code == 200
    assert r.json()["result"] == 12