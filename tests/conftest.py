# python
# tests/conftest.py
import textwrap
import pytest
from fastapi.testclient import TestClient

# 你的项目模块导入（根据你项目的实际包路径调整）
from celery_app import create_celery_app
from crypto_data_engine.common.config.config_settings import ServerConfig, CeleryConfig, settings
from startup_server import create_app

@pytest.fixture(scope="session")
def celery_eager_app():
    """
    创建一个 Celery 实例，并开启 eager 模式，避免使用真实 broker/backend。
    """
    app = create_celery_app()
    app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
    )
    return app

@pytest.fixture()
def injected_settings(monkeypatch, tmp_path):
    """
    通过 YAML/或直接构造 Settings 子对象，注入到全局 settings。
    如需严格验证 YAML->Settings 的加载链路，可在这里改为调用你的加载函数。
    """
    # 假设你希望通过 YAML 驱动（演示写入 YAML 文件）
    celery_yaml = textwrap.dedent("""
    broker_url: "redis://localhost:6379/0"
    backend_url: "redis://localhost:6379/1"
    task_serializer: "json"
    result_serializer: "json"
    accept_content: ["json"]
    task_default_queue: "cpu"
    worker_max_tasks_per_child: 200
    task_acks_late: true
    """).strip()

    server_yaml = textwrap.dedent("""
    host: "127.0.0.1"
    port: 18080
    log_level: "INFO"
    reload: false
    workers: 1
    """).strip()

    celery_yaml_path = tmp_path / "celery_settings.yaml"
    server_yaml_path = tmp_path / "server_settings.yaml"
    celery_yaml_path.write_text(celery_yaml, encoding="utf-8")
    server_yaml_path.write_text(server_yaml, encoding="utf-8")

    # 在测试中，直接以 Pydantic 模型构造并覆盖全局 settings
    celery_cfg = CeleryConfig(
        broker_url="redis://localhost:6379/0",
        backend_url="redis://localhost:6379/1",
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_default_queue="cpu",
        worker_max_tasks_per_child=200,
        task_acks_late=True,
    )
    server_cfg = ServerConfig(
        host="127.0.0.1",
        port=18080,
        log_level="INFO",
        reload=False,
        workers=1,
    )

    # 覆盖全局 settings 对象
    monkeypatch.setattr(settings, "celery_cfg", celery_cfg, raising=True)
    monkeypatch.setattr(settings, "server_cfg", server_cfg, raising=True)

    return {
        "celery_yaml": str(celery_yaml_path),
        "server_yaml": str(server_yaml_path),
    }

@pytest.fixture()
def api_client(injected_settings):
    """
    创建 FastAPI 客户端。create_app 内部会读取已注入的 settings.server_cfg。
    """
    app = create_app()
    with TestClient(app) as client:
        yield client