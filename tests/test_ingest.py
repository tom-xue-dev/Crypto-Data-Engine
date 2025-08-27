# python
# tests/test_server_worker.py
import subprocess
import types
import pytest
from startup_server import server_startup, start_worker
from crypto_data_engine.common.config.config_settings import settings

def test_server_startup_reads_settings(monkeypatch, injected_settings):
    # 拦截 uvicorn.run，验证 host/port 来源于 settings.server_cfg
    called = {}
    def fake_run(app, host, port):
        called["host"] = host
        called["port"] = port
    monkeypatch.setattr("startup_server.uvicorn.run", fake_run)

    server_startup()  # 不传 host/port，应该读取 settings.server_cfg
    assert called["host"] == settings.server_cfg.host
    assert called["port"] == settings.server_cfg.port

def test_start_worker_builds_command(monkeypatch):
    calls = {}
    def fake_run(cmd, *args, **kwargs):
        calls["cmd"] = cmd
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    start_worker("download")
    assert "--queues=download_tasks" in " ".join(calls["cmd"])
    assert "-A celery_app.celery_app" in " ".join(calls["cmd"])

    # 未知服务应退出（通过 sys.exit），这里拦截为异常
    with pytest.raises(SystemExit):
        start_worker("unknown_service")