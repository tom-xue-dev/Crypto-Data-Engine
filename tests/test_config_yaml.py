# python
# tests/test_config_yaml.py
from Config import Config  # 你提供的 YAML 简易读取类
from crypto_data_engine.common.config.config_settings import settings

def test_yaml_loader_and_settings_override(tmp_path, injected_settings):
    # 1) 用简易 Config 类读取临时 YAML，并验证键值
    celery_yaml_path = injected_settings["celery_yaml"]
    server_yaml_path = injected_settings["server_yaml"]

    cfg1 = Config(celery_yaml_path)
    assert cfg1.get("broker_url").startswith("redis://")
    assert cfg1["task_default_queue"] == "cpu"

    cfg2 = Config(server_yaml_path)
    assert cfg2["host"] == "127.0.0.1"
    assert cfg2.get("port") == 18080

    # 2) 验证注入后的 settings 生效
    assert settings.server_cfg.host == "127.0.0.1"
    assert settings.server_cfg.port == 18080
    assert settings.celery_cfg.task_default_queue == "cpu"