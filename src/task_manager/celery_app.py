from celery import Celery

from common.config.load_config import celery_cfg


def create_celery_app(app_name: str = "crypto_data_engine") -> Celery:
    app = Celery(app_name)
    app.config_from_object(celery_cfg)
    return app

celery_app = create_celery_app()

# ✅ 延迟注册，避免循环导入
from task_manager import celery_worker
celery_worker.register_tasks(celery_app)
