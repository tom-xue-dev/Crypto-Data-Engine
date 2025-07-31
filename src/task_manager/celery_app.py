from celery import Celery
from common.config.config_settings import settings


def create_celery_app(app_name: str = "quant_backtest_system") -> Celery:
    app = Celery(app_name)
    app.config_from_object(settings.celery_cfg)
    return app

celery_app = create_celery_app()

from task_manager import celery_worker
celery_worker.register_tasks(celery_app)
