from celery import Celery

import logging

logger = logging.getLogger(__name__)


def create_celery_app(app_name: str = "crypto_data_engine") -> Celery:
    """Create Celery application instance."""
    app = Celery(app_name)
    # Load configuration from settings
    # app.config_from_object(settings.celery_cfg)
    from crypto_data_engine.common.config.config_settings import settings
    # Convert settings into plain dict
    config_dict = {
        'broker_url': settings.celery_cfg.broker_url,
        'result_backend': settings.celery_cfg.result_backend,
        'task_serializer': settings.celery_cfg.task_serializer,
        'result_serializer': settings.celery_cfg.result_serializer,
        'accept_content': settings.celery_cfg.accept_content,
        'task_default_queue': settings.celery_cfg.task_default_queue,
        'worker_max_tasks_per_child': settings.celery_cfg.worker_max_tasks_per_child,
        'task_acks_late': settings.celery_cfg.task_acks_late,
    }

    # Configure using dictionary instead of object reference
    app.conf.update(config_dict)

    # Additional configuration
    app.conf.update(
        # Task routing configuration
        task_routes={
            'tick.download': {'queue': 'io_intensive'},
            'tick.health_check': {'queue': 'cpu'},
        },
        # Task execution configuration
        task_time_limit=3600,  # 1 hour hard timeout
        task_soft_time_limit=3300,  # 55 minutes soft timeout
        worker_prefetch_multiplier=1,  # Fetch one task at a time
        task_acks_late=True,  # Ack only after completion
        # Serialization configuration
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        # Result backend configuration
        result_expires=3600,  # Store results for 1 hour
        worker_redirect_stdouts=True,          # Redirect print to logs
        worker_redirect_stdouts_level="INFO",  # Record as INFO
    )

    logger.info(f"Celery app '{app_name}' initialized")
    logger.info(f"Broker: {app.conf.broker_url}")
    logger.info(f"Backend: {app.conf.result_backend}")

    return app


# Create global Celery instance
celery_app = create_celery_app()


# Lazy import to avoid circular dependency
def register_all_tasks():
    """Register all Celery tasks."""
    try:
        from task_manager import celery_worker
        tasks = celery_worker.register_tasks(celery_app)
        logger.info(f"Registered tasks: {list(tasks.keys())}")
        return tasks
    except Exception as e:
        logger.error(f"Failed to register tasks: {e}")
        raise


# Register tasks at module import time
register_all_tasks()