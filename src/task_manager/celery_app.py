from celery import Celery

import logging

logger = logging.getLogger(__name__)


def create_celery_app(app_name: str = "crypto_data_engine") -> Celery:
    """创建Celery应用实例"""
    app = Celery(app_name)
    # 从设置中加载配置
    # app.config_from_object(settings.celery_cfg)
    from crypto_data_engine.common.config.config_settings import settings
    # 获取配置并转换为普通字典
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

    # 使用字典配置而不是对象
    app.conf.update(config_dict)

    # 额外的配置
    app.conf.update(
        # 任务路由配置
        task_routes={
            'tick.download': {'queue': 'io_intensive'},
            'tick.health_check': {'queue': 'cpu'},
        },
        # 任务执行配置
        task_time_limit=3600,  # 1小时超时
        task_soft_time_limit=3300,  # 55分钟软超时
        worker_prefetch_multiplier=1,  # 每次只取一个任务
        task_acks_late=True,  # 任务完成后才确认
        # 序列化配置
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        # 结果后端配置
        result_expires=3600,  # 结果保存1小时
        worker_redirect_stdouts=True,          # 重定向 print 到日志
        worker_redirect_stdouts_level="INFO",  # 以 INFO 级别记录
    )

    logger.info(f"Celery app '{app_name}' 创建完成")
    logger.info(f"Broker: {app.conf.broker_url}")
    logger.info(f"Backend: {app.conf.result_backend}")

    return app


# 创建全局Celery实例
celery_app = create_celery_app()


# 延迟导入避免循环依赖
def register_all_tasks():
    """注册所有任务"""
    try:
        from task_manager import celery_worker
        tasks = celery_worker.register_tasks(celery_app)
        logger.info(f"已注册任务: {list(tasks.keys())}")
        return tasks
    except Exception as e:
        logger.error(f"任务注册失败: {e}")
        raise


# 在模块加载时注册任务
register_all_tasks()