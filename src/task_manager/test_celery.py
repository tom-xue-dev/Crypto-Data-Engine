from task_manager.celery_app import celery_app

r = celery_app.send_task("tick.health_check", queue="cpu")
print(r.id)
