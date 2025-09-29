import sys
import subprocess
import platform
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def start_celery_worker(queue_name: str, service_name: str):
    pool_mode = "solo" if platform.system() == "Windows" else "prefork"

    cmd = [
        "celery", "-A", "task_manager.celery_app", "worker",
        "--loglevel=info",
        f"--queues={queue_name}",
        f"--hostname={service_name}@%h",
        f"--pool={pool_mode}",
        "-E"  # Enable event reporting
    ]

    logger.info(f"🚀 Starting Celery worker: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    if len(sys.argv) != 3 or sys.argv[1] != "worker":
        logger.error("Usage: python main.py worker [download|preprocess|backtest]")
        sys.exit(1)

    service = sys.argv[2]
    queue_map = {
        "download": "io_intensive",
        "extract": "extract",
        "backtest": "backtest_tasks"
    }

    queue = queue_map.get(service)
    if not queue:
        logger.error(f"❌ Unknown worker type '{service}'")
        sys.exit(1)
    else:
        print("queue",queue)
    try:
        start_celery_worker(queue, service)
    except KeyboardInterrupt:
        logger.info("celery worker stopped")
        sys.exit(0)

if __name__ == "__main__":
    main()
