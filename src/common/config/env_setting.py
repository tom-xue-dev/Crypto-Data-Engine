# scripts/generate_env.py
from pathlib import Path

from common.config.paths import ENV_FILE_PATH

env_content = """#
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_BACKEND_URL=redis://redis:6379/1
APP_ENV=development
JWT_SECRET_KEY=changeme
"""

env_path = Path(ENV_FILE_PATH)
if env_path.exists():
    print("⚠️ .env already exists, won't overwrite.")
else:
    env_path.write_text(env_content)
    print("✅ .env file generated successfully.")
