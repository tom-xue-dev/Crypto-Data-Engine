from pathlib import Path
from crypto_data_engine.common.utils.setting_utils import find_project_root

PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
CONFIG_ROOT = PROJECT_ROOT / "data" / "config" / "config_templates"
ENV_FILE_PATH = PROJECT_ROOT / ".env"
FUTURES_DATA_ROOT = Path("E:/data")
