from pathlib import Path


def find_project_root(marker_files=("pyproject.toml", "requirements.txt", ".git")) -> Path:
    curr = Path(__file__).resolve()
    for parent in curr.parents:
        if any((parent / name).exists() for name in marker_files):
            return parent
    raise RuntimeError("❌ Cannot locate project root — no marker file found.")