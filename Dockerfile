FROM python:3.12-slim

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# ✅ 把 src/ 下的所有东西直接拷进 /app
COPY src/ /app/src/

# ✅ 现在不需要 PYTHONPATH！也不需要 ENV！
# ✅ 启动时也直接引用 task_manager 启动模块
CMD ["python", "task_manager/startup.py", "worker", "download"]
