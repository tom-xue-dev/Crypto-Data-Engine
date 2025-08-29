FROM python:3.12-slim

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# 🔥 确保安装了 flower
RUN pip install flower

# 把 src/ 下的所有东西直接拷进 /app
COPY src /app/src/

# 设置工作目录以便找到 task_manager 模块
WORKDIR /app/src

# 默认命令（会被 docker-compose 中的 command 覆盖）
CMD ["celery", "--version"]