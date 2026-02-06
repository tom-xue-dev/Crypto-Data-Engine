
FROM python:3.12-slim

RUN pip install poetry

WORKDIR /app

# 🔥 修正：去掉 ../，因为构建上下文已经是项目根目录
COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root

COPY src ./src/

ENV PYTHONPATH=/app/src

WORKDIR /app/src

# Default: start the API server
CMD ["python", "-m", "crypto_data_engine.main", "serve", "--host", "0.0.0.0", "--port", "8000"]