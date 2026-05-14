FROM python:3.12-slim

WORKDIR /app

# 从本地 COPY uv 二进制，避免拉 ghcr.io
COPY uv uvx /usr/local/bin/
RUN chmod +x /usr/local/bin/uv /usr/local/bin/uvx

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev

COPY configs/react_baseline.example.yaml /app/configs/react_baseline.example.yaml

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
