FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY . .

CMD [".venv/bin/python", "main.py"]
