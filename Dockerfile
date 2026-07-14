FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# LightGBM and XGBoost link against the OpenMP runtime (libgomp), which the
# slim base image does not ship. Without it, importing them fails with
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY . .

CMD [".venv/bin/python", "main.py"]
