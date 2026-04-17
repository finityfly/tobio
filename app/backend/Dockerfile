# Tobio backend — uv-managed deps, cached layer for fast redeploys
FROM python:3.10-slim

# cv2 needs these shared libs even with opencv-python-headless on slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libxcb-shm0 libxcb-xfixes0 libxrender1 libxext6 libsm6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Ultralytics/Matplotlib writable config dirs (avoids Render warnings)
ENV YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install deps first — this layer is only rebuilt when pyproject.toml or uv.lock changes
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# Swap GUI OpenCV (pulled by ultralytics) for headless — no display server in Docker
RUN .venv/bin/pip install --quiet opencv-python-headless && \
    .venv/bin/pip uninstall -y opencv-python 2>/dev/null || true

# Add the venv to PATH so uvicorn/python resolve without uv run
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

# Render sets PORT at runtime; default for local runs
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
