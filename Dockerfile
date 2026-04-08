FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip uv==0.10.0

WORKDIR /workspace/card-framework

COPY docker/setup_and_run_entrypoint.py /usr/local/bin/card-framework-docker-entrypoint.py

ENTRYPOINT ["python", "/usr/local/bin/card-framework-docker-entrypoint.py"]
