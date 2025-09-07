# Dockerfile.image2text
# CPU base (swap to a CUDA image if you want GPU: e.g. pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    OUTPUT_DIR=/data \
    I2T_MODEL=llava-hf/llava-onevision-qwen2-7b-ov-hf

# System deps for PIL & image IO
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget libgl1 libglib2.0-0 \
    git curl build-essential libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# (Optional) upgrade pip
RUN python3 -m pip install --upgrade pip

# --- GPU PyTorch (CUDA 12.1) BEFORE app deps; keep torch out of requirements.txt ---
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1

# Copy project
COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

RUN chmod +x /app/entrypoint.sh

EXPOSE 8001 8502
VOLUME ["/data"]

ENTRYPOINT ["/app/entrypoint.sh"]
