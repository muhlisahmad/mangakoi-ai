# RunPod Serverless worker — manga translation pipeline
#
# Base image includes CUDA + cuDNN runtime matching RunPod's GPU hosts.
# Per RunPod's worker best practices, image size directly affects cold
# start time — system deps are kept minimal and apt lists are purged.

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies:
#   python3.11           — runtime
#   fonts-comic-neue      — typesetting font (Stage 5)
#   libgl1, libglib2.0-0  — required by opencv-python-headless at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-comic-neue \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached and only
# rebuilds when requirements.txt actually changes — not on every
# pipeline code edit.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Safely catch the missing inpainting package without triggering nested dependency conflicts
RUN pip install --no-cache-dir --no-deps simple-lama-inpainting

# Application code
COPY pipeline/ ./pipeline/
COPY handler.py .

# Build-time defaults (non-sensitive). Runtime secrets — BUCKET_*,
# webhook config — are set in the RunPod console per the
# environment-variables best practices, never baked into the image.
ENV USE_4BIT_TRANSLATION="false"
ENV HF_HOME="/runpod-volume/hf_cache"

# -u: unbuffered stdout/stderr so logs reach the RunPod console in
# real time instead of being buffered until the process exits.
CMD ["python3", "-u", "handler.py"]
