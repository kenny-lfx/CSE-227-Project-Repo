FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-pip \
    ffmpeg \
    tesseract-ocr libtesseract-dev \
    libgl1 libglib2.0-0 git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip wheel setuptools \
 && pip install -r /app/requirements.txt \
 && pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision  # CUDA 12.1 版

# 如果你的 requirements 已含 torch/vision，记得用 CUDA 源覆盖安装

# docker run -it -v "${PWD}:/app" -w /app --name zoomvb zoom-vb:cpu /bin/bash
#docker run --gpus all -it --name zoomvb-gpu `
#  -v "${PWD}:/app" -w /app zoom-vb:gpu

# exit
# docker start -ai zoomvb-gpu