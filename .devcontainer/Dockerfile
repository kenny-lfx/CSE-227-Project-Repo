FROM mcr.microsoft.com/devcontainers/base:ubuntu

# System deps for OpenCV, OCR(Tesseract), ffmpeg
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg \
    tesseract-ocr libtesseract-dev \
    libgl1 libglib2.0-0 \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Optional: create venv for cleanliness
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Pre-install wheel build helpers
RUN pip install --upgrade pip wheel setuptools

WORKDIR /workspaces/${HOSTNAME}