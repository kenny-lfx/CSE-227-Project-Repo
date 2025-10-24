#!/usr/bin/env bash
set -euo pipefail

# In Codespaces, repo is mounted at /workspaces/<repo>
cd /workspaces/"$(basename $(pwd))" || true

# Install Python deps (CPU-only)
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "[setup] requirements.txt not found; installing default deps"
  pip install \
    opencv-python-headless numpy pillow tqdm scikit-image \
    mediapipe \
    torch torchvision \
    ultralytics \
    pytesseract
fi

echo "[setup] Done. You can now run:"
echo "python zoom_bg_blur_attack_pipeline.py --help"