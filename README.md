# Zoom/VC Virtual Background Blur Privacy Study (C Pipeline)

## 1. Project Overview

This repository investigates potential **privacy leakage** from virtual background blur features in modern video conferencing tools such as **Zoom**, **Microsoft Teams**, and **Google Meet**. The goal is to assess whether it is technically possible to reconstruct private background information from blurred video outputs.

A complete and reproducible pipeline is implemented, which:

* Extracts frames from blurred composite videos
* Segments and removes the foreground (person)
* Aligns background regions across time
* Aggregates temporal information to reconstruct latent background details
* Optionally applies deep learning deblurring (MPRNet, Real-ESRGAN)
* Runs object and text extraction (YOLOv8, OCR) for privacy evaluation

> **Key Finding:** For typical *defocus or Gaussian blurs* used by conferencing platforms, background recovery is visually cleaner but lacks high-frequency detail, posing **low practical privacy risk** under default conditions.

---

## 2. System Design and Methodology

### 2.1 Pipeline Overview

1. **Frame Extraction** – Sample frames at fixed FPS and resize for performance control.
2. **Segmentation** – Use either **MediaPipe SelfieSeg** (CPU) or **DeepLabV3** (GPU) for person segmentation.
3. **Alignment** – Register frames using **ORB affine matching** or **Farnebäck optical flow**.
4. **Aggregation** – Median, mean, or max aggregation to reconstruct consistent background structure.
5. **Post-processing** – Denoising and contrast-limited adaptive histogram equalization (CLAHE).
6. **Optional Deep Restoration** – Apply **MPRNet** (image deblurring) or **Real-ESRGAN** (super-resolution and sharpening).
7. **Information Extraction** – Perform object detection (YOLOv8) and OCR (Tesseract) to measure residual information leakage.

### 2.2 Design Principles

* **Foreground Suppression:** Foreground pixels are masked out before aggregation to prevent subject texture contamination.
* **Robust Alignment:** Optical flow is better for low-texture, blurred scenes; ORB performs well on structured regions.
* **Tile-based Aggregation:** Reduces memory footprint for large videos.
* **Modular Design:** Each stage can be independently replaced or ablated for research comparisons.

---

## 3. Environment and Setup

### 3.1 Docker (Recommended)

* **CPU Base Image:** `zoom-vb:cpu` (Python 3.11 slim)
* **GPU Base Image:** `zoom-vb:gpu` (CUDA 12.x or 13.x runtime, preinstalled PyTorch + TorchVision)

> Windows users: Enable **WSL 2** and **Docker GPU Support**. Ensure `nvidia-smi` runs successfully both on host and in container.

### 3.2 Project Structure

```bash
mkdir -p videos results
# place input video as videos/sample2.mp4
```

### 3.3 Full Pipeline Example (GPU)

```bash
python run_pipeline_plus_mprnet.py \
  --video /app/videos/sample2.mp4 \
  --out_dir /app/results/full_run \
  --seg_backend deeplabv3 --device cuda \
  --align_method flow --aggregate median \
  --fps 30 --resize_long 1600 --sample_every 1 \
  --denoise --clahe \
  --mprnet_repo /app/MPRNet \
  --mprnet_weights /app/MPRNet/Deblurring/pretrained_models/model_deblurring.pth \
  --yolo --ocr --ie_on_enhanced
```

### 3.4 CPU Example

```bash
python run_pipeline_plus_mprnet.py \
  --video /app/videos/sample2.mp4 \
  --out_dir /app/results/full_run_cpu \
  --seg_backend mediapipe --device cpu \
  --align_method flow --aggregate median \
  --fps 15 --resize_long 960 --sample_every 1 \
  --denoise --clahe \
  --mprnet_repo /app/MPRNet \
  --mprnet_weights /app/MPRNet/Deblurring/pretrained_models/model_deblurring.pth \
  --yolo --ocr --ie_on_enhanced
```

**Core Scripts:**

* `zoom_bg_blur_attack_pipeline.py` – main reconstruction logic
* `inference_mprnet_single.py` – single-image MPRNet inference
* `run_pipeline_plus_mprnet.py` – orchestrated end-to-end workflow
* `enhance_realesrgan_single.py` – Real-ESRGAN single-image enhancer

---

## 4. Recommended Parameters (High-End GPU)

| Parameter              | Suggested Value | Purpose                               |
| ---------------------- | --------------- | ------------------------------------- |
| `--seg_backend`        | `deeplabv3`     | cleaner masks                         |
| `--device`             | `cuda`          | faster segmentation                   |
| `--align_method`       | `flow`          | robust under blur                     |
| `--aggregate`          | `median`        | noise-resistant aggregation           |
| `--fps`                | `30`            | higher temporal resolution            |
| `--resize_long`        | `1600–1920`     | preserve details                      |
| `--denoise`, `--clahe` | on              | improves contrast and OCR reliability |

---

## 5. Experiments and Findings

* **Data:** Real and synthetic blurred meeting clips (indoor scenes, static subjects)
* **Observation:**

  * Virtual background blur acts as a **low-pass Gaussian filter**, applied per-frame to the composited background.
  * Temporal aggregation yields mild contrast recovery but no true texture restoration.
  * **MPRNet** helps motion-blurred cases but not defocus blur.
  * **Real-ESRGAN** sharpens boundaries but introduces hallucinated details.
  * OCR and object detection success rates remain low on reconstructed outputs.

**Conclusion:**
Even after advanced reconstruction and enhancement, **semantic recovery of private content remains unreliable**. While mild improvement in visibility is possible, the overall privacy threat from background blur remains **limited**.

---

## 6. Limitations and Failure Analysis

1. **Blur Type:** Zoom and similar tools use defocus-style Gaussian blur—an irreversible low-pass filter.
2. **Temporal Redundancy:** For static scenes, nearly identical frames provide minimal stochastic information for aggregation.
3. **Edge Leakage:** Foreground blending introduces artifacts near hair and shoulders, affecting background estimation.
4. **Deep Model Bias:** Generic restoration models may hallucinate details, leading to misleading or inaccurate reconstructions.

---
