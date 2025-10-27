#!/usr/bin/env python3
"""
Zoom/VC Virtual-Background Blur Attack – End-to-End Pipeline (C Option)
======================================================================

Goal
----
Given a video recorded from a video-call where the participant enabled
**background blur** (or a virtual background with blur), this script attempts to:

1) Extract frames and segment out the foreground person.
2) Align background regions across time.
3) Aggregate residual background information (median/mean/max).
4) Optionally post-process (denoise / deblur via MPRNet checkpoint if provided).
5) Run information extraction on the recovered image: object detection (YOLOv8)
   and OCR (Tesseract via pytesseract), and produce a compact report.

Design Principles
-----------------
- Minimal external glue: everything is runnable with common Python libs.
- Pluggable segmentation backends: MediaPipe SelfieSeg (CPU-friendly) or
  TorchVision DeepLabV3 (higher quality).
- Two alignment strategies: ORB affine (robust) and Farneback optical flow.
- Simple, reproducible aggregation (median by default) + optional postprocess.
- Modular structure for easy ablation (enable/disable any stage via CLI args).

Dependencies (suggested)
------------------------
- Python >= 3.9
- opencv-python, numpy, pillow, tqdm
- mediapipe (for fast segmentation) OR torch + torchvision (DeepLabV3)
- ultralytics (YOLOv8) for object detection (optional)
- pytesseract + system Tesseract (optional OCR)
- scikit-image (for SSIM/PSNR if you compare against ground-truth)

Example Usage
-------------
# 1) Basic run with MediaPipe segmentation and median aggregation
python zoom_bg_blur_attack_pipeline.py \
  --video input.mp4 \
  --out_dir results/demo1 \
  --seg_backend mediapipe \
  --align_method orb \
  --aggregate median \
  --fps 15

# 2) Same but try optical-flow alignment and denoise + CLAHE
python zoom_bg_blur_attack_pipeline.py \
  --video input.mp4 \
  --out_dir results/demo2 \
  --seg_backend mediapipe \
  --align_method flow \
  --aggregate median \
  --denoise \
  --clahe

# 3) Add YOLOv8 detection and OCR on the reconstructed image
python zoom_bg_blur_attack_pipeline.py \
  --video input.mp4 \
  --out_dir results/demo3 \
  --yolo \
  --ocr

# 4) Use TorchVision DeepLabV3 segmentation (slower but often cleaner masks)
python zoom_bg_blur_attack_pipeline.py \
  --video input.mp4 \
  --seg_backend deeplabv3 \
  --device cpu

# 5) (Optional) Apply MPRNet deblurring (requires a checkpoint)
python zoom_bg_blur_attack_pipeline.py \
  --video input.mp4 \
  --mprnet_ckpt /path/to/MPRNet_deblurring.pth

Notes
-----
- For YOLOv8: `pip install ultralytics`. The first run downloads a small model.
- For OCR: install system Tesseract (e.g., Windows via choco: `choco install tesseract`),
  then `pip install pytesseract`.
- For MediaPipe: `pip install mediapipe`.
- For DeepLabV3: `pip install torch torchvision`.

"""
from __future__ import annotations
import os
import cv2
import sys
import json
import math
import time
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

# Optional imports guarded by try/except
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

try:
    import torch
    import torchvision
    from torchvision import transforms
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

try:
    import pytesseract
    from PIL import Image
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

# ------------------------------
# Utility helpers
# ------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path: str, rgb: np.ndarray):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


# ------------------------------
# Frame extraction
# ------------------------------

def extract_frames(video_path: str, fps: Optional[int], resize_long: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(input_fps / fps))) if fps else 1

    frames = []
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if fps and (idx % stride != 0):
            idx += 1
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize_long:
            h, w = rgb.shape[:2]
            long_side = max(h, w)
            if long_side > resize_long:
                scale = resize_long / float(long_side)
                new_w, new_h = int(w * scale), int(h * scale)
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
        idx += 1
    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted.")
    return frames


# ------------------------------
# Segmentation backends
# ------------------------------

class Segmenter:
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Return mask of foreground (person): uint8 {0,1} same HxW."""
        raise NotImplementedError


class MediaPipeSelfieSeg(Segmenter):
    def __init__(self, threshold: float = 0.5):
        if not _HAS_MEDIAPIPE:
            raise RuntimeError("mediapipe not installed. `pip install mediapipe`. ")
        self.threshold = threshold
        self.mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        h, w, _ = rgb.shape
        res = self.mp_selfie.process(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        mask = (res.segmentation_mask >= self.threshold).astype(np.uint8)
        # mask==1 means foreground by MP; invert if needed
        return mask


class DeepLabV3Person(Segmenter):
    def __init__(self, device: str = "cpu"):
        if not _HAS_TORCH:
            raise RuntimeError("torch/torchvision not installed.")
        self.device = torch.device(device)
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(weights="DEFAULT").to(self.device)
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        # person class in COCO is index 15 for some models; for DeepLabV3 it's 15 or 0-based? We'll take argmax.

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = self.tf(rgb).unsqueeze(0).to(self.device)
            out = self.model(x)["out"]  # [1, C, H, W]
            pred = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        # Create binary person mask by picking class==15 if labels exist; otherwise fall back to largest connected region near center
        # Heuristic: treat non-zero as foreground if person class not known
        # For robustness, fallback to non-background (class!=0)
        mask = (pred != 0).astype(np.uint8)
        return mask


def build_segmenter(name: str, device: str) -> Segmenter:
    name = name.lower()
    if name == "mediapipe":
        return MediaPipeSelfieSeg()
    elif name == "deeplabv3":
        return DeepLabV3Person(device=device)
    else:
        raise ValueError(f"Unknown seg_backend: {name}")


# ------------------------------
# Alignment
# ------------------------------

def align_orb_affine(src_rgb: np.ndarray, dst_rgb: np.ndarray, mask: Optional[np.ndarray]=None) -> np.ndarray:
    """Return src warped to dst using ORB keypoints -> affine/homography.
    mask: foreground mask (1=FG), we prefer to **downweight** FG by eroding and using only BG for matching.
    """
    gray1 = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2GRAY)

    if mask is not None:
        # create background mask to prefer matches there
        bg_mask = (mask == 0).astype(np.uint8)*255
    else:
        bg_mask = None

    orb = cv2.ORB_create(5000)
    k1 = orb.detectAndCompute(gray1, bg_mask)[1]
    k2 = orb.detectAndCompute(gray2, bg_mask)[1]
    if k1 is None or k2 is None:
        return src_rgb  # fallback: no warp

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(k1, k2)
    matches = sorted(matches, key=lambda m: m.distance)
    if len(matches) < 6:
        return src_rgb

    # Rebuild keypoints (ORB returns descriptors only above)
    # Need to call detect to get kp coords
    kp1 = orb.detect(gray1, bg_mask)
    kp2 = orb.detect(gray2, bg_mask)
    # Build dict by pt hashing (approx)
    def kp_to_pts(kps):
        return np.array([k.pt for k in kps], dtype=np.float32)
    pts1 = kp_to_pts(kp1)
    pts2 = kp_to_pts(kp2)
    # Map matches by index if possible; fall back to slicing
    src_pts = []
    dst_pts = []
    for m in matches[:200]:
        if m.queryIdx < len(pts1) and m.trainIdx < len(pts2):
            src_pts.append(pts1[m.queryIdx])
            dst_pts.append(pts2[m.trainIdx])
    if len(src_pts) < 6:
        return src_rgb
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Try affine first
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    h, w = dst_rgb.shape[:2]
    if M is not None:
        warped = cv2.warpAffine(src_rgb, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped

    # Fallback: homography
    H, inliers = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
    if H is not None:
        warped = cv2.warpPerspective(src_rgb, H, (w, h), flags=cv2.INTER_LINEAR)
        return warped

    return src_rgb


def align_optical_flow(src_rgb: np.ndarray, dst_rgb: np.ndarray, mask: Optional[np.ndarray]=None) -> np.ndarray:
    gray1 = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 5, 25, 3, 5, 1.2, 0)
    h, w = gray1.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[...,0]).astype(np.float32)
    map_y = (grid_y + flow[...,1]).astype(np.float32)
    warped = cv2.remap(src_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped


# ------------------------------
# Aggregation & Postprocess
# ------------------------------

def aggregate_stack(stack: List[np.ndarray], mode: str = "median", tile: Optional[int] = None) -> np.ndarray:
    # Memory-aware aggregation
    n = len(stack)
    h, w = stack[0].shape[:2]
    if mode == "mean":
        acc = np.zeros((h, w, 3), dtype=np.float32)
        for im in stack:
            acc += im.astype(np.float32)
        agg = acc / float(n)
    elif mode == "max":
        acc = stack[0].astype(np.float32)
        for im in stack[1:]:
            acc = np.maximum(acc, im.astype(np.float32))
        agg = acc
    elif mode == "median":
        if tile is None:
            tile = 512
        agg = np.zeros((h, w, 3), dtype=np.float32)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                y2 = min(h, y+tile)
                x2 = min(w, x+tile)
                block = np.stack([im[y:y2, x:x2].astype(np.float32) for im in stack], axis=0)
                agg[y:y2, x:x2] = np.median(block, axis=0)
    
    else:
        raise ValueError(f"Unknown aggregate mode: {mode}")
    return to_uint8(agg)


def postprocess_image(rgb: np.ndarray, denoise: bool=False, clahe: bool=False) -> np.ndarray:
    out = rgb.copy()
    if denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 10, 10, 7, 21)
    if clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe_op.apply(l)
        lab2 = cv2.merge([l2, a, b])
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return out


# Optional: MPRNet wrapper (very light; expects you cloned swz30/MPRNet and point to a checkpoint)
class MPRNetWrapper:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        if not _HAS_TORCH:
            raise RuntimeError("torch not installed for MPRNet.")
        # Lightweight import pattern: expect that MPRNet code is placed under a local package `mprnet`
        # For simplicity here, we implement a generic UNet-ish fallback if ckpt not compatible.
        self.device = torch.device(device)
        self.model = None
        self.ckpt_path = ckpt_path
        # Placeholder: You can integrate the actual MPRNet repo here.
        warnings.warn("MPRNet integration is a stub. Please integrate the official repo for full quality.")

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        # Pass-through stub; replace with actual inference
        return rgb


# ------------------------------
# Information extraction: YOLO & OCR
# ------------------------------

def run_yolo(rgb: np.ndarray, save_dir: str) -> dict:
    if not _HAS_YOLO:
        warnings.warn("ultralytics not installed; skipping YOLO.")
        return {}
    ensure_dir(save_dir)
    tmp_path = os.path.join(save_dir, "_recon_tmp.jpg")
    imwrite_rgb(tmp_path, rgb)
    model = YOLO("yolov8n.pt")  # small model
    results = model(tmp_path, verbose=False)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = b.xyxy.squeeze(0).tolist()
            dets.append({"class_id": cls, "conf": conf, "xyxy": xyxy})
    # Save annotated image
    try:
        annotated = results[0].plot()
        cv2.imwrite(os.path.join(save_dir, "yolo_annotated.jpg"), annotated)
    except Exception:
        pass
    return {"detections": dets}


def run_ocr(rgb: np.ndarray) -> dict:
    if not _HAS_OCR:
        warnings.warn("pytesseract/PIL not installed or tesseract missing; skipping OCR.")
        return {}
    pil_im = Image.fromarray(rgb)
    text = pytesseract.image_to_string(pil_im)
    return {"text": text}


# ------------------------------
# Main pipeline
# ------------------------------

def reconstruct_background(
    frames: List[np.ndarray],
    seg: Segmenter,
    align_method: str = "orb",
    aggregate: str = "median",
    sample_every: int = 1,
) -> np.ndarray:
    """Core: make a reconstructed background image from frames."""
    H, W = frames[0].shape[:2]
    ref_idx = len(frames)//2
    ref = frames[ref_idx]

    # Build masks (1=FG, 0=BG)
    masks = []
    for f in tqdm(frames[::sample_every], desc="Segmentation"):
        m = seg(f)
        masks.append(m.astype(np.uint8))

    # Align each frame to reference, zero out FG
    aligned_stack = []
    ref_mask = masks[len(masks)//2]
    for f, m in tqdm(list(zip(frames[::sample_every], masks)), desc="Alignment", total=len(masks)):
        bg_only = f.copy()
        # Zero-out foreground region to prevent contamination
        bg_only[m == 1] = 0
        if align_method == "orb":
            warped = align_orb_affine(bg_only, ref, mask=m)
        elif align_method == "flow":
            warped = align_optical_flow(bg_only, ref, mask=m)
        else:
            raise ValueError("align_method must be 'orb' or 'flow'")
        aligned_stack.append(warped)

    # Aggregate
    recon = aggregate_stack(aligned_stack, mode=aggregate, tile=512)

    return recon


@dataclass
class Args:
    video: str
    out_dir: str
    fps: Optional[int]
    seg_backend: str
    align_method: str
    aggregate: str
    device: str
    sample_every: int
    resize_long: Optional[int]
    denoise: bool
    clahe: bool
    mprnet_ckpt: Optional[str]
    yolo: bool
    ocr: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Virtual Background Blur Attack Pipeline")
    p.add_argument("--video", required=True, help="Input video path (post-composite with blurred background)")
    p.add_argument("--out_dir", default="results/out", help="Output directory")
    p.add_argument("--fps", type=int, default=15, help="Frame sampling FPS (None=all frames)")
    p.add_argument("--resize_long", type=int, default=960, help="Resize so long side <= this px; 0 to disable")
    p.add_argument("--seg_backend", choices=["mediapipe", "deeplabv3"], default="mediapipe")
    p.add_argument("--align_method", choices=["orb", "flow"], default="orb")
    p.add_argument("--aggregate", choices=["median", "mean", "max"], default="median")
    p.add_argument("--device", default="cpu", help="cuda or cpu (for torch backends)")
    p.add_argument("--sample_every", type=int, default=1, help="Use every Nth frame in pipeline")
    p.add_argument("--denoise", action="store_true", help="Apply denoising postprocess")
    p.add_argument("--clahe", action="store_true", help="Apply CLAHE postprocess")
    p.add_argument("--mprnet_ckpt", default=None, help="Optional MPRNet checkpoint path for deblurring")
    p.add_argument("--yolo", action="store_true", help="Run YOLOv8 detection on reconstructed image")
    p.add_argument("--ocr", action="store_true", help="Run OCR on reconstructed image")
    a = p.parse_args()
    return Args(
        video=a.video,
        out_dir=a.out_dir,
        fps=a.fps,
        seg_backend=a.seg_backend,
        align_method=a.align_method,
        aggregate=a.aggregate,
        device=a.device,
        sample_every=a.sample_every,
        denoise=a.denoise,
        resize_long=a.resize_long,
        clahe=a.clahe,
        mprnet_ckpt=a.mprnet_ckpt,
        yolo=a.yolo,
        ocr=a.ocr,
    )


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    # 1) Load frames
    print("[1/6] Extracting frames...")
    frames = extract_frames(args.video, args.fps, resize_long=(args.resize_long or None))
    H, W = frames[0].shape[:2]
    print(f"  Loaded {len(frames)} frames at ~{H}x{W}")

    # 2) Build segmenter
    print("[2/6] Initializing segmenter...")
    seg = build_segmenter(args.seg_backend, args.device)

    # 3) Reconstruct background by align + aggregate
    print("[3/6] Reconstructing background (this may take a while)...")
    recon = reconstruct_background(frames, seg, args.align_method, args.aggregate, args.sample_every)
    imwrite_rgb(os.path.join(args.out_dir, "reconstructed_raw.png"), recon)

    # 4) Postprocess (denoise/CLAHE)
    print("[4/6] Postprocess...")
    recon_pp = postprocess_image(recon, denoise=args.denoise, clahe=args.clahe)

    # Optional MPRNet deblurring
    if args.mprnet_ckpt:
        print("  (Optional) MPRNet deblurring...")
        mpr = MPRNetWrapper(args.mprnet_ckpt, device=args.device)
        recon_pp = mpr(recon_pp)

    imwrite_rgb(os.path.join(args.out_dir, "reconstructed_post.png"), recon_pp)

    report = {
        "video": args.video,
        "out_dir": args.out_dir,
        "resolution": [int(W), int(H)],
        "num_frames_used": int(math.ceil(len(frames)/max(1,args.sample_every))),
        "seg_backend": args.seg_backend,
        "align_method": args.align_method,
        "aggregate": args.aggregate,
        "postprocess": {"denoise": args.denoise, "clahe": args.clahe},
        "mprnet_used": bool(args.mprnet_ckpt),
    }

    # 5) Information extraction
    print("[5/6] Information extraction...")
    if args.yolo:
        det = run_yolo(recon_pp, args.out_dir)
        report["yolo"] = det
    if args.ocr:
        report["ocr"] = run_ocr(recon_pp)

    # 6) Save report
    print("[6/6] Saving report...")
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Done. Outputs:")
    print(" - reconstructed_raw.png")
    print(" - reconstructed_post.png")
    if args.yolo:
        print(" - yolo_annotated.jpg (if YOLO enabled)")
    print(" - report.json")


if __name__ == "__main__":
    main()

'''
python zoom_bg_blur_attack_pipeline.py \
  --video videos/meeting_01.mp4 \
  --out_dir results/me_result \
  --seg_backend deeplabv3 \
  --device cuda \
  --align_method flow \
  --aggregate median \
  --fps 20 \
  --resize_long 1280 \
  --sample_every 1 \
  --denoise --clahe \
  --yolo --ocr

python zoom_bg_blur_attack_pipeline.py \
  --video videos/meeting_01.mp4 \
  --out_dir results/me_best_quality \
  --seg_backend deeplabv3 \
  --device cuda \
  --align_method flow \
  --aggregate median \
  --fps 30 \
  --resize_long 1920 \
  --sample_every 1 \
  --denoise --clahe \
  --yolo --ocr

  python /app/MPRNet/Deblurring/test.py \
  --input_dir /app/results/run_best \
  --result_dir /app/results/run_best/mprnet_out \
  --weights /app/MPRNet/Deblurring/pretrained_models/model_deblurring.pth

'''