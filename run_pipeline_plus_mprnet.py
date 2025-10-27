#!/usr/bin/env python3
"""
Run full pipeline + MPRNet enhancement + optional YOLO/OCR on enhanced image.
=============================================================================

This script orchestrates three stages:
  1) Run your main pipeline (zoom_bg_blur_attack_pipeline.py)
  2) Run MPRNet single-image deblurring on reconstructed_post.png
  3) (Optional) Run YOLO/OCR on the enhanced image and save an enhanced report

Example (inside container):

  python run_pipeline_plus_mprnet.py \
    --video /app/videos/sample2.mp4 \
    --out_dir /app/results/full_run \
    --seg_backend deeplabv3 --device cuda \
    --align_method flow --aggregate median \
    --fps 30 --resize_long 1600 --sample_every 1 \
    --denoise --clahe \
    --mprnet_repo /app/MPRNet \
    --mprnet_weights /app/MPRNet/Deblurring/pretrained_models/model_deblurring.pth \
    --yolo --ocr

Outputs of interest under --out_dir:
  reconstructed_post.png                (from stage 1)
  reconstructed_post_mprnet.png         (from stage 2)
  yolo_mprnet_annotated.jpg (if --yolo)
  report.json                           (from stage 1)
  report_enhanced.json                  (stage 3 results on MPRNet output)
"""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys


def run_cmd(cmd: str):
    print(f"[RUN] {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise SystemExit(f"Command failed (exit {ret}): {cmd}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pipeline + MPRNet + optional YOLO/OCR on enhanced image")
    # Main pipeline args
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seg_backend", choices=["mediapipe","deeplabv3"], default="mediapipe")
    p.add_argument("--device", default="cpu")
    p.add_argument("--align_method", choices=["orb","flow"], default="flow")
    p.add_argument("--aggregate", choices=["median","mean","max"], default="median")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--resize_long", type=int, default=1280)
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--clahe", action="store_true")
    p.add_argument("--yolo", action="store_true", help="Also run YOLO on base pipeline output (stage 1)")
    p.add_argument("--ocr", action="store_true", help="Also run OCR on base pipeline output (stage 1)")
    # MPRNet
    p.add_argument("--mprnet_repo", required=True)
    p.add_argument("--mprnet_weights", required=True)
    # Extra: run detections on enhanced image too
    p.add_argument("--ie_on_enhanced", action="store_true", help="Run YOLO/OCR on MPRNet output and write report_enhanced.json")
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # 1) Main pipeline
    base_cmd = [
        sys.executable, 'zoom_bg_blur_attack_pipeline.py',
        '--video', a.video,
        '--out_dir', a.out_dir,
        '--seg_backend', a.seg_backend,
        '--device', a.device,
        '--align_method', a.align_method,
        '--aggregate', a.aggregate,
        '--fps', str(a.fps),
        '--resize_long', str(a.resize_long),
        '--sample_every', str(a.sample_every)
    ]
    if a.denoise: base_cmd.append('--denoise')
    if a.clahe: base_cmd.append('--clahe')
    if a.yolo: base_cmd.append('--yolo')
    if a.ocr: base_cmd.append('--ocr')

    run_cmd(' '.join(shlex.quote(x) for x in base_cmd))

    # 2) MPRNet single-image inference
    src_png = os.path.join(a.out_dir, 'reconstructed_post.png')
    out_png = os.path.join(a.out_dir, 'reconstructed_post_mprnet.png')
    if not os.path.exists(src_png):
        raise SystemExit(f"Missing base image for MPRNet: {src_png}")

    mpr_cmd = [
        sys.executable, 'inference_mprnet_single.py',
        '--repo', a.mprnet_repo,
        '--weights', a.mprnet_weights,
        '--input', src_png,
        '--output', out_png,
        '--device', a.device
    ]
    run_cmd(' '.join(shlex.quote(x) for x in mpr_cmd))

    # 3) Optionally run YOLO/OCR on enhanced image and write report_enhanced.json
    if a.ie_on_enhanced:
        # We call a tiny inline Python to use ultralytics/pytesseract if available.
        inline = f"""
from ultralytics import YOLO
import json, os
from PIL import Image
import pytesseract
import cv2

inp = {out_png!r}
out_json = os.path.join({a.out_dir!r}, 'report_enhanced.json')
out_anno = os.path.join({a.out_dir!r}, 'yolo_mprnet_annotated.jpg')

rep = {{'enhanced_image': inp}}

# YOLO
try:
    m = YOLO('yolov8n.pt')
    r = m(inp, verbose=False)
    dets = []
    for x in r:
        if x.boxes is None: continue
        for b in x.boxes:
            dets.append({{
                'class_id': int(b.cls.item()),
                'conf': float(b.conf.item()),
                'xyxy': [float(v) for v in b.xyxy.squeeze(0).tolist()]
            }})
    rep['yolo'] = {{'detections': dets}}
    # save annotated
    cv2.imwrite(out_anno, r[0].plot())
except Exception as e:
    rep['yolo_error'] = str(e)

# OCR
try:
    txt = pytesseract.image_to_string(Image.open(inp))
    rep['ocr'] = {{'text': txt}}
except Exception as e:
    rep['ocr_error'] = str(e)

with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print('Enhanced IE saved:', out_json)
print('Enhanced YOLO annotated:', out_anno)
"""
        run_cmd(f"{shlex.quote(sys.executable)} - <<'PY'\n{inline}\nPY")

    print("\nAll done. Key outputs:")
    print(f" - {src_png}")
    print(f" - {out_png}")
    if a.ie_on_enhanced:
        print(f" - {os.path.join(a.out_dir, 'report_enhanced.json')}")
        print(f" - {os.path.join(a.out_dir, 'yolo_mprnet_annotated.jpg')}")


if __name__ == '__main__':
    main()
