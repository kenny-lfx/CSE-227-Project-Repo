#!/usr/bin/env python3
"""
Single-image inference wrapper for MPRNet (Deblurring)
======================================================

This script runs **one image** through MPRNet's deblurring model without
requiring the dataset directory structure expected by `Deblurring/test.py`.

Usage (inside your Docker container):

  python inference_mprnet_single.py \
    --repo /app/MPRNet \
    --weights /app/MPRNet/Deblurring/pretrained_models/model_deblurring.pth \
    --input /app/results/run_best/reconstructed_post.png \
    --output /app/results/run_best/reconstructed_post_mprnet.png \
    --device cuda

Notes
-----
- `--repo` should point to the root of the cloned MPRNet repo.
- Works with typical MPRNet structure: MPRNet/Deblurring/{MPRNet.py, utils.py, ...}
- Pads the input to multiples of 8 (reflect) and then unpads back to original size.
- Saves PNG to `--output` path.
"""
from __future__ import annotations
import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte


def add_repo_to_path(repo_root: str) -> None:
    """Ensure MPRNet repo is importable regardless of CWD.
    Expected layout:
      <repo_root>/Deblurring/MPRNet.py
      <repo_root>/Deblurring/utils.py
    """
    repo_root = os.path.abspath(repo_root)
    debl_dir = os.path.join(repo_root, "Deblurring")
    # Add both root and Deblurring to sys.path to cover different import styles
    for p in [repo_root, debl_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-image MPRNet deblurring inference")
    p.add_argument("--repo", required=True, help="Path to cloned MPRNet repo root")
    p.add_argument("--weights", required=True, help="Path to MPRNet deblurring checkpoint .pth")
    p.add_argument("--input", required=True, help="Input image path (PNG/JPG)")
    p.add_argument("--output", required=True, help="Output image path (PNG)")
    p.add_argument("--device", default="cuda", choices=["cuda","cpu"], help="Device for inference")
    p.add_argument("--pad_factor", type=int, default=8, help="Pad image to multiple of this value")
    return p.parse_args()


def load_mprnet(repo_root: str, weights: str, device: str = "cuda"):
    add_repo_to_path(repo_root)
    # Try the common import patterns
    try:
        from MPRNet import MPRNet as MPRNetModel  # when CWD is Deblurring/
        import utils as mutils
    except Exception:
        try:
            from Deblurring.MPRNet import MPRNet as MPRNetModel
            from Deblurring import utils as mutils
        except Exception:
            try:
                from MPRNet.Deblurring.MPRNet import MPRNet as MPRNetModel
                import MPRNet.Deblurring.utils as mutils
            except Exception as e:
                raise ImportError("Cannot import MPRNet/ utils from repo. Check --repo path.") from e

    model = MPRNetModel()
    # utils.load_checkpoint returns (start_epoch) in some repos; we only need state load side effect
    _ = mutils.load_checkpoint(model, weights)
    model.to(device)
    model.eval()
    return model


def read_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_image_rgb(path: str, rgb_u8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        path = path + ".png"
    cv2.imwrite(path, bgr)


def pad_to_multiple(img: torch.Tensor, factor: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = img.shape
    H = (h + factor - 1) // factor * factor
    W = (w + factor - 1) // factor * factor
    pad_h, pad_w = H - h, W - w
    if pad_h or pad_w:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
    return img, pad_h, pad_w


def main():
    args = parse_args()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Load model
    print(f"[MPRNet] repo={args.repo}\n          weights={args.weights}\n          device={device}")
    model = load_mprnet(args.repo, args.weights, device)

    # Read and prepare image
    rgb = read_image_rgb(args.input).astype(np.float32) / 255.0
    img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # Pad and infer
    img_pad, ph, pw = pad_to_multiple(img, args.pad_factor)
    with torch.no_grad():
        out_list = model(img_pad)
        # Many repos return a list/tuple of outputs by stage; use the first
        if isinstance(out_list, (list, tuple)):
            out = out_list[0]
        else:
            out = out_list
        out = torch.clamp(out, 0, 1)
        if ph or pw:
            _, _, h, w = img.shape
            out = out[:, :, :h, :w]
    out_np = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    out_u8 = img_as_ubyte(out_np)

    write_image_rgb(args.output, out_u8)
    print(f"[MPRNet] Saved: {args.output}")


if __name__ == "__main__":
    main()
