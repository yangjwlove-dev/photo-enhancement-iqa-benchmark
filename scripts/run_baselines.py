#!/usr/bin/env python3
"""Pure local open-source baseline image processor for Open IQA Benchmark (v1.0.0).

Zero external API calls. Deterministic local algorithms only:
  - opencv_fastnlmeans_clahe : OpenCV fastNlMeansDenoisingColored + LAB CLAHE
  - lanczos_2x_unsharp       : Pillow Lanczos 2x + mild UnsharpMask
  - lanczos_4x               : Pillow Lanczos 4x
  - wiener_unsharp_deblur    : Reflect-padded Wiener deconvolution (Gaussian PSF sigma=2.0)
  - opencv_nlm_deblock       : OpenCV fastNlMeansDenoisingColored for JPEG Q30 deblocking
  - clahe_nlm_unsharp_text   : Mild OpenCV NLM + LAB CLAHE + mild UnsharpMask for archival print
  - nearest_neighbor_4x_rgba : Exact 4x integer nearest-neighbor scaling preserving RGBA & palette
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from skimage.restoration import wiener


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def gaussian_psf(sigma: float = 2.0, size: int = 13) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def apply_opencv_nlm_clahe(before_path: Path, after_path: Path, h_val: float = 6.0, clip_limit: float = 2.0) -> None:
    bgr = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {before_path}")
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h_val, h_val, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enh = clahe.apply(l_ch)
    merged = cv2.merge((l_enh, a_ch, b_ch))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    after_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(after_path, format="PNG", optimize=True)


def apply_lanczos_2x_unsharp(before_path: Path, after_path: Path) -> None:
    with Image.open(before_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        up = rgb.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        sharp = up.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
        after_path.parent.mkdir(parents=True, exist_ok=True)
        sharp.save(after_path, format="PNG", optimize=True)


def apply_lanczos_4x(before_path: Path, after_path: Path) -> None:
    with Image.open(before_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        up = rgb.resize((w * 4, h * 4), Image.Resampling.LANCZOS)
        after_path.parent.mkdir(parents=True, exist_ok=True)
        up.save(after_path, format="PNG", optimize=True)


def apply_wiener_deblur(before_path: Path, after_path: Path) -> None:
    with Image.open(before_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    psf = gaussian_psf(sigma=2.0, size=13)
    pad = 16
    out = np.empty_like(rgb)
    for c in range(3):
        ch_pad = np.pad(rgb[:, :, c], pad, mode="reflect")
        deconv = wiener(ch_pad, psf, balance=0.05, clip=True)
        out[:, :, c] = deconv[pad:-pad, pad:-pad]
    out_u8 = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    after_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_u8).save(after_path, format="PNG", optimize=True)


def apply_opencv_nlm_deblock(before_path: Path, after_path: Path) -> None:
    bgr = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {before_path}")
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 7, 7, 7, 21)
    rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    after_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(after_path, format="PNG", optimize=True)


def apply_clahe_nlm_unsharp_text(before_path: Path, after_path: Path) -> None:
    bgr = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {before_path}")
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 4, 4, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_enh = clahe.apply(l_ch)
    rgb = cv2.cvtColor(cv2.merge((l_enh, a_ch, b_ch)), cv2.COLOR_LAB2RGB)
    pil_img = Image.fromarray(rgb).filter(ImageFilter.UnsharpMask(radius=1.2, percent=90, threshold=3))
    after_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(after_path, format="PNG", optimize=True)


def apply_nearest_neighbor_4x_rgba(before_path: Path, after_path: Path) -> None:
    with Image.open(before_path) as img:
        rgba = img.convert("RGBA")
        w, h = rgba.size
        up = rgba.resize((w * 4, h * 4), Image.Resampling.NEAREST)
        after_path.parent.mkdir(parents=True, exist_ok=True)
        up.save(after_path, format="PNG", optimize=True)


def run_method(method: str, before_path: Path, after_path: Path) -> None:
    if method == "opencv_fastnlmeans_clahe":
        apply_opencv_nlm_clahe(before_path, after_path)
    elif method == "lanczos_2x_unsharp":
        apply_lanczos_2x_unsharp(before_path, after_path)
    elif method == "lanczos_4x":
        apply_lanczos_4x(before_path, after_path)
    elif method == "wiener_unsharp_deblur":
        apply_wiener_deblur(before_path, after_path)
    elif method == "opencv_nlm_deblock":
        apply_opencv_nlm_deblock(before_path, after_path)
    elif method == "clahe_nlm_unsharp_text":
        apply_clahe_nlm_unsharp_text(before_path, after_path)
    elif method == "nearest_neighbor_4x_rgba":
        apply_nearest_neighbor_4x_rgba(before_path, after_path)
    else:
        raise ValueError(f"Unsupported baseline_method: {method}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic local open-source baselines for all 40 samples.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root")
    parser.add_argument("--samples", type=Path, default=None, help="Path to samples.csv")
    parser.add_argument("--update-csv", action="store_true", help="Update after_sha256 in samples.csv")
    args = parser.parse_args()

    root = args.root.resolve()
    csv_path = (args.samples or (root / "data" / "samples.csv")).resolve()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for row in rows:
        sid = row["sample_id"]
        method = row["baseline_method"]
        before_path = root / row["before_path"]
        after_path = root / row["after_path"]
        run_method(method, before_path, after_path)
        if args.update_csv:
            row["after_sha256"] = sha256_file(after_path)
        print(f"[BASELINE] {sid} | {row['category']:24s} | {method:25s} -> {row['after_path']}")

    if args.update_csv:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[UPDATED] after_sha256 synced to {csv_path}")

    print(f"Done. Generated {len(rows)} local baseline outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
