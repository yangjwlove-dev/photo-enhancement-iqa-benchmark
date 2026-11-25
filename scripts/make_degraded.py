#!/usr/bin/env python3
"""Deterministic controlled degradation generator for EnhancePhoto IQA Benchmark.

Reads data/samples.csv and generates images/before/ from images/reference/
for all synthetic degradation rows:
  - synthetic_bicubic_2x    : 2x bicubic downsampling (saved as lossless PNG)
  - synthetic_bicubic_4x    : 4x bicubic downsampling (saved as lossless PNG)
  - synthetic_gaussian_blur : Gaussian blur radius=2.0 (saved as lossless PNG)
  - synthetic_jpeg_q30      : JPEG compression quality=30, 4:2:0 subsampling
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

BLUR_RADIUS = 2.0
JPEG_QUALITY = 30
JPEG_SUBSAMPLING = 2  # 4:2:0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_degradation(ref_path: Path, out_path: Path, deg_type: str) -> tuple[tuple[int, int], tuple[int, int]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(ref_path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        orig_size = img.size
        w, h = orig_size

        if deg_type == "synthetic_bicubic_2x":
            target = (max(1, w // 2), max(1, h // 2))
            degraded = img.resize(target, Image.Resampling.BICUBIC)
            degraded.save(out_path, format="PNG", optimize=True)
        elif deg_type == "synthetic_bicubic_4x":
            target = (max(1, w // 4), max(1, h // 4))
            degraded = img.resize(target, Image.Resampling.BICUBIC)
            degraded.save(out_path, format="PNG", optimize=True)
        elif deg_type == "synthetic_gaussian_blur":
            degraded = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
            degraded.save(out_path, format="PNG", optimize=True)
        elif deg_type == "synthetic_jpeg_q30":
            rgb = img.convert("RGB")
            rgb.save(
                out_path,
                format="JPEG",
                quality=JPEG_QUALITY,
                subsampling=JPEG_SUBSAMPLING,
                optimize=True,
            )
            degraded = rgb
        else:
            raise ValueError(f"Unsupported degradation_type: {deg_type}")

        return orig_size, degraded.size


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate controlled degraded inputs from reference images.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root")
    parser.add_argument("--samples", type=Path, default=None, help="Path to samples.csv")
    parser.add_argument("--update-csv", action="store_true", help="Update before_sha256/reference_sha256 in samples.csv")
    args = parser.parse_args()

    root = args.root.resolve()
    csv_path = (args.samples or (root / "data" / "samples.csv")).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"samples.csv not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    processed = 0
    for row in rows:
        deg_type = (row.get("degradation_type") or "").strip()
        if deg_type == "natural":
            if args.update_csv and row.get("before_path"):
                bp = root / row["before_path"]
                if bp.exists():
                    row["before_sha256"] = sha256_file(bp)
            continue

        ref_rel = (row.get("reference_path") or "").strip()
        before_rel = (row.get("before_path") or "").strip()
        if not ref_rel or not before_rel:
            raise ValueError(f"Sample {row.get('sample_id')} missing reference_path or before_path")

        ref_path = root / ref_rel
        before_path = root / before_rel
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image missing for {row.get('sample_id')}: {ref_path}")

        orig_size, deg_size = apply_degradation(ref_path, before_path, deg_type)
        processed += 1
        print(f"[OK] {row.get('sample_id')} ({deg_type}): {orig_size[0]}x{orig_size[1]} -> {deg_size[0]}x{deg_size[1]} -> {before_rel}")

        if args.update_csv:
            row["before_sha256"] = sha256_file(before_path)
            if "reference_sha256" in fieldnames:
                row["reference_sha256"] = sha256_file(ref_path)

    if args.update_csv:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[UPDATED] SHA-256 hashes synced to {csv_path}")

    print(f"Done. Generated {processed} controlled degraded image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
