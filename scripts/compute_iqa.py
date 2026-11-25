#!/usr/bin/env python3
"""Deterministic objective IQA & pixel-art metric suite for Open IQA Benchmark (v1.0.0).

Computes zero-external-weight, 100% reproducible open-source metrics using
scikit-image, OpenCV, SciPy, NumPy, and Pillow:
  - Full-reference metrics (for controlled degradation samples with reference_path):
      * PSNR (dB, higher is better)
      * SSIM (0..1, higher is better)
      * For upscale_2x / upscale_4x, `before` FR metrics are evaluated on naive
        bicubic upscaling to reference dimensions so deltas quantify improvement
        (or regression) relative to standard bicubic interpolation.
  - No-reference deterministic spatial metrics (for all 35 natural/archival images):
      * Laplacian Variance Sharpness (higher = sharper high-frequency detail)
      * Tenengrad Gradient Energy (Sobel magnitude squared mean)
      * RMS Luminance Contrast (standard deviation of grayscale intensity)
      * Shannon Entropy (8-bit luminance entropy in bits)
  - Pixel-art exactness & color-pollution checks (for all 5 pixel_art_upscale sprites):
      * Exact 4x integer dimensions
      * Unique RGBA color count & Delta Unique Colors (must be 0 for nearest-neighbor)
      * Exact 4x RGBA alpha channel Kronecker integrity
      * Comparison against naive Bicubic 4x scaling (quantifying color pollution)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_image_stats(path: Path) -> tuple[int, int, float]:
    kb = round(path.stat().st_size / 1024.0, 2)
    with Image.open(path) as img:
        w, h = img.size
    return w, h, kb


def compute_nr_metrics(path: Path) -> dict[str, float]:
    with Image.open(path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))

    rms_contrast = float(np.std(gray))

    hist, _ = np.histogram(gray.astype(np.uint8), bins=256, range=(0, 256))
    probs = hist[hist > 0] / float(gray.size)
    entropy = float(-np.sum(probs * np.log2(probs)))

    return {
        "laplacian_var": round(lap_var, 4),
        "tenengrad": round(tenengrad, 4),
        "rms_contrast": round(rms_contrast, 4),
        "shannon_entropy": round(entropy, 4),
    }


def compute_fr_metrics(test_arr: np.ndarray, ref_arr: np.ndarray) -> tuple[float, float]:
    psnr = float(peak_signal_noise_ratio(ref_arr, test_arr, data_range=255))
    ssim = float(structural_similarity(ref_arr, test_arr, channel_axis=2, data_range=255))
    return round(psnr, 4), round(ssim, 4)


def analyze_pixel_art(before_path: Path, after_path: Path) -> dict[str, Any]:
    with Image.open(before_path) as b_img:
        b_has_alpha = b_img.mode in ("RGBA", "LA") or ("transparency" in b_img.info)
        b_rgba = b_img.convert("RGBA")
        b_arr = np.asarray(b_rgba, dtype=np.uint8)
        bw, bh = b_rgba.size
        b_unique = int(np.unique(b_arr.reshape(-1, 4), axis=0).shape[0])
        b_trans_ratio = round(float(np.mean(b_arr[:, :, 3] == 0)), 4)

        # Counterfactual naive Bicubic 4x scaling to quantify palette contamination
        bicubic_4x = b_rgba.resize((bw * 4, bh * 4), Image.Resampling.BICUBIC)
        bic_arr = np.asarray(bicubic_4x, dtype=np.uint8)
        bic_unique = int(np.unique(bic_arr.reshape(-1, 4), axis=0).shape[0])

    with Image.open(after_path) as a_img:
        a_has_alpha = a_img.mode in ("RGBA", "LA") or ("transparency" in a_img.info)
        a_rgba = a_img.convert("RGBA")
        a_arr = np.asarray(a_rgba, dtype=np.uint8)
        aw, ah = a_rgba.size
        a_unique = int(np.unique(a_arr.reshape(-1, 4), axis=0).shape[0])

    exact_4x = (aw == bw * 4) and (ah == bh * 4)
    alpha_ok = bool(b_has_alpha and a_has_alpha)
    if alpha_ok and exact_4x:
        expected_alpha = np.repeat(np.repeat(b_arr[:, :, 3], 4, axis=0), 4, axis=1)
        alpha_ok = bool(np.array_equal(a_arr[:, :, 3], expected_alpha))

    return {
        "pixel_unique_colors_before": b_unique,
        "pixel_unique_colors_after": a_unique,
        "pixel_delta_colors": a_unique - b_unique,
        "pixel_exact_4x": "true" if exact_4x else "false",
        "pixel_alpha_intact": "true" if alpha_ok else "false",
        "pixel_transparent_ratio_before": b_trans_ratio,
        "pixel_bicubic_4x_colors": bic_unique,
        "pixel_bicubic_4x_delta_colors": bic_unique - b_unique,
    }


def fmt_delta(a: float, b: float) -> float:
    return round(a - b, 4)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute deterministic open-source IQA & pixel-art metrics.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root")
    parser.add_argument("--samples", type=Path, default=None, help="Path to samples.csv")
    parser.add_argument("--output", type=Path, default=None, help="Output metrics CSV path")
    parser.add_argument("--summary", type=Path, default=None, help="Output summary JSON path")
    args = parser.parse_args()

    root = args.root.resolve()
    csv_path = (args.samples or (root / "data" / "samples.csv")).resolve()
    out_csv = (args.output or (root / "data" / "metrics_iqa.csv")).resolve()
    summary_path = (args.summary or (root / "data" / "summary.json")).resolve()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    fieldnames = [
        "sample_id",
        "category",
        "baseline_method",
        "degradation_type",
        "before_w",
        "before_h",
        "after_w",
        "after_h",
        "before_kb",
        "after_kb",
        "psnr_before",
        "psnr_after",
        "psnr_delta",
        "ssim_before",
        "ssim_after",
        "ssim_delta",
        "laplacian_var_before",
        "laplacian_var_after",
        "laplacian_var_delta",
        "tenengrad_before",
        "tenengrad_after",
        "tenengrad_delta",
        "rms_contrast_before",
        "rms_contrast_after",
        "rms_contrast_delta",
        "shannon_entropy_before",
        "shannon_entropy_after",
        "shannon_entropy_delta",
        "pixel_unique_colors_before",
        "pixel_unique_colors_after",
        "pixel_delta_colors",
        "pixel_exact_4x",
        "pixel_alpha_intact",
        "pixel_transparent_ratio_before",
        "pixel_bicubic_4x_colors",
        "pixel_bicubic_4x_delta_colors",
        "notes",
    ]

    results: list[dict[str, Any]] = []
    for row in rows:
        sid = row["sample_id"]
        cat = row["category"]
        method = row["baseline_method"]
        deg = row["degradation_type"]
        before_path = root / row["before_path"]
        after_path = root / row["after_path"]
        ref_rel = (row.get("reference_path") or "").strip()
        ref_path = (root / ref_rel) if ref_rel else None

        bw, bh, bkb = get_image_stats(before_path)
        aw, ah, akb = get_image_stats(after_path)

        rec: dict[str, Any] = {k: "" for k in fieldnames}
        rec.update(
            {
                "sample_id": sid,
                "category": cat,
                "baseline_method": method,
                "degradation_type": deg,
                "before_w": bw,
                "before_h": bh,
                "after_w": aw,
                "after_h": ah,
                "before_kb": bkb,
                "after_kb": akb,
            }
        )

        if cat == "pixel_art_upscale":
            px = analyze_pixel_art(before_path, after_path)
            rec.update(px)
            rec["notes"] = "exact_integer_4x_vs_bicubic_counterfactual"
            print(
                f"[PIXEL] {sid} | {bw}x{bh}->{aw}x{ah} | colors={px['pixel_unique_colors_before']}->"
                f"{px['pixel_unique_colors_after']} (delta={px['pixel_delta_colors']}) "
                f"| bicubic_4x_colors={px['pixel_bicubic_4x_colors']} (+{px['pixel_bicubic_4x_delta_colors']})"
            )
        else:
            b_nr = compute_nr_metrics(before_path)
            a_nr = compute_nr_metrics(after_path)
            for m in ("laplacian_var", "tenengrad", "rms_contrast", "shannon_entropy"):
                rec[f"{m}_before"] = b_nr[m]
                rec[f"{m}_after"] = a_nr[m]
                rec[f"{m}_delta"] = fmt_delta(a_nr[m], b_nr[m])

            if ref_path is not None and ref_path.exists():
                with Image.open(ref_path) as ref_im:
                    ref_arr = np.asarray(ref_im.convert("RGB"), dtype=np.uint8)
                with Image.open(after_path) as aft_im:
                    aft_arr = np.asarray(aft_im.convert("RGB"), dtype=np.uint8)
                with Image.open(before_path) as bef_im:
                    bef_rgb = bef_im.convert("RGB")
                    if bef_rgb.size != (ref_arr.shape[1], ref_arr.shape[0]):
                        # Upscale baseline comparison: naive Bicubic upsampling to reference dimensions
                        bef_eval = bef_rgb.resize((ref_arr.shape[1], ref_arr.shape[0]), Image.Resampling.BICUBIC)
                        rec["notes"] = "fr_before_is_bicubic_upsample_to_ref"
                    else:
                        bef_eval = bef_rgb
                        rec["notes"] = "fr_same_resolution_vs_ref"
                    bef_arr = np.asarray(bef_eval, dtype=np.uint8)

                psnr_b, ssim_b = compute_fr_metrics(bef_arr, ref_arr)
                psnr_a, ssim_a = compute_fr_metrics(aft_arr, ref_arr)
                rec["psnr_before"] = psnr_b
                rec["psnr_after"] = psnr_a
                rec["psnr_delta"] = fmt_delta(psnr_a, psnr_b)
                rec["ssim_before"] = ssim_b
                rec["ssim_after"] = ssim_a
                rec["ssim_delta"] = fmt_delta(ssim_a, ssim_b)
                print(
                    f"[FR+NR] {sid} ({cat:12s}) | PSNR {psnr_b:.2f}->{psnr_a:.2f} ({rec['psnr_delta']:+.2f} dB) "
                    f"| SSIM {ssim_b:.4f}->{ssim_a:.4f} ({rec['ssim_delta']:+.4f}) "
                    f"| LapVar {b_nr['laplacian_var']:.1f}->{a_nr['laplacian_var']:.1f}"
                )
            else:
                rec["notes"] = "no_reference_archival_scan"
                print(
                    f"[NR]    {sid} ({cat:23s}) | LapVar {b_nr['laplacian_var']:.1f}->{a_nr['laplacian_var']:.1f} "
                    f"| RMS {b_nr['rms_contrast']:.1f}->{a_nr['rms_contrast']:.1f} "
                    f"| Entropy {b_nr['shannon_entropy']:.2f}->{a_nr['shannon_entropy']:.2f}"
                )

        results.append(rec)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[SAVED] {out_csv} ({len(results)} rows)")

    # Build aggregated summary.json
    numeric_keys = [
        "psnr_before", "psnr_after", "psnr_delta",
        "ssim_before", "ssim_after", "ssim_delta",
        "laplacian_var_before", "laplacian_var_after", "laplacian_var_delta",
        "tenengrad_before", "tenengrad_after", "tenengrad_delta",
        "rms_contrast_before", "rms_contrast_after", "rms_contrast_delta",
        "shannon_entropy_before", "shannon_entropy_after", "shannon_entropy_delta",
        "pixel_unique_colors_before", "pixel_unique_colors_after", "pixel_delta_colors",
        "pixel_bicubic_4x_colors", "pixel_bicubic_4x_delta_colors",
    ]

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    agg_by_cat: dict[str, Any] = {}
    for cat, cat_rows in by_cat.items():
        entry: dict[str, Any] = {
            "count": len(cat_rows),
            "baseline_methods": sorted({r["baseline_method"] for r in cat_rows}),
        }
        for k in numeric_keys:
            vals = [float(r[k]) for r in cat_rows if r[k] != ""]
            if vals:
                entry[f"{k}_mean"] = round(statistics.mean(vals), 4)
                entry[f"{k}_median"] = round(statistics.median(vals), 4)
        if cat == "pixel_art_upscale":
            entry["exact_4x_pass_count"] = sum(1 for r in cat_rows if r["pixel_exact_4x"] == "true")
            entry["alpha_intact_pass_count"] = sum(1 for r in cat_rows if r["pixel_alpha_intact"] == "true")
            entry["zero_color_pollution_count"] = sum(1 for r in cat_rows if int(r["pixel_delta_colors"]) == 0)
        agg_by_cat[cat] = entry

    # Also aggregate deblur_jpeg by sub-degradation (gaussian blur vs jpeg q30)
    deblur_sub: dict[str, Any] = {}
    for sub in ("synthetic_gaussian_blur", "synthetic_jpeg_q30"):
        sub_rows = [r for r in results if r["category"] == "deblur_jpeg" and r["degradation_type"] == sub]
        if sub_rows:
            s_entry: dict[str, Any] = {"count": len(sub_rows), "baseline_method": sub_rows[0]["baseline_method"]}
            for k in ("psnr_before", "psnr_after", "psnr_delta", "ssim_before", "ssim_after", "ssim_delta", "laplacian_var_delta"):
                vals = [float(r[k]) for r in sub_rows if r[k] != ""]
                if vals:
                    s_entry[f"{k}_mean"] = round(statistics.mean(vals), 4)
                    s_entry[f"{k}_median"] = round(statistics.median(vals), 4)
            deblur_sub[sub] = s_entry

    checksums = {}
    for rel in (
        "data/samples.csv",
        "data/metrics_iqa.csv",
        "scripts/fetch_sources.py",
        "scripts/make_degraded.py",
        "scripts/run_baselines.py",
        "scripts/verify_licenses.py",
        "scripts/compute_iqa.py",
    ):
        p = root / rel
        if p.exists():
            checksums[rel] = sha256_file(p)

    summary_obj = {
        "dataset_title": "Open Image Enhancement, Deblurring & Pixel-Art Scaling Baseline Dataset (v1.0.0)",
        "version": "1.0.0",
        "total_samples": len(results),
        "metric_backend": "deterministic_open_source (scikit-image PSNR/SSIM, OpenCV Laplacian/Tenengrad, NumPy RMS/Entropy, Pillow/NumPy Pixel-Art Palette & Alpha)",
        "by_category": agg_by_cat,
        "deblur_jpeg_subgroups": deblur_sub,
        "file_sha256": checksums,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    print(f"[SAVED] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
