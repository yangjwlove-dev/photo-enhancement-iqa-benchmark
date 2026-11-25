#!/usr/bin/env python3
"""License, provenance, integrity, and quota verifier for Open IQA Benchmark (v1.0.0)."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from PIL import Image

REQUIRED_COLUMNS = [
    "sample_id",
    "category",
    "baseline_method",
    "related_reference_url",
    "source_archive",
    "source_url",
    "source_license",
    "degradation_type",
    "before_path",
    "after_path",
    "reference_path",
    "before_sha256",
    "after_sha256",
    "captured_at_utc",
]

ALLOWED_LICENSES = {
    "CC0 1.0",
    "Public Domain (PD-US-expired)",
    "Public Domain (PD-old-100)",
    "Public Domain (NASA)",
    "Public Domain (LOC No Known Restrictions)",
    "Public Domain (Wikimedia PD)",
}

FORBIDDEN_DOMAINS = (
    "unsplash.com",
    "pexels.com",
    "pixabay.com",
    "shutterstock.com",
    "gettyimages.com",
    "istockphoto.com",
    "stock.adobe.com",
    "alamy.com",
    "freepik.com",
    "123rf.com",
    "dreamstime.com",
)

EXPECTED_CATEGORY_COUNTS = {
    "old_photo_denoise": 8,
    "upscale_2x": 8,
    "upscale_4x": 8,
    "deblur_jpeg": 6,
    "historical_text_clarity": 5,
    "pixel_art_upscale": 5,
}

ALLOWED_DEGRADATIONS = {
    "natural",
    "synthetic_bicubic_2x",
    "synthetic_bicubic_4x",
    "synthetic_gaussian_blur",
    "synthetic_jpeg_q30",
}

ALLOWED_PIXEL_SIZES = {16, 32, 64}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_image_file(path: Path) -> tuple[tuple[int, int], str, str | None]:
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        img.load()
        return img.size, img.mode, img.format


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all 40 benchmark samples, licenses, and hashes.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root")
    parser.add_argument("--samples", type=Path, default=None, help="Path to samples.csv")
    parser.add_argument(
        "--allow-missing-after",
        action="store_true",
        help="Allow missing after images (for Stage 1 verification before running baselines)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    csv_path = (args.samples or (root / "data" / "samples.csv")).resolve()
    errors: list[str] = []

    if not csv_path.exists():
        print(f"[FAIL] Missing samples.csv: {csv_path}")
        return 1

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing_cols:
        errors.append(f"Missing required CSV columns: {missing_cols}")

    if len(rows) != 40:
        errors.append(f"Expected exactly 40 rows, found {len(rows)}")

    seen_ids: set[str] = set()
    seen_before_paths: set[str] = set()
    seen_after_paths: set[str] = set()
    seen_source_urls: set[str] = set()
    cat_counts: Counter[str] = Counter()
    deblur_jpeg_sub: Counter[str] = Counter()
    passed_rows = 0

    for idx, row in enumerate(rows, start=1):
        sid = (row.get("sample_id") or "").strip()
        cat = (row.get("category") or "").strip()
        baseline_method = (row.get("baseline_method") or "").strip()
        ref_url = (row.get("related_reference_url") or "").strip()
        src_archive = (row.get("source_archive") or "").strip()
        src_url = (row.get("source_url") or "").strip()
        src_lic = (row.get("source_license") or "").strip()
        deg = (row.get("degradation_type") or "").strip()
        before_rel = (row.get("before_path") or "").strip()
        before_sha = (row.get("before_sha256") or "").strip()
        after_rel = (row.get("after_path") or "").strip()
        after_sha = (row.get("after_sha256") or "").strip()
        ref_rel = (row.get("reference_path") or "").strip()
        ref_sha = (row.get("reference_sha256") or "").strip()

        row_errors: list[str] = []

        if not sid:
            row_errors.append("empty sample_id")
        elif sid in seen_ids:
            row_errors.append(f"duplicate sample_id {sid}")
        seen_ids.add(sid)

        cat_counts[cat] += 1
        if cat not in EXPECTED_CATEGORY_COUNTS:
            row_errors.append(f"unknown category '{cat}'")

        if not baseline_method:
            row_errors.append("empty baseline_method")
        elif "enhancephoto" in baseline_method.lower():
            row_errors.append(f"baseline_method must name local open-source algorithm, got '{baseline_method}'")

        if not ref_url.startswith("https://enhancephoto.org/"):
            row_errors.append(f"invalid related_reference_url '{ref_url}'")

        if not src_archive:
            row_errors.append("empty source_archive")

        if not src_url.startswith("https://"):
            row_errors.append(f"source_url must start with https:// (got '{src_url}')")
        elif src_url in seen_source_urls and cat != "pixel_art_upscale":
            row_errors.append(f"duplicate source_url '{src_url}'")
        seen_source_urls.add(src_url)

        combined_src = f"{src_archive} {src_url}".lower()
        for bad in FORBIDDEN_DOMAINS:
            if bad in combined_src:
                row_errors.append(f"forbidden stock domain '{bad}' detected")

        if src_lic not in ALLOWED_LICENSES:
            row_errors.append(f"disallowed source_license '{src_lic}'")

        if deg not in ALLOWED_DEGRADATIONS:
            row_errors.append(f"invalid degradation_type '{deg}'")

        if cat == "deblur_jpeg":
            deblur_jpeg_sub[deg] += 1

        if not before_rel:
            row_errors.append("empty before_path")
        elif before_rel in seen_before_paths:
            row_errors.append(f"duplicate before_path '{before_rel}'")
        seen_before_paths.add(before_rel)

        before_path = root / before_rel if before_rel else None
        bw = bh = 0
        if before_path is None or not before_path.exists():
            row_errors.append(f"before_path does not exist: {before_rel}")
        else:
            size_bytes = before_path.stat().st_size
            if size_bytes < 100:
                row_errors.append(f"before_path too small ({size_bytes} bytes)")
            actual_sha = sha256_file(before_path)
            if before_sha != actual_sha:
                row_errors.append(f"before_sha256 mismatch (csv={before_sha[:12]}, actual={actual_sha[:12]})")
            try:
                (bw, bh), mode, fmt = verify_image_file(before_path)
                if cat == "pixel_art_upscale":
                    if fmt != "PNG":
                        row_errors.append(f"pixel art must be PNG, got {fmt}")
                    if mode != "RGBA":
                        row_errors.append(f"pixel art must be RGBA, got {mode}")
                    if bw not in ALLOWED_PIXEL_SIZES or bh not in ALLOWED_PIXEL_SIZES:
                        row_errors.append(f"pixel art size {bw}x{bh} not in {sorted(ALLOWED_PIXEL_SIZES)}")
                else:
                    if min(bw, bh) < 150 or max(bw, bh) > 8000:
                        row_errors.append(f"natural image size out of range: {bw}x{bh}")
            except Exception as exc:
                row_errors.append(f"failed to decode before_path: {exc}")

        if deg != "natural":
            if not ref_rel:
                row_errors.append("controlled degradation sample missing reference_path")
            else:
                ref_path = root / ref_rel
                if not ref_path.exists():
                    row_errors.append(f"reference_path does not exist: {ref_rel}")
                else:
                    actual_ref_sha = sha256_file(ref_path)
                    if ref_sha and ref_sha != actual_ref_sha:
                        row_errors.append(f"reference_sha256 mismatch (csv={ref_sha[:12]}, actual={actual_ref_sha[:12]})")
                    try:
                        (rw, rh), _, _ = verify_image_file(ref_path)
                        if bw and bh:
                            if deg == "synthetic_bicubic_2x" and (bw != rw // 2 or bh != rh // 2):
                                row_errors.append(f"2x downsample mismatch: ref={rw}x{rh}, before={bw}x{bh}")
                            elif deg == "synthetic_bicubic_4x" and (bw != rw // 4 or bh != rh // 4):
                                row_errors.append(f"4x downsample mismatch: ref={rw}x{rh}, before={bw}x{bh}")
                            elif deg in ("synthetic_gaussian_blur", "synthetic_jpeg_q30") and (bw != rw or bh != rh):
                                row_errors.append(f"{deg} dimension mismatch: ref={rw}x{rh}, before={bw}x{bh}")
                    except Exception as exc:
                        row_errors.append(f"failed to decode reference_path: {exc}")

        if not after_rel:
            if not args.allow_missing_after:
                row_errors.append("empty after_path")
        else:
            if after_rel in seen_after_paths:
                row_errors.append(f"duplicate after_path '{after_rel}'")
            seen_after_paths.add(after_rel)
            after_path = root / after_rel
            if not after_path.exists():
                if not args.allow_missing_after:
                    row_errors.append(f"after_path does not exist: {after_rel}")
            else:
                if after_path.stat().st_size < 100:
                    row_errors.append(f"after_path too small ({after_path.stat().st_size} bytes)")
                actual_after_sha = sha256_file(after_path)
                if not after_sha:
                    row_errors.append("missing after_sha256")
                elif after_sha != actual_after_sha:
                    row_errors.append(f"after_sha256 mismatch (csv={after_sha[:12]}, actual={actual_after_sha[:12]})")
                try:
                    verify_image_file(after_path)
                except Exception as exc:
                    row_errors.append(f"failed to decode after_path: {exc}")

        if row_errors:
            errors.append(f"Row {idx} ({sid}): " + "; ".join(row_errors))
        else:
            passed_rows += 1

    for cat, expected in EXPECTED_CATEGORY_COUNTS.items():
        actual = cat_counts.get(cat, 0)
        if actual != expected:
            errors.append(f"Category '{cat}' count mismatch: expected {expected}, got {actual}")

    if deblur_jpeg_sub.get("synthetic_gaussian_blur", 0) != 3 or deblur_jpeg_sub.get("synthetic_jpeg_q30", 0) != 3:
        errors.append(
            f"deblur_jpeg must have 3 synthetic_gaussian_blur + 3 synthetic_jpeg_q30, got {dict(deblur_jpeg_sub)}"
        )

    print("=== Open IQA Benchmark License & Integrity Verification ===")
    for cat, expected in EXPECTED_CATEGORY_COUNTS.items():
        print(f"  - {cat:25s}: {cat_counts.get(cat, 0)}/{expected}")

    if errors:
        print(f"\n[FAIL] {passed_rows}/{len(rows)} rows passed, {len(errors)} error(s) found:")
        for e in errors:
            print(f"  * {e}")
        return 1

    print(f"\n[PASS] {passed_rows}/40 PASS - All files, SHA-256 hashes, resolutions, and CC0/PD licenses verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
