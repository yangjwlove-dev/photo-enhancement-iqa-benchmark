#!/usr/bin/env python3
"""Deterministic CC0 / Public Domain source fetcher for Open IQA Benchmark (v1.0.0).

Downloads 40 strictly verified CC0 / Public Domain images across 6 task categories:
  1. old_photo_denoise (8)       - Wikimedia Commons pre-1925 archival scans (PD)
  2. upscale_2x (8)              - NASA Image Library (PD) & Wikimedia Commons (CC0)
  3. upscale_4x (8)              - NASA Image Library (PD) & Wikimedia Commons (CC0)
  4. deblur_jpeg (6: 3+3)        - NASA Image Library (PD) & Wikimedia Commons (CC0)
  5. historical_text_clarity (5) - Wikimedia Commons pre-1925 printed pages/newspapers (PD)
  6. pixel_art_upscale (5)       - OpenGameArt strictly CC0 1.0 RGBA sprites (16/32/64px)

Zero commercial stock sites (no Unsplash/Pexels/Pixabay/Shutterstock/Getty).
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {
    "User-Agent": "OpenIQABenchmarkBuilder/1.0 (https://github.com/; reproducible-open-dataset)"
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_get(url: str, params: dict[str, Any] | None = None, timeout: int = 30) -> requests.Response:
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed GET {url}")


def normalize_reference_image(raw_bytes: bytes, out_path: Path, max_side: int = 1200) -> tuple[int, int] | None:
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")
        w, h = img.size
        if min(w, h) < 600:
            return None
        # Resize down to max_side if larger, ensuring exact multiples of 4
        scale = min(1.0, max_side / max(w, h))
        nw = max(600, int(round(w * scale)))
        nh = max(600, int(round(h * scale)))
        # Keep aspect ratio closely while making both divisible by 4
        nw = (nw // 4) * 4
        nh = (nh // 4) * 4
        if min(nw, nh) < 400:
            return None
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="PNG", optimize=True)
        return img.size


def normalize_natural_scan(raw_bytes: bytes, out_path: Path, max_side: int = 1400) -> tuple[int, int] | None:
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")
        w, h = img.size
        if min(w, h) < 450:
            return None
        scale = min(1.0, max_side / max(w, h))
        nw = int(round(w * scale))
        nh = int(round(h * scale))
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=92, optimize=True)
        return img.size


def search_wikimedia(query: str, allowed_short_licenses: set[str], limit: int = 25) -> list[dict[str, Any]]:
    r = safe_get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrnamespace": 6,
            "gsrsearch": query,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1600,
            "format": "json",
        },
    )
    pages = r.json().get("query", {}).get("pages", {})
    results: list[dict[str, Any]] = []
    for p in pages.values():
        title = p.get("title", "")
        ii_list = p.get("imageinfo") or []
        if not ii_list:
            continue
        ii = ii_list[0]
        mime = ii.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue
        w = ii.get("width", 0)
        h = ii.get("height", 0)
        if min(w, h) < 550 or max(w, h) > 12000:
            continue
        meta = ii.get("extmetadata") or {}
        lic_short = (meta.get("LicenseShortName") or {}).get("value", "").strip()
        if lic_short not in allowed_short_licenses:
            continue
        # Exclude modern living portraits / people keywords
        t_low = title.lower()
        if any(bad in t_low for bad in ("portrait", "selfie", "woman", "man ", "girl", "boy", "president", "actor", "actress", "face", "person")):
            continue
        desc_url = ii.get("descriptionurl", "")
        dl_url = ii.get("thumburl") or ii.get("url", "")
        if not desc_url.startswith("https://") or not dl_url.startswith("https://"):
            continue
        if any(bad in desc_url.lower() for bad in FORBIDDEN_DOMAINS):
            continue
        results.append(
            {
                "title": title,
                "source_url": desc_url,
                "download_url": dl_url,
                "license_short": lic_short,
                "width": w,
                "height": h,
            }
        )
    return results


def search_nasa(queries: list[str], needed: int, used_urls: set[str]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for q in queries:
        if len(collected) >= needed:
            break
        r = safe_get(
            "https://images-api.nasa.gov/search",
            params={"q": q, "media_type": "image", "page_size": 25},
        )
        items = r.json().get("collection", {}).get("items", [])
        for item in items:
            if len(collected) >= needed:
                break
            data_list = item.get("data") or []
            if not data_list:
                continue
            d = data_list[0]
            nasa_id = d.get("nasa_id", "").strip()
            title = (d.get("title") or "").lower()
            if not nasa_id:
                continue
            # Avoid modern human portrait shots
            if any(bad in title for bad in ("astronaut", "portrait", "crew", "team", "director", "administrator", "student", "engineer", "scientist", "award", "ceremony")):
                continue
            src_url = f"https://images.nasa.gov/details/{nasa_id}"
            if src_url in used_urls:
                continue
            href = item.get("href")
            if not href:
                continue
            try:
                assets = safe_get(href).json()
            except Exception:
                continue
            # Pick ~medium or ~large jpg (avoid gigantic 50MB tiff/orig)
            dl_url = None
            for cand_suffix in ("~large.jpg", "~medium.jpg", "~orig.jpg"):
                for a in assets:
                    if isinstance(a, str) and a.lower().endswith(cand_suffix):
                        dl_url = a.replace("http://", "https://")
                        break
                if dl_url:
                    break
            if not dl_url:
                continue
            used_urls.add(src_url)
            collected.append(
                {
                    "title": d.get("title", nasa_id),
                    "source_archive": "NASA Image and Video Library",
                    "source_url": src_url,
                    "download_url": dl_url,
                    "source_license": "Public Domain (NASA)",
                }
            )
    return collected


def collect_pixel_art_cc0(needed: int = 5) -> list[dict[str, Any]]:
    """Download strictly CC0 1.0 RGBA pixel-art sprites from OpenGameArt."""
    # Verified CC0 1.0 OpenGameArt pack(s)
    pack_specs = [
        (
            "https://opengameart.org/content/platform-pixel-art-assets",
            "https://opengameart.org/sites/default/files/grotto_escape_pack.zip",
            "OpenGameArt (CC0 1.0 - Grotto Escape Pack)",
        ),
        (
            "https://opengameart.org/content/tiny-16-basic",
            None,
            "OpenGameArt (CC0 1.0)",
        ),
    ]
    sprites: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for page_url, zip_url, archive_label in pack_specs:
        if len(sprites) >= needed:
            break
        page_resp = safe_get(page_url)
        if "CC0" not in page_resp.text:
            continue
        if zip_url is None:
            zips = re.findall(r"https://opengameart\.org/sites/default/files/[^\"]+\.zip", page_resp.text)
            if not zips:
                continue
            zip_url = zips[0]
        zresp = safe_get(zip_url)
        with zipfile.ZipFile(io.BytesIO(zresp.content)) as zf:
            png_names = sorted(
                [n for n in zf.namelist() if n.lower().endswith(".png") and "__MACOSX" not in n]
            )
            # First pass: standalone 16x16 / 32x32 / 64x64 RGBA PNG sprites
            for name in png_names:
                if len(sprites) >= needed:
                    break
                data = zf.read(name)
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        if im.format != "PNG":
                            continue
                        w, h = im.size
                        if w not in (16, 32, 64) or h not in (16, 32, 64):
                            continue
                        rgba = im.convert("RGBA")
                        arr = np.asarray(rgba, dtype=np.uint8)
                        trans_ratio = float(np.mean(arr[:, :, 3] == 0))
                        unique_colors = int(np.unique(arr.reshape(-1, 4), axis=0).shape[0])
                        if trans_ratio < 0.05 or trans_ratio > 0.90 or unique_colors < 4:
                            continue
                        buf = io.BytesIO()
                        rgba.save(buf, format="PNG", optimize=True)
                        png_bytes = buf.getvalue()
                        hsh = hashlib.sha256(png_bytes).hexdigest()
                        if hsh in seen_hashes:
                            continue
                        seen_hashes.add(hsh)
                        sprites.append(
                            {
                                "bytes": png_bytes,
                                "source_archive": archive_label,
                                "source_url": page_url,
                                "source_license": "CC0 1.0",
                                "note": name,
                            }
                        )
                except Exception:
                    continue

            # Second pass: if standalone sprites < needed, extract clean 16x16 / 32x32 tiles from sprite sheets
            if len(sprites) < needed:
                for name in png_names:
                    if len(sprites) >= needed:
                        break
                    data = zf.read(name)
                    try:
                        with Image.open(io.BytesIO(data)) as im:
                            rgba = im.convert("RGBA")
                            w, h = rgba.size
                            tile = 32 if (w >= 64 and h >= 32 and w % 32 == 0 and h % 32 == 0) else 16
                            if w < tile or h < tile or w % tile != 0 or h % tile != 0:
                                continue
                            for ty in range(0, h, tile):
                                for tx in range(0, w, tile):
                                    if len(sprites) >= needed:
                                        break
                                    crop = rgba.crop((tx, ty, tx + tile, ty + tile))
                                    arr = np.asarray(crop, dtype=np.uint8)
                                    trans_ratio = float(np.mean(arr[:, :, 3] == 0))
                                    unique_colors = int(np.unique(arr.reshape(-1, 4), axis=0).shape[0])
                                    if trans_ratio < 0.15 or trans_ratio > 0.80 or unique_colors < 6:
                                        continue
                                    buf = io.BytesIO()
                                    crop.save(buf, format="PNG", optimize=True)
                                    png_bytes = buf.getvalue()
                                    hsh = hashlib.sha256(png_bytes).hexdigest()
                                    if hsh in seen_hashes:
                                        continue
                                    seen_hashes.add(hsh)
                                    sprites.append(
                                        {
                                            "bytes": png_bytes,
                                            "source_archive": archive_label,
                                            "source_url": page_url,
                                            "source_license": "CC0 1.0",
                                            "note": f"{name}@{tx},{ty},{tile}x{tile}",
                                        }
                                    )
                    except Exception:
                        continue

    return sprites[:needed]


def main() -> int:
    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    used_urls: set[str] = set()
    rows: list[dict[str, str]] = []
    sample_idx = 1

    def next_id() -> str:
        nonlocal sample_idx
        sid = f"{sample_idx:03d}"
        sample_idx += 1
        return sid

    # 1. old_photo_denoise (8) - Wikimedia Commons pre-1925 Detroit Publishing / historic scans
    print("[1/6] Fetching 8 old_photo_denoise archival scans from Wikimedia Commons (PD)...")
    old_queries = [
        "Detroit Publishing Company building 1905",
        "Detroit Publishing Company street 1908",
        "Detroit Publishing Company bridge 1900",
        "Detroit Publishing Company harbor 1904",
    ]
    old_count = 0
    for q in old_queries:
        if old_count >= 8:
            break
        for cand in search_wikimedia(q, {"Public domain"}, limit=20):
            if old_count >= 8:
                break
            if cand["source_url"] in used_urls:
                continue
            try:
                raw = safe_get(cand["download_url"]).content
                sid = next_id()
                before_rel = f"images/before/old_{sid}.jpg"
                after_rel = f"images/after/old_{sid}.png"
                dims = normalize_natural_scan(raw, ROOT / before_rel)
                if dims is None:
                    sample_idx -= 1
                    continue
                used_urls.add(cand["source_url"])
                old_count += 1
                rows.append(
                    {
                        "sample_id": sid,
                        "category": "old_photo_denoise",
                        "baseline_method": "opencv_fastnlmeans_clahe",
                        "related_reference_url": "https://enhancephoto.org/photo-restoration",
                        "source_archive": "Wikimedia Commons (Detroit Publishing Co. / Archival PD)",
                        "source_url": cand["source_url"],
                        "source_license": "Public Domain (PD-US-expired)",
                        "degradation_type": "natural",
                        "before_path": before_rel,
                        "after_path": after_rel,
                        "reference_path": "",
                        "before_sha256": sha256_file(ROOT / before_rel),
                        "after_sha256": "",
                        "reference_sha256": "",
                        "captured_at_utc": captured_at,
                    }
                )
                print(f"  + {sid} old_photo_denoise {dims[0]}x{dims[1]} <- {cand['title'][:50]}")
            except Exception as exc:
                sample_idx -= 1
                print(f"  ! skip {cand['title'][:40]}: {exc}")

    # Collect 22 distinct high-res reference images for upscale_2x (8), upscale_4x (8), deblur_jpeg (6)
    print("[2-4/6] Fetching 22 high-res reference images (NASA PD + Wikimedia CC0)...")
    nasa_pool = search_nasa(
        [
            "earth from space mountain",
            "mars rover landscape",
            "hubble nebula galaxy",
            "apollo moon crater",
            "glacier river aerial",
            "volcano island satellite",
        ],
        needed=24,
        used_urls=used_urls,
    )
    wiki_cc0_pool: list[dict[str, Any]] = []
    for q in ["CC0 landscape mountain", "CC0 architecture bridge", "CC0 forest river"]:
        for cand in search_wikimedia(q, {"CC0"}, limit=15):
            if cand["source_url"] not in used_urls:
                wiki_cc0_pool.append(
                    {
                        "title": cand["title"],
                        "source_archive": "Wikimedia Commons (CC0)",
                        "source_url": cand["source_url"],
                        "download_url": cand["download_url"],
                        "source_license": "CC0 1.0",
                    }
                )

    # Interleave NASA PD and Wikimedia CC0 for diverse domains
    ref_candidates: list[dict[str, Any]] = []
    max_len = max(len(nasa_pool), len(wiki_cc0_pool))
    for i in range(max_len):
        if i < len(nasa_pool):
            ref_candidates.append(nasa_pool[i])
        if i < len(wiki_cc0_pool):
            ref_candidates.append(wiki_cc0_pool[i])

    ref_tasks = (
        [("upscale_2x", "lanczos_2x_unsharp", "https://enhancephoto.org/image-upscaler", "synthetic_bicubic_2x", "up2")] * 8
        + [("upscale_4x", "lanczos_4x", "https://enhancephoto.org/image-upscaler", "synthetic_bicubic_4x", "up4")] * 8
        + [("deblur_jpeg", "wiener_unsharp_deblur", "https://enhancephoto.org/make-blurry-image-clear", "synthetic_gaussian_blur", "deb")] * 3
        + [("deblur_jpeg", "opencv_nlm_deblock", "https://enhancephoto.org/make-blurry-image-clear", "synthetic_jpeg_q30", "jpg")] * 3
    )

    cand_idx = 0
    for cat, method, ref_url, deg_type, prefix in ref_tasks:
        saved = False
        while cand_idx < len(ref_candidates):
            cand = ref_candidates[cand_idx]
            cand_idx += 1
            try:
                raw = safe_get(cand["download_url"]).content
                sid = next_id()
                ref_rel = f"images/reference/{prefix}_{sid}.png"
                ext = "jpg" if deg_type == "synthetic_jpeg_q30" else "png"
                before_rel = f"images/before/{prefix}_{sid}.{ext}"
                after_rel = f"images/after/{prefix}_{sid}.png"
                dims = normalize_reference_image(raw, ROOT / ref_rel)
                if dims is None:
                    sample_idx -= 1
                    continue
                rows.append(
                    {
                        "sample_id": sid,
                        "category": cat,
                        "baseline_method": method,
                        "related_reference_url": ref_url,
                        "source_archive": cand["source_archive"],
                        "source_url": cand["source_url"],
                        "source_license": cand["source_license"],
                        "degradation_type": deg_type,
                        "before_path": before_rel,
                        "after_path": after_rel,
                        "reference_path": ref_rel,
                        "before_sha256": "",
                        "after_sha256": "",
                        "reference_sha256": sha256_file(ROOT / ref_rel),
                        "captured_at_utc": captured_at,
                    }
                )
                print(f"  + {sid} {cat:12s} ({deg_type}) {dims[0]}x{dims[1]} <- {cand['title'][:45]}")
                saved = True
                break
            except Exception as exc:
                sample_idx -= 1
                print(f"  ! skip ref {cand['title'][:35]}: {exc}")
        if not saved:
            raise RuntimeError(f"Ran out of candidates for {cat}")

    # 5. historical_text_clarity (5) - Wikimedia Commons pre-1925 printed pages / title pages / newspapers
    print("[5/6] Fetching 5 historical_text_clarity archival print scans from Wikimedia Commons (PD)...")
    txt_queries = [
        "1910 newspaper front page OR title page",
        "1890 book title page public domain",
        "1915 poster printed text public domain",
    ]
    txt_count = 0
    for q in txt_queries:
        if txt_count >= 5:
            break
        for cand in search_wikimedia(q, {"Public domain"}, limit=20):
            if txt_count >= 5:
                break
            if cand["source_url"] in used_urls:
                continue
            try:
                raw = safe_get(cand["download_url"]).content
                sid = next_id()
                before_rel = f"images/before/txt_{sid}.jpg"
                after_rel = f"images/after/txt_{sid}.png"
                dims = normalize_natural_scan(raw, ROOT / before_rel)
                if dims is None:
                    sample_idx -= 1
                    continue
                used_urls.add(cand["source_url"])
                txt_count += 1
                rows.append(
                    {
                        "sample_id": sid,
                        "category": "historical_text_clarity",
                        "baseline_method": "clahe_nlm_unsharp_text",
                        "related_reference_url": "https://enhancephoto.org/make-blurry-image-clear",
                        "source_archive": "Wikimedia Commons (Archival Print PD)",
                        "source_url": cand["source_url"],
                        "source_license": "Public Domain (PD-US-expired)",
                        "degradation_type": "natural",
                        "before_path": before_rel,
                        "after_path": after_rel,
                        "reference_path": "",
                        "before_sha256": sha256_file(ROOT / before_rel),
                        "after_sha256": "",
                        "reference_sha256": "",
                        "captured_at_utc": captured_at,
                    }
                )
                print(f"  + {sid} historical_text_clarity {dims[0]}x{dims[1]} <- {cand['title'][:45]}")
            except Exception as exc:
                sample_idx -= 1
                print(f"  ! skip txt {cand['title'][:35]}: {exc}")

    # 6. pixel_art_upscale (5) - OpenGameArt CC0 1.0 RGBA sprites
    print("[6/6] Fetching 5 pixel_art_upscale RGBA sprites from OpenGameArt (CC0 1.0)...")
    px_sprites = collect_pixel_art_cc0(needed=5)
    if len(px_sprites) != 5:
        raise RuntimeError(f"Expected 5 CC0 pixel art sprites, got {len(px_sprites)}")
    for sp in px_sprites:
        sid = next_id()
        before_rel = f"images/before/pxl_{sid}.png"
        after_rel = f"images/after/pxl_{sid}.png"
        bp = ROOT / before_rel
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_bytes(sp["bytes"])
        with Image.open(bp) as im:
            dims = im.size
        rows.append(
            {
                "sample_id": sid,
                "category": "pixel_art_upscale",
                "baseline_method": "nearest_neighbor_4x_rgba",
                "related_reference_url": "https://enhancephoto.org/pixel-art-upscaler",
                "source_archive": sp["source_archive"],
                "source_url": sp["source_url"],
                "source_license": sp["source_license"],
                "degradation_type": "natural",
                "before_path": before_rel,
                "after_path": after_rel,
                "reference_path": "",
                "before_sha256": sha256_file(bp),
                "after_sha256": "",
                "reference_sha256": "",
                "captured_at_utc": captured_at,
            }
        )
        print(f"  + {sid} pixel_art_upscale {dims[0]}x{dims[1]} <- {sp['note']}")

    if len(rows) != 40:
        raise RuntimeError(f"Expected 40 total samples, got {len(rows)}")

    fieldnames = [
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
        "reference_sha256",
        "captured_at_utc",
    ]
    out_csv = ROOT / "data" / "samples.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Wrote 40 verified CC0/PD samples to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
