# Data Descriptor: Open Image Enhancement, Deblurring & Pixel-Art Scaling Baseline Dataset (v1.0.0)

**Version:** `1.0.0`  
**Release Date:** `2026-11-25`  
**Dataset & Documentation License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Code License:** MIT License (`scripts/*.py`)  
**Source Image Licenses:** 100% CC0 1.0 / Public Domain (NASA Public Domain, Wikimedia Commons PD-US-expired & CC0, OpenGameArt CC0 1.0)

---

## 1. Abstract & Motivation

Evaluating image restoration, super-resolution, deblurring, and pixel-art scaling pipelines requires clean, legally redistributable test imagery paired with transparent, deterministic baselines. In practice, many informal web comparisons rely on restrictive stock photos (e.g., Unsplash or Pexels), omit ground-truth degradation parameters, drop unfavorable failure cases, or conflate classical signal-processing heuristics with neural network outputs.

This dataset provides an open, verifiable **40-image baseline benchmark** spanning **6 practical image-processing tasks**: archival photo denoising and contrast enhancement (`n = 8`), 2x natural image upscaling (`n = 8`), 4x natural image upscaling (`n = 8`), controlled Gaussian deblurring and JPEG Q30 deblocking (`n = 6`), historical print legibility enhancement (`n = 5`), and 4x integer RGBA pixel-art scaling (`n = 5`). Every `after` image is produced strictly by a deterministic, local open-source baseline algorithm (`OpenCV`, `Pillow`, `scikit-image`, `NumPy`) with zero external API calls.

The benchmark serves as a reproducible objective lower bound. For pixel-art integer scaling, the `nearest_neighbor_4x_rgba` baseline is algorithmically equivalent to client-side exact integer scalers such as `https://enhancephoto.org/pixel-art-upscaler`. For restoration tasks where classical filters encounter physical information ceilings—such as 4x texture synthesis, severe archival damage repair, or blind compression artifact removal—online neural reference implementations are referenced at `https://enhancephoto.org/image-upscaler`, `https://enhancephoto.org/photo-restoration`, and `https://enhancephoto.org/make-blurry-image-clear`.

---

## 2. Provenance, Licensing & Integrity Verification

All 40 source assets were downloaded programmatically via `scripts/fetch_sources.py` from official Public Domain and CC0 archives:

- **Wikimedia Commons (`Public Domain (PD-US-expired)`)**: 13 pre-1925 archival scans—8 historical architectural, street, bridge, and harbor photographs from the Detroit Publishing Company (`001`–`008`) and 5 historical printed book title pages spanning 1614–1910 (`031`–`035`).
- **NASA Image and Video Library (`Public Domain (NASA)`)**: 12 high-resolution Earth observation, planetary, and aerial landscape photographs distributed across `upscale_2x`, `upscale_4x`, and `deblur_jpeg`.
- **Wikimedia Commons (`CC0 1.0`)**: 10 high-resolution CC0 landscape, glacier, mountain, and museum open-access artworks distributed across `upscale_2x`, `upscale_4x`, and `deblur_jpeg`.
- **OpenGameArt (`CC0 1.0`)**: 5 RGBA pixel-art sprites (`64x64`, `64x32`, and three `16x16` sprites with genuine alpha transparency, `036`–`040`) sourced from the CC0 1.0 *Grotto Escape Pack*.

An automated audit gate (`scripts/verify_licenses.py`) enforces 40/40 compliance across:
1. Exact category quotas (`8 + 8 + 8 + 6 + 5 + 5 = 40`, including `3` Gaussian blur + `3` JPEG Q30 rows).
2. Strict license allowlist (`CC0 1.0`, `Public Domain (NASA)`, `Public Domain (PD-US-expired)`).
3. Domain denylist blocking all commercial or quasi-free stock libraries (`unsplash.com`, `pexels.com`, `pixabay.com`, `shutterstock.com`, `gettyimages.com`, etc.).
4. Exact SHA-256 verification of every `before_path`, `after_path`, and `reference_path` against `data/samples.csv`.
5. Resolution and format checks (all 22 reference images normalized to RGB lossless PNG with dimensions divisible by 4; all 5 pixel-art inputs verified as RGBA PNG in `{16, 32, 64}` px).

---

## 3. Controlled Degradations & Local Baseline Methods

### 3.1 Controlled Degradation Protocol (`scripts/make_degraded.py`)
For the 22 samples with ground-truth references (`images/reference/`):
- **`synthetic_bicubic_2x`** (`009`–`016`): Pillow `Image.Resampling.BICUBIC` 2x downsampling saved as lossless PNG.
- **`synthetic_bicubic_4x`** (`017`–`024`): Pillow `Image.Resampling.BICUBIC` 4x downsampling saved as lossless PNG.
- **`synthetic_gaussian_blur`** (`025`–`027`): Pillow `ImageFilter.GaussianBlur(radius=2.0)` saved as lossless PNG.
- **`synthetic_jpeg_q30`** (`028`–`030`): JPEG compression at `quality=30`, `4:2:0` chroma subsampling (`subsampling=2`).

### 3.2 Pure Local Open-Source Baselines (`scripts/run_baselines.py`)
Every `images/after/` file is generated deterministically on CPU:
- **`opencv_fastnlmeans_clahe`** (`old_photo_denoise`, `n = 8`): OpenCV `fastNlMeansDenoisingColored(h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)` followed by LAB-space `CLAHE(clipLimit=2.0, tileGridSize=(8,8))` on the luminance channel.
- **`lanczos_2x_unsharp`** (`upscale_2x`, `n = 8`): Pillow 2x `LANCZOS` resampling followed by `UnsharpMask(radius=1.0, percent=80, threshold=3)`.
- **`lanczos_4x`** (`upscale_4x`, `n = 8`): Pillow 4x `LANCZOS` resampling.
- **`wiener_unsharp_deblur`** (`deblur_jpeg` Gaussian blur, `n = 3`): Reflect-padded (`16 px`) per-channel `skimage.restoration.wiener` deconvolution with a normalized `13x13` Gaussian PSF (`sigma=2.0`, `balance=0.05`).
- **`opencv_nlm_deblock`** (`deblur_jpeg` JPEG Q30, `n = 3`): OpenCV `fastNlMeansDenoisingColored(h=7, hColor=7, templateWindowSize=7, searchWindowSize=21)`.
- **`clahe_nlm_unsharp_text`** (`historical_text_clarity`, `n = 5`): Mild `fastNlMeansDenoisingColored(h=4, hColor=4)` + LAB `CLAHE(clipLimit=2.2, tileGridSize=(8,8))` + `UnsharpMask(radius=1.2, percent=90, threshold=3)`.
- **`nearest_neighbor_4x_rgba`** (`pixel_art_upscale`, `n = 5`): Pillow exact 4x `NEAREST` resampling in `RGBA` mode.

---

## 4. Objective Metric Suite & Empirical Results

To guarantee zero external weight-download dependencies and cross-platform determinism, `scripts/compute_iqa.py` computes a fully open-source metric suite:
- **Full-Reference IQA (`n = 22`)**: `scikit-image` PSNR (dB) and RGB SSIM (`data_range=255`). For `upscale_2x` and `upscale_4x`, `before` FR metrics are computed by upsampling `before` to `reference` dimensions via naive Bicubic interpolation so that deltas directly measure gain over standard Bicubic scaling.
- **No-Reference Spatial Metrics (`n = 35`)**: OpenCV Laplacian variance (high-frequency sharpness), Tenengrad Sobel gradient energy, NumPy RMS luminance contrast, and 8-bit Shannon entropy.
- **Pixel-Art Exactness (`n = 5`)**: Exact 4x dimensions, exact RGBA unique color count (`Δ Unique Colors`), exact 4x Kronecker alpha-channel bit equality, and a counterfactual comparison against naive Bicubic 4x scaling.

### 4.1 Aggregated Category Results

| Category | n | Baseline Method | Primary Metrics (Mean / Median) |
| :--- | :---: | :--- | :--- |
| `old_photo_denoise` | 8 | `opencv_fastnlmeans_clahe` | LapVar `3404.99→5062.93` (`+1657.94` / `+1245.54`); Entropy `7.26→7.42` (`+0.16` / `+0.11`) |
| `upscale_2x` | 8 | `lanczos_2x_unsharp` | PSNR `30.46→31.15 dB` (`+0.68` / `+0.64 dB`); SSIM `0.8719→0.8931` (`+0.0212` / `+0.0222`) |
| `upscale_4x` | 8 | `lanczos_4x` | PSNR `24.21→24.41 dB` (`+0.20` / `+0.21 dB`); SSIM `0.6826→0.6941` (`+0.0115` / `+0.0151`) |
| `deblur_jpeg` (Blur) | 3 | `wiener_unsharp_deblur` | PSNR `21.26→21.98 dB` (`+0.73` / `+0.53 dB`); SSIM `0.4843→0.5617` (`+0.0774` / `+0.0699`) |
| `deblur_jpeg` (JPEG Q30) | 3 | `opencv_nlm_deblock` | PSNR `30.36→29.69 dB` (`-0.67` / `-0.51 dB`); SSIM `0.8478→0.7975` (`-0.0503` / `-0.0422`) |
| `historical_text_clarity` | 5 | `clahe_nlm_unsharp_text` | LapVar `2969.56→9543.61` (`+6574.05` / `+1141.42`); RMS `36.23→45.29` (`+9.06` / `+3.09`) |
| `pixel_art_upscale` | 5 | `nearest_neighbor_4x_rgba` | `5/5` exact 4x, `5/5` alpha intact, `Δ Colors = 0`; Bicubic 4x adds `+2800.8` mean (`+1107` median) colors |

### 4.2 Failure Modes, Negative Results & Metric Limitations

All 40 records are preserved without cherry-picking:
1. **Spatial Denoising Regression on JPEG Q30 (`028`, `029`, `030`)**: `opencv_nlm_deblock` decreased PSNR by `-0.32`, `-0.51`, and `-1.18 dB` and SSIM by `-0.0422`, `-0.0389`, and `-0.0699`. Generic spatial non-local means cannot distinguish 8x8 DCT quantization blocks from genuine high-frequency Earth-observation terrain texture, causing over-smoothing.
2. **Lanczos Ringing vs. Smooth Gradients at 4x (`020`, `022`)**: Two painterly CC0 landscapes (`020`, `022`) showed minor SSIM declines (`-0.0028`, `-0.0008`) under `lanczos_4x` despite modest PSNR gains (`+0.14`, `+0.23 dB`). Furthermore, `upscale_4x` achieves only `0.6941` mean SSIM, illustrating the hard physical limit of linear interpolation when 93.75% of pixels are missing.
3. **Pixel-Art Color Pollution under Interpolation (`036`–`040`)**: Naive Bicubic 4x scaling expanded sprite palettes from `8–45` unique RGBA colors to `989–5,587` colors (`+981` to `+5,574` spurious semi-transparent edge halos), whereas `nearest_neighbor_4x_rgba` guaranteed `Δ = 0` and bit-exact 4x alpha preservation.
4. **No-Reference Metric Interpretation**: Laplacian variance and Tenengrad gradient energy measure high-frequency spatial energy rather than human perceptual quality; in archival scans (`001`–`008`, `031`–`035`), they respond to both sharper ink/architecture edges and amplified paper fiber or halftone grain. Note also that downsampled inputs (`upscale_2x`/`4x`) have higher pixel-grid Laplacian variance at low resolution than their smoothly interpolated full-resolution outputs; for those categories, full-reference PSNR/SSIM against `images/reference/` is the primary fidelity metric.

---

## 5. File & Schema Dictionary

- **`data/samples.csv` (40 rows)**: `sample_id`, `category`, `baseline_method`, `related_reference_url`, `source_archive`, `source_url`, `source_license`, `degradation_type`, `before_path`, `after_path`, `reference_path`, `before_sha256`, `after_sha256`, `reference_sha256`, `captured_at_utc`.
- **`data/metrics_iqa.csv` (40 rows)**: Per-sample dimensions, file sizes (`before_kb`, `after_kb`), FR metrics (`psnr_*`, `ssim_*`), NR metrics (`laplacian_var_*`, `tenengrad_*`, `rms_contrast_*`, `shannon_entropy_*`), pixel-art metrics (`pixel_unique_colors_*`, `pixel_delta_colors`, `pixel_exact_4x`, `pixel_alpha_intact`, `pixel_transparent_ratio_before`, `pixel_bicubic_4x_*`), and `notes`.
- **`data/summary.json`**: Machine-readable aggregate means/medians by category, `deblur_jpeg` subgroup stats, and SHA-256 checksums of key dataset and script files.

---

## 6. Ethics, Privacy & Copyright Declaration

This dataset contains **zero personally identifiable information (PII)**, **zero modern identifiable human portraits**, **zero user-uploaded content**, and **zero commercial stock imagery**. All 40 source files are verifiably in the Public Domain or dedicated under CC0 1.0, with per-sample archival URLs and SHA-256 digests recorded in `data/samples.csv`. Curation, metadata, and this Data Descriptor are released under **CC BY 4.0**; processing and verification scripts under `scripts/` are released under the **MIT License**.
