#!/usr/bin/env python3
"""Build a clean, publication-ready 3-4 page DATA_DESCRIPTOR.pdf using ReportLab."""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(
        0.75 * inch,
        0.48 * inch,
        "Open Image Enhancement, Deblurring & Pixel-Art Scaling Baseline Dataset (v1.0.0) — Data Descriptor",
    )
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.48 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    summary = json.loads((ROOT / "data" / "summary.json").read_text(encoding="utf-8"))
    by_cat = summary["by_category"]
    sub = summary["deblur_jpeg_subgroups"]

    out_pdf = ROOT / "DATA_DESCRIPTOR.pdf"
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Data Descriptor: Open Image Enhancement, Deblurring & Pixel-Art Scaling Baseline Dataset (v1.0.0)",
        author="Open IQA Baseline Benchmark Project",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16.5,
        leading=21,
        textColor=colors.HexColor("#111827"),
        alignment=0,
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#374151"),
        spaceAfter=10,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#1E3A5F"),
        spaceBefore=12,
        spaceAfter=5,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=4,
    )
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11,
        textColor=colors.HexColor("#111827"),
    )
    cell_hdr = ParagraphStyle(
        "CellHdr",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    story = []

    # Header
    story.append(
        Paragraph(
            "Data Descriptor: Open Image Enhancement, Deblurring &amp; Pixel-Art Scaling Baseline Dataset (v1.0.0)",
            title_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Version:</b> 1.0.0 &nbsp;|&nbsp; <b>Release Date:</b> 2026-11-25 &nbsp;|&nbsp; "
            "<b>Dataset License:</b> CC BY 4.0 &nbsp;|&nbsp; <b>Scripts License:</b> MIT<br/>"
            "<b>Source Imagery:</b> 100% CC0 1.0 &amp; Public Domain (NASA, Wikimedia Commons PD/CC0, OpenGameArt CC0)<br/>"
            "<b>Author:</b> Open IQA Baseline Benchmark Project",
            meta_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#D1D5DB"), spaceAfter=10))

    # Section 1
    story.append(Paragraph("1. Abstract &amp; Motivation", h1_style))
    story.append(
        Paragraph(
            "Reproducible evaluation of image restoration, super-resolution, deblurring, and pixel-art scaling requires "
            "legally redistributable imagery paired with deterministic, transparent baselines. Informal web comparisons "
            "frequently rely on restrictive stock photos, omit synthetic degradation parameters, discard failure cases, "
            "or conflate classical signal-processing filters with neural network outputs.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "This release provides a verified <b>40-image open baseline dataset</b> covering <b>6 image-processing tasks</b>: "
            "archival photo denoising and contrast enhancement (n = 8), 2x natural upscaling (n = 8), 4x natural upscaling "
            "(n = 8), controlled Gaussian deblurring and JPEG Q30 deblocking (n = 6), historical print clarity (n = 5), and "
            "4x integer RGBA pixel-art scaling (n = 5). All 40 <i>images/after/</i> outputs are computed strictly on local CPU "
            "using deterministic open-source algorithms (OpenCV, Pillow, scikit-image, NumPy) with zero paid API calls.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Reference Scope:</b> The 4x integer nearest-neighbor pixel-art baseline is algorithmically equivalent to the "
            "client-side browser utility <i>https://enhancephoto.org/pixel-art-upscaler</i>. For tasks where classical "
            "interpolation or spatial filtering hits physical ceilings (4x texture synthesis, severe archival restoration, "
            "and blind compression removal), online neural reference implementations are documented at "
            "<i>https://enhancephoto.org/image-upscaler</i>, <i>https://enhancephoto.org/photo-restoration</i>, and "
            "<i>https://enhancephoto.org/make-blurry-image-clear</i>.",
            body_style,
        )
    )

    # Section 2
    story.append(Paragraph("2. Provenance, Licensing &amp; Integrity Gates", h1_style))
    story.append(
        Paragraph(
            "Every source image was programmatically fetched via <i>scripts/fetch_sources.py</i> from official Public Domain "
            "or CC0 repositories, excluding all commercial stock sites (Unsplash, Pexels, Pixabay, Shutterstock, Getty, etc.) "
            "and modern identifiable portraits:",
            body_style,
        )
    )
    prov_bullets = [
        "<b>Wikimedia Commons (Public Domain — PD-US-expired, n = 13):</b> 8 pre-1925 Detroit Publishing Company architectural/street/harbor scans (001–008) and 5 historical printed book title pages spanning 1614–1910 (031–035).",
        "<b>NASA Image and Video Library (Public Domain — NASA, n = 12):</b> High-resolution Earth observation, planetary, and aerial terrain photographs distributed across 2x/4x upscaling and deblur/JPEG benchmarks.",
        "<b>Wikimedia Commons (CC0 1.0, n = 10):</b> High-resolution CC0 landscape, glacier, and museum open-access artworks distributed across 2x/4x upscaling and deblur/JPEG benchmarks.",
        "<b>OpenGameArt (CC0 1.0, n = 5):</b> Five RGBA PNG pixel-art sprites (64x64, 64x32, and three 16x16 sprites with verified alpha transparency, 036–040) from the CC0 1.0 Grotto Escape Pack.",
    ]
    for b in prov_bullets:
        story.append(Paragraph(f"&bull; {b}", bullet_style))

    story.append(
        Paragraph(
            "An automated verification gate (<i>scripts/verify_licenses.py</i>) validates 40/40 quota balance, license "
            "allowlist compliance, stock-domain denylist exclusion, image decodability, exact 4-divisible reference geometry, "
            "RGBA pixel-art constraints, and SHA-256 checksums across all before, after, and reference files.",
            body_style,
        )
    )

    # Section 3
    story.append(Paragraph("3. Controlled Degradations &amp; Local Open-Source Baselines", h1_style))
    story.append(Paragraph("3.1 Controlled Degradation Protocol (n = 22 Reference Pairs)", h2_style))
    deg_bullets = [
        "<b>synthetic_bicubic_2x (009–016):</b> Pillow BICUBIC 2x downsampling from lossless PNG ground truth.",
        "<b>synthetic_bicubic_4x (017–024):</b> Pillow BICUBIC 4x downsampling from lossless PNG ground truth.",
        "<b>synthetic_gaussian_blur (025–027):</b> Pillow GaussianBlur(radius=2.0) saved as lossless PNG.",
        "<b>synthetic_jpeg_q30 (028–030):</b> JPEG compression at quality=30 with 4:2:0 chroma subsampling.",
    ]
    for b in deg_bullets:
        story.append(Paragraph(f"&bull; {b}", bullet_style))

    story.append(Paragraph("3.2 Deterministic Local Baseline Algorithms (n = 40 Outputs)", h2_style))
    base_rows = [
        [Paragraph("Category", cell_hdr), Paragraph("n", cell_hdr), Paragraph("Baseline Method", cell_hdr), Paragraph("Algorithm Specification", cell_hdr)],
        [Paragraph("old_photo_denoise", cell_style), Paragraph("8", cell_style), Paragraph("opencv_fastnlmeans_clahe", cell_style), Paragraph("OpenCV fastNlMeansDenoisingColored(h=6) + LAB CLAHE(clipLimit=2.0, 8x8)", cell_style)],
        [Paragraph("upscale_2x", cell_style), Paragraph("8", cell_style), Paragraph("lanczos_2x_unsharp", cell_style), Paragraph("Pillow 2x LANCZOS + UnsharpMask(radius=1.0, percent=80, threshold=3)", cell_style)],
        [Paragraph("upscale_4x", cell_style), Paragraph("8", cell_style), Paragraph("lanczos_4x", cell_style), Paragraph("Pillow 4x LANCZOS resampling to exact reference dimensions", cell_style)],
        [Paragraph("deblur_jpeg (Blur)", cell_style), Paragraph("3", cell_style), Paragraph("wiener_unsharp_deblur", cell_style), Paragraph("Reflect-padded (16px) scikit-image Wiener deconvolution (13x13 PSF, sigma=2.0, balance=0.05)", cell_style)],
        [Paragraph("deblur_jpeg (JPEG)", cell_style), Paragraph("3", cell_style), Paragraph("opencv_nlm_deblock", cell_style), Paragraph("OpenCV fastNlMeansDenoisingColored(h=7, hColor=7, 7, 21)", cell_style)],
        [Paragraph("historical_text_clarity", cell_style), Paragraph("5", cell_style), Paragraph("clahe_nlm_unsharp_text", cell_style), Paragraph("Mild OpenCV NLM(h=4) + LAB CLAHE(clipLimit=2.2) + UnsharpMask(1.2, 90, 3)", cell_style)],
        [Paragraph("pixel_art_upscale", cell_style), Paragraph("5", cell_style), Paragraph("nearest_neighbor_4x_rgba", cell_style), Paragraph("Pillow exact 4x NEAREST resampling in RGBA mode (zero palette blending)", cell_style)],
    ]
    t_base = Table(base_rows, colWidths=[1.35 * inch, 0.35 * inch, 1.55 * inch, 3.75 * inch], repeatRows=1)
    t_base.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(Spacer(1, 3))
    story.append(t_base)
    story.append(PageBreak())

    # Section 4
    story.append(Paragraph("4. Empirical Benchmark Results (All 40 Samples Retained)", h1_style))
    story.append(
        Paragraph(
            "Metrics are computed deterministically by <i>scripts/compute_iqa.py</i> without external neural weight downloads: "
            "full-reference PSNR and RGB SSIM (scikit-image) for all 22 reference-backed pairs; no-reference Laplacian "
            "variance, Tenengrad gradient energy, RMS luminance contrast, and Shannon entropy for all 35 natural/archival "
            "images; and exact 4x geometry, RGBA alpha-channel bit equality, and palette-size deltas (plus naive Bicubic 4x "
            "counterfactuals) for all 5 pixel-art sprites. For <i>upscale_2x</i> and <i>upscale_4x</i>, before FR metrics "
            "represent naive Bicubic upscaling to reference dimensions.",
            body_style,
        )
    )

    u2 = by_cat["upscale_2x"]
    u4 = by_cat["upscale_4x"]
    old = by_cat["old_photo_denoise"]
    txt = by_cat["historical_text_clarity"]
    px = by_cat["pixel_art_upscale"]
    gb = sub["synthetic_gaussian_blur"]
    jp = sub["synthetic_jpeg_q30"]

    res_rows = [
        [Paragraph("Category / Subgroup", cell_hdr), Paragraph("n", cell_hdr), Paragraph("Metric", cell_hdr), Paragraph("Before Mean / Med", cell_hdr), Paragraph("After Mean / Med", cell_hdr), Paragraph("Delta Mean / Med", cell_hdr)],
        [Paragraph("old_photo_denoise", cell_style), Paragraph("8", cell_style), Paragraph("Laplacian Var<br/>Entropy (bits)", cell_style), Paragraph(f"{old['laplacian_var_before_mean']:.1f} / {old['laplacian_var_before_median']:.1f}<br/>{old['shannon_entropy_before_mean']:.2f} / {old['shannon_entropy_before_median']:.2f}", cell_style), Paragraph(f"{old['laplacian_var_after_mean']:.1f} / {old['laplacian_var_after_median']:.1f}<br/>{old['shannon_entropy_after_mean']:.2f} / {old['shannon_entropy_after_median']:.2f}", cell_style), Paragraph(f"+{old['laplacian_var_delta_mean']:.1f} / +{old['laplacian_var_delta_median']:.1f}<br/>+{old['shannon_entropy_delta_mean']:.2f} / +{old['shannon_entropy_delta_median']:.2f}", cell_style)],
        [Paragraph("upscale_2x (vs Bicubic)", cell_style), Paragraph("8", cell_style), Paragraph("PSNR (dB)<br/>SSIM", cell_style), Paragraph(f"{u2['psnr_before_mean']:.2f} / {u2['psnr_before_median']:.2f}<br/>{u2['ssim_before_mean']:.4f} / {u2['ssim_before_median']:.4f}", cell_style), Paragraph(f"{u2['psnr_after_mean']:.2f} / {u2['psnr_after_median']:.2f}<br/>{u2['ssim_after_mean']:.4f} / {u2['ssim_after_median']:.4f}", cell_style), Paragraph(f"+{u2['psnr_delta_mean']:.2f} / +{u2['psnr_delta_median']:.2f}<br/>+{u2['ssim_delta_mean']:.4f} / +{u2['ssim_delta_median']:.4f}", cell_style)],
        [Paragraph("upscale_4x (vs Bicubic)", cell_style), Paragraph("8", cell_style), Paragraph("PSNR (dB)<br/>SSIM", cell_style), Paragraph(f"{u4['psnr_before_mean']:.2f} / {u4['psnr_before_median']:.2f}<br/>{u4['ssim_before_mean']:.4f} / {u4['ssim_before_median']:.4f}", cell_style), Paragraph(f"{u4['psnr_after_mean']:.2f} / {u4['psnr_after_median']:.2f}<br/>{u4['ssim_after_mean']:.4f} / {u4['ssim_after_median']:.4f}", cell_style), Paragraph(f"+{u4['psnr_delta_mean']:.2f} / +{u4['psnr_delta_median']:.2f}<br/>+{u4['ssim_delta_mean']:.4f} / +{u4['ssim_delta_median']:.4f}", cell_style)],
        [Paragraph("deblur_jpeg (Gaussian)", cell_style), Paragraph("3", cell_style), Paragraph("PSNR (dB)<br/>SSIM", cell_style), Paragraph(f"{gb['psnr_before_mean']:.2f} / {gb['psnr_before_median']:.2f}<br/>{gb['ssim_before_mean']:.4f} / {gb['ssim_before_median']:.4f}", cell_style), Paragraph(f"{gb['psnr_after_mean']:.2f} / {gb['psnr_after_median']:.2f}<br/>{gb['ssim_after_mean']:.4f} / {gb['ssim_after_median']:.4f}", cell_style), Paragraph(f"+{gb['psnr_delta_mean']:.2f} / +{gb['psnr_delta_median']:.2f}<br/>+{gb['ssim_delta_mean']:.4f} / +{gb['ssim_delta_median']:.4f}", cell_style)],
        [Paragraph("deblur_jpeg (JPEG Q30)", cell_style), Paragraph("3", cell_style), Paragraph("PSNR (dB)<br/>SSIM", cell_style), Paragraph(f"{jp['psnr_before_mean']:.2f} / {jp['psnr_before_median']:.2f}<br/>{jp['ssim_before_mean']:.4f} / {jp['ssim_before_median']:.4f}", cell_style), Paragraph(f"{jp['psnr_after_mean']:.2f} / {jp['psnr_after_median']:.2f}<br/>{jp['ssim_after_mean']:.4f} / {jp['ssim_after_median']:.4f}", cell_style), Paragraph(f"{jp['psnr_delta_mean']:.2f} / {jp['psnr_delta_median']:.2f}<br/>{jp['ssim_delta_mean']:.4f} / {jp['ssim_delta_median']:.4f}", cell_style)],
        [Paragraph("historical_text_clarity", cell_style), Paragraph("5", cell_style), Paragraph("Laplacian Var<br/>RMS Contrast", cell_style), Paragraph(f"{txt['laplacian_var_before_mean']:.1f} / {txt['laplacian_var_before_median']:.1f}<br/>{txt['rms_contrast_before_mean']:.2f} / {txt['rms_contrast_before_median']:.2f}", cell_style), Paragraph(f"{txt['laplacian_var_after_mean']:.1f} / {txt['laplacian_var_after_median']:.1f}<br/>{txt['rms_contrast_after_mean']:.2f} / {txt['rms_contrast_after_median']:.2f}", cell_style), Paragraph(f"+{txt['laplacian_var_delta_mean']:.1f} / +{txt['laplacian_var_delta_median']:.1f}<br/>+{txt['rms_contrast_delta_mean']:.2f} / +{txt['rms_contrast_delta_median']:.2f}", cell_style)],
        [Paragraph("pixel_art_upscale", cell_style), Paragraph("5", cell_style), Paragraph("Unique Colors<br/>Bicubic 4x Colors", cell_style), Paragraph(f"{px['pixel_unique_colors_before_mean']:.1f} / {px['pixel_unique_colors_before_median']:.1f}<br/>(5/5 RGBA alpha)", cell_style), Paragraph(f"{px['pixel_unique_colors_after_mean']:.1f} / {px['pixel_unique_colors_after_median']:.1f}<br/>{px['pixel_bicubic_4x_colors_mean']:.1f} / {px['pixel_bicubic_4x_colors_median']:.1f}", cell_style), Paragraph(f"0.0 / 0.0 (5/5 exact)<br/>+{px['pixel_bicubic_4x_delta_colors_mean']:.1f} / +{px['pixel_bicubic_4x_delta_colors_median']:.1f}", cell_style)],
    ]
    t_res = Table(res_rows, colWidths=[1.45 * inch, 0.32 * inch, 1.05 * inch, 1.38 * inch, 1.38 * inch, 1.42 * inch], repeatRows=1)
    t_res.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(Spacer(1, 3))
    story.append(t_res)

    story.append(Paragraph("4.1 Honest Negative Cases, Boundaries &amp; Metric Limitations", h2_style))
    lim_bullets = [
        "<b>JPEG Q30 Deblocking Regression (028–030):</b> Generic spatial Non-Local Means (h=7) reduced PSNR by -0.32 to -1.18 dB and SSIM by -0.0389 to -0.0699 by over-smoothing fine terrain textures in satellite imagery. Retaining these rows highlights why blind spatial filters underperform DCT-aware or neural artifact restoration.",
        "<b>4x Upscaling Information Ceiling (020, 022):</b> Lanczos 4x gained only +0.20 dB mean PSNR over naive Bicubic 4x and regressed slightly in SSIM on two smooth painterly artworks (-0.0028 on 020; -0.0008 on 022) due to ringing, confirming that 4x super-resolution requires learned image priors.",
        "<b>Pixel-Art Color Pollution (036–040):</b> Nearest-neighbor 4x achieved 5/5 exact geometry, 5/5 bit-exact RGBA alpha preservation, and 0 added colors. Naive Bicubic 4x injected +981 to +5,574 spurious semi-transparent interpolated colors per sprite.",
        "<b>No-Reference Sharpness Trade-Offs:</b> Laplacian variance and Tenengrad gradient energy reward high-frequency edge contrast in archival scans (001–008, 031–035) but cannot separate ink/structure edges from amplified paper fiber or halftone grain.",
    ]
    for b in lim_bullets:
        story.append(Paragraph(f"&bull; {b}", bullet_style))

    story.append(PageBreak())

    # Section 5 & 6
    story.append(Paragraph("5. File &amp; Schema Dictionary", h1_style))
    schema_bullets = [
        "<b>data/samples.csv (40 rows):</b> sample_id, category, baseline_method, related_reference_url, source_archive, source_url, source_license, degradation_type, before_path, after_path, reference_path, before_sha256, after_sha256, reference_sha256, captured_at_utc.",
        "<b>data/metrics_iqa.csv (40 rows):</b> Per-sample image dimensions, file sizes (KB), full-reference PSNR/SSIM (before, after, delta), no-reference Laplacian variance, Tenengrad energy, RMS contrast, Shannon entropy, pixel-art palette &amp; alpha exactness checks, and Bicubic 4x counterfactual palette counts.",
        "<b>data/summary.json:</b> Category-level mean and median aggregates, deblur_jpeg subgroup breakdowns, metric backend metadata, and SHA-256 verification checksums for dataset tables and pipeline scripts.",
        "<b>images/reference/ (22 files), images/before/ (40 files), images/after/ (40 files):</b> Complete paired image corpus matching all paths and SHA-256 digests in data/samples.csv.",
        "<b>scripts/:</b> Reproducible Python pipeline — fetch_sources.py, make_degraded.py, run_baselines.py, verify_licenses.py, compute_iqa.py, and build_descriptor_pdf.py.",
    ]
    for b in schema_bullets:
        story.append(Paragraph(f"&bull; {b}", bullet_style))

    story.append(Paragraph("5.1 Key File SHA-256 Integrity Manifest", h2_style))
    sha_rows = [[Paragraph("Relative Path", cell_hdr), Paragraph("SHA-256 Digest", cell_hdr)]]
    for rel, digest in summary.get("file_sha256", {}).items():
        sha_rows.append([Paragraph(rel, cell_style), Paragraph(f"<font face='Courier' size='7.8'>{digest}</font>", cell_style)])
    t_sha = Table(sha_rows, colWidths=[2.1 * inch, 4.9 * inch], repeatRows=1)
    t_sha.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_sha)

    story.append(Paragraph("6. Ethics, Privacy, Copyright &amp; Reproducibility", h1_style))
    story.append(
        Paragraph(
            "<b>Zero PII &amp; Zero Modern Portraits:</b> The benchmark contains no personally identifiable information, "
            "no user-uploaded photographs, no modern identifiable human portraits, and no commercial stock imagery. "
            "All 40 source assets are verifiably CC0 1.0 or Public Domain with archive URLs logged in <i>data/samples.csv</i>.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Dual Licensing:</b> Dataset curation, tabular annotations, baseline outputs, and this Data Descriptor are "
            "released under <b>CC BY 4.0</b>. All Python processing, verification, and evaluation scripts under "
            "<i>scripts/</i> are released under the <b>MIT License</b>. Anyone can deterministically reproduce all 22 "
            "controlled degradations, all 40 local baseline outputs, and all objective IQA metrics offline with zero API keys.",
            body_style,
        )
    )

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    reader = PdfReader(str(out_pdf))
    pages = len(reader.pages)
    text_len = sum(len(p.extract_text() or "") for p in reader.pages)
    size_kb = round(out_pdf.stat().st_size / 1024.0, 2)
    print(f"[PDF] Built {out_pdf.name}: {pages} pages, {size_kb} KB, {text_len} extracted chars")
    if pages < 3 or text_len < 1500:
        raise RuntimeError(f"PDF validation failed: pages={pages}, text_len={text_len}")
    return out_pdf


if __name__ == "__main__":
    build_pdf()
