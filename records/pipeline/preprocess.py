"""
Computer-vision enhancement pipeline for degraded land-record scans.

Steps
-----
1.  Grayscale conversion
2.  Stamp/seal ink softening (Telea inpainting with stamp mask)
3.  **Gamma correction** (automatic or manual) - recovers faded / dark scans
4.  Denoising (Non-Local Means / median fallback)
5.  CLAHE local contrast normalisation
6.  Adaptive binarization (Gaussian adaptive threshold)
7.  Speckle (noise blob) removal (Indic-safe)
8.  Deskew (min-area-rect angle estimation)

Every step records diagnostics that are surfaced in the UI so operators can
see exactly which enhancements were applied.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 3200          # downscale very large scans for speed
DESKEW_MAX_ANGLE = 12.0  # ignore implausible rotations


# ---------------------------------------------------------------------------
# Loading (images and legacy PDFs)
# ---------------------------------------------------------------------------
def load_pages(path: str | Path, max_pages: int = 5, pdf_zoom: float = 2.0):
    """Return ``(pages, source_page_count)`` from an image file or PDF.

    ``pages`` holds at most ``max_pages`` BGR ndarrays (the ones actually
    enhanced + OCR'd). ``source_page_count`` is the *true* number of pages
    in the source file, uncapped.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf as pdfium
        except ImportError:
            try:
                import fitz as pdfium  # legacy module name
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyMuPDF is required to rasterise PDFs") from exc
        pages = []
        with pdfium.open(str(path)) as pdf_doc:
            source_page_count = pdf_doc.page_count
            for i, page in enumerate(pdf_doc):
                if i >= max_pages:
                    break
                pix = page.get_pixmap(matrix=pdfium.Matrix(pdf_zoom, pdf_zoom),
                                      alpha=False)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                pages.append(arr[:, :, ::-1].copy())          # RGB -> BGR
        return pages, source_page_count

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not decode image: {path.name}")
    return [img], 1


# ---------------------------------------------------------------------------
# Gamma correction
# ---------------------------------------------------------------------------
def auto_gamma_value(gray: np.ndarray) -> float:
    """Pick gamma so mid-tones land at 0.5. gamma>1 brightens, <1 darkens."""
    mean = float(np.mean(gray)) / 255.0
    mean = min(max(mean, 0.05), 0.95)
    gamma = math.log(mean) / math.log(0.5)
    return float(np.clip(gamma, 0.4, 2.8))


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction with a 256-entry look-up table."""
    if gamma is None or abs(gamma - 1.0) < 1e-3:
        return gray
    inv = 1.0 / gamma
    table = (np.power(np.arange(256) / 255.0, inv) * 255).astype("uint8")
    return cv2.LUT(gray, table)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _estimate_skew(binary: np.ndarray) -> float:
    """Estimate dominant text angle from the dark foreground pixels."""
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) < 300:
        return 0.0
    sample = coords
    if len(coords) > 20000:                       # cap for speed
        idx = np.random.default_rng(42).choice(len(coords), 20000,
                                               replace=False)
        sample = coords[idx]
    rect = cv2.minAreaRect(sample[:, ::-1].astype(np.float32))
    angle = float(rect[-1])
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > DESKEW_MAX_ANGLE:
        return 0.0
    return angle


def _rotate(image: np.ndarray, angle: float, border=(255, 255, 255)):
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


def _despeckle(binary: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Remove tiny isolated black blobs (scan noise).
    
    min_area set to 6 (Indic-safe) to preserve Malayalam/Kannada/Telugu dots.
    """
    inv = cv2.bitwise_not(binary)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    out = np.zeros_like(inv)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return cv2.bitwise_not(out)


def _remove_rules(binary: np.ndarray):
    """Remove long horizontal/vertical table rules and page frames."""
    inv = cv2.bitwise_not(binary)
    h, w = binary.shape[:2]
    klen = max(40, w // 18)
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1)))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen)))
    mask = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    removed = int(cv2.countNonZero(mask))
    if removed < 60:
        return binary, 0
    cleaned = cv2.inpaint(binary, mask, 3, cv2.INPAINT_TELEA)
    _, cleaned = cv2.threshold(cleaned, 180, 255, cv2.THRESH_BINARY)
    return cleaned, removed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def preprocess_image(bgr: np.ndarray, gamma="auto", gamma_value: float | None = None,
                     denoise: bool = True, binarize: bool = True,
                     deskew: bool = True,
                     stamp_mask: np.ndarray | None = None) -> dict:
    """Run the full enhancement chain.

    ``gamma``: ``"auto"`` | ``"off"`` | ``"manual"`` (with ``gamma_value``).
    ``stamp_mask``: Optional binary mask of stamp/seal ink for Telea inpainting.
    Returns ``{processed, enhanced_gray, gray, info}``.
    """
    t0 = time.perf_counter()
    steps = []

    # 0. resize
    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        steps.append({"name": "Resize", "detail": f"scaled x{scale:.2f}"})

    original_brightness = float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))

    # 1. grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    steps.append({"name": "Grayscale", "detail": "single-channel luminance"})

    # 1.5. stamp/seal ink inpainting (if mask supplied)
    if stamp_mask is not None and getattr(stamp_mask, "size", 0) > 0:
        try:
            m = stamp_mask
            if m.shape[:2] != (gray.shape[0], gray.shape[1]):
                m = cv2.resize(m, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            if cv2.countNonZero(m) > 0:
                gray = cv2.inpaint(gray, m, 3, cv2.INPAINT_TELEA)
                steps.append({"name": "Stamp Inpaint", "detail": "softened official seal/stamp ink"})
        except Exception:
            pass

    # 2. gamma correction ------------------------------------------------
    if gamma == "auto":
        g_used = auto_gamma_value(gray)
        mode = "auto"
    elif gamma == "manual" and gamma_value:
        g_used = float(np.clip(gamma_value, 0.3, 3.0))
        mode = "manual"
    else:
        g_used, mode = 1.0, "off"
    enhanced = apply_gamma(gray, g_used) if mode != "off" else gray.copy()
    new_brightness = float(np.mean(enhanced))
    steps.append({
        "name": "Gamma correction",
        "detail": (f"{mode} gamma={g_used:.2f}, mean brightness "
                   f"{original_brightness:.0f} -> {new_brightness:.0f}"),
    })

    # 3. denoise ---------------------------------------------------------
    if denoise:
        if enhanced.size > 3_000_000:
            enhanced = cv2.medianBlur(enhanced, 3)
            steps.append({"name": "Denoise", "detail": "median filter (large scan)"})
        else:
            enhanced = cv2.fastNlMeansDenoising(enhanced, None, h=8,
                                                templateWindowSize=7,
                                                searchWindowSize=21)
            steps.append({"name": "Denoise", "detail": "non-local means h=8"})

    # 4. CLAHE local contrast -------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(enhanced)
    steps.append({"name": "CLAHE", "detail": "local contrast, clip=2.0"})

    # 5. binarize --------------------------------------------------------
    if binarize:
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=35, C=15)
        steps.append({"name": "Binarize", "detail": "adaptive Gaussian (35, 15)"})
        binary = _despeckle(binary, min_area=6)
        steps.append({"name": "Despeckle", "detail": "remove blobs < 6 px"})
        binary, removed = _remove_rules(binary)
        if removed:
            steps.append({"name": "De-rule",
                          "detail": f"removed {removed} px of table ruling/frame lines"})
    else:
        binary = enhanced

    # 6. deskew ----------------------------------------------------------
    angle = 0.0
    if deskew:
        small = binary
        factor = 1.0
        if max(binary.shape[:2]) > 1200:
            factor = 1200.0 / max(binary.shape[:2])
            small = cv2.resize(binary, None, fx=factor, fy=factor,
                               interpolation=cv2.INTER_AREA)
        angle = _estimate_skew(small)
        if abs(angle) > 0.25:
            binary = _rotate(binary, angle, border=255)
            enhanced = _rotate(enhanced, angle)
            gray = _rotate(gray, angle)
        steps.append({"name": "Deskew",
                      "detail": f"estimated angle {angle:+.2f} deg"
                                + ("" if abs(angle) > 0.25 else " (skipped)")})

    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    info = {
        "gamma_mode": gamma if gamma != "manual" else "manual",
        "gamma_used": round(g_used, 3),
        "skew_angle": round(angle, 2),
        "input_size": [w, h],
        "output_size": [processed.shape[1], processed.shape[0]],
        "mean_brightness_before": round(original_brightness, 1),
        "mean_brightness_after": round(new_brightness, 1),
        "steps": steps,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return {"processed": processed, "enhanced_gray": enhanced,
            "gray": gray, "binary": binary, "info": info}