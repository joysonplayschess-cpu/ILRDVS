"""
Cheap, pre-preprocessing document quality assessment.

Runs only fast NumPy / OpenCV calls (no NL-Means, no inpainting, no OCR).
Produces a structured report and a recommended preprocessing level:

    "minimal"  -> preprocess.preprocess_minimal
    "moderate" -> preprocess.preprocess_moderate
    "heavy"    -> preprocess.preprocess_image   (existing full pipeline)

Stamp detection is intentionally *not* skipped here -- a cheap pre-check
sets `stamp_hint` so the orchestrator can still run the full stamp
detector + inpainting even on otherwise clean pages.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np
from django.conf import settings

# --- defaults (also live in settings.PREPROCESSING_CONFIG; override there) ---
_DEFAULTS = {
    "min_width": 900,          # below this, likely low-DPI phone scan
    "min_height": 900,
    "max_side": 3200,          # matches preprocess.MAX_SIDE
    "blur_threshold": 90.0,    # variance of Laplacian; <90 = blurry
    "noise_threshold": 12.0,   # std of high-pass residual
    "contrast_threshold": 45.0,  # std of gray; <45 = flat
    "dark_pixel_pct": 0.35,    # >35% very dark pixels -> dark scan
    "bright_pixel_pct": 0.55,  # >55% near-white -> faded scan
    "skew_threshold_deg": 1.5,
    "edge_density_low": 0.010, # <1% edges -> almost empty
    "edge_density_high": 0.22, # >22% edges -> speckle / heavy noise
    "stamp_ink_ratio": 0.010,  # colored-ink pixels > 1% of page
}

def _cfg(key: str):
    src = getattr(settings, "PREPROCESSING_CONFIG", {}) or {}
    return src.get(key, _DEFAULTS[key])

@dataclass
class QualityReport:
    width: int
    height: int
    resolution_score: float
    brightness: float
    contrast: float
    dark_pct: float
    bright_pct: float
    noise: float
    blur: float                  # variance of Laplacian (higher = sharper)
    edge_density: float
    skew_estimate_deg: float
    stamp_hint: bool
    quality_score: float         # 0..1, higher = cleaner
    level: str                   # "minimal" | "moderate" | "heavy"
    reasons: list[str]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _cheap_stamp_hint(bgr: np.ndarray) -> tuple[bool, float]:
    """Very cheap colored-ink ratio; NOT a replacement for stamps.detect_stamps."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ranges = (
        ((0, 60, 60), (12, 255, 255)),      # red low
        ((165, 60, 60), (180, 255, 255)),   # red high
        ((85, 50, 50), (135, 255, 255)),    # blue/purple
    )
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    ratio = float(cv2.countNonZero(m)) / float(m.size or 1)
    return ratio >= _cfg("stamp_ink_ratio"), ratio

def _quick_skew(gray: np.ndarray) -> float:
    """Coarse skew estimate on a downscaled binarized copy (bounded time)."""
    h, w = gray.shape[:2]
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(gray, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    _, binv = cv2.threshold(small, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binv > 0))
    if len(coords) < 400:
        return 0.0
    if len(coords) > 15000:
        idx = np.random.default_rng(7).choice(len(coords), 15000, replace=False)
        coords = coords[idx]
    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    ang = float(rect[-1])
    if ang < -45:
        ang = -(90 + ang)
    else:
        ang = -ang
    return 0.0 if abs(ang) > 12.0 else ang

def assess_document_quality(bgr: np.ndarray) -> QualityReport:
    """Return a QualityReport and recommended preprocessing level."""
    t0 = time.perf_counter()
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return QualityReport(0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.0, False, 0.0,
                             "heavy", ["empty image -> heavy"], 0.0)

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # brightness / contrast / dark-bright distribution
    brightness = float(gray.mean())
    contrast = float(gray.std())
    dark_pct = float((gray < 60).mean())
    bright_pct = float((gray > 220).mean())

    # sharpness (Laplacian variance)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # noise proxy: gray - median-blurred gray
    med = cv2.medianBlur(gray, 3)
    noise = float(np.abs(gray.astype(np.int16) - med.astype(np.int16)).std())

    # edge density
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(cv2.countNonZero(edges)) / float(edges.size or 1)

    # skew (coarse)
    skew = _quick_skew(gray)

    # resolution score (short side vs min_width)
    short_side = min(h, w)
    res_score = max(0.0, min(1.0, short_side / float(_cfg("min_width"))))

    stamp_hint, _stamp_ratio = _cheap_stamp_hint(bgr)

    reasons: list[str] = []

    # ---- Level decision (documented, configurable) ----
    level = "minimal"

    if short_side < int(_cfg("min_width") * 0.8):
        level = "heavy"
        reasons.append(f"short_side {short_side}px < 0.8*min_width")

    if dark_pct > _cfg("dark_pixel_pct"):
        level = "heavy"
        reasons.append(f"dark_pct {dark_pct:.2f} > {_cfg('dark_pixel_pct')}")
    if bright_pct > _cfg("bright_pixel_pct") and contrast < _cfg("contrast_threshold"):
        level = "heavy"
        reasons.append(f"faded (bright_pct {bright_pct:.2f}, contrast {contrast:.1f})")

    if level == "minimal" and contrast < _cfg("contrast_threshold"):
        level = "moderate"
        reasons.append(f"low contrast {contrast:.1f}")

    if blur < _cfg("blur_threshold") * 0.6:
        level = "heavy"
        reasons.append(f"very blurry (lap-var {blur:.1f})")
    elif level == "minimal" and blur < _cfg("blur_threshold"):
        level = "moderate"
        reasons.append(f"soft (lap-var {blur:.1f})")

    if noise > _cfg("noise_threshold") * 1.5:
        level = "heavy"
        reasons.append(f"noisy (residual std {noise:.1f})")
    elif level == "minimal" and noise > _cfg("noise_threshold"):
        level = "moderate"
        reasons.append(f"mild noise ({noise:.1f})")

    if abs(skew) > _cfg("skew_threshold_deg") * 2:
        level = "heavy"
        reasons.append(f"large skew {skew:+.2f} deg")
    elif level == "minimal" and abs(skew) > _cfg("skew_threshold_deg"):
        level = "moderate"
        reasons.append(f"skew {skew:+.2f} deg")

    if edge_density < _cfg("edge_density_low"):
        if level == "heavy":
            reasons.append("edge density very low, downgrading heavy -> moderate")
            level = "moderate"
    if edge_density > _cfg("edge_density_high"):
        level = "heavy"
        reasons.append(f"speckle-heavy edges {edge_density:.3f}")

    # Born-digital / clean scan heuristic:
    # sharp, decent contrast, white background, low noise, little dark fill
    digital_clean = (
        blur >= _cfg("blur_threshold")
        and contrast >= _cfg("contrast_threshold")
        and noise <= _cfg("noise_threshold")
        and dark_pct < 0.12
        and bright_pct > 0.40
        and short_side >= int(_cfg("min_width") * 0.9)
    )
    if digital_clean:
        level = "minimal"
        reasons = ["digital/clean scan heuristic -> minimal (skip heavy CV)"]

    if not reasons:
        reasons.append("clean scan -> minimal")

    q = 1.0
    q *= 1.0 if contrast >= _cfg("contrast_threshold") else max(0.4, contrast / _cfg("contrast_threshold"))
    q *= 1.0 if blur >= _cfg("blur_threshold") else max(0.4, blur / _cfg("blur_threshold"))
    q *= 1.0 if noise <= _cfg("noise_threshold") else max(0.4, _cfg("noise_threshold") / max(noise, 1))
    q *= max(0.5, 1.0 - dark_pct * 0.6)
    q *= max(0.5, 1.0 - max(0.0, bright_pct - 0.5))
    q *= res_score
    q = float(max(0.0, min(1.0, q)))

    return QualityReport(
        width=w, height=h,
        resolution_score=round(res_score, 3),
        brightness=round(brightness, 1),
        contrast=round(contrast, 1),
        dark_pct=round(dark_pct, 3),
        bright_pct=round(bright_pct, 3),
        noise=round(noise, 2),
        blur=round(blur, 2),
        edge_density=round(edge_density, 4),
        skew_estimate_deg=round(skew, 2),
        stamp_hint=bool(stamp_hint),
        quality_score=round(q, 3),
        level=level,
        reasons=reasons,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )