"""
Stamp / seal detection for land-record scans.

Round revenue seals and rectangular office stamps are found with colour-
ink masks, morphological closing, and contour analysis. Validation flags
stamps that sit on top of OCR text, implausible stamp counts, and missing
certification blocks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from django.conf import settings

CERT_TERMS = (
    "signature", "signed", "lekhpal", "certified", "certify", "certificate",
    "attest", "attested", "seal",
    "लेखपाल", "प्रमाणित", "हस्ताक्षर", "मुहर", "सत्यापित",
)


@dataclass
class StampBox:
    """Axis-aligned stamp location in source-image pixels."""
    x: int
    y: int
    w: int
    h: int
    circularity: float = 0.0
    kind: str = "rect"  # "round" | "rect"

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class StampReport:
    boxes: list[StampBox] = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.uint8))
    circularity: float = 0.0  # mean circularity of detected stamps (0 if none)


def detect_stamps(image_bgr, min_ratio: float = 0.002,
                  max_ratio: float = 0.12) -> StampReport:
    """Detect round seals and rectangular stamps on a BGR document image.

    ``min_ratio`` / ``max_ratio`` bound contour area as a fraction of the
    page. Returns a :class:`StampReport` with boxes, a binary mask, and the
    mean circularity of accepted detections.
    """
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return StampReport()

    img = image_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    page_area = float(max(h * w, 1))
    min_area = max(min_ratio * page_area, 25.0)
    max_area = max_ratio * page_area

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_mask = _ink_color_mask(hsv)
    ink_mask = _dark_ink_mask(gray)
    combined = cv2.bitwise_or(color_mask, ink_mask)

    k_round = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    k_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 11))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.bitwise_or(
        cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_round),
        cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_rect),
    )
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

    stamp_mask = np.zeros((h, w), dtype=np.uint8)
    boxes: list[StampBox] = []

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        circularity = 0.0
        if peri > 1e-6:
            circularity = float(4.0 * np.pi * area / (peri * peri))
            circularity = float(np.clip(circularity, 0.0, 1.5))
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 8 or bh < 8:
            continue
        aspect = bw / float(bh)
        extent = area / float(max(bw * bh, 1))
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        is_round = circularity >= 0.62 and 0.65 <= aspect <= 1.45
        is_rect = (
            0.35 <= aspect <= 3.2
            and extent >= 0.35
            and (len(approx) in (4, 5, 6) or (circularity < 0.62 and extent >= 0.45))
        )
        if not (is_round or is_rect):
            continue
        # Skip page-border frames (touching most of the image edge).
        if x <= 2 and y <= 2 and bw >= w * 0.9 and bh >= h * 0.9:
            continue
        kind = "round" if is_round else "rect"
        boxes.append(StampBox(x=int(x), y=int(y), w=int(bw), h=int(bh),
                              circularity=circularity, kind=kind))
        cv2.drawContours(stamp_mask, [cnt], -1, 255, thickness=-1)

    for circle in _hough_seals(gray, min_area, max_area):
        if not _overlaps_any(circle.bbox, [b.bbox for b in boxes], 0.35):
            boxes.append(circle)
            x, y, bw, bh = circle.bbox
            cv2.ellipse(stamp_mask, (x + bw // 2, y + bh // 2),
                        (bw // 2, bh // 2), 0, 0, 360, 255, -1)

    boxes = _nms(boxes, iou_thresh=0.4)
    mean_c = (sum(b.circularity for b in boxes) / len(boxes)) if boxes else 0.0
    return StampReport(boxes=boxes, mask=stamp_mask, circularity=mean_c)


def validate_stamps(report: StampReport, word_boxes, ocr_text: str) -> list[dict]:
    """Return validation issue dicts for ``report`` against OCR geometry/text."""
    issues: list[dict] = []
    boxes = list(getattr(report, "boxes", None) or [])
    words = list(word_boxes or [])
    try:
        iou_thresh = float(getattr(settings, "STAMP_OVERLAP_TEXT_IOU", 0.15))
    except (TypeError, ValueError):
        iou_thresh = 0.15

    for i, stamp in enumerate(boxes):
        sb = _as_xywh(stamp)
        hits = 0
        for wb in words:
            wb_xywh = _as_xywh(wb)
            if wb_xywh is None:
                continue
            if _stamp_word_overlap(sb, wb_xywh, iou_thresh):
                hits += 1
            if hits >= 3:
                break
        if hits >= 3:
            issues.append({
                "rule_code": "STAMP_OVERLAP",
                "field_name": "",
                "severity": "error",
                "message": (
                    f"Stamp {i + 1} overlaps {hits} or more OCR word boxes "
                    f"(IoU > {iou_thresh:.2f}); the seal may cover text."
                ),
            })

    if len(boxes) >= 3:
        issues.append({
            "rule_code": "STAMP_MULTIPLE",
            "field_name": "",
            "severity": "warning",
            "message": f"{len(boxes)} stamps detected; confirm which seals are valid.",
        })

    require_cert = bool(getattr(settings, "STAMP_REQUIRE_CERT_BLOCK", True))
    if require_cert and not boxes and not _has_cert_terms(ocr_text, words):
        issues.append({
            "rule_code": "STAMP_MISSING",
            "field_name": "",
            "severity": "error",
            "message": (
                "No stamp/seal detected and no signature, lekhpal, or "
                "certified wording was found in the OCR text."
            ),
        })

    return issues


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def _ink_color_mask(hsv: np.ndarray) -> np.ndarray:
    """Red / blue / purple / green office-ink hues (typical stamp colours)."""
    ranges = (
        ((0, 40, 40), (12, 255, 255)),
        ((165, 40, 40), (180, 255, 255)),
        ((85, 35, 40), (135, 255, 255)),
        ((125, 35, 40), (165, 255, 255)),
        ((35, 50, 40), (85, 255, 255)),
    )
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
    return mask


def _dark_ink_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Drop speckle / single-glyph text; keep denser stamp ink.
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return inv


def _hough_seals(gray: np.ndarray, min_area: float,
                 max_area: float) -> list[StampBox]:
    h, w = gray.shape[:2]
    min_r = max(8, int((min_area / np.pi) ** 0.5))
    max_r = max(min_r + 1, int((max_area / np.pi) ** 0.5))
    max_r = min(max_r, min(h, w) // 2)
    blurred = cv2.medianBlur(gray, 5)
    try:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(30, min_r),
            param1=80, param2=28, minRadius=min_r, maxRadius=max_r,
        )
    except cv2.error:
        return []
    found: list[StampBox] = []
    if circles is None:
        return found
    for cx, cy, r in np.round(circles[0]).astype(int):
        area = float(np.pi * r * r)
        if area < min_area or area > max_area:
            continue
        x = int(max(0, cx - r))
        y = int(max(0, cy - r))
        bw = int(min(w - x, 2 * r))
        bh = int(min(h - y, 2 * r))
        found.append(StampBox(x=x, y=y, w=bw, h=bh, circularity=1.0, kind="round"))
    return found


def _nms(boxes: list[StampBox], iou_thresh: float) -> list[StampBox]:
    if len(boxes) <= 1:
        return boxes
    remaining = sorted(boxes, key=lambda b: b.w * b.h, reverse=True)
    kept: list[StampBox] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [
            b for b in remaining
            if _iou(best.bbox, b.bbox) < iou_thresh
        ]
    return kept


def _overlaps_any(box, others, thresh: float) -> bool:
    return any(_iou(box, other) >= thresh for other in others)


# ---------------------------------------------------------------------------
# Geometry / validation helpers
# ---------------------------------------------------------------------------
def _as_xywh(obj) -> tuple[int, int, int, int] | None:
    if obj is None:
        return None
    if isinstance(obj, StampBox):
        return obj.bbox
    if isinstance(obj, dict):
        if "bbox" in obj and obj["bbox"] is not None:
            return _as_xywh(obj["bbox"])
        if all(k in obj for k in ("x", "y", "w", "h")):
            return (int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"]))
        if all(k in obj for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (int(obj["x1"]), int(obj["y1"]),
                              int(obj["x2"]), int(obj["y2"]))
            return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 4:
        return (int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]))
    return None


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = float(aw * ah + bw * bh - inter)
    return inter / union if union > 0 else 0.0


def _stamp_word_overlap(stamp_xywh, word_xywh, thresh: float) -> bool:
    """True if classic IoU or word-coverage exceeds ``thresh``.

    Word boxes are much smaller than seals, so intersection/union alone
    often stays below 0.15 even when a word sits fully inside the stamp.
    """
    if _iou(stamp_xywh, word_xywh) > thresh:
        return True
    sx, sy, sw, sh = stamp_xywh
    wx, wy, ww, wh = word_xywh
    word_area = float(max(ww * wh, 1))
    x1, y1 = max(sx, wx), max(sy, wy)
    x2, y2 = min(sx + sw, wx + ww), min(sy + sh, wy + wh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return (inter / word_area) > thresh


def _has_cert_terms(ocr_text: str, word_boxes) -> bool:
    blob = (ocr_text or "").lower()
    extra = []
    for wb in word_boxes or []:
        if isinstance(wb, dict) and wb.get("text"):
            extra.append(str(wb["text"]))
    blob = blob + "\n" + " ".join(extra).lower()
    if not blob.strip():
        return False
    for term in CERT_TERMS:
        if term.lower() in blob:
            return True
        if re.search(rf"\b{re.escape(term)}\b", blob, re.IGNORECASE):
            return True
    return False
