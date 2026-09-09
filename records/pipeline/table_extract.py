"""Ruled-table cell segmentation + per-cell OCR.

Land-record tables (Tamil Nadu patta/chitta extracts, khatauni grids, etc.)
put a field's label in a header ROW and its value in a DATA row below,
with header cells often wrapping onto a second OCR line before the real
data row even starts. Tesseract's line grouping has no notion of ruled
grid lines, so records/pipeline/extract.py's row/column geometry
heuristics can still misread a wrapped header continuation as if it were
the value (confirmed on a real Tamil Nadu patta scan: father_name came
back as "Soil Remark" / "sName Type" -- neighbouring header cells'
wrapped text, not real data).

This module sidesteps that failure mode entirely: it finds the ruled grid
directly in the image (independent of how Tesseract grouped its OCR
lines) and OCRs each cell on its own, so a header cell and a data cell are
never at risk of being read as one blob.

Returned fields are merged into records/pipeline/extract.py's output by
records/pipeline/service.py, keeping whichever source has higher
confidence per field -- this module is a supplement to, not a
replacement for, the line/label-based extractor (it only produces values
for fields whose column header it recognises).
"""
from __future__ import annotations

import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from . import ocr as ocr_module
from .extract import parse_value

# Normalized table header -> canonical FIELD_KEYS name (see
# records/constants.py). "_area_*" are pseudo-keys resolved into
# plot_area/area_unit below rather than being field names themselves.
TABLE_HEADER_ALIASES = {
    "survey no": "survey_number",
    "survey number": "survey_number",
    "s no": "survey_number",
    "sno": "survey_number",
    "khasra no": "khasra_number",
    "khata no": "khata_number",
    "patta no": "khata_number",
    "patta number": "khata_number",
    "owner name": "owner_name",
    "father s name": "father_name",
    "fathers name": "father_name",
    "father name": "father_name",
    "hectares": "_area_hectare",
    "ares": "_area_are",
    "cents": "_area_cent",
}

_AREA_UNIT_BY_KEY = {"_area_hectare": "hectare", "_area_are": "are", "_area_cent": "cent"}
# Tamil Nadu patta tables commonly split area into hectares/ares/cents
# columns; prefer the finest-grained one present on a row.
_AREA_PRIORITY = ["_area_cent", "_area_are", "_area_hectare"]


def _normalize_header(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", raw.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _line_positions(bin_img: np.ndarray, axis: int, min_run: int):
    """Coordinates of ruling lines: axis=0 -> row boundaries (horizontal
    rules), axis=1 -> column boundaries (vertical rules)."""
    if axis == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_run, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_run))
    lines = cv2.erode(bin_img, kernel)
    lines = cv2.dilate(lines, kernel)
    profile = lines.sum(axis=1) if axis == 0 else lines.sum(axis=0)
    positions = [i for i, v in enumerate(profile) if v > 0]
    collapsed: list[list[int]] = []
    for p in positions:
        if not collapsed or p - collapsed[-1][-1] > 3:
            collapsed.append([p])
        else:
            collapsed[-1].append(p)
    return [int(np.mean(run)) for run in collapsed]


def detect_table_regions(gray: np.ndarray):
    """Bounding boxes of ruled-line tables. Requires BOTH multiple
    horizontal AND multiple vertical rules inside the box, so a single
    underline or a plain page-border box is never mistaken for a data
    grid."""
    h, w = gray.shape[:2]
    inv = 255 - gray
    bin_img = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 15, -2)
    horiz_min_run = max(20, w // 30)
    vert_min_run = max(20, h // 60)
    horiz = cv2.erode(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_min_run, 1)))
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_min_run, 1)))
    vert = cv2.erode(bin_img, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_min_run)))
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_min_run)))
    grid = cv2.dilate(cv2.bitwise_or(horiz, vert),
                      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if (bw * bh) / img_area < 0.01:
            continue
        roi_h, roi_v = horiz[y:y + bh, x:x + bw], vert[y:y + bh, x:x + bw]
        if (np.count_nonzero(roi_h.sum(axis=1) > 0) < 2
                or np.count_nonzero(roi_v.sum(axis=0) > 0) < 2):
            continue
        boxes.append((x, y, x + bw, y + bh))
    return boxes


def segment_cells(table_crop: np.ndarray):
    """grid[row][col] = (x1, y1, x2, y2) cell bbox from ruling-line
    intersections. Falls back to one whole-crop cell if no usable grid is
    found."""
    h, w = table_crop.shape[:2]
    if h < 2 or w < 2:
        return [[(0, 0, w, h)]]

    inv = 255 - table_crop
    bin_img = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 15, -2)
    rows = _line_positions(bin_img, 0, max(20, w // 3))
    cols = _line_positions(bin_img, 1, max(20, h // 4))
    if 0 not in rows:
        rows = [0] + rows
    if (h - 1) not in rows:
        rows = rows + [h - 1]
    if 0 not in cols:
        cols = [0] + cols
    if (w - 1) not in cols:
        cols = cols + [w - 1]
    rows, cols = sorted(set(rows)), sorted(set(cols))
    if len(rows) < 2 or len(cols) < 2:
        return [[(0, 0, w, h)]]

    grid = []
    for ri in range(len(rows) - 1):
        y1, y2 = rows[ri], rows[ri + 1]
        if y2 - y1 < 10:
            # A real text row needs meaningfully more room than this; a
            # thin sliver here is a rounding artifact at the crop edge
            # (e.g. the gap between the last ruling line and the bottom
            # of the detected table box), not an actual row.
            continue
        row_cells = []
        for ci in range(len(cols) - 1):
            x1, x2 = cols[ci], cols[ci + 1]
            if x2 - x1 < 10:
                continue
            row_cells.append((x1, y1, x2, y2))
        if row_cells:
            grid.append(row_cells)
    return grid or [[(0, 0, w, h)]]


def _inset(bbox, margin: int, w: int, h: int):
    """Shrink a cell bbox inward so the ruling lines forming its own
    border aren't included in the OCR crop -- left in, they render as
    stray "[" / "|" glyphs that corrupt recognition (confirmed: header
    cells like "Survey No" were coming back as "[ee |" before this)."""
    x1, y1, x2, y2 = bbox
    nx1, ny1 = min(x1 + margin, x2 - 1), min(y1 + margin, y2 - 1)
    nx2, ny2 = max(x2 - margin, nx1 + 1), max(y2 - margin, ny1 + 1)
    return max(nx1, 0), max(ny1, 0), min(nx2, w), min(ny2, h)


def _ocr_cell(crop: np.ndarray, langs: str) -> tuple[str, float]:
    """PSM 6 (uniform block of text) rather than PSM 7 (single line) --
    a header cell here is often two lines wrapped within one cell (e.g.
    "Survey" / "No"), and PSM 7 forcing a single-line read on that
    dropped the text entirely (confirmed: came back empty). Small cells
    are also upscaled 2x with cubic interpolation first -- Tesseract's
    accuracy on a tiny native-resolution crop is known to be poor,
    upscaling is the standard mitigation. Cells come from the raw (not
    despeckled) scan, so a light per-cell Otsu threshold is applied
    first -- cheap, and safe at this small a crop size unlike the
    page-wide despeckle that erases the table's ruling lines."""
    if crop.size == 0:
        return "", 0.0
    crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    if crop.std() > 1:  # skip Otsu on a near-blank cell (would just amplify noise)
        _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    try:
        data = pytesseract.image_to_data(padded, lang=langs, config="--psm 6 --oem 1",
                                         output_type=Output.DICT)
    except Exception:
        return "", 0.0

    words, confs = [], []
    for t, c in zip(data.get("text", []), data.get("conf", [])):
        if (t or "").strip():
            words.append(t.strip())
            try:
                cv_ = float(c)
            except (TypeError, ValueError):
                cv_ = -1
            if cv_ >= 0:
                confs.append(cv_)
    text = " ".join(words).strip()
    conf = (sum(confs) / len(confs) / 100.0) if confs else (0.5 if text else 0.0)
    return text, conf


def extract_table_fields(raw_bgr: np.ndarray, lang: str = "eng") -> dict:
    """Returns {field_key: {"value", "confidence", "source", "method"}}
    built directly from ruled-table cell geometry. Only produces entries
    for column headers this module recognises (see
    TABLE_HEADER_ALIASES) -- callers merge this into the regular
    line/label-based extraction, keeping whichever side is more
    confident per field.

    Deliberately takes the RAW page image, not the fully preprocessed one
    (records/pipeline/preprocess.py's despeckle + heavy binarization is
    tuned for Tesseract's line-based OCR and reliably erases a ruled
    table's thin border lines as "noise" -- verified: after that pipeline
    runs, a real Tamil Nadu patta scan's table grid has zero surviving
    ruling lines, so line detection here would find nothing on it.
    Line/grid structure has to be read before that cleanup happens."""
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    langs = ocr_module._available_langs(ocr_module.LANG_MODELS.get(lang, lang or "eng"))

    results: dict[str, dict] = {}
    for (x1, y1, x2, y2) in detect_table_regions(gray):
        crop = gray[y1:y2, x1:x2]
        grid = segment_cells(crop)
        if len(grid) < 2:
            continue  # no data row beneath the header row -- nothing to extract

        header_cells = []
        for (hx1, hy1, hx2, hy2) in grid[0]:
            ix1, iy1, ix2, iy2 = _inset((hx1, hy1, hx2, hy2), 4, crop.shape[1], crop.shape[0])
            header_cells.append(_ocr_cell(crop[iy1:iy2, ix1:ix2], langs))
        headers = [(text.strip() or f"col_{i}") for i, (text, _c) in enumerate(header_cells)]
        canonical = [TABLE_HEADER_ALIASES.get(_normalize_header(h)) for h in headers]
        if not any(canonical):
            continue  # this ruled grid doesn't look like a recognised field table

        for data_row in grid[1:]:
            row_area_candidates: dict[str, tuple[str, float]] = {}
            for i, (cx1, cy1, cx2, cy2) in enumerate(data_row):
                key = canonical[i] if i < len(canonical) else None
                if not key:
                    continue
                ix1, iy1, ix2, iy2 = _inset((cx1, cy1, cx2, cy2), 4, crop.shape[1], crop.shape[0])
                text, conf = _ocr_cell(crop[iy1:iy2, ix1:ix2], langs)
                text = text.strip()
                if not text or text in ("-", "--", "---", "\u2014"):
                    continue

                if key in _AREA_UNIT_BY_KEY:
                    row_area_candidates[key] = (text, conf)
                    continue

                value, quality, _extra = parse_value(key, text)
                if not value or quality <= 0:
                    continue
                existing = results.get(key)
                score = conf * quality
                if existing is None or score > existing["confidence"]:
                    results[key] = {
                        "value": value, "confidence": round(score, 4),
                        "source": "table_cell", "method": "ruled-table cell OCR",
                    }

            for area_key in _AREA_PRIORITY:
                if area_key not in row_area_candidates:
                    continue
                text, conf = row_area_candidates[area_key]
                m = re.search(r"[\d.]+", text)
                if not m:
                    continue
                existing = results.get("plot_area")
                if existing is None or conf > existing["confidence"]:
                    results["plot_area"] = {
                        "value": m.group(0), "confidence": round(conf, 4),
                        "source": "table_cell", "method": "ruled-table cell OCR",
                    }
                    results["area_unit"] = {
                        "value": _AREA_UNIT_BY_KEY[area_key], "confidence": round(conf, 4),
                        "source": "table_cell", "method": "ruled-table cell OCR",
                    }
                break  # only the highest-priority area column per row

    return results
