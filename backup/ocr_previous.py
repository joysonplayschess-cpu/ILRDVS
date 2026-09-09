"""
OCR engine abstraction.

Primary engine: **Tesseract** (free & open-source,
https://github.com/tesseract-ocr/tesseract) invoked through ``pytesseract``.
Supports English and the installed Indian language packs
(hin, ben, tam, tel, kan, guj, mar, pan).  An ``auto`` mode runs the combined
``eng+hin`` model and the script detector reports the dominant language.
"""
from __future__ import annotations

import re
import unicodedata

import pytesseract
from pytesseract import Output

AVAILABLE_LANGS = ["eng", "hin", "ben", "tam", "tel", "kan", "guj", "mar",
                   "pan"]
AUTO_LANGS = "eng+hin"
# Bilingual record forms: pair Indic packs with the Latin model.
LANG_MODELS = {"auto": AUTO_LANGS, "hin": "hin+eng", "ben": "ben+eng",
               "tam": "tam+eng", "tel": "tel+eng", "kan": "kan+eng",
               "guj": "guj+eng", "mar": "mar+eng", "pan": "pan+eng",
               "eng": "eng"}

_SCRIPT_RANGES = {
    "hin": ("\u0900", "\u097F"),   # Devanagari
    "ben": ("\u0980", "\u09FF"),
    "tam": ("\u0B80", "\u0BFF"),
    "tel": ("\u0C00", "\u0C7F"),
    "kan": ("\u0C80", "\u0CFF"),
    "guj": ("\u0A80", "\u0AFF"),
    "pan": ("\u0A00", "\u0A7F"),   # Gurmukhi
}


def detect_script(text: str) -> str:
    """Return dominant language code based on Unicode script coverage."""
    counts = {code: 0 for code in _SCRIPT_RANGES}
    latin = 0
    for ch in text:
        for code, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= ch <= hi:
                counts[code] += 1
                break
        else:
            if ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN"):
                latin += 1
    best = max(counts, key=counts.get)
    if counts[best] > 5 and counts[best] >= latin:
        return best
    return "eng"


def tesseract_version() -> str:
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return "unknown"


def _image_to_data(img, langs, psm):
    config = f"--psm {psm} --oem 1"
    return pytesseract.image_to_data(img, lang=langs, config=config,
                                     output_type=Output.DICT)


def ocr_image(img, lang: str = "auto", psm: int = 4) -> dict:
    """Run OCR and return words-with-confidence, text and diagnostics.

    Page-segmentation strategy: PSM 4 (column of variable-size text) suits
    ruled land-record forms; if it returns suspiciously little text we
    automatically retry with fully automatic PSM 3 and keep the richer
    result.
    """
    langs = LANG_MODELS.get(lang, lang) or AUTO_LANGS
    try:
        data = _image_to_data(img, langs, psm)
        n_words = sum(1 for t in data["text"] if (t or "").strip())
        if n_words < 45:
            retry = _image_to_data(img, langs, 3)
            n_retry = sum(1 for t in retry["text"] if (t or "").strip())
            if n_retry > n_words:
                data, psm = retry, 3
    except pytesseract.TesseractError:
        # requested language pack missing -> graceful fallback to English
        langs = "eng"
        data = _image_to_data(img, langs, psm)

    words = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:      # tesseract uses -1 for structural rows
            conf = 45.0   # structure rows that kept text anyway
        words.append({
            "text": txt,
            "conf": min(conf, 100.0),
            "x": int(data["left"][i]), "y": int(data["top"][i]),
            "w": int(data["width"][i]), "h": int(data["height"][i]),
            "block": int(data["block_num"][i]),
            "par": int(data["par_num"][i]),
            "line": int(data["line_num"][i]),
        })

    # group into lines (used by the field extractor and to rebuild text)
    lines_map: dict[tuple, list] = {}
    for wd in words:
        lines_map.setdefault((wd["block"], wd["par"], wd["line"]), []).append(wd)
    lines = []
    for key in sorted(lines_map):
        ws = sorted(lines_map[key], key=lambda w: w["x"])
        text = " ".join(w["text"] for w in ws)
        conf = sum(w["conf"] for w in ws) / max(len(ws), 1)
        lines.append({"id": key, "text": text, "words": ws, "conf": conf})

    full_text = "\n".join(l["text"] for l in lines)
    full_text = re.sub(r"[ \t]{2,}", " ", full_text).strip()
    total_words = max(len(words), 1)
    avg_conf = sum(w["conf"] for w in words) / total_words / 100.0

    return {
        "words": words,
        "lines": lines,
        "text": full_text,
        "avg_conf": round(avg_conf, 4),
        "lang_used": langs,
        "psm": psm,
        "detected_language": detect_script(full_text),
        "engine": f"Tesseract {tesseract_version()}",
        "word_count": len(words),
    }
