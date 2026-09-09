"""
OCR engine abstraction for the ILRDVS project.

Supported languages are exactly the languages configured in config/settings.py:
English, Hindi, Bengali, Tamil, Telugu, Kannada, Gujarati, Marathi and Punjabi.
Auto mode uses all installed supported language packs and reports the dominant
Indic script detected in the OCR output.
"""
from __future__ import annotations

import re
import unicodedata

import pytesseract
from pytesseract import Output

AVAILABLE_LANGS = ["eng", "hin", "ben", "tam", "tel", "kan", "guj", "mar", "pan"]
OCR_LANGUAGES = {
    "eng": "English",
    "hin": "Hindi / हिन्दी" ,
    "ben": "Bengali / বাংলা" ,
    "tam": "Tamil / தமிழ்" ,
    "tel": "Telugu / తెలుగు" ,
    "kan": "Kannada / ಕನ್ನಡ" ,
    "guj": "Gujarati / ગુજરાતી" ,
    "mar": "Marathi / मराठी" ,
    "pan": "Punjabi / ਪੰਜਾਬੀ" ,
}

# A bilingual model is normally more accurate for land records because most
# forms contain English field names, numbers, dates and an Indic language.
LANG_MODELS = {
    "eng": "eng",
    "hin": "hin+eng",
    "ben": "ben+eng",
    "tam": "tam+eng",
    "tel": "tel+eng",
    "kan": "kan+eng",
    "guj": "guj+eng",
    "mar": "mar+eng",
    "pan": "pan+eng",
}

# Auto mode is deliberately limited to the languages in settings.py.
# Tesseract accepts a + separated language list.  If one installed pack is
# missing, _available_langs() below removes it instead of crashing the app.
AUTO_CANDIDATES = ["eng", "hin", "ben", "tam", "tel", "kan", "guj", "mar", "pan"]
AUTO_LANGS = "+".join(AUTO_CANDIDATES)

_SCRIPT_RANGES = {
    "hin": ("\u0900", "\u097F"),
    "ben": ("\u0980", "\u09FF"),
    "pan": ("\u0A00", "\u0A7F"),
    "guj": ("\u0A80", "\u0AFF"),
    "tam": ("\u0B80", "\u0BFF"),
    "tel": ("\u0C00", "\u0C7F"),
    "kan": ("\u0C80", "\u0CFF"),
}

# Malayalam is U+0D00..U+0D7F and must be explicitly included.
_SCRIPT_RANGES["mal"] = ("\u0D00", "\u0D7F")

_SCRIPT_TO_LANG = {
    "hin": "hin", "ben": "ben", "pan": "pan", "guj": "guj",
    "tam": "tam", "tel": "tel", "kan": "kan", "mal": "mal",
}


def _installed_languages() -> set[str]:
    """Return installed Tesseract language codes when possible."""
    try:
        return set(pytesseract.get_languages(config=""))
    except Exception:
        return {"eng"}


def _available_langs(requested: str) -> str:
    """Keep only requested language packs that are actually installed."""
    installed = _installed_languages()
    requested_codes = [x for x in requested.split("+") if x]
    usable = [x for x in requested_codes if x in installed]
    if "eng" in installed and "eng" not in usable:
        usable.append("eng")
    return "+".join(dict.fromkeys(usable)) or "eng"


def detect_script(text: str) -> str:
    """Return dominant supported language code from Unicode script coverage.

    Devanagari is reported as Hindi because Unicode script alone cannot
    distinguish Hindi from Marathi.  The Marathi model remains selectable
    explicitly through settings.py.
    """
    counts = {code: 0 for code in _SCRIPT_RANGES}
    latin = 0
    for ch in text:
        matched = False
        for code, (lo, hi) in _SCRIPT_RANGES.items():
            if ord(lo) <= ord(ch) <= ord(hi):
                counts[code] += 1
                matched = True
                break
        if not matched and ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN"):
            latin += 1
    best = max(counts, key=counts.get)
    if counts[best] >= 3 and counts[best] >= latin:
        return _SCRIPT_TO_LANG.get(best, best)
    return "eng"


def tesseract_version() -> str:
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return "unknown"


def _image_to_data(img, langs: str, psm: int):
    config = f"--psm {psm} --oem 1"
    return pytesseract.image_to_data(img, lang=langs, config=config,
                                     output_type=Output.DICT)
def _word_count(data) -> int:
    return sum(1 for t in data.get("text", []) if (t or "").strip())


def _run_once(img, langs: str, psm: int):
    data = _image_to_data(img, langs, psm)
    count = _word_count(data)
    confs = []
    for value, text in zip(data.get("conf", []), data.get("text", [])):
        if not (text or "").strip():
            continue
        try:
            c = float(value)
        except (TypeError, ValueError):
            continue
        if c >= 0:
            confs.append(c)
    avg = sum(confs) / len(confs) if confs else 0.0
    return data, count, avg


def _select_auto_result(img, psm: int):
    """Try the supported packs and select the strongest useful OCR result.

    This is slower than one huge multi-language Tesseract call, but it avoids
    the common problem where a nine-language model produces poor text because
    too many script models compete for the same glyphs.
    """
    installed = _installed_languages()
    candidates = [c for c in AUTO_CANDIDATES if c in installed]
    if not candidates:
        candidates = ["eng"]

    results = []
    for code in candidates:
        langs = LANG_MODELS.get(code, code)
        langs = _available_langs(langs)
        try:
            data, count, avg = _run_once(img, langs, psm)
        except pytesseract.TesseractError:
            continue
        text = " ".join(t for t in data.get("text", []) if (t or "").strip())
        script = detect_script(text)
        # Prefer actual script coverage, then OCR confidence and word count.
        script_chars = sum(
            1 for ch in text
            for lo, hi in _SCRIPT_RANGES.values()
            if ord(lo) <= ord(ch) <= ord(hi)
        )
        score = avg + min(count, 100) * 0.15 + min(script_chars, 100) * 0.35
        if script == "eng" and code != "eng":
            score -= 8
        results.append((score, data, count, avg, langs, script))

    if not results:
        langs = _available_langs("eng")
        data, count, avg = _run_once(img, langs, psm)
        return data, psm, langs, "eng", count, avg

    _, data, count, avg, langs, script = max(results, key=lambda x: x[0])
    return data, psm, langs, script, count, avg


def ocr_image(img, lang: str = "auto", psm: int = 4) -> dict:
    """Run OCR and return words, lines, text and diagnostics."""
    lang = (lang or "auto").strip().lower()

    if lang == "auto":
        try:
            data, used_psm, langs, detected, n_words, avg_raw = _select_auto_result(img, psm)
            if n_words < 5 and used_psm != 3:
                data, used_psm, langs, detected, n_words, avg_raw = _select_auto_result(img, 3)
        except pytesseract.TesseractError:
            langs = _available_langs("eng")
            data, n_words, avg_raw = _run_once(img, langs, psm)
            used_psm = psm
            detected = "eng"
    else:
        requested = LANG_MODELS.get(lang, lang)
        langs = _available_langs(requested)
        try:
            data, n_words, avg_raw = _run_once(img, langs, psm)
            used_psm = psm
            if n_words < 5 and psm != 3:
                retry, retry_count, retry_avg = _run_once(img, langs, 3)
                if retry_count > n_words or retry_avg > avg_raw + 5:
                    data, n_words, avg_raw, used_psm = retry, retry_count, retry_avg, 3
        except pytesseract.TesseractError:
            langs = _available_langs("eng")
            data, n_words, avg_raw = _run_once(img, langs, psm)
            used_psm = psm
        text_for_detection = " ".join(t for t in data.get("text", []) if (t or "").strip())
        detected = lang if lang != "eng" else detect_script(text_for_detection)

    words = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            conf = 45.0
        words.append({
            "text": txt,
            "conf": min(conf, 100.0),
            "x": int(data["left"][i]), "y": int(data["top"][i]),
            "w": int(data["width"][i]), "h": int(data["height"][i]),
            "block": int(data["block_num"][i]),
            "par": int(data["par_num"][i]),
            "line": int(data["line_num"][i]),
        })

    lines_map = {}
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
    avg_conf = sum(w["conf"] for w in words) / max(len(words), 1) / 100.0

    return {
        "words": words,
        "lines": lines,
        "text": full_text,
        "avg_conf": round(avg_conf, 4),
        "lang_used": langs,
        "psm": used_psm,
        "detected_language": detected,
        "engine": f"Tesseract {tesseract_version()}",
        "word_count": len(words),
    }
