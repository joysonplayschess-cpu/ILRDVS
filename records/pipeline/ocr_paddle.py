"""
PaddleOCR client -- the third-choice OCR engine (after Bhashini, then
Tesseract). PaddleOCR is an optional dependency (it pulls in paddlepaddle,
which is large); if it isn't installed this module simply raises
PaddleUnavailable so ocr.py's fallback chain skips it silently.

Install with:  pip install paddleocr paddlepaddle
"""
from __future__ import annotations

import functools

# Our internal language codes -> PaddleOCR language codes. PaddleOCR's
# multilingual models don't cover every Indic script we support; codes with
# no supported Paddle model fall back to "en" (still better than crashing).
LANG_TO_PADDLE = {
    "eng": "en", "hin": "hi", "mar": "mr", "tam": "ta", "tel": "te",
    "kan": "ka", "guj": "en",  # no dedicated Gujarati PP-OCR model
    "ben": "en",               # no dedicated Bengali PP-OCR model
    "pan": "en",               # no dedicated Punjabi PP-OCR model
}


class PaddleUnavailable(Exception):
    """Raised when paddleocr isn't installed, or the recognition call
    itself fails. ocr.py catches this and treats it as end-of-fallback-chain."""


@functools.lru_cache(maxsize=8)
def _get_engine(paddle_lang: str):
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise PaddleUnavailable(
            "paddleocr is not installed (pip install paddleocr paddlepaddle)"
        ) from exc
    try:
        return PaddleOCR(use_angle_cls=True, lang=paddle_lang, show_log=False)
    except Exception as exc:  # model download / init failure
        raise PaddleUnavailable(f"could not initialise PaddleOCR: {exc}") from exc


def ocr_image(img, lang: str = "auto") -> dict:
    """Run PaddleOCR on a BGR/greyscale numpy image (OpenCV-style).

    Returns the same dict shape as records.pipeline.ocr.ocr_image().
    Raises PaddleUnavailable if paddleocr isn't installed or recognition
    fails outright.
    """
    paddle_lang = LANG_TO_PADDLE.get(lang, "en")
    engine = _get_engine(paddle_lang)

    try:
        result = engine.ocr(img, cls=True)
    except Exception as exc:
        raise PaddleUnavailable(f"PaddleOCR inference failed: {exc}") from exc

    words, lines = [], []
    # PaddleOCR returns a list-per-image of [ [box, (text, conf)], ... ]
    detections = (result or [[]])[0] or []
    for i, det in enumerate(detections):
        try:
            box, (text, conf) = det
        except (ValueError, TypeError):
            continue
        text = (text or "").strip()
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        word = {"text": text, "conf": min(float(conf) * 100, 100.0),
               "x": int(min(xs)), "y": int(min(ys)),
               "w": int(max(xs) - min(xs)), "h": int(max(ys) - min(ys)),
               "block": 0, "par": 0, "line": i}
        words.append(word)
        lines.append({"id": (0, 0, i), "text": text, "words": [word],
                      "conf": word["conf"]})

    lines.sort(key=lambda l: l["words"][0]["y"])
    full_text = "\n".join(l["text"] for l in lines)
    avg_conf = (sum(w["conf"] for w in words) / len(words) / 100.0
               if words else 0.0)

    return {
        "words": words, "lines": lines, "text": full_text,
        "avg_conf": round(avg_conf, 4), "lang_used": paddle_lang,
        "psm": None, "detected_language": paddle_lang,
        "engine": "PaddleOCR", "word_count": len(words),
    }
