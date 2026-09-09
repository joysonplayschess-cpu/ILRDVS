"""
Backward-compatible wrappers around :class:`records.pipeline.bhashini_ocr.BhashiniOCR`.

New code should import ``BhashiniOCR`` / ``OcrResult`` from ``bhashini_ocr``.
"""
from __future__ import annotations

from records.pipeline.bhashini_ocr import BhashiniOCR, LANG_TO_ULCA, OcrResult

LANG_TO_BHASHINI = LANG_TO_ULCA


class BhashiniUnavailable(Exception):
    """Raised by the legacy ``ocr_image()`` helper when Bhashini cannot run."""


def is_configured() -> bool:
    return BhashiniOCR().is_configured()


def ocr_image(img, lang: str = "auto") -> dict:
    result = BhashiniOCR().recognize(img, lang=lang)
    if result is None:
        raise BhashiniUnavailable("Bhashini OCR unavailable")
    return result.to_pipeline_dict()
