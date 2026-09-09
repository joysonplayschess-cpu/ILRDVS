"""
Helper that turns plain text (e.g. a fixture, a pasted OCR dump, or a text
layer already embedded in a PDF) into the same word/line bundle that
``ocr.ocr_image`` returns.

Geometry is synthesised from character positions with a fixed-width metric,
which keeps column alignment intact -- this is what the layout-aware
extractor needs (labels in the left column, values in the right column, or
labels on one row and values on the row underneath).

Used by the test-suite and by ``service`` when a PDF already contains a
digital text layer, so the NLP stage always receives the same structure.
"""
from __future__ import annotations

CHAR_W = 10
LINE_H = 26
LINE_GAP = 12


def bundle_from_text(text: str, conf: float = 92.0, page: int = 0) -> dict:
    """Build an ``{words, lines, text, ...}`` bundle from plain text."""
    lines = []
    words = []
    y = 40
    for li, raw in enumerate(text.splitlines()):
        if not raw.strip():
            y += LINE_H + LINE_GAP
            continue
        ws = []
        col = 0
        for token in raw.split(" "):
            if not token:
                col += 1
                continue
            wd = {
                "text": token,
                "conf": float(conf),
                "x": col * CHAR_W,
                "y": y,
                "w": len(token) * CHAR_W,
                "h": LINE_H,
                "block": 1,
                "par": 1,
                "line": li,
                "page": page,
            }
            ws.append(wd)
            words.append(wd)
            col += len(token) + 1
        line_text = " ".join(w["text"] for w in ws)
        lines.append({"id": (1, 1, li), "text": line_text, "words": ws,
                      "conf": float(conf), "page": page,
                      "x": ws[0]["x"], "y": y, "h": LINE_H})
        y += LINE_H + LINE_GAP

    full = "\n".join(l["text"] for l in lines)
    return {
        "words": words,
        "lines": lines,
        "text": full,
        "avg_conf": round(conf / 100.0, 4),
        "lang_used": "eng",
        "psm": 4,
        "detected_language": "eng",
        "engine": "text-layer",
        "word_count": len(words),
    }
