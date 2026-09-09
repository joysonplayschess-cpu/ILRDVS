"""
Bhashini / ULCA Indic OCR client.

Calls MeitY ULCA ``getModelsPipeline`` to resolve a compute endpoint, then
posts the page image to the inference API. Missing credentials or any API
failure return ``None`` so the caller can fall through to another engine.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# Internal Tesseract-style codes -> ULCA ISO-639-1 codes.
LANG_TO_ULCA = {
    "eng": "en",
    "hin": "hi",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "mal": "ml",
    "ben": "bn",
    "guj": "gu",
    "mar": "mr",
    "pan": "pa",
}


@dataclass
class OcrWord:
    text: str
    confidence: float  # 0.0 – 1.0
    bbox: tuple[int, int, int, int]  # x, y, w, h in source-image pixels


@dataclass
class OcrResult:
    text: str
    words: list[OcrWord] = field(default_factory=list)
    mean_confidence: float = 0.0  # 0.0 – 1.0
    lang_used: str = ""
    engine: str = "Bhashini (Digital India / MeitY ULCA)"

    def to_pipeline_dict(self) -> dict:
        """Shape expected by ``records.pipeline.ocr`` / field extraction."""
        words = []
        for i, word in enumerate(self.words):
            x, y, w, h = word.bbox
            conf = word.confidence
            if conf <= 1.0:
                conf = conf * 100.0
            words.append({
                "text": word.text,
                "conf": min(float(conf), 100.0),
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "block": 0, "par": 0, "line": i,
            })
        lines = []
        for i, word in enumerate(words):
            lines.append({
                "id": (0, 0, i),
                "text": word["text"],
                "words": [word],
                "conf": word["conf"],
            })
        avg = self.mean_confidence
        if avg > 1.0:
            avg = avg / 100.0
        return {
            "words": words,
            "lines": lines,
            "text": self.text,
            "avg_conf": round(float(avg), 4),
            "lang_used": self.lang_used,
            "psm": None,
            "detected_language": self.lang_used,
            "engine": self.engine,
            "word_count": len(words),
        }


class BhashiniOCR:
    """Indic OCR via ULCA pipeline-config + inference endpoints."""

    def is_configured(self) -> bool:
        user_id, api_key, _pipeline_id = self._creds()
        return bool(user_id and api_key)

    def recognize(self, img, lang: str = "auto") -> OcrResult | None:
        """Run OCR. Returns ``None`` if keys are missing or the API fails."""
        if not self.is_configured():
            logger.info("Bhashini OCR skipped: BHASHINI_USER_ID / BHASHINI_API_KEY not set")
            return None
        try:
            ulca_lang = self._ulca_lang(lang)
            img_b64, scale = self._encode_image(img)
            if not img_b64:
                return None
            cfg = self._pipeline_config(ulca_lang)
            if cfg is None:
                return None
            payload = self._infer(img_b64, ulca_lang, cfg)
            if payload is None:
                return None
            return self._parse_response(payload, ulca_lang, scale)
        except Exception:
            logger.warning("Bhashini OCR failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _creds(self) -> tuple[str, str, str]:
        return (
            (getattr(settings, "BHASHINI_USER_ID", "") or "").strip(),
            (getattr(settings, "BHASHINI_API_KEY", "") or "").strip(),
            (getattr(settings, "BHASHINI_PIPELINE_ID", "") or "").strip(),
        )

    def _timeout(self) -> int:
        try:
            return int(getattr(settings, "BHASHINI_OCR_TIMEOUT", 120) or 120)
        except (TypeError, ValueError):
            return 120

    def _pipeline_url(self) -> str:
        return (getattr(settings, "BHASHINI_PIPELINE_URL", "")
                or "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline")

    def _fallback_infer_url(self) -> str:
        return (getattr(settings, "BHASHINI_API_URL", "")
                or "https://dhruva-api.bhashini.gov.in/services/inference")

    def _ulca_lang(self, lang: str) -> str:
        code = (lang or "auto").strip().lower()
        if code in ("", "auto"):
            return "en"
        return LANG_TO_ULCA.get(code, code if len(code) == 2 else "en")

    def _auth_headers(self) -> dict:
        user_id, api_key, _ = self._creds()
        return {"userID": user_id, "ulcaApiKey": api_key}

    def _encode_image(self, img) -> tuple[str | None, float]:
        import cv2

        if img is None:
            return None, 1.0
        work = img
        h, w = work.shape[:2]
        try:
            max_side = int(getattr(settings, "BHASHINI_MAX_IMAGE_SIDE", 2000) or 2000)
        except (TypeError, ValueError):
            max_side = 2000
        scale = 1.0
        longest = max(h, w)
        if longest > max_side > 0:
            scale = max_side / float(longest)
            work = cv2.resize(
                work,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", work, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            logger.warning("Bhashini OCR: could not encode image")
            return None, 1.0
        return base64.b64encode(buf.tobytes()).decode("ascii"), scale

    def _pipeline_config(self, ulca_lang: str) -> dict | None:
        _user_id, _api_key, pipeline_id = self._creds()
        body = {
            "pipelineTasks": [{
                "taskType": "ocr",
                "config": {"language": {"sourceLanguage": ulca_lang}},
            }],
            "pipelineRequestConfig": {"pipelineId": pipeline_id},
        }
        try:
            resp = requests.post(
                self._pipeline_url(),
                json=body,
                headers=self._auth_headers(),
                timeout=self._timeout(),
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Bhashini pipeline config call failed: %s", exc)
            return None

        try:
            ocr_task = next(
                t for t in data["pipelineResponseConfig"]
                if t.get("taskType") == "ocr"
            )
            service_id = ocr_task["config"][0]["serviceId"]
            endpoint = data["pipelineInferenceAPIEndPoint"]
            compute_url = endpoint.get("callbackUrl") or self._fallback_infer_url()
            key = endpoint.get("inferenceApiKey") or {}
            name = key.get("name") or "Authorization"
            value = key.get("value") or ""
            inference_header = {name: value} if value else {}
        except (KeyError, IndexError, StopIteration, TypeError) as exc:
            logger.warning("Bhashini pipeline config shape unexpected: %s", exc)
            return None

        return {
            "service_id": service_id,
            "compute_url": compute_url,
            "inference_header": inference_header,
        }

    def _infer(self, img_b64: str, ulca_lang: str, cfg: dict) -> dict | None:
        body = {
            "pipelineTasks": [{
                "taskType": "ocr",
                "config": {
                    "language": {"sourceLanguage": ulca_lang},
                    "serviceId": cfg["service_id"],
                    "modality": "print-word-level",
                },
            }],
            "inputData": {"image": [{"imageContent": img_b64}]},
        }
        headers = {"Content-Type": "application/json", **cfg.get("inference_header", {})}
        try:
            resp = requests.post(
                cfg["compute_url"],
                json=body,
                headers=headers,
                timeout=self._timeout(),
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Bhashini inference call failed: %s", exc)
            return None

    def _parse_response(self, data: dict, ulca_lang: str, scale: float) -> OcrResult | None:
        try:
            outputs = data["pipelineResponse"][0]["output"]
        except (KeyError, IndexError, TypeError):
            outputs = data.get("output") or data.get("pipelineResponse") or data

        words: list[OcrWord] = []
        inv = (1.0 / scale) if scale else 1.0

        def walk(node, prefer_children: bool = True):
            if isinstance(node, list):
                for item in node:
                    walk(item, prefer_children)
                return
            if not isinstance(node, dict):
                return
            children = (
                node.get("regions")
                or node.get("tokens")
                or node.get("words")
                or node.get("lines")
            )
            if prefer_children and children:
                walk(children, prefer_children)
                return
            text = node.get("source") or node.get("text") or node.get("target") or ""
            if not isinstance(text, str):
                text = str(text) if text else ""
            text = text.strip()
            if not text:
                if children:
                    walk(children, prefer_children)
                return
            conf = node.get("confidence", node.get("conf", 0.9))
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.9
            if conf > 1.0:
                conf = conf / 100.0
            bbox = self._bbox(node.get("boundingBox") or node.get("bbox"), inv)
            words.append(OcrWord(text=text, confidence=max(0.0, min(conf, 1.0)), bbox=bbox))

        walk(outputs)
        if not words:
            return None

        mean = sum(w.confidence for w in words) / len(words)
        text = "\n".join(w.text for w in words)
        return OcrResult(
            text=text,
            words=words,
            mean_confidence=mean,
            lang_used=ulca_lang,
        )

    def _bbox(self, raw, inv: float) -> tuple[int, int, int, int]:
        if not raw:
            return (0, 0, 0, 0)
        vertices = []
        if isinstance(raw, dict):
            vertices = raw.get("vertices") or raw.get("points") or []
            if not vertices:
                x1 = raw.get("x1", raw.get("x", 0)) or 0
                y1 = raw.get("y1", raw.get("y", 0)) or 0
                x2 = raw.get("x2", x1) or x1
                y2 = raw.get("y2", y1) or y1
                w = raw.get("w", raw.get("width", (x2 - x1))) or 0
                h = raw.get("h", raw.get("height", (y2 - y1))) or 0
                return (
                    int(float(x1) * inv),
                    int(float(y1) * inv),
                    int(float(w) * inv),
                    int(float(h) * inv),
                )
        elif isinstance(raw, (list, tuple)):
            vertices = raw
        xs, ys = [], []
        for v in vertices:
            if isinstance(v, dict):
                xs.append(float(v.get("x", 0) or 0))
                ys.append(float(v.get("y", 0) or 0))
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                xs.append(float(v[0]))
                ys.append(float(v[1]))
        if not xs or not ys:
            return (0, 0, 0, 0)
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        return (
            int(x0 * inv),
            int(y0 * inv),
            int((x1 - x0) * inv),
            int((y1 - y0) * inv),
        )
