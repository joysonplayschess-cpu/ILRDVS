"""
Pipeline orchestration.

``process_document`` drives a Document through
stamp detection -> adaptive/minimal-safe enhance -> OCR (+ rescue) ->
field extraction -> validation -> record creation.

``apply_verification`` persists human corrections, feeds the learning
memory, re-runs validation and updates the record status.
"""
from __future__ import annotations

import time
import traceback

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone

from records.constants import FIELD_KEYS, FIELD_WEIGHTS
from records.models import (Document, DocumentPage, FieldExtraction,
                            LandRecord, ValidationIssue)
from records.pipeline import extract, ocr, preprocess, stamps, table_extract, validate


def _log(doc: Document, stage: str, **extra):
    entry = {"time": timezone.now().strftime("%H:%M:%S"), "stage": stage}
    entry.update({k: v for k, v in extra.items() if v is not None})
    doc.processing_log = (doc.processing_log or []) + [entry]


def _looks_digitally_clean(bgr) -> bool:
    """
    Fast heuristic: sharp, decent contrast, mostly white background, little dark fill.
    Used when quality.py is missing OR to force minimal path on e-Services PDFs.
    """
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark_pct = float((gray < 60).mean())
    bright_pct = float((gray > 220).mean())
    med = cv2.medianBlur(gray, 3)
    noise = float(np.abs(gray.astype(np.int16) - med.astype(np.int16)).std())
    return (
        blur >= 90.0
        and contrast >= 40.0
        and noise <= 14.0
        and dark_pct < 0.15
        and bright_pct > 0.35
        and min(gray.shape[:2]) >= 700
    )


def _run_preprocess(page, gamma_mode, gamma_value, stamp_mask, force_level: str | None):
    """
    force_level: "minimal" | "moderate" | "heavy" | None (auto).
    Prefers preprocess_minimal / preprocess_moderate if present; else safe fallbacks.
    """
    level = force_level or "heavy"
    has_minimal = hasattr(preprocess, "preprocess_minimal")
    has_moderate = hasattr(preprocess, "preprocess_moderate")

    if level == "minimal" and has_minimal:
        pre = preprocess.preprocess_minimal(page, stamp_mask=stamp_mask)
        return pre, "minimal"
    if level == "moderate" and has_moderate:
        pre = preprocess.preprocess_moderate(
            page, stamp_mask=stamp_mask, gamma=gamma_mode, gamma_value=gamma_value)
        return pre, "moderate"

    # heavy = existing full pipeline
    try:
        pre = preprocess.preprocess_image(
            page, gamma=gamma_mode, gamma_value=gamma_value, stamp_mask=stamp_mask)
    except TypeError:
        pre = preprocess.preprocess_image(page, gamma=gamma_mode, gamma_value=gamma_value)
    return pre, "heavy"


def _ocr_score(res: dict) -> tuple:
    return (int(res.get("word_count") or 0), float(res.get("avg_conf") or 0.0))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_document(doc: Document, gamma_mode: str = "auto",
                     gamma_value: float | None = None,
                     lang: str | None = None) -> LandRecord | None:
    """Run the complete AI pipeline for one uploaded document."""
    from records.utils import log_audit

    lang = lang or doc.language or "auto"
    doc.status = "PROCESSING"
    doc.error_message = ""
    doc.processing_log = []
    doc.save(update_fields=["status", "error_message", "processing_log"])
    _log(doc, "queued", message="Pipeline started",
         gamma_mode=str(gamma_mode), language=lang)

    try:
        # ---------------- 1. load -------------------------------------
        pages, source_page_count = preprocess.load_pages(
            doc.file.path, max_pages=settings.OCR_MAX_PAGES)
        doc.page_count = source_page_count
        doc.processed_page_count = len(pages)
        if source_page_count > len(pages):
            _log(doc, "load", pages=len(pages), source_pages=source_page_count,
                 message=f"Only the first {len(pages)} of "
                         f"{source_page_count} page(s) were processed.")
        else:
            _log(doc, "load", pages=len(pages))

        # ---------------- 2. stamps + enhance + OCR each page ---------
        merged_lines, merged_words = [], []
        merged_text, confs, engine, used_lang = [], [], "", ""
        detected = ""
        pre_info = {}
        table_fields_all: dict[str, dict] = {}
        all_stamp_issues: list[dict] = []

        DocumentPage.objects.filter(document=doc).delete()

        for i, page in enumerate(pages):
            page_t0 = time.perf_counter()

            # --- 2a. stamp detection (always) ---
            stamp_report = stamps.detect_stamps(page)
            # Only pass mask if valid stamps were found (prevents false masks on tables)
            use_mask = stamp_report.mask if getattr(stamp_report, "boxes", None) else None
            if stamp_report.boxes:
                _log(doc, "stamp_detect", page=i + 1,
                     stamps_found=len(stamp_report.boxes),
                     circularity=f"{getattr(stamp_report, 'circularity', 0):.2f}")
            else:
                _log(doc, "stamp_detect", page=i + 1, stamps_found=0)

            # --- 2b. choose preprocess level ---
            chosen_level = "heavy"
            try:
                from records.pipeline.quality import assess_document_quality
                if getattr(settings, "ADAPTIVE_PREPROCESS_ENABLED", True):
                    q = assess_document_quality(page)
                    chosen_level = q.level
                    _log(doc, "quality", page=i + 1, level=q.level,
                         quality_score=q.quality_score, reasons=q.reasons,
                         assess_ms=q.elapsed_ms)
                else:
                    chosen_level = "heavy"
            except Exception:
                if _looks_digitally_clean(page):
                    chosen_level = "minimal"
                    _log(doc, "quality", page=i + 1, level="minimal",
                         reasons=["digital/clean heuristic (no quality.py)"])
                else:
                    chosen_level = "heavy"
                    _log(doc, "quality", page=i + 1, level="heavy",
                         reasons=["not digital-clean / quality.py unavailable"])

            # Real stamp present → upgrade to moderate to allow inpainting
            if stamp_report.boxes and chosen_level == "minimal":
                chosen_level = "moderate"
                _log(doc, "quality", page=i + 1,
                     message="stamp present -> upgrade minimal to moderate")

            # --- 2c. preprocess ---
            p_t0 = time.perf_counter()
            try:
                pre, used_level = _run_preprocess(
                    page, gamma_mode, gamma_value, use_mask, chosen_level)
            except Exception as pe:
                _log(doc, "preprocess_error", page=i + 1, error=str(pe)[:160])
                gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
                pre = {
                    "processed": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                    "info": {
                        "level": "original_fallback",
                        "gamma_used": 1.0,
                        "skew_angle": 0.0,
                        "mean_brightness_before": float(gray.mean()),
                        "mean_brightness_after": float(gray.mean()),
                        "steps": [{"name": "Fallback", "detail": "original gray"}],
                    },
                }
                used_level = "original_fallback"
            p_ms = (time.perf_counter() - p_t0) * 1000

            pre_info = pre.get("info") or {}
            _log(doc, "enhance", page=i + 1,
                 level=used_level,
                 gamma=pre_info.get("gamma_used"),
                 skew=pre_info.get("skew_angle"),
                 brightness=(
                     f"{pre_info.get('mean_brightness_before', 0):.0f} -> "
                     f"{pre_info.get('mean_brightness_after', 0):.0f}"
                 ),
                 preprocess_ms=round(p_ms, 1))

            ok, buf = cv2.imencode(".png", pre["processed"])
            page_row = None
            if ok:
                page_row = DocumentPage(document=doc, page_number=i + 1)
                page_row.processed_image.save(
                    f"processed_{doc.pk}_p{i + 1}.png",
                    ContentFile(buf.tobytes()), save=False)
            if i == 0:
                if ok:
                    doc.processed_image.save(
                        f"processed_{doc.pk}.png",
                        ContentFile(buf.tobytes()), save=False)
                doc.preprocess_info = pre_info

            # --- 2d. OCR ---
            o_t0 = time.perf_counter()
            try:
                ocr_res = ocr.ocr_image(pre["processed"], lang=lang)
            except Exception as ocr_exc:
                _log(doc, "ocr_error", page=i + 1, error=str(ocr_exc)[:160])
                if page_row is not None:
                    page_row.save()
                continue

            # --- 2e. OCR RESCUE: destroyed enhance must never win ---
            wc = int(ocr_res.get("word_count") or 0)
            ac = float(ocr_res.get("avg_conf") or 0.0)
            if wc < 8 or (wc < 25 and ac < 0.20):
                _log(doc, "ocr_rescue", page=i + 1,
                     message=f"weak OCR after {used_level} "
                             f"({wc} words, conf={ac:.2f}); retrying light paths")

                candidates = [("enhanced", ocr_res, pre)]

                # A) minimal (no binarize) if available
                try:
                    if hasattr(preprocess, "preprocess_minimal"):
                        light = preprocess.preprocess_minimal(page, stamp_mask=None)
                    else:
                        g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
                        light = {
                            "processed": cv2.cvtColor(g, cv2.COLOR_GRAY2BGR),
                            "info": {
                                "level": "minimal_fallback",
                                "gamma_used": 1.0, "skew_angle": 0.0,
                                "mean_brightness_before": float(g.mean()),
                                "mean_brightness_after": float(g.mean()),
                                "steps": [{"name": "Minimal", "detail": "gray only"}],
                            },
                        }
                    ocr_light = ocr.ocr_image(light["processed"], lang=lang)
                    candidates.append(("minimal", ocr_light, light))
                except Exception as e:
                    _log(doc, "ocr_rescue_error", page=i + 1, path="minimal",
                         error=str(e)[:120])

                # B) near-original grayscale
                try:
                    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
                    raw_img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                    ocr_raw = ocr.ocr_image(raw_img, lang=lang)
                    raw_pre = {
                        "processed": raw_img,
                        "info": {
                            "level": "original_rescue",
                            "gamma_used": 1.0, "skew_angle": 0.0,
                            "mean_brightness_before": float(g.mean()),
                            "mean_brightness_after": float(g.mean()),
                            "steps": [{"name": "Rescue", "detail": "original gray"}],
                        },
                    }
                    candidates.append(("original", ocr_raw, raw_pre))
                except Exception as e:
                    _log(doc, "ocr_rescue_error", page=i + 1, path="original",
                         error=str(e)[:120])

                best_name, ocr_res, pre = max(
                    candidates, key=lambda t: _ocr_score(t[1]))
                used_level = f"{best_name} (rescue)" if best_name != "enhanced" else used_level
                pre_info = pre.get("info") or pre_info

                # refresh UI preview if rescue won
                if best_name != "enhanced":
                    ok2, buf2 = cv2.imencode(".png", pre["processed"])
                    if ok2 and page_row is not None:
                        page_row.processed_image.save(
                            f"processed_{doc.pk}_p{i + 1}.png",
                            ContentFile(buf2.tobytes()), save=False)
                    if i == 0 and ok2:
                        doc.processed_image.save(
                            f"processed_{doc.pk}.png",
                            ContentFile(buf2.tobytes()), save=False)
                        doc.preprocess_info = pre_info

                _log(doc, "ocr_rescue_pick", page=i + 1, picked=best_name,
                     words=ocr_res.get("word_count"),
                     conf=ocr_res.get("avg_conf"))

            o_ms = (time.perf_counter() - o_t0) * 1000

            engine = ocr_res.get("engine") or engine
            used_lang = ocr_res.get("lang_used") or used_lang
            detected = ocr_res.get("detected_language") or detected
            confs.append(float(ocr_res.get("avg_conf") or 0.0))
            merged_text.append(ocr_res.get("text") or "")

            if page_row is not None:
                page_row.ocr_confidence = ocr_res.get("avg_conf") or 0.0
                page_row.word_count = ocr_res.get("word_count") or 0
                page_row.save()

            # Tag geometry with page number
            for wd in ocr_res.get("words") or []:
                wd["page"] = i
            for ln in ocr_res.get("lines") or []:
                ln["page"] = i

            merged_lines.extend(ocr_res.get("lines") or [])
            merged_words.extend(ocr_res.get("words") or [])

            # Stamp validation against OCR boxes
            page_stamp_issues = stamps.validate_stamps(
                stamp_report,
                word_boxes=ocr_res.get("words") or [],
                ocr_text=ocr_res.get("text") or "",
            )
            if page_stamp_issues:
                all_stamp_issues.extend(page_stamp_issues)
                _log(doc, "stamp_validate", page=i + 1, issues=len(page_stamp_issues))

            _log(doc, "ocr", page=i + 1,
                 words=ocr_res.get("word_count"),
                 confidence=f"{float(ocr_res.get('avg_conf') or 0) * 100:.1f}%",
                 language=used_lang,
                 level=used_level,
                 ocr_ms=round(o_ms, 1))

            print(
                f"[ILRDVS] p{i + 1} Mode={used_level} "
                f"Stamp={'YES' if stamp_report.boxes else 'NO'} "
                f"Pre={p_ms:.0f}ms OCR={o_ms:.0f}ms "
                f"Words={ocr_res.get('word_count')} "
                f"Conf={float(ocr_res.get('avg_conf') or 0) * 100:.1f}%"
            )

            # Ruled-table cell extraction on ORIGINAL page (not destroyed enhance)
            try:
                page_table_fields = table_extract.extract_table_fields(
                    page, lang=used_lang)
            except Exception as tbl_exc:
                page_table_fields = {}
                _log(doc, "table_extract_error", page=i + 1,
                     error=str(tbl_exc)[:160])
            for k, v in page_table_fields.items():
                existing = table_fields_all.get(k)
                if existing is None or v["confidence"] > existing["confidence"]:
                    table_fields_all[k] = v

            _log(doc, "page_timing", page=i + 1,
                 total_ms=round((time.perf_counter() - page_t0) * 1000, 1),
                 level=used_level)

        text_all = "\n".join(merged_text)
        avg_conf = sum(confs) / max(len(confs), 1)

        # ---------------- 3. field extraction (NLP) --------------------
        ocr_bundle = {"lines": merged_lines, "words": merged_words,
                      "text": text_all}
        if not merged_lines and not (text_all or "").strip():
            _log(doc, "ocr_empty",
                 message="OCR returned no readable text - the record is "
                         "created empty and queued for manual entry.")
        try:
            fields = extract.extract_fields(ocr_bundle)
        except Exception as nlp_exc:
            _log(doc, "extract_error",
                 error=f"{nlp_exc.__class__.__name__}: {nlp_exc}")
            fields = extract.extract_fields({"lines": [], "words": [], "text": ""})

        threshold = getattr(settings, "OCR_CONFIDENCE_THRESHOLD", 0.75)
        table_hits = 0
        for k, v in table_fields_all.items():
            existing = fields.get(k)
            if existing is not None and v["confidence"] <= existing["confidence"]:
                continue
            fields[k] = {
                "value": v["value"], "confidence": v["confidence"],
                "source": v["source"], "method": v["method"],
                "needs_review": v["confidence"] < threshold,
            }
            table_hits += 1
        if table_hits:
            _log(doc, "table_extract", fields_replaced=table_hits)

        learned_hits = sum(1 for f in fields.values() if f["source"] == "learned")
        found = sum(1 for k in FIELD_KEYS if fields[k]["value"])
        review = sum(1 for k in FIELD_KEYS if fields[k]["needs_review"])
        _log(doc, "extract", fields=f"{found}/{len(FIELD_KEYS)}",
             needs_review=review, learned_corrections=learned_hits)

        # ---------------- 4. persist record ---------------------------
        with transaction.atomic():
            record, _ = LandRecord.objects.get_or_create(document=doc)
            try:
                record.plot_area = (float(fields["plot_area"]["value"])
                                    if fields["plot_area"]["value"] else None)
            except (TypeError, ValueError):
                record.plot_area = None
            for k in FIELD_KEYS:
                if k == "plot_area":
                    continue
                value = fields[k]["value"] or ""
                max_len = getattr(LandRecord._meta.get_field(k), "max_length", None)
                if max_len:
                    value = value[:max_len]
                setattr(record, k, value)
            record.status = "PENDING"
            record.save()

            FieldExtraction.objects.filter(record=record).delete()
            FieldExtraction.objects.bulk_create([
                FieldExtraction(
                    record=record, field_name=k,
                    value=(str(record.plot_area) if k == "plot_area"
                           else fields[k]["value"]),
                    confidence=fields[k]["confidence"],
                    source=fields[k]["source"],
                    method=fields[k].get("method", ""),
                    needs_review=fields[k]["needs_review"])
                for k in FIELD_KEYS
            ])

        # ---------------- 5. validation --------------------------------
        revalidate(record, extra_issues=all_stamp_issues)
        n_err = record.issues.filter(severity="error", resolved=False).count()
        _log(doc, "validate", issues=record.issues.count(), errors=n_err,
             duplicate=record.is_duplicate)

        # ---------------- 6. wrap up ----------------------------------
        record.overall_confidence = _aggregate_confidence(record)
        record.save(update_fields=["overall_confidence"])
        doc.ocr_engine = engine
        doc.ocr_language = used_lang
        doc.detected_language = detected
        doc.ocr_text = text_all
        doc.ocr_confidence = round(avg_conf, 4)
        doc.status = "PROCESSED"
        doc.processed_at = timezone.now()
        doc.save()
        _log(doc, "done",
             overall_confidence=f"{record.overall_confidence * 100:.1f}%")
        doc.save(update_fields=["processing_log"])
        log_audit(
            None, "PROCESS", doc,
            f"OCR pipeline completed: {found}/{len(FIELD_KEYS)} fields, "
            f"confidence {record.overall_confidence * 100:.1f}%, "
            f"{record.issues.count()} validation issue(s).")
        return record

    except Exception as exc:  # noqa: BLE001
        doc.status = "FAILED"
        doc.error_message = f"{exc.__class__.__name__}: {exc}"
        _log(doc, "error", error=doc.error_message)
        doc.processing_log.append(
            {"stage": "traceback", "trace": traceback.format_exc()[-1500:]})
        doc.save()
        log_audit(None, "PROCESS_FAILED", doc, doc.error_message)
        return None


def _aggregate_confidence(record: LandRecord) -> float:
    rows = {f.field_name: f for f in record.fields.all()}
    num = den = 0.0
    for key, weight in FIELD_WEIGHTS.items():
        if weight < 0.7 and not rows.get(key):
            continue
        den += weight
        row = rows.get(key)
        if row is not None and (row.corrected_value or row.value):
            c = max(row.confidence, 0.95 if row.source == "manual" else 0)
            num += weight * c
    return round(num / den, 4) if den else 0.0


def revalidate(record: LandRecord, extra_issues: list[dict] | None = None) -> int:
    record.issues.all().delete()
    issues = validate.validate_record(record)
    if extra_issues:
        seen = {(i["rule_code"], i.get("message", "")) for i in issues}
        for ei in extra_issues:
            key = (ei["rule_code"], ei.get("message", ""))
            if key not in seen:
                issues.append(ei)
                seen.add(key)

    ValidationIssue.objects.bulk_create(
        [ValidationIssue(record=record, **i) for i in issues])
    record.is_duplicate = any(i["rule_code"] == "DUPLICATE_PLOT" for i in issues)
    record.overall_confidence = _aggregate_confidence(record)
    record.save(update_fields=["is_duplicate", "overall_confidence", "updated_at"])
    return sum(1 for i in issues if i["severity"] == "error")


def apply_verification(record: LandRecord, cleaned: dict, user,
                       note: str = "", request=None) -> LandRecord:
    from records.pipeline import learn
    from records.utils import log_audit

    changes = []
    with transaction.atomic():
        for key, new_value in cleaned.items():
            new_value = (new_value or "").strip()
            old_value = getattr(record, key)
            old_str = "" if old_value is None else str(old_value)
            if key == "plot_area":
                try:
                    parsed = float(new_value) if new_value else None
                except ValueError:
                    parsed = None
                if parsed != record.plot_area:
                    changes.append((key, old_str, new_value))
                    record.plot_area = parsed
                    _update_field_row(record, key, old_str, new_value)
                continue
            if new_value != old_str:
                changes.append((key, old_str, new_value))
                setattr(record, key, new_value)
                _update_field_row(record, key, old_str, new_value)
                if old_str:
                    learn.record_correction(key, old_str, new_value)

        record.save()
        record.review_note = note or record.review_note
        n_errors = revalidate(record)

        for key, old, new in changes:
            log_audit(user, "FIELD_CORRECTED", record,
                      f"{key}: '{old}' -> '{new}'", request=request)

        if n_errors == 0:
            record.status = "VERIFIED"
            record.verified_by = user if user.is_authenticated else None
            record.verified_at = timezone.now()

        record.overall_confidence = _aggregate_confidence(record)
        record.save()
        log_audit(user, "VERIFY", record,
                  f"Verified with {len(changes)} correction(s); "
                  f"status={record.status}", request=request)
    return record


def _update_field_row(record, key, old, new):
    row = record.fields.filter(field_name=key).first()
    if row is None:
        row = record.fields.create(field_name=key, value=old)
    row.corrected_value = new
    row.source = "manual"
    row.needs_review = False
    row.confidence = 0.97
    row.method = "manual verification by officer"
    row.save()


def reject_record(record: LandRecord, user, note: str, request=None):
    from records.utils import log_audit
    record.status = "REJECTED"
    record.review_note = note
    record.verified_by = user if user.is_authenticated else None
    record.verified_at = timezone.now()
    record.save(update_fields=["status", "review_note", "verified_by",
                               "verified_at", "updated_at"])
    log_audit(user, "REJECT", record, note or "record rejected", request=request)
    return record