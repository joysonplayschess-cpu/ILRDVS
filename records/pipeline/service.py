"""
Pipeline orchestration.

``process_document`` drives a Document through
enhancement -> stamp detection -> OCR -> field extraction -> validation -> record creation.

``apply_verification`` persists human corrections, feeds the learning
memory, re-runs validation and updates the record status.
"""
from __future__ import annotations

import traceback
from datetime import datetime

import cv2
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
                         f"{source_page_count} page(s) were processed "
                         f"(OCR_MAX_PAGES={settings.OCR_MAX_PAGES}); the "
                         f"rest were not enhanced, OCR'd, or extracted.")
        else:
            _log(doc, "load", pages=len(pages))

        # ---------------- 2. detect stamps, enhance + OCR each page ----
        merged_lines, merged_words = [], []
        merged_text, confs, engine, used_lang = [], [], "", ""
        detected = ""
        pre_info = {}
        table_fields_all: dict[str, dict] = {}
        all_stamp_issues: list[dict] = []
        
        DocumentPage.objects.filter(document=doc).delete()
        for i, page in enumerate(pages):
            # Stamp & seal detection on original page
            stamp_report = stamps.detect_stamps(page)
            if stamp_report.boxes:
                _log(doc, "stamp_detect", page=i + 1,
                     stamps_found=len(stamp_report.boxes),
                     circularity=f"{stamp_report.circularity:.2f}")

            # Enhance image (passing stamp mask if supported by preprocess)
            try:
                pre = preprocess.preprocess_image(
                    page, gamma=gamma_mode, gamma_value=gamma_value,
                    stamp_mask=stamp_report.mask
                )
            except TypeError:
                pre = preprocess.preprocess_image(
                    page, gamma=gamma_mode, gamma_value=gamma_value
                )

            pre_info = pre["info"]
            _log(doc, "enhance", page=i + 1, gamma=pre_info["gamma_used"],
                 skew=pre_info["skew_angle"],
                 brightness=f"{pre_info['mean_brightness_before']:.0f} -> "
                            f"{pre_info['mean_brightness_after']:.0f}")

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

            try:
                ocr_res = ocr.ocr_image(pre["processed"], lang=lang)
            except Exception as ocr_exc:      # engine missing / page unreadable
                _log(doc, "ocr_error", page=i + 1, error=str(ocr_exc)[:160])
                if page_row is not None:
                    page_row.save()
                continue

            engine, used_lang = ocr_res["engine"], ocr_res["lang_used"]
            detected = ocr_res["detected_language"]
            confs.append(ocr_res["avg_conf"])
            merged_text.append(ocr_res["text"])
            if page_row is not None:
                page_row.ocr_confidence = ocr_res["avg_conf"]
                page_row.word_count = ocr_res["word_count"]
                page_row.save()

            # Tag geometry with page number
            for wd in ocr_res["words"]:
                wd["page"] = i
            for ln in ocr_res["lines"]:
                ln["page"] = i
            merged_lines.extend(ocr_res["words"])
            merged_words.extend(ocr_res["words"])

            # Validate stamps against page OCR bounding boxes & text
            page_stamp_issues = stamps.validate_stamps(
                stamp_report, word_boxes=ocr_res["words"], ocr_text=ocr_res["text"]
            )
            if page_stamp_issues:
                all_stamp_issues.extend(page_stamp_issues)
                _log(doc, "stamp_validate", page=i + 1, issues=len(page_stamp_issues))

            _log(doc, "ocr", page=i + 1, words=ocr_res["word_count"],
                 confidence=f"{ocr_res['avg_conf'] * 100:.1f}%",
                 language=used_lang)

            # Ruled-table cell extraction
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

        text_all = "\n".join(merged_text)
        avg_conf = sum(confs) / max(len(confs), 1)

        # ---------------- 3. field extraction (NLP) --------------------
        ocr_bundle = {"lines": merged_lines, "words": merged_words,
                      "text": text_all}
        if not merged_lines:
            _log(doc, "ocr_empty",
                 message="OCR returned no readable text - the record is "
                         "created empty and queued for manual entry.")
        try:
            fields = extract.extract_fields(ocr_bundle)
        except Exception as nlp_exc:          # never lose the OCR result
            _log(doc, "extract_error", error=f"{nlp_exc.__class__.__name__}: "
                                             f"{nlp_exc}")
            fields = extract.extract_fields({"lines": [], "words": [],
                                             "text": ""})

        # Table cell values override if confidence is higher
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

        learned_hits = sum(1 for f in fields.values()
                           if f["source"] == "learned")
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
                max_len = getattr(LandRecord._meta.get_field(k),
                                  "max_length", None)
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
        _log(doc, "done", overall_confidence=f"{record.overall_confidence * 100:.1f}%")
        doc.save(update_fields=["processing_log"])
        log_audit(None, "PROCESS", doc,
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
    """Weighted mean of field confidences (missing fields count as 0)."""
    rows = {f.field_name: f for f in record.fields.all()}
    num = den = 0.0
    for key, weight in FIELD_WEIGHTS.items():
        if weight < 0.7 and not rows.get(key):
            continue        # optional fields only count when present
        den += weight
        row = rows.get(key)
        if row is not None and (row.corrected_value or row.value):
            c = max(row.confidence, 0.95 if row.source == "manual" else 0)
            num += weight * c
    return round(num / den, 4) if den else 0.0


def revalidate(record: LandRecord, extra_issues: list[dict] | None = None) -> int:
    """Re-run the rule engine; returns the number of open error issues."""
    record.issues.all().delete()
    issues = validate.validate_record(record)
    if extra_issues:
        # Merge stamp validation issues deduplicated by rule_code & message
        seen = {(i["rule_code"], i.get("message", "")) for i in issues}
        for ei in extra_issues:
            key = (ei["rule_code"], ei.get("message", ""))
            if key not in seen:
                issues.append(ei)
                seen.add(key)

    ValidationIssue.objects.bulk_create(
        [ValidationIssue(record=record, **i) for i in issues])
    record.is_duplicate = any(i["rule_code"] == "DUPLICATE_PLOT"
                              for i in issues)
    record.overall_confidence = _aggregate_confidence(record)
    record.save(update_fields=["is_duplicate", "overall_confidence",
                               "updated_at"])
    return sum(1 for i in issues if i["severity"] == "error")


# ---------------------------------------------------------------------------
# Human verification
# ---------------------------------------------------------------------------
def apply_verification(record: LandRecord, cleaned: dict, user,
                       note: str = "", request=None) -> LandRecord:
    """Persist officer corrections, feed the learning memory, re-validate."""
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
    log_audit(user, "REJECT", record, note or "record rejected",
              request=request)
    return record