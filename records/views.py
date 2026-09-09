"""HTML views: dashboard, upload, documents, verification, records, audit."""

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Q, Sum
from django.db.models.functions import TruncMonth
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .constants import FIELD_DEFS, FIELD_LABELS
from .models import (AuditLog, Document, FieldExtraction, LandRecord,
                     LearnedCorrection, ValidationIssue)
from .pipeline import service
from .utils import get_role, log_audit, role_required

ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",
               ".webp"}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
@login_required
def dashboard(request):
    docs = Document.objects.all()
    recs = LandRecord.objects.all()

    n_docs = docs.count()
    n_processed = docs.filter(status="PROCESSED").count()
    kpis = {
        "documents": n_docs,
        "processed": n_processed,
        "processed_pct": round(100 * n_processed / n_docs, 1) if n_docs else 0,
        "avg_confidence": round(
            (recs.aggregate(a=Avg("overall_confidence"))["a"] or 0) * 100, 1),
        "verified": recs.filter(status="VERIFIED").count(),
        "pending": recs.filter(status="PENDING").count(),
        "duplicates": recs.filter(is_duplicate=True).count(),
        "failed": docs.filter(status="FAILED").count(),
        "fields_review": FieldExtraction.objects.filter(
            needs_review=True, record__status="PENDING").count(),
        "learned": LearnedCorrection.objects.count(),
        "learned_applied": LearnedCorrection.objects.aggregate(
            s=Sum("times_applied"))["s"] or 0,
        "corrections": FieldExtraction.objects.filter(
            source="manual").exclude(corrected_value="").count(),
    }

    # ---- chart datasets --------------------------------------------------
    status_rows = docs.values("status").annotate(n=Count("id"))
    rec_status_rows = recs.values("status").annotate(n=Count("id"))

    lang_rows = (docs.exclude(detected_language="")
                 .values("detected_language").annotate(n=Count("id")))

    trend = list(recs.order_by("created_at")
                 .values_list("pk", "overall_confidence"))[-40:]

    states = (docs.exclude(state_name="")
              .values("state_name")
              .annotate(total=Count("id"),
                        processed=Count("id", filter=Q(status="PROCESSED"))))
    verified_by_state = (recs.filter(status="VERIFIED")
                         .exclude(state="")
                         .values("state").annotate(n=Count("id")))
    ver_map = {r["state"]: r["n"] for r in verified_by_state}
    state_labels = [s["state_name"] for s in states]

    issue_rows = (ValidationIssue.objects.filter(resolved=False)
                  .values("rule_code").annotate(n=Count("id")).order_by("-n"))
    sev_rows = (ValidationIssue.objects.filter(resolved=False)
                .values("severity").annotate(n=Count("id")))

    monthly = (docs.annotate(m=TruncMonth("uploaded_at"))
               .values("m").annotate(n=Count("id")).order_by("m"))

    chart_data = {
        "doc_status": {"labels": [dict(Document.STATUS_CHOICES).get(r["status"], r["status"])
                                  for r in status_rows],
                       "data": [r["n"] for r in status_rows]},
        "rec_status": {"labels": [dict(LandRecord.STATUS_CHOICES).get(r["status"], r["status"])
                                  for r in rec_status_rows],
                       "data": [r["n"] for r in rec_status_rows]},
        "languages": {"labels": [r["detected_language"].upper() for r in lang_rows],
                      "data": [r["n"] for r in lang_rows]},
        "accuracy": {"labels": [f"#{pk}" for pk, _ in trend],
                     "data": [round(c * 100, 1) for _, c in trend]},
        "states": {"labels": state_labels,
                   "processed": [s["processed"] for s in states],
                   "verified": [ver_map.get(s["state_name"], 0) for s in states],
                   "total": [s["total"] for s in states]},
        "issues": {"labels": [r["rule_code"] for r in issue_rows],
                   "data": [r["n"] for r in issue_rows]},
        "severity": {"labels": [r["severity"] for r in sev_rows],
                     "data": [r["n"] for r in sev_rows]},
        "monthly": {"labels": [m["m"].strftime("%b %Y") for m in monthly],
                    "data": [m["n"] for m in monthly]},
    }

    recent_docs = docs[:6]
    queue = (recs.filter(status="PENDING")
             .order_by("overall_confidence")
             .select_related("document")[:6])
    learned_recent = LearnedCorrection.objects.order_by("-created_at")[:6]

    return render(request, "records/dashboard.html", {
        "kpis": kpis, "chart_data": chart_data,
        "recent_docs": recent_docs, "queue": queue,
        "learned_recent": learned_recent, "role": get_role(request.user),
    })


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
@role_required("ADMIN", "OPERATOR")
def upload(request):
    if request.method == "POST":
        files = request.FILES.getlist("files")
        language = request.POST.get("language", "auto")
        gamma_mode = request.POST.get("gamma_mode", "auto")
        gamma_value = request.POST.get("gamma_value") or None
        state_name = request.POST.get("state_name", "").strip()
        district_name = request.POST.get("district_name", "").strip()
        try:
            gamma_value = float(gamma_value) if gamma_value else None
        except ValueError:
            gamma_value = None

        if not files:
            messages.error(request, "Please select at least one file.")
            return redirect("upload")

        processed_docs, failures = [], 0
        for fh in files:
            ext = "." + fh.name.rsplit(".", 1)[-1].lower() if "." in fh.name else ""
            if ext not in ALLOWED_EXT:
                messages.error(request, f"{fh.name}: unsupported file type {ext!r}.")
                failures += 1
                continue

            # Deduplication guard: prevent double-clicks from creating duplicate jobs
            active_job = Document.objects.filter(
                original_name=fh.name,
                file_size=fh.size,
                status="PROCESSING",
                uploaded_by=request.user
            ).first()

            if active_job:
                messages.warning(
                    request,
                    f"'{fh.name}' is already being processed (Document #{active_job.pk}). Skipping duplicate submission."
                )
                continue

            doc = Document.objects.create(
                file=fh, original_name=fh.name, file_size=fh.size,
                language=language, state_name=state_name,
                district_name=district_name, uploaded_by=request.user)
            log_audit(request.user, "UPLOAD", doc,
                      f"Uploaded '{fh.name}' ({fh.size / 1024:.0f} KB)",
                      request=request)
            
            try:
                record = service.process_document(
                    doc, gamma_mode=gamma_mode, gamma_value=gamma_value,
                    lang=language)
            except Exception as exc:
                doc.status = "FAILED"
                doc.error_message = f"Processing error: {exc}"
                doc.save(update_fields=["status", "error_message"])
                record = None

            if record is None:
                messages.error(request, f"{fh.name}: processing failed - "
                                        f"see document log for details.")
                failures += 1
            else:
                processed_docs.append(doc)

        if processed_docs:
            messages.success(
                request,
                f"{len(processed_docs)} document(s) digitized successfully."
                + (f" {failures} failed." if failures else ""))
            if len(processed_docs) == 1:
                return redirect("document_detail", pk=processed_docs[0].pk)
        return redirect("document_list")

    return render(request, "records/upload.html", {
        "languages": settings.OCR_LANGUAGES, "role": get_role(request.user),
    })


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@login_required
def document_list(request):
    qs = Document.objects.all()
    status = request.GET.get("status", "")
    q = request.GET.get("q", "").strip()
    if status:
        qs = qs.filter(status=status)
    if q:
        qs = qs.filter(Q(original_name__icontains=q) |
                       Q(district_name__icontains=q) |
                       Q(state_name__icontains=q) |
                       Q(record__owner_name__icontains=q))
    page = Paginator(qs, 12).get_page(request.GET.get("page"))
    return render(request, "records/document_list.html", {
        "page": page, "status": status, "q": q,
        "statuses": Document.STATUS_CHOICES, "role": get_role(request.user),
    })


@login_required
def document_detail(request, pk):
    doc = get_object_or_404(Document.objects.select_related(), pk=pk)
    record = getattr(doc, "record", None)
    fields = []
    issues = []
    if record:
        rows = {f.field_name: f for f in record.fields.all()}
        for key, label, *_ in FIELD_DEFS:
            row = rows.get(key)
            fields.append({
                "key": key, "label": label,
                "value": getattr(record, key),
                "row": row,
            })
        issues = record.issues.all()
    audits = AuditLog.objects.filter(
        Q(object_type="Document", object_id=str(doc.pk)) |
        Q(object_type="LandRecord",
          object_id=str(record.pk) if record else "-1")
    )[:30]
    pages = doc.pages.all().order_by("page_number")

    return render(request, "records/document_detail.html", {
        "doc": doc, "record": record, "fields": fields, "issues": issues,
        "pages": pages, "audits": audits, "role": get_role(request.user),
        "threshold_pct": settings.OCR_CONFIDENCE_THRESHOLD * 100,
    })


@require_POST
@role_required("ADMIN", "OPERATOR")
def document_reprocess(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    gamma_mode = request.POST.get("gamma_mode", "auto")
    gamma_value = request.POST.get("gamma_value")
    try:
        gamma_value = float(gamma_value) if gamma_value else None
    except ValueError:
        gamma_value = None
    log_audit(request.user, "REPROCESS", doc, "Re-run of the OCR pipeline",
              request=request)
    try:
        record = service.process_document(doc, gamma_mode=gamma_mode,
                                          gamma_value=gamma_value,
                                          lang=doc.language)
    except Exception as exc:
        doc.status = "FAILED"
        doc.error_message = f"Reprocessing failed: {exc}"
        doc.save(update_fields=["status", "error_message"])
        record = None

    if record:
        messages.success(request, "Document re-processed successfully.")
    else:
        messages.error(request, "Re-processing failed - check the log.")
    return redirect("document_detail", pk=doc.pk)


# ---------------------------------------------------------------------------
# Verification workflow (human-in-the-loop)
# ---------------------------------------------------------------------------
@role_required("ADMIN", "VERIFIER", "OPERATOR")
def verify_queue(request):
    qs = (LandRecord.objects.filter(status="PENDING")
          .select_related("document")
          .annotate(review_fields=Count("fields", filter=Q(fields__needs_review=True)),
                    open_errors=Count("issues", filter=Q(issues__severity="error",
                                                         issues__resolved=False)))
          .order_by("overall_confidence"))
    page = Paginator(qs, 12).get_page(request.GET.get("page"))
    return render(request, "records/verify_queue.html", {
        "page": page, "role": get_role(request.user),
        "threshold_pct": settings.OCR_CONFIDENCE_THRESHOLD * 100,
    })


@role_required("ADMIN", "VERIFIER")
def verify_record(request, pk):
    record = get_object_or_404(LandRecord.objects.select_related("document"),
                               pk=pk)
    if request.method == "POST":
        action = request.POST.get("action", "verify")
        note = request.POST.get("note", "").strip()
        if action == "reject":
            service.reject_record(record, request.user, note, request=request)
            messages.warning(request, f"Record #{record.pk} rejected.")
            return redirect("verify_queue")
        cleaned = {}
        for key, _label, _t, _req, _w in FIELD_DEFS:
            cleaned[key] = request.POST.get(key, "")
        service.apply_verification(record, cleaned, request.user, note,
                                   request=request)
        if record.status == "VERIFIED":
            messages.success(request,
                             f"Record #{record.pk} verified and locked. "
                             f"Safe OCR repairs were added to the correction memory.")
        else:
            messages.warning(request,
                             f"Record #{record.pk} saved, but validation "
                             f"errors remain - it stays in the queue.")
        return redirect("verify_queue")

    rows = {f.field_name: f for f in record.fields.all()}
    fields = []
    for key, label, wtype, required, _w in FIELD_DEFS:
        row = rows.get(key)
        fields.append({
            "key": key, "label": label, "type": wtype, "required": required,
            "value": getattr(record, key) or "",
            "ocr_value": row.value if row else "",
            "confidence": row.confidence if row else 0.0,
            "source": row.source if row else "ocr",
            "method": row.method if row else "",
            "needs_review": bool(row and row.needs_review),
        })
    
    pages = record.document.pages.all().order_by("page_number")

    return render(request, "records/verify.html", {
        "record": record, "doc": record.document, "fields": fields,
        "pages": pages,
        "issues": record.issues.filter(resolved=False),
        "field_labels": FIELD_LABELS, "role": get_role(request.user),
    })


# ---------------------------------------------------------------------------
# Records browser + export
# ---------------------------------------------------------------------------
def _filtered_records(request):
    qs = LandRecord.objects.select_related("document").all()
    q = request.GET.get("q", "").strip()
    status = request.GET.get("status", "")
    state = request.GET.get("state", "").strip()
    district = request.GET.get("district", "").strip()
    if q:
        qs = qs.filter(Q(owner_name__icontains=q) |
                       Q(survey_number__icontains=q) |
                       Q(khasra_number__icontains=q) |
                       Q(khata_number__icontains=q) |
                       Q(village__icontains=q) | Q(district__icontains=q))
    if status:
        qs = qs.filter(status=status)
    if state:
        qs = qs.filter(state__iexact=state)
    if district:
        qs = qs.filter(district__iexact=district)
    return qs


@login_required
def record_list(request):
    qs = _filtered_records(request)
    page = Paginator(qs, 15).get_page(request.GET.get("page"))
    states = (LandRecord.objects.exclude(state="").values_list("state", flat=True)
              .distinct().order_by("state"))
    districts = (LandRecord.objects.exclude(district="")
                 .values_list("district", flat=True).distinct()
                 .order_by("district"))
    return render(request, "records/record_list.html", {
        "page": page, "states": states, "districts": districts,
        "params": request.GET, "role": get_role(request.user),
    })


@login_required
def records_export_csv(request):
    import csv
    from django.http import HttpResponse

    qs = _filtered_records(request)
    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = 'attachment; filename="land_records.csv"'
    writer = csv.writer(resp)
    header = [label for _k, label, *_ in FIELD_DEFS] + ["Status", "Confidence %"]
    writer.writerow(header)
    for r in qs:
        writer.writerow([getattr(r, k) or "" for k, *_ in FIELD_DEFS]
                        + [r.get_status_display(), r.overall_confidence_pct])
    log_audit(request.user, "EXPORT", None,
              f"CSV export of {qs.count()} record(s)", request=request)
    return resp


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
@role_required("ADMIN", "VERIFIER")
def audit_list(request):
    qs = AuditLog.objects.all()
    action = request.GET.get("action", "")
    if action:
        qs = qs.filter(action=action)
    actions = (AuditLog.objects.values_list("action", flat=True)
               .distinct().order_by("action"))
    page = Paginator(qs, 40).get_page(request.GET.get("page"))
    return render(request, "records/audit.html", {
        "page": page, "actions": actions, "action": action,
        "role": get_role(request.user),
    })