"""
Integration API (JSON) for LRMS / DILRMP / GIS style consumers.

Authentication: an active web session, or the ``X-API-Key`` HTTP header
(keys configured in ``settings.API_KEYS``).  All mutating calls are audited.
"""
import json

from django.conf import settings
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .constants import FIELD_DEFS, FIELD_LABELS, FIELD_KEYS
from .models import Document, LandRecord, ValidationIssue
from .pipeline import service
from .utils import log_audit


# ---------------------------------------------------------------------------
# Auth plumbing
# ---------------------------------------------------------------------------
def _client_label(request):
    key = request.headers.get("X-API-Key", "")
    return settings.API_KEYS.get(key) if key else None


def api_auth(view):
    """Session-authenticated user OR valid X-API-Key header.

    API consumers authenticate with a shared key rather than session cookies,
    so every API view is CSRF-exempt (``wrapper.csrf_exempt = True``).
    """
    def wrapper(request, *args, **kwargs):
        if request.user.is_authenticated:
            return view(request, *args, **kwargs)
        label = _client_label(request)
        if label:
            request._api_client = label
            return view(request, *args, **kwargs)
        return JsonResponse(
            {"error": "Unauthorized. Log in or pass a valid X-API-Key header."},
            status=401)
    wrapper.__name__ = view.__name__
    wrapper.csrf_exempt = True
    return wrapper


def _paginate(request, qs, serialize, per_page=20):
    page = Paginator(qs, per_page).get_page(request.GET.get("page", 1))
    return JsonResponse({
        "count": page.paginator.count,
        "page": page.number,
        "pages": page.paginator.num_pages,
        "results": [serialize(o) for o in page],
    })


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------
def doc_json(doc: Document) -> dict:
    record = getattr(doc, "record", None)
    return {
        "id": doc.pk, "filename": doc.filename, "status": doc.status,
        "language": doc.language,
        "detected_language": doc.detected_language or None,
        "pages": doc.page_count, "file_size_bytes": doc.file_size,
        "ocr_engine": doc.ocr_engine or None,
        "ocr_confidence": doc.ocr_confidence,
        "state": doc.state_name or None, "district": doc.district_name or None,
        "uploaded_at": doc.uploaded_at.isoformat(),
        "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
        "record_id": record.pk if record else None,
        "record_status": record.status if record else None,
        "preprocessing": doc.preprocess_info or None,
    }


def record_json(rec: LandRecord, include_fields=True) -> dict:
    data = {key: getattr(rec, key) for key in FIELD_KEYS}
    data.update({
        "id": rec.pk, "status": rec.status,
        "is_duplicate": rec.is_duplicate,
        "overall_confidence": rec.overall_confidence,
        "document_id": rec.document_id,
        "created_at": rec.created_at.isoformat(),
        "verified_at": rec.verified_at.isoformat() if rec.verified_at else None,
        "verified_by": rec.verified_by.username if rec.verified_by else None,
    })
    if include_fields:
        data["fields"] = [
            {"field": f.field_name, "label": FIELD_LABELS.get(f.field_name),
             "value": f.value, "corrected_value": f.corrected_value or None,
             "confidence": round(f.confidence, 3), "source": f.source,
             "method": f.method or None,
             "needs_review": f.needs_review}
            for f in rec.fields.all().order_by("id")
        ]
        data["validation_issues"] = [
            {"rule": i.rule_code, "field": i.field_name or None,
             "severity": i.severity, "message": i.message,
             "resolved": i.resolved}
            for i in rec.issues.all().order_by("severity")
        ]
    return data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@api_auth
def api_stats(request):
    from django.db.models import Avg, Count
    docs = Document.objects.all()
    recs = LandRecord.objects.all()
    return JsonResponse({
        "documents": {
            "total": docs.count(),
            "by_status": {r["status"]: r["n"] for r in
                          docs.values("status").annotate(n=Count("id"))},
        },
        "records": {
            "total": recs.count(),
            "by_status": {r["status"]: r["n"] for r in
                          recs.values("status").annotate(n=Count("id"))},
            "avg_confidence": round(
                recs.aggregate(a=Avg("overall_confidence"))["a"] or 0, 4),
            "duplicates": recs.filter(is_duplicate=True).count(),
        },
    })


@api_auth
@csrf_exempt
@require_http_methods(["GET", "POST"])
def api_documents(request):
    if request.method == "POST":
        fh = request.FILES.get("file")
        if not fh:
            return JsonResponse({"error": "multipart field 'file' is required."},
                                status=400)
        doc = Document.objects.create(
            file=fh, original_name=fh.name, file_size=fh.size,
            language=request.POST.get("language", "auto"),
            state_name=request.POST.get("state", ""),
            district_name=request.POST.get("district", ""),
            uploaded_by=request.user if request.user.is_authenticated else None)
        gamma_mode = request.POST.get("gamma", "auto")
        gamma_value = request.POST.get("gamma_value")
        try:
            gamma_value = float(gamma_value) if gamma_value else None
        except ValueError:
            gamma_value = None
        log_audit(request.user, "API_UPLOAD", doc,
                  f"Upload via API client "
                  f"'{getattr(request, '_api_client', request.user)}'",
                  request=request)
        service.process_document(doc, gamma_mode=gamma_mode,
                                 gamma_value=gamma_value, lang=doc.language)
        return JsonResponse(doc_json(doc), status=201)
    qs = Document.objects.all()
    status = request.GET.get("status")
    if status:
        qs = qs.filter(status=status)
    return _paginate(request, qs, doc_json)


@api_auth
@require_http_methods(["GET"])
def api_document_detail(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    data = doc_json(doc)
    data["ocr_text"] = doc.ocr_text
    data["processing_log"] = doc.processing_log
    if getattr(doc, "record", None):
        data["record"] = record_json(doc.record)
    return JsonResponse(data)


@api_auth
@csrf_exempt
@require_http_methods(["POST"])
def api_document_reprocess(request, pk):
    doc = get_object_or_404(Document, pk=pk)
    service.process_document(doc, lang=doc.language)
    log_audit(request.user, "API_REPROCESS", doc, "Reprocess via API",
              request=request)
    return JsonResponse(doc_json(doc))


@api_auth
@require_http_methods(["GET"])
def api_records(request):
    qs = LandRecord.objects.all().order_by("-created_at")
    for param, lookup in (("status", "status"), ("state", "state__iexact"),
                          ("district", "district__iexact"),
                          ("village", "village__icontains"),
                          ("survey_number", "survey_number__iexact"),
                          ("owner", "owner_name__icontains")):
        v = request.GET.get(param)
        if v:
            qs = qs.filter(**{lookup: v})
    if request.GET.get("duplicates") == "1":
        qs = qs.filter(is_duplicate=True)
    return _paginate(request, qs, lambda r: record_json(r, include_fields=False))


@api_auth
@require_http_methods(["GET"])
def api_record_detail(request, pk):
    return JsonResponse(record_json(get_object_or_404(LandRecord, pk=pk)))


@api_auth
@csrf_exempt
@require_http_methods(["POST"])
def api_record_verify(request, pk):
    """Body (JSON): {"fields": {field: value, ...}, "note": "..."}"""
    rec = get_object_or_404(LandRecord, pk=pk)
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)
    fields = payload.get("fields") or {}
    if not isinstance(fields, dict):
        return JsonResponse({"error": "'fields' must be an object."}, status=400)
    cleaned = {str(k): str(v) for k, v in fields.items() if k in FIELD_KEYS}
    service.apply_verification(rec, cleaned, request.user,
                               note=payload.get("note", ""), request=request)
    return JsonResponse(record_json(rec))


@api_auth
@require_http_methods(["GET"])
def api_field_schema(request):
    return JsonResponse({
        "fields": [{"name": k, "label": l, "type": t, "required": req,
                    "weight": w} for k, l, t, req, w in FIELD_DEFS],
    })
