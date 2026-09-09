"""Database models for the land-record digitization platform."""
import uuid

from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone


# ---------------------------------------------------------------------------
# Users & roles
# ---------------------------------------------------------------------------
class UserProfile(models.Model):
    """Role-based access control on top of Django auth."""

    ROLES = [
        ("ADMIN", "Administrator"),
        ("OPERATOR", "Digitization Operator"),
        ("VERIFIER", "Verification Officer"),
        ("VIEWER", "Read-only Viewer"),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE,
                                related_name="profile")
    role = models.CharField(max_length=16, choices=ROLES, default="VIEWER")
    organization = models.CharField(max_length=120, blank=True)
    phone = models.CharField(max_length=20, blank=True)

    def __str__(self):
        return f"{self.user.username} ({self.get_role_display()})"


@receiver(post_save, sender=User)
def _create_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)
    elif hasattr(instance, "profile"):
        instance.profile.save()


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
def document_upload_path(instance, filename):
    return f"documents/{timezone.now():%Y/%m}/{uuid.uuid4().hex[:12]}_{filename}"


class Document(models.Model):
    """An uploaded land-record artifact (scan / photo / legacy PDF)."""

    STATUS_CHOICES = [
        ("UPLOADED", "Uploaded"),
        ("PROCESSING", "Processing"),
        ("PROCESSED", "Processed"),
        ("FAILED", "Failed"),
    ]
    LANGUAGE_CHOICES = [
        ("auto", "Auto-detect"),
        ("eng", "English"),
        ("hin", "Hindi"),
        ("ben", "Bengali"),
        ("tam", "Tamil"),
        ("tel", "Telugu"),
        ("kan", "Kannada"),
        ("guj", "Gujarati"),
        ("mar", "Marathi"),
        ("pan", "Punjabi"),
    ]

    file = models.FileField(upload_to=document_upload_path)
    processed_image = models.ImageField(upload_to="processed/%Y/%m/",
                                        null=True, blank=True)
    original_name = models.CharField(max_length=255, blank=True)
    file_size = models.BigIntegerField(default=0)
    # True page count of the source file (uncapped -- e.g. a 12-page PDF
    # reports 12 here even though only OCR_MAX_PAGES of them are enhanced).
    page_count = models.PositiveIntegerField(default=1)
    # How many of those pages were actually enhanced + OCR'd + fed into
    # field extraction (<= page_count, capped by settings.OCR_MAX_PAGES).
    processed_page_count = models.PositiveIntegerField(default=1)

    language = models.CharField(max_length=8, choices=LANGUAGE_CHOICES,
                                default="auto")
    detected_language = models.CharField(max_length=8, blank=True)

    # Optional cataloguing metadata (drives state/district progress charts).
    state_name = models.CharField(max_length=120, blank=True, db_index=True)
    district_name = models.CharField(max_length=120, blank=True, db_index=True)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES,
                              default="UPLOADED", db_index=True)
    ocr_engine = models.CharField(max_length=80, blank=True)
    ocr_language = models.CharField(max_length=40, blank=True)
    ocr_confidence = models.FloatField(default=0.0)
    ocr_text = models.TextField(blank=True)
    preprocess_info = models.JSONField(default=dict, blank=True)
    processing_log = models.JSONField(default=list, blank=True)
    error_message = models.TextField(blank=True)

    uploaded_by = models.ForeignKey(User, null=True, blank=True,
                                    on_delete=models.SET_NULL,
                                    related_name="documents")
    uploaded_at = models.DateTimeField(auto_now_add=True, db_index=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    # -- helpers ---------------------------------------------------------
    @property
    def filename(self):
        return self.original_name or (
            self.file.name.rsplit("/", 1)[-1] if self.file else "-")

    @property
    def is_pdf(self):
        return self.filename.lower().endswith(".pdf")

    @property
    def is_image(self):
        return not self.is_pdf

    @property
    def ocr_confidence_pct(self):
        return round(self.ocr_confidence * 100, 1)

    def __str__(self):
        return f"Document #{self.pk} - {self.filename}"


class DocumentPage(models.Model):
    """One page's enhanced preview + OCR stats. Every page of a multi-page
    PDF is enhanced and OCR'd by the pipeline (records/pipeline/service.py
    already loops over and merges all of them into field extraction) --
    this model exists purely so the UI can *show* that, instead of only
    ever displaying page 1's preview and leaving the rest invisible."""
    document = models.ForeignKey(Document, on_delete=models.CASCADE,
                                 related_name="pages")
    page_number = models.PositiveIntegerField()
    processed_image = models.ImageField(upload_to="processed/%Y/%m/pages/")
    ocr_confidence = models.FloatField(default=0.0)
    word_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["page_number"]
        unique_together = [("document", "page_number")]

    @property
    def ocr_confidence_pct(self):
        return round(self.ocr_confidence * 100, 1)


# ---------------------------------------------------------------------------
# Structured land records
# ---------------------------------------------------------------------------
class LandRecord(models.Model):
    """Structured information extracted (and verified) from a Document."""

    STATUS_CHOICES = [
        ("PENDING", "Pending verification"),
        ("VERIFIED", "Verified"),
        ("REJECTED", "Rejected"),
    ]

    document = models.OneToOneField(Document, on_delete=models.CASCADE,
                                    related_name="record")

    owner_name = models.CharField(max_length=200, blank=True)
    father_name = models.CharField(max_length=200, blank=True)
    survey_number = models.CharField(max_length=120, blank=True, db_index=True)
    subdivision_number = models.CharField(max_length=60, blank=True)
    khasra_number = models.CharField(max_length=120, blank=True)
    khata_number = models.CharField(max_length=120, blank=True)
    plot_area = models.FloatField(null=True, blank=True)
    area_unit = models.CharField(max_length=32, blank=True)
    village = models.CharField(max_length=120, blank=True, db_index=True)
    tehsil = models.CharField(max_length=120, blank=True)
    district = models.CharField(max_length=120, blank=True, db_index=True)
    state = models.CharField(max_length=120, blank=True, db_index=True)
    pincode = models.CharField(max_length=10, blank=True)
    address = models.CharField(max_length=300, blank=True)
    land_classification = models.CharField(max_length=160, blank=True)
    ownership_type = models.CharField(max_length=80, blank=True)
    mutation_number = models.CharField(max_length=60, blank=True)
    mutation_date = models.CharField(max_length=20, blank=True)
    registration_number = models.CharField(max_length=60, blank=True)
    registration_date = models.CharField(max_length=20, blank=True)

    overall_confidence = models.FloatField(default=0.0)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES,
                              default="PENDING", db_index=True)
    is_duplicate = models.BooleanField(default=False)

    review_note = models.TextField(blank=True)
    verified_by = models.ForeignKey(User, null=True, blank=True,
                                    on_delete=models.SET_NULL,
                                    related_name="verified_records")
    verified_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def overall_confidence_pct(self):
        return round(self.overall_confidence * 100, 1)

    @property
    def primary_plot_id(self):
        return (self.survey_number or self.khasra_number or "-")

    def __str__(self):
        return (f"Record #{self.pk} - {self.owner_name or '?'} "
                f"({self.primary_plot_id})")


class FieldExtraction(models.Model):
    """Per-field extraction result with confidence and provenance."""

    SOURCE_CHOICES = [
        ("ocr", "OCR + rules"),
        ("learned", "Verified-correction memory"),
        ("manual", "Manual verification"),
    ]

    record = models.ForeignKey(LandRecord, on_delete=models.CASCADE,
                               related_name="fields")
    field_name = models.CharField(max_length=40)
    value = models.TextField(blank=True)
    confidence = models.FloatField(default=0.0)
    source = models.CharField(max_length=12, choices=SOURCE_CHOICES,
                              default="ocr")
    needs_review = models.BooleanField(default=False, db_index=True)
    method = models.CharField(max_length=90, blank=True)
    corrected_value = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("record", "field_name")

    @property
    def confidence_pct(self):
        return round(self.confidence * 100, 1)

    def __str__(self):
        return f"{self.record_id}:{self.field_name}={self.value!r}"


class ValidationIssue(models.Model):
    """Output of the business-rule / cross-validation engine."""

    SEVERITIES = [("error", "Error"), ("warning", "Warning"), ("info", "Info")]

    record = models.ForeignKey(LandRecord, on_delete=models.CASCADE,
                               related_name="issues")
    rule_code = models.CharField(max_length=40, db_index=True)
    field_name = models.CharField(max_length=40, blank=True)
    severity = models.CharField(max_length=10, choices=SEVERITIES,
                                default="warning")
    message = models.CharField(max_length=400)
    resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["severity", "rule_code"]

    def __str__(self):
        return f"[{self.rule_code}] {self.message}"


# ---------------------------------------------------------------------------
# Learning memory
# ---------------------------------------------------------------------------
class LearnedCorrection(models.Model):
    """A fuzzy-match memory created when an officer corrects an OCR value.

    Future extractions auto-apply these corrections (AI-driven learning).
    """

    field_name = models.CharField(max_length=40)
    raw_value = models.TextField()
    corrected_value = models.TextField()
    times_applied = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ("field_name", "raw_value", "corrected_value")

    def __str__(self):
        return f"{self.field_name}: {self.raw_value!r} -> {self.corrected_value!r}"


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
class AuditLog(models.Model):
    """Immutable audit trail for every sensitive action."""

    user = models.ForeignKey(User, null=True, blank=True,
                             on_delete=models.SET_NULL)
    user_label = models.CharField(max_length=120, blank=True)
    action = models.CharField(max_length=60, db_index=True)
    object_type = models.CharField(max_length=30, blank=True)
    object_id = models.CharField(max_length=40, blank=True)
    detail = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.created_at:%Y-%m-%d %H:%M} {self.action} {self.detail[:40]}"
