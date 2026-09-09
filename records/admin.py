from django.contrib import admin

from .models import (AuditLog, Document, FieldExtraction, LandRecord,
                     LearnedCorrection, UserProfile, ValidationIssue)


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    extra = 0


class FieldExtractionInline(admin.TabularInline):
    model = FieldExtraction
    extra = 0
    readonly_fields = ("confidence", "source")


class ValidationIssueInline(admin.TabularInline):
    model = ValidationIssue
    extra = 0


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "filename", "language", "status",
                    "ocr_confidence_pct", "state_name", "district_name",
                    "uploaded_by", "uploaded_at")
    list_filter = ("status", "language", "state_name")
    search_fields = ("original_name", "ocr_text")
    readonly_fields = ("processing_log", "preprocess_info")


@admin.register(LandRecord)
class LandRecordAdmin(admin.ModelAdmin):
    list_display = ("id", "owner_name", "primary_plot_id", "village",
                    "district", "state", "overall_confidence_pct", "status",
                    "is_duplicate")
    list_filter = ("status", "is_duplicate", "state", "district")
    search_fields = ("owner_name", "survey_number", "khasra_number",
                     "khata_number", "village", "district")
    inlines = [FieldExtractionInline, ValidationIssueInline]


@admin.register(ValidationIssue)
class ValidationIssueAdmin(admin.ModelAdmin):
    list_display = ("rule_code", "severity", "field_name", "record",
                    "resolved", "created_at")
    list_filter = ("rule_code", "severity", "resolved")


@admin.register(LearnedCorrection)
class LearnedCorrectionAdmin(admin.ModelAdmin):
    list_display = ("field_name", "raw_value", "corrected_value",
                    "times_applied", "last_used")
    list_filter = ("field_name",)


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user_label", "action", "object_type",
                    "object_id", "ip_address")
    list_filter = ("action", "object_type")
    search_fields = ("detail", "user_label")
    readonly_fields = ("created_at",)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "organization", "phone")
    list_filter = ("role",)
