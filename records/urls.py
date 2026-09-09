from django.urls import path

from . import api, views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload, name="upload"),
    path("documents/", views.document_list, name="document_list"),
    path("documents/<int:pk>/", views.document_detail, name="document_detail"),
    path("documents/<int:pk>/reprocess/", views.document_reprocess,
         name="document_reprocess"),
    path("verify/", views.verify_queue, name="verify_queue"),
    path("records/<int:pk>/verify/", views.verify_record, name="verify_record"),
    path("records/", views.record_list, name="record_list"),
    path("records/export.csv", views.records_export_csv, name="records_export"),
    path("audit/", views.audit_list, name="audit_list"),

    # ---- integration API ------------------------------------------------------------
    path("api/v1/stats/", api.api_stats, name="api_stats"),
    path("api/v1/documents/", api.api_documents, name="api_documents"),
    path("api/v1/documents/<int:pk>/", api.api_document_detail,
         name="api_document_detail"),
    path("api/v1/documents/<int:pk>/reprocess/", api.api_document_reprocess,
         name="api_document_reprocess"),
    path("api/v1/records/", api.api_records, name="api_records"),
    path("api/v1/records/<int:pk>/", api.api_record_detail,
         name="api_record_detail"),
    path("api/v1/records/<int:pk>/verify/", api.api_record_verify,
         name="api_record_verify"),
    path("api/v1/schema/fields/", api.api_field_schema,
         name="api_field_schema"),
]
