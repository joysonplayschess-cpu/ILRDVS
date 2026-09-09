"""
Django settings for the Intelligent Land Record Digitization and
Validation System (ILRDVS).

Developed as a reference implementation aligned with the objectives of the
Digital India Land Records Modernization Programme (DILRMP).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Minimal .env loader (no extra dependency): if a .env file sits next to
# manage.py, load KEY=VALUE lines into os.environ (without overriding
# anything already exported in the real shell environment). See
# .env.example for the Bhashini keys this is meant for.
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip())

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
SECRET_KEY = "django-insecure-ilrdvs-dev-key-change-me-in-production"
DEBUG = True
ALLOWED_HOSTS = ["*"]

# The sandbox live-preview is served from *.e2b.app - trust it for POSTs.
CSRF_TRUSTED_ORIGINS = [
    "https://*.e2b.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://0.0.0.0:8000",
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "records",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ---------------------------------------------------------------------------
# Database - SQLite for the demo; switch to PostgreSQL/PostGIS in production.
# ---------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ---------------------------------------------------------------------------
# Internationalisation
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static / media
# ---------------------------------------------------------------------------
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "login"

# ---------------------------------------------------------------------------
# ILRDVS domain configuration
# ---------------------------------------------------------------------------
# Minimum per-field confidence before the field is flagged for manual review.
OCR_CONFIDENCE_THRESHOLD = 0.75

# Language packs available for OCR processing
OCR_LANGUAGES = {
    "auto": "Auto-detect (eng+hin)",
    "eng": "English",
    "hin": "Hindi / हिन्दी",
    "tam": "Tamil / தமிழ்",
    "tel": "Telugu / తెలుగు",
    "kan": "Kannada / ಕನ್ನಡ",
    "mal": "Malayalam / മലയാളം",
    "ben": "Bengali / বাংলা",
    "guj": "Gujarati / ગુજરાતી",
    "mar": "Marathi / मराठी",
    "pan": "Punjabi / ਪੰਜਾਬੀ",
}
# Tesseract codes used for the auto-detect pass.
OCR_AUTO_LANGS = "eng+hin"

# Maximum PDF pages processed per document.
OCR_MAX_PAGES = int(os.environ.get("OCR_MAX_PAGES", "20"))

# ---------------------------------------------------------------------------
# OCR engine priority + Bhashini (Digital India / MeitY ULCA) credentials
# ---------------------------------------------------------------------------
def _env(key, default=""):
    """Read an env var, treating missing or blank values as `default`."""
    value = os.environ.get(key)
    if value is None or str(value).strip() == "":
        return default
    return value.strip()

BHASHINI_USER_ID = _env("BHASHINI_USER_ID", "")
BHASHINI_API_KEY = _env("BHASHINI_API_KEY", "")
BHASHINI_PIPELINE_ID = _env("BHASHINI_PIPELINE_ID", "")
BHASHINI_API_URL = _env(
    "BHASHINI_API_URL",
    "https://dhruva-api.bhashini.gov.in/services/inference",
)
BHASHINI_PIPELINE_URL = _env(
    "BHASHINI_PIPELINE_URL",
    "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline",
)
BHASHINI_OCR_TIMEOUT = int(_env("BHASHINI_OCR_TIMEOUT", "120"))
BHASHINI_MAX_IMAGE_SIDE = int(_env("BHASHINI_MAX_IMAGE_SIDE", "2000"))

# OCR engine execution priority order
OCR_ENGINE_PRIORITY = ["bhashini", "tesseract", "paddle"]

# ---------------------------------------------------------------------------
# Stamp detection (revenue/office stamps on land-record scans)
# ---------------------------------------------------------------------------
STAMP_DETECTION_ENABLED = True
STAMP_MIN_AREA_RATIO = 0.002
STAMP_MAX_AREA_RATIO = 0.12
STAMP_REQUIRE_CERT_BLOCK = True
STAMP_OVERLAP_TEXT_IOU = 0.15

# Simple shared API keys for machine-to-machine integration
API_KEYS = {
    "lr-demo-key-2026": "Demo LRMS integration client",
}