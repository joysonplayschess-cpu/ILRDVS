# ILRDVS — Intelligent Land Record Digitization & Validation System

An AI-powered Django platform that converts scanned land registers, Khatauni/RoR forms, cadastral printouts and legacy PDFs into **structured, validated, audit-trailed digital records**, aligned with the objectives of the **Digital India Land Records Modernization Programme (DILRMP)**.

Primary languages: **English, Hindi, Tamil, Telugu, Kannada, Malayalam** (plus Bengali, Gujarati, Marathi, Punjabi where packs/API support allow).

```text
scan/PDF
    │
    ▼
per-page quality assessment (cheap)
    │
    ▼
stamp/seal detection (always; conservative)
    │
    ▼
adaptive preprocessing
    • minimal  — clean / born-digital pages
    • moderate — mild fade / noise / skew
    • heavy    — dark, noisy, ruled, difficult scans
    │
    ▼
OCR: Bhashini ULCA (primary) → Tesseract → PaddleOCR (fallback)
    │  + OCR rescue if enhance yields near-empty text
    ▼
NLP field extraction (labels, layout, table cells, learning memory)
    │
    ▼
validation (business rules, stamp issues, duplicates, state↔district)
    │
    ▼
HITL verification console → audited digital register
```

## 1. Quick start

### Bhashini setup (primary OCR engine)

Engine order (`settings.OCR_ENGINE_PRIORITY`): **Bhashini → Tesseract → PaddleOCR**

If Bhashini keys are missing or a call fails, the pipeline falls through automatically. Tesseract needs no key.

```bash
cp .env.example .env
# set:
#   BHASHINI_USER_ID=...
#   BHASHINI_API_KEY=...
#   BHASHINI_PIPELINE_ID=...   # if required by your ULCA app
```

Or export env vars / edit `config/settings.py`. Blank keys = skip Bhashini.

> **Tip:** On upload, pin language (`eng`, `hin`, `tam`, `tel`, `kan`, `mal`) instead of `auto` for faster, more stable runs.

### Adaptive preprocessing

Each page is assessed independently (`records/pipeline/quality.py`):

| Level | When | What runs |
|---|---|---|
| minimal | Clean / digital e-Services style pages | Resize, grayscale, optional stamp inpaint, light deskew |
| moderate | Mild quality issues | + gamma, light denoise, CLAHE, deskew (no hard binarize) |
| heavy | Dark, noisy, faded, difficult scans | Full existing pipeline: gamma, NL-Means, CLAHE, binarize, despeckle, de-rule, deskew |

Stamp detection always runs. A real seal can upgrade minimal → moderate so ink can be inpainted. False stamp storms are capped (see Stamp section).

If OCR after enhance returns almost no words, OCR rescue retries minimal/original grayscale so a bad enhance cannot leave a blank record.

### Stamp / seal handling

- Conservative detector: prefers coloured office ink; ignores table rules and body text.
- Max plausible stamps guard (drops absurd counts like "112 stamps" on clean forms).
- Issues: `STAMP_OVERLAP`, `STAMP_MULTIPLE`, `STAMP_MISSING` (softened for digital e-signed cues).
- Console log: `Stamp=YES` / `Stamp=NO` means detector ran; `NO` = zero seals found, not "feature off".

### Multi-page PDFs

`OCR_MAX_PAGES` (default 20, env-overridable). Each page is assessed and OCR'd separately; fields merge into one `LandRecord`. `Document.page_count` is the true source count; `processed_page_count` is how many pages were processed.

### System dependencies

```bash
sudo apt-get install -y tesseract-ocr \
  tesseract-ocr-hin tesseract-ocr-ben tesseract-ocr-tam tesseract-ocr-tel \
  tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-guj tesseract-ocr-mar \
  tesseract-ocr-pan

# optional better LSTM models:
# ./scripts/install_tessdata_best.sh eng hin tam tel kan mal ...

# optional third OCR engine:
pip install paddleocr paddlepaddle
```

### Python setup

```bash
cd landrecords
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo
python manage.py runserver 0.0.0.0:8000
```

| Account | Password | Role |
|---|---|---|
| admin | admin123 | Administrator |
| operator1 | operate123 | Digitization Operator |
| verifier1 | verify123 | Verification Officer |
| viewer1 | view123 | Viewer |

```bash
python manage.py test

# optional ground-truth eval if fixtures present:
# python manage.py eval_khatauni_gt
```

Pre-seed correction memory (deterministic cache, not model training):

```bash
python manage.py seed_demo --reset --train-passes 10
```

## 2. What to try

- **Dashboard** — KPIs, confidence, issues, learning log.
- **Clean digital Patta/RoR PDF** — should often use minimal/moderate, `Stamp=NO`, non-zero word counts; Enhanced should still look readable.
- **Dark/faded scan** — should escalate to heavy.
- **Stamp on text** — real coloured seals may show `Stamp=YES` and overlap rules.
- **Verification queue** — correct fields; corrections enter learning memory.
- **Duplicate plot** — same survey/khasra + village + district flags duplicate.
- **Upload** with pinned language and gamma auto/manual/off.
- **API** under `/api/v1/` with session or `X-API-Key: lr-demo-key-2026`.

### Pipeline timing log (server console)

```text
[ILRDVS] p1 Mode=moderate Stamp=NO Pre=2248ms OCR=33660ms Words=153 Conf=67.5%
```

- **Mode** — preprocess level used (or `… (rescue)` if OCR rescue won).
- **Stamp=YES/NO** — seals found on that page (feature is always on).
- **Pre / OCR** — local CV vs OCR engine time (Bhashini is often the slow part).
- **Words / Conf** — OCR yield for that page.

## 3. Integration API

Base: `/api/v1/` — browser session or header `X-API-Key: lr-demo-key-2026` (`settings.API_KEYS`).

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/stats/` | KPIs |
| GET/POST | `/api/v1/documents/` | list / upload + process |
| GET | `/api/v1/documents/<id>/` | status, OCR, record |
| POST | `/api/v1/documents/<id>/reprocess/` | re-run pipeline |
| GET | `/api/v1/records/` | search |
| GET | `/api/v1/records/<id>/` | record + confidences + issues |
| POST | `/api/v1/records/<id>/verify/` | verify/edit fields |
| GET | `/api/v1/schema/fields/` | canonical field schema |

```bash
curl -H "X-API-Key: lr-demo-key-2026" \
  -F file=@khatauni_scan.pdf -F language=eng -F district=Chennai -F gamma=auto \
  http://localhost:8000/api/v1/documents/
```

## 4. Project layout

```text
landrecords/
├── config/                 # settings, urls, wsgi
├── records/
│   ├── models.py
│   ├── constants.py
│   ├── views.py / api.py
│   ├── pipeline/
│   │   ├── quality.py      # cheap per-page quality → minimal|moderate|heavy
│   │   ├── stamps.py       # seal detection + validation issues
│   │   ├── preprocess.py   # minimal / moderate / heavy enhance
│   │   ├── bhashini_ocr.py # ULCA Bhashini client
│   │   ├── ocr.py          # engine chain + language handling
│   │   ├── extract.py      # NLP / layout field extraction
│   │   ├── table_extract.py
│   │   ├── indic_labels.py
│   │   ├── validate.py
│   │   ├── learn.py
│   │   └── service.py      # orchestration + OCR rescue + timing
│   └── management/commands/
├── tools/                  # nlp_bench, e2e_bench (if present)
├── media/
├── static/
├── templates/
├── PROJECT_REPORT.md
└── requirements.txt
```

## 5. Important settings (`config/settings.py`)

| Setting | Role |
|---|---|
| `OCR_ENGINE_PRIORITY` | `["bhashini", "tesseract", "paddle"]` |
| `BHASHINI_*` | ULCA credentials and timeouts |
| `OCR_CONFIDENCE_THRESHOLD` | Default 0.75 → HITL review |
| `OCR_MAX_PAGES` | Cap pages processed per document |
| `OCR_LANGUAGES` | Includes `mal` (Malayalam) and other Indic packs |
| `ADAPTIVE_PREPROCESS_ENABLED` | Per-page minimal/moderate/heavy routing |
| `PREPROCESSING_CONFIG` | Tunable quality thresholds |
| `STAMP_*` | Stamp area ratios, overlap IoU, cert-block policy |
| `PIPELINE_DEBUG` | Optional per-stage debug images under `media/debug/` |
| `API_KEYS` | Machine clients |

**Speed while testing:** pin language; lower `OCR_MAX_PAGES`; or temporarily set `OCR_ENGINE_PRIORITY = ["tesseract"]` to skip Bhashini latency.

**Accuracy mode:** Bhashini on + pinned language.

## 6. Validation rules (selected)

| Code | Meaning |
|---|---|
| `REQUIRED_FIELD` | Mandatory land fields empty |
| `ID_FORMAT` / `AREA_*` / `PINCODE_*` / `DATE_*` | Format and range checks |
| `STATE_DISTRICT` | District vs state master list |
| `DUPLICATE_PLOT` | Same plot id + village + district |
| `STAMP_OVERLAP` | Seal overlaps OCR text |
| `STAMP_MULTIPLE` | Several seals (warning) |
| `STAMP_MISSING` | No seal and no cert cues (skipped for many digital e-sign patterns) |

## 7. Notes

- Demo corpus in `media/seed/` is synthetic; real pipeline results are not hard-coded.
- SQLite for demo; use PostgreSQL/PostGIS in production.
- Pipeline is synchronous per upload (multi-page = sum of page times).
- For bulk volume, wrap `service.process_document` with Celery/RQ.
- Upload UI dedupes concurrent "already PROCESSING" same file for the same user (avoids duplicate jobs from double-click / retries).
- Sensitive test files: keep under a gitignored path (e.g. `tests/fixtures/sensitive/`).
- Field accuracy depends on document quality, language pin, and OCR engine availability. Measure on your own ground-truth set rather than assuming a fixed percentage.

## 8. Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| Enhanced white / 0 words | Heavy CV on clean PDF (old path) | `quality.py`, `preprocess_minimal`, service OCR rescue |
| "100+ stamps" | Old dark-ink stamp OR | Updated `stamps.py` (colour-first + max-stamp guard) |
| Very slow multi-page | Bhashini per page | Console `OCR=…ms`; pin lang; raise patience or use Tess for smoke tests |
| Many PROCESSING rows | Double upload | Delete stuck rows; use single upload; dedupe in views |
| Empty district/village | No OCR text yet | Fix OCR first; labels already include Taluk/Patta/District |

Clear stuck jobs (example):

```bash
python manage.py shell -c "from records.models import Document; Document.objects.filter(status='PROCESSING').delete()"
```

---

### Files checklist (for your own audit)

| File | In README? | Purpose |
|------|------------|---------|
| `stamps.py` | Yes | Stamp=NO/YES behaviour |
| `quality.py` | Yes | Adaptive levels |
| `preprocess.py` | Yes | minimal / moderate / heavy |
| `service.py` | Yes | Orchestration + rescue + log line |
| `ocr.py` / `bhashini_ocr.py` | Yes | Engine chain |
| `extract.py` / `validate.py` / `views.py` | Yes | Downstream |
| `settings.py` | Yes | Knobs |
| `PROJECT_REPORT.md` | Mention only | Optional separate doc |
