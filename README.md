# ILRDVS — Intelligent Land Record Digitization & Validation System

An AI-powered Django platform that converts scanned land registers, handwritten-era
Khatauni/RoR forms, cadastral printouts and legacy PDFs into **structured, validated,
audit-trailed digital records** — aligned with the objectives of the **Digital India
Land Records Modernization Programme (DILRMP)**.

```
  scan/PDF  ──▶  enhancement (auto γ-correction · denoise · CLAHE ·
 | image          de-rule · despeckle · deskew)
              ──▶  OCR (Bhashini ▸ Tesseract ▸ PaddleOCR fallback chain,
                        eng + 8 Indic languages, word-level conf)
              ──▶  NLP extraction (multilingual labels, typed parsers,
                                   learned-correction memory)
              ──▶  validation (business rules · state↔district cross-check ·
                               duplicate detection)
              ──▶  HITL verification console  ──▶  audited digital register
```

---

## 1. Quick start

### Bhashini setup (primary OCR engine)
The pipeline tries OCR engines in this order: **Bhashini → Tesseract →
PaddleOCR** (`settings.OCR_ENGINE_PRIORITY`), falling through automatically
whenever one isn't configured/installed or fails on a page. Tesseract needs
no key, so the app runs fine with nothing configured — Bhashini is just
preferred first when available.

To enable Bhashini: get a free key at https://bhashini.gov.in → sign in →
**My Profile** → generate ULCA API key (you'll get a **User ID** and an
**API key**). Then either:

```bash
cp .env.example .env
# edit .env and fill in:
#   BHASHINI_USER_ID=...
#   BHASHINI_API_KEY=...
```
— or set them directly as environment variables before running the server
(`export BHASHINI_USER_ID=... BHASHINI_API_KEY=...`) — or paste them
straight into `config/settings.py` at the `BHASHINI_USER_ID` /
`BHASHINI_API_KEY` lines if you'd rather not use a `.env` file. Leaving
them blank just skips Bhashini with no other changes needed.

### Multi-page PDFs (`OCR_MAX_PAGES`)
Each page of a PDF is separately enhanced, OCR'd and fed into extraction --
that costs one more enhance+OCR pass per page, so `config/settings.py`
caps how many pages of any single document get processed
(`OCR_MAX_PAGES`, default **20**, overridable via the `OCR_MAX_PAGES` env
var). Pages beyond the cap are skipped entirely (not enhanced, not shown,
not searched for fields) -- the document detail / verify pages will show a
warning banner ("N of M pages processed") whenever this happens, and
`Document.page_count` always reports the *true* page count of the source
file even when `Document.processed_page_count` is lower. Raise the cap if
your real scans regularly run longer than 20 pages; each extra page adds
roughly one enhance+OCR pass of processing time.

### Improving Hindi/Devanagari OCR accuracy
The distro `hin.traineddata` that ships with `tesseract-ocr-hin` is usable
but noticeably weaker than the LSTM-trained `tessdata_best` model,
especially on faded/noisy scans. Swap it in with:
```bash
./scripts/install_tessdata_best.sh          # installs eng+hin by default
./scripts/install_tessdata_best.sh hin ben tam tel kan guj mar pan  # or pick languages
```
This overwrites the language files Tesseract already has installed (back
up `/usr/share/tesseract-ocr/*/tessdata/*.traineddata` first if you want
to be able to revert). No code or settings changes are needed afterwards
-- `pipeline/ocr.py` just gets better recognition from the same call.

### System dependencies
```bash
sudo apt-get install -y tesseract-ocr \
  tesseract-ocr-hin tesseract-ocr-ben tesseract-ocr-tam tesseract-ocr-tel \
  tesseract-ocr-kan tesseract-ocr-guj tesseract-ocr-mar tesseract-ocr-pan
# only needed for regenerating demo scans:
sudo apt-get install -y fonts-deva fonts-lohit-deva
# optional third-choice OCR engine (skipped automatically if absent):
pip install paddleocr paddlepaddle
```

### Python setup
```bash
cd landrecords
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo        # demo users + 12 synthetic scans, fully processed
python manage.py runserver 0.0.0.0:8000
```

Open **http://localhost:8000** and sign in:

| Account | Password | Role | Can do |
|---|---|---|---|
| `admin` | `admin123` | Administrator | everything + Django admin + audit |
| `operator1` | `operate123` | Digitization Operator | upload, reprocess, view queue |
| `verifier1` | `verify123` | Verification Officer | review queue, verify/reject records |
| `viewer1` | `view123` | Viewer | read-only dashboards & registers |

Run the test suite: `python manage.py test` (29 tests).

Want the demo corpus's verified-correction memory pre-seeded (so repeated
OCR misreads on names/places auto-correct instead of needing review every
time)? Reset and rebuild with several verify passes:
```bash
python manage.py seed_demo --reset --train-passes 10
```
This reprocesses + re-verifies the demo subset 10 times with the correct
ground-truth values (same mechanism as an officer correcting the same
mistake repeatedly in the UI) — it's a deterministic correction cache, not
model retraining; see `records/pipeline/learn.py`.

---

## 2. What to try

1. **Dashboard** — KPIs, state-wise progress, confidence trend, issue mix, the
   AI-learning log (every verifier correction appears here and is re-applied to
   future uploads).
2. **Documents → #1 (UP_Varanasi_RoR_dark.pdf)** — a very dark scan. Open the
   **Enhanced (γ-corrected)** tab and compare with the original; check the
   *pipeline log* chips (`γ 1.06`, brightness before/after, deskew).
3. **Verification Queue** — the damaged Hindi scan (`RJ_Sanganer_damaged.png`)
   sits at ~2 % confidence with missing fields; correct them, approve, and watch
   the record lock and the corrections enter the learning memory.
4. **Duplicate detection** — document #2 (`UP_Varanasi_rescan_clean.png`) is a
   second scan of the same plot; the system links it to record #1 and flags it.
5. **Upload** your own PDF/JPG/PNG/TIFF and choose *gamma: auto/manual/off*.
6. **API** (below) from curl or any LRMS/GIS client.

---

## 3. Integration API

All JSON endpoints live under `/api/v1/` and authenticate with **either** a
browser session **or** the header `X-API-Key: lr-demo-key-2026`
(configure keys in `settings.API_KEYS`).

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/stats/` | programme KPIs |
| GET/POST | `/api/v1/documents/` | list / **upload + auto-process** (`multipart: file, language, state, district, gamma`) |
| GET | `/api/v1/documents/<id>/` | status, pre-processing info, OCR text, full record |
| POST | `/api/v1/documents/<id>/reprocess/` | re-run pipeline |
| GET | `/api/v1/records/` | search (`?district=Varanasi&status=VERIFIED`, `?survey_number=331/2`, `?duplicates=1`…) |
| GET | `/api/v1/records/<id>/` | record + per-field confidence + issues |
| POST | `/api/v1/records/<id>/verify/` | `{"fields": {...}, "note": "..."}` — verify/edit |
| GET | `/api/v1/schema/fields/` | canonical field schema for integrators |

```bash
curl -H "X-API-Key: lr-demo-key-2026" \
     -F file=@khatauni_scan.pdf -F language=hin -F district=Bhopal -F gamma=auto \
     http://localhost:8000/api/v1/documents/
```

---

## 4. Project layout

```
landrecords/
├── config/                 # Django project (settings/urls/wsgi)
├── records/
│   ├── models.py           # Document, LandRecord, FieldExtraction,
│   │                       # ValidationIssue, LearnedCorrection, AuditLog, UserProfile
│   ├── constants.py        # canonical FIELD_DEFS (18 land-record fields)
│   ├── pipeline/
│   │   ├── preprocess.py   # ⭐ gamma correction, denoise, CLAHE, adaptive
│   │   │                   #    binarize, de-rule, despeckle, deskew
│   │   ├── ocr.py          # Tesseract wrapper: per-language auto selection
│   │   │                   #   over every installed pack (eng, hin, ben, tam,
│   │   │                   #   tel, kan, guj, mar, pan, +mal detection),
│   │   │                   #   PSM-4 with PSM-3 rescue, graceful fallback
│   │   ├── extract.py      # ⭐ NLP: normalise → exact/fuzzy label match →
│   │   │                   #    layout (bbox) context → entity parsing →
│   │   │                   #    confidence scoring
│   │   ├── ocr_text.py     # text → word/line bundle (PDF text layer, tests)
│   │   ├── nlp_corpus.py   # 28-case NLP regression corpus
│   │   ├── indic_labels.py # multilingual label catalogue (9 Indian languages)
│   │   ├── validate.py     # business rules, cross-checks, duplicate detection
│   │   ├── learn.py        # field-specific verified-correction memory
│   │   └── service.py      # orchestration + verification transactions
│   ├── views.py / api.py   # web UI + JSON integration API
│   └── management/commands/seed_demo.py
├── docs/uploaded_reference/  # verbatim reference copies of supplied modules
├── backup/                   # previous version of ocr.py (pre-merge)
├── tools/
│   ├── nlp_bench.py        # NLP scorecard  (python3 tools/nlp_bench.py -v)
│   └── e2e_bench.py        # scan→OCR→NLP accuracy vs ground truth
├── static/ css+js (self-contained, Chart.js vendored)
├── templates/
├── PROJECT_REPORT.md       # scope-of-study & technology tables, architecture
└── requirements.txt
```

## 5. Notes

- Demo corpus in `media/seed/` is **entirely synthetic** — generated offline; the OCR,
  enhancement and validation results are real.
- SQLite is used for the demo; swap `DATABASES` for PostgreSQL/PostGIS in production.
- The pipeline is synchronous for clarity; `service.process_document` drops straight
  into Celery/RQ for high-volume deployments.
- `settings.OCR_CONFIDENCE_THRESHOLD` (default 0.75) governs what enters the
  human-verification queue.
- Extraction quality checks:
  `python3 manage.py test records` (28 tests),
  `python3 tools/nlp_bench.py` (25-case NLP corpus, 103/103 fields),
  `python3 tools/e2e_bench.py` (rendered scans → real Tesseract → NLP).
- `auto` language mode runs one OCR pass per installed language pack and keeps
  the strongest result (accurate for unknown-language scans, slower); pick an
  explicit language on the upload form for the fast path.
- Every extracted value stores *how* it was found (`FieldExtraction.method`);
  the document page, verification console and `/api/v1/records/<id>/` all show
  the real pipeline output — there are no hard-coded demo values.
