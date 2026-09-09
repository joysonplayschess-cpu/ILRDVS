# Intelligent Land Record Digitization & Validation System (ILRDVS)
### Project Report — AI-powered digitization of legacy land records for the DILRMP ecosystem

> **Reference implementation:** Django + Tesseract OCR + OpenCV + NLP extraction,
> with gamma-correction image enhancement, confidence scoring, human-assisted
> verification, self-learning corrections, audit trails and an integration API.

---

## 1. Background and Problem Statement

Land records form the backbone of land administration, property ownership, taxation,
land acquisition, dispute resolution and infrastructure planning. Across India, a
significant portion of historical land records continues to exist as handwritten
registers, scanned documents, cadastral maps and legacy PDFs maintained at various
administrative levels. These records suffer from poor image quality, inconsistent
formats, faded text, damaged pages, multiple regional languages and handwritten
annotations — making manual digitization slow, costly and error-prone, and blocking
integration with modern Land Records Management Systems (LRMS), GIS platforms and
citizen-centric services under DILRMP.

## 2. Proposed Solution

An intelligent AI-based platform that automatically extracts structured information
from unstructured scans, classifies it into canonical land-record fields
(owner, survey / khasra / khata numbers, plot area, village, tehsil, district,
classification, ownership, mutation and registration details), validates it with
business rules and cross-database checks, and routes only genuinely uncertain
records to human verification officers — with every correction fed back to improve
the models over time.

## 3. Scope of Study

| # | Scope Area | In Scope (this implementation) | Techniques Used | Boundary / Future Extension |
|---|---|---|---|---|
| 1 | Document ingestion | Scanned PDFs, JPG/PNG/TIFF/BMP/WEBP scans & photos; multi-page PDFs (first 5 pages) | PyMuPDF rasterization, OpenCV decode | Bulk watch-folders, scanner (TWAIN/SANE) integration |
| 2 | Image enhancement | Auto/manual **gamma correction**, denoising (NL-Means), CLAHE local contrast, adaptive binarization, table-rule removal, despeckling, deskew | OpenCV pipeline with per-document diagnostics | GAN/deep-learning based restoration, bleed-through removal |
| 3 | OCR | Printed text in English + Hindi, Bengali, Tamil, Telugu, Kannada, Gujarati, Marathi, Punjabi; auto script detection; word-level confidence | Free & open-source **Tesseract 5** (`pytesseract`), PSM auto-retry | Handwriting (HWTR/TrOCR), layout-aware LSTM models, fine-tuned Indic packs |
| 4 | Field extraction | 18 canonical fields (owner, father, survey/khasra/khata, area+unit, village, tehsil, district, state, PIN, classification, ownership, mutation no./date, registration no./date) | Rule-based NLP: multilingual label spotting, layout geometry, typed parsers (numbers, units, dates, PIN) | NER with IndicBERT/LLMs, layout transformers (LayoutLM), geocoding |
| 5 | Validation | Mandatory-field checks, ID/area/PIN/date formats, plausible ranges, state↔district master cross-check, **duplicate detection** (plot+village+district) | Deterministic rules engine; severity levels | Live API cross-verification with state LRMS/DILRMP services, Aadhaar/UIDAI eKYC-based owner checks |
| 6 | Human-in-the-loop | Verification queue for low-confidence records, side-by-side console, approve/reject with notes | Role-gated Django views; per-field confidence surfaces | Two-person rule, supervisor escalation, sampling-based QC |
| 7 | AI learning | Correction memory (fuzzy auto-application on later documents), usage counters, learning log on dashboard | `difflib` similarity ≥ 0.86, per-field memories | Online model retraining, active-learning queues |
| 8 | Interoperability | JSON REST-style API with API-key auth; CSV export; canonical field schema endpoint | Django views; versioned `/api/v1/` | DILRMP/Bhulekh connectors, WFS/WMS (GeoServer), Bhuvan/NIC APIs, STAC for imagery |
| 9 | Security & audit | RBAC (Admin/Operator/Verifier/Viewer), immutable audit trail with IP, CSRF, secure cookies in prod | Django auth + custom profile roles | OAuth2/OIDC (NIC SSO), field-level encryption, data-residency controls |
| 10 | Analytics | Dashboards: documents processed, extraction confidence, validation status, pending cases, error mix, state/district-wise progress, language mix | Chart.js (vendored), aggregate SQL | GIS map widgets, SLA monitoring, per-tehsil drill-downs |
| 11 | Repository | Secure document store with metadata, pipeline logs, processed-image lineage | Django FileField storage + JSON metadata | Object storage (S3/MinIO), WORM retention, DMS standards (eGazette/DigiLocker linking) |
| 12 | Performance | Per-document synchronous processing (~3–8 s/page CPU) | NumPy/OpenCV vectorisation | Celery/RQ workers, GPU OCR nodes, horizontal scaling |

## 4. System Architecture

```
┌────────────┐   ┌────────────────────────  DJANGO APPLICATION  ───────────────────────────┐
│ Web / API  │   │                                                                          │
│ clients    │──▶│  RBAC & Auth ── Upload ──▶ PIPELINE SERVICE                               │
└────────────┘   │                    │      ├─ Preprocessor (γ-correct, denoise, CLAHE,    │
                 │                    ▼      │   binarize, de-rule, deskew)                 │
                 │             Document store│                                             │
                 │                    │      ├─ OCR engine (Tesseract, eng+hin+8 langs,     │
                 │                    ▼      │   word confidences, script detection)        │
                 │             ┌─────────┐   │                                             │
                 │             │ Audit   │───┼─ Field extractor (labels, geometry, typed)  │
                 │             │ trail   │   │                                             │
                 │             └─────────┘   ├─ Learning memory (correction fuzzy-match)   │
                 │                    │      │                                             │
                 │                    ▼      ├─ Validator (business rules, cross-checks,   │
                 │             Records DB ◀──│   duplicates)                                │
                 │                    │      │                                             │
                 │                    ▼      └─ HITL verification console                  │
                 │             Dashboards / CSV / /api/v1/                                  │
                 └──────────────────────────────────────────────────────────────────────────┘
```

## 5. Module-wise Data Flow

1. **Enhancement (`pipeline/preprocess.py`)** — grayscale → auto gamma
   (`γ = log(mean)/log(0.5)`, clamped 0.4–2.8) → NL-Means denoise → CLAHE →
   adaptive Gaussian threshold → connected-component despeckle → ruled-line
   removal via morphological opening + inpainting → min-area-rect deskew
   (±0.25° trigger).  Diagnostics (gamma used, brightness before/after, skew,
   per-step log, elapsed ms) are stored per document and shown in the UI.
2. **OCR (`pipeline/ocr.py`)** — Tesseract with paired Indic–English models,
   PSM-4 primary with automatic PSM-3 rescue on suspiciously low word counts,
   Unicode-script language detection, word-level confidence capture.
3. **Information extraction / NLP (`pipeline/extract.py`)** — a seven-stage
   layout-aware extractor (details in §5a).
4. **Validation (`pipeline/validate.py`)** — 7 rule classes (see §7).
5. **Correction memory (`pipeline/learn.py`)** — officer corrections are
   memorised **per field** and re-applied to later documents when the new OCR
   value is an exact or very close (≥0.88) match of a remembered raw value.
   Identifiers, areas, PIN codes and dates are deliberately *not* learnable,
   and a replacement that is not an OCR-repair of the same entity
   (similarity < 0.55) is never generalised.  It is a deterministic,
   auditable correction cache — not a trained model — and the UI says so.
6. **Verification (`views.verify_record`)** — editable per-field console with
   confidence bars, original-OCR recall chips, issue alerts, approve/reject.

### 5a. Extraction (NLP) pipeline in detail

| Stage | What happens | Code |
|---|---|---|
| 1. Normalisation | NFKC folding, OCR confusable repair (`0→o, 1→l, 5→s, |→l …`) applied to *label candidates only*, never to stored values | `_fold`, `_clean_text` |
| 2. Exact label matching | ~230 aliases across 20 fields (English, transliterated and Indic: *Owner Name / Khatedar / Patta Holder / Land Owner / मालिक का नाम / பட்டாதாரர்*) | `LABELS`, `ALIAS_INDEX` |
| 3. Fuzzy label matching | Token-window similarity (1–4 tokens) in two modes: whole-window ratio and word-aligned ratio, so `0wner Nane`, `Ownr Name`, `Villaqe`, `Distrist`, `Reglstration No` still resolve.  Longer words that merely *contain* an alias (`Tehsildar`, `Khatauni`) are penalised | `_match_alias` |
| 4. Contextual / layout analysis | Value is taken (a) inline to the right of the label and **cut at the next label on the line** – so `District: Chennai Village: Sholinganallur` can never bleed, (b) from the next table column on the same row band using OCR bounding boxes, or (c) from the line below the label when the form is stacked | `_inline_candidate`, `_column_candidate`, `_below_candidate` |
| 5. Entity / pattern parsing | Typed parsers: plot ids (`123`, `123/4`, `123-A`, `123/4A`, `S.No. 123/4`), areas + unit normalisation (acre/acres, hectare, sq.ft, sq ft, cents, bigha, guntha, kanal…), dates (`12/05/2024`, `12-05-2024`, `12.05.2024`, `12 May 2024`, `2024-05-12` → `DD/MM/YYYY`), PIN codes, names (boiler-plate, enum and character-soup rejection), free text.  Label-free fallbacks: gazetteer state/district lookup, PIN pattern, number-with-unit, date on the identifier's line | `parse_*`, `_pattern_fallbacks` |
| 6. Confidence scoring | `OCR word confidence × label-match quality × value plausibility × position prior`, per field, with the winning candidate chosen by the same product.  Every field also stores a human-readable `method` (e.g. *"fuzzy label 'owner name' (0.89) + value on line below label"*) and the OCR line it came from | `_score`, `FieldExtraction.method` |
| 7. Hand-off to validation | Anything below `OCR_CONFIDENCE_THRESHOLD` (0.75) or missing-but-required is flagged `needs_review` and enters the human queue instead of being silently trusted | `_finalise`, `validate.py` |

Measured on the bundled corpora: **103/103 fields** on the synthetic OCR-text
regression corpus (`records/pipeline/nlp_corpus.py`, 25 cases) and **14/14
fields on every legible rendered scan** in the end-to-end bench
(`tools/e2e_bench.py`); overall 131/168 including the deliberately destroyed
and Devanāgarī scans, where the losses are OCR character errors (all of them
flagged for review, none silently stored).

## 6. Suggested Components-wise Technology

| # | Component | This Implementation (free / open-source) | Production-grade Alternatives |
|---|---|---|---|
| 1 | **Frontend** | Django Templates + custom CSS + vanilla JS | React/Next.js, Angular |
| 2 | **Charts / dashboards** | Chart.js (vendored, offline-safe) | Apache ECharts, Metabase/Superset, Grafana |
| 3 | **Web framework / API** | Django 5/6 + json API (this project) | Django + Django REST Framework, FastAPI |
| 4 | **Image preprocessing** | OpenCV + NumPy (γ-correction, CLAHE, NLM denoise, morph ops) | OpenCV + scikit-image; deep restoration (Real-ESRGAN) |
| 5 | **OCR engine (free)** | **Tesseract 5** + `pytesseract`, 9 Indic language packs | **Free APIs/engines:** OCR.space free tier, Google ML Kit, PaddleOCR, EasyOCR; commercial: Azure/Google/AWS Textract |
| 6 | **Indic NLP** | Rule + geometry based extraction; fuzzy learning (`difflib`) | IndicBERT, IndicNER, spaCy + custom rules, LLM extraction (Llama/GPT) |
| 7 | **Handwritten Text Recognition** | — (roadmap) | TrOCR, HWTR-CRNN, Google Document AI |
| 8 | **Relational DB** | SQLite (demo) → PostgreSQL | PostgreSQL + partitioning for national scale |
| 9 | **GIS / cadastral maps** | metadata hooks (state/district) | PostGIS + GeoDjango, GeoServer, QGIS, MapServer, Leaflet/MapLibre GL |
| 10 | **Task queue** | synchronous (demo) | Celery + Redis/RabbitMQ, Dramatiq |
| 11 | **Document store** | filesystem (`media/`) | S3/MinIO, WORM compliance storage |
| 12 | **Authentication** | Django auth + profile roles + API key header | OAuth2/OIDC (Keycloak), eParivartan/NIC SSO |
| 13 | **Logging / audit** | `AuditLog` model (immutable) | ELK/OpenSearch, audit-log services |
| 14 | **Deployment** | `runserver` (demo) | Gunicorn/Uvicorn + Nginx, Docker, Kubernetes |
| 15 | **Monitoring** | pipeline timings per document | Prometheus + Grafana, Sentry |
| 16 | **Integration targets** | `/api/v1/*` for LRMS/DILRMP/GIS | DILRMP NIC connectors, Bhuvan, DigiLocker, eDistrict |

## 7. Validation Rule Catalogue

| Rule code | Check | Severity |
|---|---|---|
| `REQUIRED_FIELD` | owner, survey/khasra, village, district, plot area present | error |
| `ID_FORMAT` | plot identifiers match digit/letter/slash-dash grammar | warning |
| `AREA_RANGE` | area normalised to hectares within plausible bounds | error |
| `AREA_UNIT` | unit recognised (ha/acre/sq m/bigha/biswa/guntha/cent…) | warning |
| `PINCODE_FORMAT` | valid 6-digit Indian PIN | error |
| `DATE_FORMAT` / `DATE_FUTURE` | DD/MM/YYYY parse; no future mutation/registration dates | error |
| `STATE_DISTRICT` | district belongs to state per reference master data | warning |
| `DUPLICATE_PLOT` | same plot id + village + district already digitized | error |

## 8. Data Model (summary)

`UserProfile(role)` — RBAC over Django users.
`Document` — file, enhanced image, OCR text/confidence/engine/language, preprocessing
diagnostics, pipeline log, status (`UPLOADED→PROCESSING→PROCESSED/FAILED`).
`LandRecord` — the 18 canonical fields + `overall_confidence`, `status`
(`PENDING/VERIFIED/REJECTED`), `is_duplicate`, verifier linkage.
`FieldExtraction` — per-field value, confidence, provenance (`ocr`/`learned`/`manual`),
review flag.  `ValidationIssue` — rule/severity/message/resolved.
`LearnedCorrection` — field, raw → corrected memory with usage counters.
`AuditLog` — actor, action, object, detail, IP, timestamp (immutable).

## 9. API Examples

```bash
# upload + auto-digitize a faded Hindi scan
curl -H "X-API-Key: lr-demo-key-2026" \
     -F file=@ror_scan.pdf -F language=hin -F state="Madhya Pradesh" \
     -F district=Bhopal -F gamma=auto \
     http://HOST/api/v1/documents/

# poll processing / fetch structured record
curl -H "X-API-Key: lr-demo-key-2026" http://HOST/api/v1/documents/7/

# search verified records for a plot
curl -H "X-API-Key: lr-demo-key-2026" \
     "http://HOST/api/v1/records/?district=Varanasi&survey_number=331/2"

# officer-side correction + verification from a back-office system
curl -H "X-API-Key: lr-demo-key-2026" -H "Content-Type: application/json" \
     -X POST http://HOST/api/v1/records/7/verify/ \
     -d '{"fields":{"owner_name":"Geeta Devi","survey_number":"47/2"},"note":"register 12/44"}'
```

## 10. Security, Privacy & Governance

- Role-based access control (Admin / Operator / Verifier / Viewer) with friendly
  in-app guard rails; Django admin reserved for superusers.
- Full audit trail (uploads, pipeline runs, every field edit, verifications,
  rejections, exports, API calls) with actor + IP + timestamp.
- CSRF protection, auth-required pages, API keys for machine consumers;
  production hardening path documented (TLS, OIDC, WORM storage, PII minimisation).

## 11. Measurable Outcomes (demo corpus)

- 11 synthetic scans (clean/dark/faded/noisy/damaged, English + Hindi, PDF + images)
  processed fully automatically; dark scans recovered from ~30 % word recognition to
  **>90 % mean confidence after auto gamma correction**.
- Exact-match extraction of all 18 fields on clean forms; corrupted glyphs surface as
  low-confidence instead of silent errors → targeted human review only where needed.
- Duplicate rescan automatically linked and quarantined (rejected in audit trail).
- Verifier corrections demonstrably re-applied to subsequent extractions
  (learning log, `times_applied` counters).

## 12. Limitations & Future Work

Handwritten annotations (TrOCR/HTR), full cadastral-map vectorization (GeoDjango +
deep segmentation), live state LRMS master-data calls, async worker farm, and
model fine-tuning on state-specific form templates are the natural next steps;
the pipeline seams (`service.process_document`, `pipeline.*`) are already factored
for those swaps.
