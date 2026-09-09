#!/usr/bin/env python3
"""End-to-end bench: rendered scan -> gamma enhance -> Tesseract -> NLP.

Renders the synthetic sample corpus (clean / dark / faded / noisy / damaged,
Latin + Devanagari), runs the *real* pipeline stages and scores the extracted
fields against the ground truth used to render each page.

    python3 tools/e2e_bench.py            # all samples
    python3 tools/e2e_bench.py Bihar      # one sample
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django  # noqa: E402

django.setup()

from records.pipeline import extract, ocr, preprocess, sample_docs  # noqa: E402

only = sys.argv[1] if len(sys.argv) > 1 else None

out_dir = Path(tempfile.mkdtemp(prefix="e2e_"))
built = sample_docs.build(out_dir)
by_name = {s["filename"]: s for s in sample_docs.SAMPLES}

FIELD_OF = {
    "district": lambda d: d.get("district_hi") if d.get("script") == "devanagari" else d["district"],
    "tehsil": lambda d: d.get("tehsil_hi") if d.get("script") == "devanagari" else d["tehsil"],
    "village": lambda d: d.get("village_hi") if d.get("script") == "devanagari" else d["village"],
    "khata_number": lambda d: d["khata"],
    "khasra_number": lambda d: d["khasra"],
    "survey_number": lambda d: d["survey"],
    "owner_name": lambda d: d.get("owner_hi") if d.get("script") == "devanagari" else d["owner"],
    "father_name": lambda d: d.get("father_hi") if d.get("script") == "devanagari" else d["father"],
    "plot_area": lambda d: f'{float(d["area"]):g}',
    "mutation_number": lambda d: d["mutation"],
    "mutation_date": lambda d: d["mutation_date"],
    "registration_number": lambda d: d["reg"],
    "registration_date": lambda d: d["reg_date"],
    "pincode": lambda d: d["pin"],
}

grand_ok = grand_total = 0
for item in built:
    name = item["filename"]
    if only and only.lower() not in name.lower():
        continue
    truth = by_name[name]
    pages = preprocess.load_pages(str(item["path"]), max_pages=1)
    pre = preprocess.preprocess_image(pages[0], gamma="auto")
    res = ocr.ocr_image(pre["processed"], lang=truth.get("language", "auto"))
    for w in res["words"]:
        w["page"] = 0
    for l in res["lines"]:
        l["page"] = 0
    fields = extract.extract_fields(res, apply_learning=False)

    ok = total = 0
    misses = []
    for key, getter in FIELD_OF.items():
        want = str(getter(truth) or "").strip()
        if not want:
            continue
        got = fields[key]["value"].strip()
        total += 1
        if got.lower() == want.lower():
            ok += 1
        else:
            misses.append((key, want, got, round(fields[key]["confidence"], 2),
                           fields[key]["needs_review"]))
    grand_ok += ok
    grand_total += total
    print(f"{name:34s} gamma={pre['info']['gamma_used']:<6} "
          f"ocrconf={res['avg_conf'] * 100:5.1f}%  fields {ok}/{total}")
    for key, want, got, conf, review in misses:
        print(f"      MISS {key:20s} want={want!r:26s} got={got!r} "
              f"conf={conf} review={review}")

print(f"\nend-to-end field accuracy: {grand_ok}/{grand_total} = "
      f"{100.0 * grand_ok / max(grand_total, 1):.1f}%")
