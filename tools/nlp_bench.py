#!/usr/bin/env python3
"""NLP regression bench: runs the extraction corpus and prints a scorecard.

    python3 tools/nlp_bench.py            # summary
    python3 tools/nlp_bench.py -v         # show every field
    python3 tools/nlp_bench.py <case>     # one case, verbose
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django  # noqa: E402

django.setup()

from records.pipeline import extract, ocr_text          # noqa: E402
from records.pipeline.nlp_corpus import CASES           # noqa: E402

only = None
verbose = "-v" in sys.argv
for a in sys.argv[1:]:
    if not a.startswith("-"):
        only = a

total = hits = 0
failed_cases = []
for case in CASES:
    if only and only not in case["name"]:
        continue
    bundle = ocr_text.bundle_from_text(case["text"], conf=case["conf"] or 1)
    got = extract.extract_fields(bundle, apply_learning=False)
    bad = []
    for field, want in case["expect"].items():
        total += 1
        actual = got[field]["value"]
        ok = (actual.strip().lower() == str(want).strip().lower())
        hits += ok
        if not ok:
            bad.append((field, want, actual, got[field]["method"]))
    # nothing should be hallucinated for empty/garbage input
    if not case["expect"]:
        filled = {k: v["value"] for k, v in got.items() if v["value"]}
        total += 1
        if filled:
            bad.append(("<no hallucination>", "{}", str(filled), ""))
        else:
            hits += 1
    flag = "PASS" if not bad else "FAIL"
    if bad:
        failed_cases.append(case["name"])
    print(f"[{flag}] {case['name']}")
    for field, want, actual, method in bad:
        print(f"        {field:22s} want={want!r:28s} got={actual!r} "
              f"({method})")
    if verbose:
        for field, item in got.items():
            if item["value"]:
                print(f"        · {field:22s} = {item['value']!r:30s} "
                      f"conf={item['confidence']:.2f} review="
                      f"{item['needs_review']} :: {item['method']}")

print(f"\nfield accuracy: {hits}/{total} = {100.0 * hits / max(total,1):.1f}%")
if failed_cases:
    print("failing cases:", ", ".join(failed_cases))
sys.exit(0 if not failed_cases else 1)
