"""
Correction memory ("learning from verification").

What this module *actually* does -- no more, no less:

* When a verification officer changes an extracted value, the pair
  ``(field, ocr_value) -> corrected_value`` is stored, **but only** for
  fields where a value repeats across documents (village, tehsil, district,
  state, owner/father name, land class, ownership type) and only when the
  correction looks like an OCR repair of the *same* entity rather than a
  completely different value.
* On later extractions the memory is consulted **per field**: an exact
  normalised hit is applied directly; otherwise a close fuzzy hit
  (SequenceMatcher >= 0.88, similar length) is applied.
* Identifiers, numbers, areas, PIN codes and dates are deliberately NOT
  learnable -- silently rewriting "123/4" into "123/5" because some other
  document was corrected that way would be dangerous.

It is a deterministic correction cache, and the UI describes it exactly that
way ("verified-correction memory"), not as a self-training model.
"""
from __future__ import annotations

import difflib

from django.db.models import F
from django.utils import timezone

from records.constants import LEARNABLE_FIELDS

SIMILARITY_THRESHOLD = 0.88     # ocr value -> remembered raw value
RELATEDNESS_THRESHOLD = 0.55    # raw -> corrected (guards against nonsense)
MIN_LENGTH = 3


def _norm(value: str) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def is_learnable(field_name: str) -> bool:
    return field_name in LEARNABLE_FIELDS


def suggest(field_name: str, value: str):
    """Return ``(corrected_value, kind)`` or ``(None, "")``.

    ``kind`` is ``"exact"`` or ``"fuzzy"`` and is surfaced in the UI.
    """
    from records.models import LearnedCorrection  # avoid app-loading issues

    if not is_learnable(field_name):
        return None, ""
    target = _norm(value)
    if len(target) < MIN_LENGTH:
        return None, ""

    rows = list(LearnedCorrection.objects.filter(field_name=field_name))
    if not rows:
        return None, ""

    # 1. exact (normalised) memory hit
    for row in rows:
        if _norm(row.raw_value) == target:
            if _norm(row.corrected_value) == target:
                return None, ""
            return row.corrected_value, "exact"

    # 2. never "correct" a value that is already a known good value
    for row in rows:
        if _norm(row.corrected_value) == target:
            return None, ""

    # 3. close fuzzy hit, with a length guard
    best, best_score = None, 0.0
    for row in rows:
        raw = _norm(row.raw_value)
        if abs(len(raw) - len(target)) > max(3, int(len(target) * 0.3)):
            continue
        score = difflib.SequenceMatcher(None, target, raw).ratio()
        if score > best_score:
            best, best_score = row, score
    if best is not None and best_score >= SIMILARITY_THRESHOLD:
        return best.corrected_value, "fuzzy"
    return None, ""


def apply_and_count(field_name: str, raw_value: str):
    """Suggest a correction and bump its usage counter when applied."""
    from records.models import LearnedCorrection

    corrected, _kind = suggest(field_name, raw_value)
    if corrected:
        LearnedCorrection.objects.filter(
            field_name=field_name, corrected_value=corrected).update(
            times_applied=F("times_applied") + 1, last_used=timezone.now())
    return corrected


def record_correction(field_name: str, raw_value: str, corrected_value: str):
    """Memorise an officer correction (called from the verification flow).

    Returns the stored row, or ``None`` when the correction is not safe to
    generalise (non-learnable field, too short, or a completely different
    value rather than an OCR repair).
    """
    from records.models import LearnedCorrection

    if not is_learnable(field_name):
        return None
    raw = _norm(raw_value)
    corrected = " ".join(str(corrected_value or "").split()).strip()
    if len(raw) < MIN_LENGTH or len(corrected) < MIN_LENGTH:
        return None
    if raw == _norm(corrected):
        return None
    relatedness = difflib.SequenceMatcher(None, raw, _norm(corrected)).ratio()
    if relatedness < RELATEDNESS_THRESHOLD:
        # e.g. the officer replaced "Ram Kumar" with "Sita Devi": that is a
        # data fix for this document, not a reusable OCR-repair rule.
        return None
    obj, created = LearnedCorrection.objects.get_or_create(
        field_name=field_name, raw_value=raw, corrected_value=corrected)
    if not created:
        obj.last_used = timezone.now()
        obj.save(update_fields=["last_used"])
    return obj
