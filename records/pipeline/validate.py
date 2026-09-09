"""
Automated validation engine.

Combines business rules (required fields, formats, plausible ranges,
date sanity), cross-database style consistency checks (state <-> district
reference data) and duplicate detection across the digitised corpus.
"""
from __future__ import annotations

import re
from datetime import date

from records.constants import AREA_UNITS, FIELD_LABELS, REQUIRED_FIELDS

# Cross-database reference sample (in production: the DILRMP master data
# services / state land-record APIs would be queried in real time).
STATE_DISTRICTS = {
    "Uttar Pradesh": ["Varanasi", "Lucknow", "Prayagraj", "Agra", "Kanpur Nagar",
                      "Gorakhpur"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Sehore"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Ajmer"],
    "West Bengal": ["Kolkata", "Howrah", "Darjeeling", "Hooghly"],
    "Maharashtra": ["Pune", "Mumbai", "Nagpur", "Nashik"],
    "Karnataka": ["Bengaluru Urban", "Mysuru", "Belagavi", "Hubballi"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Thanjavur"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala", "Jalandhar"],
    "Telangana": ["Hyderabad", "Warangal", "Karimnagar"],
}
STATE_ALIASES = {"U.P.": "Uttar Pradesh", "UP": "Uttar Pradesh",
                 "M.P.": "Madhya Pradesh", "MP": "Madhya Pradesh",
                 "W.B.": "West Bengal", "TN": "Tamil Nadu",
                 "T.N.": "Tamil Nadu"}

PLOT_ID_RE = re.compile(r"^[0-9A-Za-z]+([/\-][0-9A-Za-z]+)*$")
DATE_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")


def _parse_date(value: str):
    m = DATE_RE.match((value or "").strip())
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000 if y < 50 else 1900
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _norm(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def validate_record(record) -> list[dict]:
    """Return a list of issue dicts for ``record`` (a LandRecord instance)."""
    issues = []

    def add(code, field="", severity="warning", message=""):
        issues.append({"rule_code": code, "field_name": field,
                       "severity": severity, "message": message})

    # -- 1. required fields ----------------------------------------------
    for field in REQUIRED_FIELDS:
        value = getattr(record, field)
        if value in (None, ""):
            add("REQUIRED_FIELD", field, "error",
                f"Mandatory field '{FIELD_LABELS[field]}' is empty.")

    # -- 2. identifiers must be plausible ---------------------------------
    for field in ("survey_number", "khasra_number", "khata_number"):
        value = getattr(record, field) or ""
        if value and not PLOT_ID_RE.match(value.replace(" ", "")):
            add("ID_FORMAT", field, "warning",
                f"'{value}' does not look like a valid "
                f"{FIELD_LABELS[field]} (allowed: digits, letters, / and -).")

    # -- 3. plot area sanity ----------------------------------------------
    if record.plot_area is not None:
        unit = record.area_unit or "hectare"
        factor = AREA_UNITS.get(unit)
        if factor is None:
            add("AREA_UNIT", "area_unit", "warning",
                f"Unknown area unit '{unit}'.")
            hectares = record.plot_area
        else:
            hectares = record.plot_area * factor
        if hectares <= 0.0001 or hectares > 60000:
            add("AREA_RANGE", "plot_area", "error",
                f"Plot area {record.plot_area:g} {unit} "
                f"(~{hectares:.4f} ha) is outside the plausible range.")

    # -- 4. pincode format --------------------------------------------------
    if record.pincode and not re.fullmatch(r"[1-9]\d{5}", record.pincode):
        add("PINCODE_FORMAT", "pincode", "error",
            f"PIN code '{record.pincode}' is not a valid 6-digit Indian PIN.")

    # -- 5. date sanity -----------------------------------------------------
    for field, label in (("mutation_date", "Mutation date"),
                         ("registration_date", "Registration date")):
        raw = getattr(record, field) or ""
        if raw:
            parsed = _parse_date(raw)
            if parsed is None:
                add("DATE_FORMAT", field, "error",
                    f"{label} '{raw}' is not a recognisable DD/MM/YYYY date.")
            elif parsed > date.today():
                add("DATE_FUTURE", field, "error",
                    f"{label} {raw} lies in the future.")

    # -- 6. state <-> district cross-check ----------------------------------
    state = STATE_ALIASES.get(record.state, record.state)
    if state and record.district:
        known = STATE_DISTRICTS.get(state)
        if known is not None and record.district not in known:
            add("STATE_DISTRICT", "district", "warning",
                f"'{record.district}' is not listed under {state} in the "
                f"reference master data - please confirm.")

    # -- 7. duplicate detection --------------------------------------------
    from records.models import LandRecord
    plot_ids = [pid for pid in (record.survey_number, record.khasra_number)
                if pid]
    if plot_ids and record.village and record.district:
        qs = LandRecord.objects.exclude(pk=record.pk).exclude(status="REJECTED")
        qs = qs.filter(village__iexact=_norm(record.village),
                       district__iexact=_norm(record.district))
        matches = []
        for other in qs.only("pk", "survey_number", "khasra_number",
                             "owner_name", "status"):
            other_ids = {p.upper() for p in (other.survey_number,
                                             other.khasra_number) if p}
            if other_ids & {p.upper() for p in plot_ids}:
                matches.append(other)
        if matches:
            refs = ", ".join(f"#{m.pk} ({m.status.lower()})" for m in matches)
            add("DUPLICATE_PLOT", "survey_number", "error",
                f"Another record with the same plot id + village + district "
                f"already exists: {refs}.")

    return issues
