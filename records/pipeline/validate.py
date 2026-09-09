"""
Automated validation engine.

Combines business rules (required fields, formats, plausible ranges,
date sanity), cross-database style consistency checks (state <-> district
reference data) and duplicate detection across the digitised corpus.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

from records.constants import AREA_UNITS, FIELD_LABELS, REQUIRED_FIELDS

# Cross-database reference master data for Indic states & districts
STATE_DISTRICTS = {
    "Tamil Nadu": [
        "Chennai", "Coimbatore", "Madurai", "Thanjavur", "Kanchipuram",
        "Chengalpattu", "Tiruvallur", "Salem", "Tiruchirappalli", "Tirunelveli",
        "Erode", "Vellore", "Dharmapuri", "Kanyakumari", "Nilgiris", "Dindigul",
        "Karur", "Namakkal", "Perambalur", "Pudukkottai", "Ramanathapuram",
        "Sivaganga", "Theni", "Thoothukudi", "Tiruppur", "Tiruvannamalai",
        "Tiruvarur", "Viluppuram", "Virudhunagar", "Ariyalur", "Krishnagiri",
        "Nagapattinam", "Ranipet", "Tenkasi", "Tirupathur", "Mayiladuthurai"
    ],
    "Telangana": [
        "Hyderabad", "Warangal", "Karimnagar", "Ranga Reddy", "Medchal-Malkajgiri",
        "Sangareddy", "Nizamabad", "Khammam", "Nalgonda", "Mahbubnagar",
        "Adilabad", "Bhadradri Kothagudem", "Jagtial", "Jangaon", "Jayashankar",
        "Jogulamba", "Kamareddy", "Komaram Bheem", "Mahabubabad", "Mancherial",
        "Medak", "Mulugu", "Nagarkurnool", "Narayanpet", "Nirmal", "Peddapalli",
        "Rajanna Sircilla", "Siddipet", "Suryapet", "Vikarabad", "Wanaparthy",
        "Yadadri Bhuvanagiri"
    ],
    "Karnataka": [
        "Bengaluru Urban", "Bengaluru Rural", "Mysuru", "Belagavi", "Hubballi-Dharwad",
        "Dharwad", "Dakshina Kannada", "Udupi", "Hassan", "Mandya", "Tumakuru",
        "Ballari", "Bidar", "Chamarajanagar", "Chikkaballapura", "Chikkamagaluru",
        "Chitradurga", "Davangere", "Gadag", "Kalaburagi", "Kodagu", "Kolar",
        "Koppal", "Ramanagara", "Shivamogga", "Haveri", "Uttara Kannada",
        "Vijayapura", "Yadgir", "Vijayanagara"
    ],
    "Kerala": [
        "Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha", "Kottayam",
        "Idukki", "Ernakulam", "Thrissur", "Palakkad", "Malappuram", "Kozhikode",
        "Wayanad", "Kannur", "Kasaragod"
    ],
    "Andhra Pradesh": [
        "Visakhapatnam", "Vijayawada", "Guntur", "Tirupati", "Kakinada", "Nellore",
        "Kurnool", "Anantapur", "Kadapa", "Chittoor", "East Godavari", "West Godavari",
        "Prakasam", "Srikakulam", "Vizianagaram"
    ],
    "Uttar Pradesh": [
        "Varanasi", "Lucknow", "Prayagraj", "Agra", "Kanpur Nagar", "Gorakhpur",
        "Ayodhya", "Meerut", "Ghaziabad", "Gautam Buddha Nagar", "Bareilly",
        "Aligarh", "Mathura", "Jhansi", "Saharanpur", "Moradabad", "Bhopal"
    ],
    "Madhya Pradesh": [
        "Bhopal", "Indore", "Gwalior", "Jabalpur", "Sehore", "Ujjain", "Sagar",
        "Rewa", "Satna", "Dhar", "Dewas"
    ],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga", "Purnia"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Ajmer", "Kota", "Bikaner"],
    "West Bengal": ["Kolkata", "Howrah", "Darjeeling", "Hooghly", "North 24 Parganas", "South 24 Parganas"],
    "Maharashtra": ["Pune", "Mumbai", "Mumbai Suburban", "Nagpur", "Nashik", "Thane", "Aurangabad", "Solapur"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar", "Bhavnagar"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala", "Jalandhar", "Bathinda", "Mohali"],
}

STATE_ALIASES = {
    "U.P.": "Uttar Pradesh", "UP": "Uttar Pradesh",
    "M.P.": "Madhya Pradesh", "MP": "Madhya Pradesh",
    "W.B.": "West Bengal",
    "TN": "Tamil Nadu", "T.N.": "Tamil Nadu", "Tamilnadu": "Tamil Nadu",
    "TS": "Telangana", "T.S.": "Telangana", "TG": "Telangana",
    "KA": "Karnataka", "K.A.": "Karnataka",
    "KL": "Kerala", "K.L.": "Kerala",
    "AP": "Andhra Pradesh", "A.P.": "Andhra Pradesh"
}

# Unicode-aware plot ID pattern (allows native digits, slashes, hyphens)
PLOT_ID_RE = re.compile(r"^[0-9A-Za-z\u0900-\u0DFF]+([/\-][0-9A-Za-z\u0900-\u0DFF]+)*$")
DATE_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})$")


def _parse_date(value: str):
    val = unicodedata.normalize("NFKC", (value or "").strip())
    m = DATE_RE.match(val)
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
    cleaned = unicodedata.normalize("NFKC", text or "")
    return " ".join(cleaned.split()).strip().lower()


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
    state_normalized = STATE_ALIASES.get(record.state, record.state)
    if state_normalized and record.district:
        known_districts = STATE_DISTRICTS.get(state_normalized)
        if known_districts is not None:
            # Case-insensitive search across known districts
            district_match = any(_norm(record.district) == _norm(d) for d in known_districts)
            if not district_match:
                add("STATE_DISTRICT", "district", "warning",
                    f"'{record.district}' is not listed under {state_normalized} in the "
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