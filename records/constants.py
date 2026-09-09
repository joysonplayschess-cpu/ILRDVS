"""
Canonical land-record field definitions.

Single source of truth used by the NLP extractor (pipeline/extract.py),
the verification UI, CSV export and the public API.  Every field carries a
weight used when aggregating the record-level confidence score.
"""

# (key, label, widget-type, required, weight)
FIELD_DEFS = [
    ("owner_name",           "Owner Name",            "text",   True,  1.0),
    ("father_name",          "Father / Spouse Name",  "text",   False, 0.5),
    ("survey_number",        "Survey Number",         "text",   True,  1.0),
    ("subdivision_number",   "Subdivision / Hissa No.", "text", False, 0.5),
    ("khasra_number",        "Khasra Number",         "text",   False, 0.8),
    ("khata_number",         "Khata Number",          "text",   False, 0.8),
    ("plot_area",            "Plot Area",             "number", True,  1.0),
    ("area_unit",            "Area Unit",             "text",   False, 0.4),
    ("village",              "Village / Mauza",       "text",   True,  1.0),
    ("tehsil",               "Tehsil / Taluka",       "text",   False, 0.7),
    ("district",             "District",              "text",   True,  1.0),
    ("state",                "State",                 "text",   False, 0.7),
    ("pincode",              "PIN Code",              "text",   False, 0.4),
    ("address",              "Address",               "text",   False, 0.4),
    ("land_classification",  "Land Classification",   "text",   False, 0.6),
    ("ownership_type",       "Ownership Type",        "text",   False, 0.5),
    ("mutation_number",      "Mutation (Dakhil Kharij) No.", "text", False, 0.5),
    ("mutation_date",        "Mutation Date",         "text",   False, 0.5),
    ("registration_number",  "Registration Number",   "text",   False, 0.5),
    ("registration_date",    "Registration Date",     "text",   False, 0.5),
]

FIELD_KEYS = [f[0] for f in FIELD_DEFS]
FIELD_LABELS = {f[0]: f[1] for f in FIELD_DEFS}
FIELD_WEIGHTS = {f[0]: f[4] for f in FIELD_DEFS}
REQUIRED_FIELDS = [f[0] for f in FIELD_DEFS if f[3]]

# Fields for which the "learning" subsystem maintains correction memories.
LEARNABLE_FIELDS = {"owner_name", "village", "tehsil", "district",
                    "land_classification", "ownership_type", "father_name"}

AREA_UNITS = {
    "hectare": 1.0, "acre": 0.404686, "sq.m": 0.0001, "sq.ft": 0.0000092903,
    "sq.yd": 0.00008361, "bigha": 0.2529, "biswa": 0.0126, "guntha": 0.0101,
    "cent": 0.00405, "are": 0.01, "kanal": 0.05059, "marla": 0.00253,
    "ground": 0.0223,
}

DOC_STATUS = {
    "UPLOADED": "Uploaded",
    "PROCESSING": "Processing",
    "PROCESSED": "Processed",
    "FAILED": "Failed",
}

RECORD_STATUS = {
    "PENDING": "Pending verification",
    "VERIFIED": "Verified",
    "REJECTED": "Rejected",
}
