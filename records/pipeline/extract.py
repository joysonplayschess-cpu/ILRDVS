"""
Layout-aware information-extraction (NLP) layer.

Pipeline implemented here, in this order:

    1. text normalisation      -- unicode folding, OCR confusable repair
    2. exact label matching    -- alias lexicon (EN + transliterated + Indic)
    3. fuzzy label matching    -- token-window similarity (difflib), so
                                  "0wner Nane" still matches "owner name"
    4. contextual analysis     -- value right of the label, in the next
                                  table column (same row band, using OCR
                                  bounding boxes), or on the line below
    5. entity/pattern parsing  -- typed parsers for ids, areas+units, dates,
                                  PIN codes, names, free text
    6. confidence scoring      -- OCR word confidence x label-match quality
                                  x value plausibility, per field
    7. hand-off to validation  -- low confidence / implausible values are
                                  flagged ``needs_review`` for the HITL queue

Every field returned carries ``value``, ``confidence``, ``method`` (a plain
description of *how* it was found) and ``evidence`` (the OCR line it came
from) so nothing shown in the UI is invented.

Input is the OCR bundle produced by ``pipeline.ocr.ocr_image`` (or
``pipeline.ocr_text.bundle_from_text``): ``{"lines": [...], "words": [...],
"text": str}`` where each word has ``text, conf, x, y, w, h``.
"""
from __future__ import annotations

import difflib
import functools
import re
import unicodedata
from statistics import mean

from django.conf import settings

from records.constants import FIELD_KEYS, LEARNABLE_FIELDS
from . import learn
from .indic_labels import INDIC_LABELS

# ---------------------------------------------------------------------------
# 1. Normalisation helpers
# ---------------------------------------------------------------------------
# Characters Tesseract habitually swaps on degraded scans.  Applied ONLY when
# comparing label candidates -- never to the stored value.
_CONFUSABLES = str.maketrans({
    "0": "o", "1": "l", "5": "s", "8": "b", "6": "g", "9": "g", "2": "z",
    "|": "l", "!": "l", "@": "a", "$": "s", "£": "e", "¢": "c", "€": "e",
    "§": "s", "©": "c", "—": "-", "–": "-", "“": '"', "”": '"', "’": "'",
})


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _fold(text: str) -> str:
    """Aggressive fold used for label comparison only."""
    text = _nfkc(text).lower().translate(_CONFUSABLES)
    return "".join(ch for ch in text if ch.isalnum())


def _clean_text(text: str) -> str:
    """Light clean-up applied to *values* (keeps original characters)."""
    text = _nfkc(text)
    text = text.replace("|", " ").replace("_", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" \t:;,-–—.=~*<>\"'`/\\")


@functools.lru_cache(maxsize=200_000)
def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# 2. Label lexicon
# ---------------------------------------------------------------------------
# field -> list of (alias, specificity).  Specificity < 1.0 marks generic
# words ("name", "area", "no") that must lose against a specific alias.
LABELS: dict[str, list[tuple[str, float]]] = {
    "owner_name": [
        ("owner name", 1.0), ("owners name", 1.0), ("owner's name", 1.0),
        ("name of owner", 1.0), ("name of the owner", 1.0),
        ("name of holder", 1.0), ("holder name", 1.0), ("land owner", 1.0),
        ("landowner", 1.0), ("khatedar", 1.0), ("khatedar name", 1.0),
        ("khatadar", 1.0), ("khata holder", 1.0), ("patta holder", 1.0),
        ("pattadar", 1.0), ("pattedar", 1.0), ("patta holder name", 1.0),
        ("raiyat", 1.0), ("raiyat name", 1.0), ("bhumidhar", 1.0),
        ("account holder", 1.0), ("name of tenant", 1.0),
        ("owner", 0.9), ("name", 0.55), ("name of khatedar", 1.0),
        ("मालिक का नाम", 1.0), ("खातेदार का नाम", 1.0), ("भूमि स्वामी", 1.0),
        ("खातेदार", 0.95), ("मालिक", 0.9), ("स्वामी", 0.9), ("धारक का नाम", 1.0),
        ("பட்டாதாரர்", 1.0), ("உரிமையாளர்", 1.0),
    ],
    "father_name": [
        ("father name", 1.0), ("fathers name", 1.0), ("father's name", 1.0),
        ("name of father", 1.0), ("husband name", 1.0),
        ("husbands name", 1.0), ("father husband name", 1.0),
        ("father or husband name", 1.0), ("spouse name", 1.0),
        ("guardian name", 1.0), ("s/o", 0.9), ("w/o", 0.9), ("d/o", 0.9),
        ("son of", 0.9), ("wife of", 0.9), ("daughter of", 0.9),
        ("father", 0.85), ("husband", 0.85),
        ("पिता का नाम", 1.0), ("पति का नाम", 1.0), ("पिता", 0.85), ("पति", 0.85),
        ("தந்தை பெயர்", 1.0),
    ],
    "survey_number": [
        ("survey number", 1.0), ("survey no", 1.0), ("survey nos", 1.0),
        ("s no", 0.9), ("sy no", 1.0), ("sy number", 1.0),
        ("resurvey number", 1.0), ("re survey no", 1.0),
        ("old survey no", 1.0), ("new survey no", 1.0),
        ("field number", 0.9), ("gat number", 1.0), ("gat no", 1.0),
        ("plot number", 0.9), ("plot no", 0.9), ("survey", 0.85),
        ("सर्वे नंबर", 1.0), ("सर्वे क्रमांक", 1.0), ("गट क्रमांक", 1.0),
        ("भूमि क्रमांक", 1.0), ("सर्वे", 0.85), ("புல எண்", 1.0),
    ],
    "subdivision_number": [
        ("subdivision number", 1.0), ("subdivision no", 1.0),
        ("sub division no", 1.0), ("sub division number", 1.0),
        ("subdivision", 0.95), ("sub div no", 1.0), ("hissa number", 1.0),
        ("hissa no", 1.0), ("hissa", 0.9), ("उपविभाग", 1.0), ("हिस्सा", 0.9),
        ("உட்பிரிவு", 1.0),
    ],
    "khasra_number": [
        ("khasra number", 1.0), ("khasra no", 1.0), ("khasra", 0.95),
        ("खसरा नंबर", 1.0), ("खसरा", 0.95),
    ],
    "khata_number": [
        ("khata number", 1.0), ("khata no", 1.0), ("khewat number", 1.0),
        ("khewat no", 1.0), ("khatauni number", 1.0), ("khatauni no", 1.0),
        ("account number", 0.9), ("account no", 0.9), ("patta number", 1.0),
        ("patta no", 1.0), ("patta", 0.9), ("khata", 0.9), ("khewat", 0.9),
        ("खाता नंबर", 1.0), ("खाता", 0.9), ("पट्टा संख्या", 1.0),
        ("பட்டா எண்", 1.0),
    ],
    "plot_area": [
        ("total area", 1.0), ("area of plot", 1.0), ("plot area", 1.0),
        ("land area", 1.0), ("extent of land", 1.0), ("extent", 0.95),
        ("measurement", 0.85), ("rakba", 1.0), ("rakaba", 1.0),
        ("area", 0.8), ("क्षेत्रफल", 1.0), ("रकबा", 1.0), ("क्षेत्र", 0.85),
        ("பரப்பளவு", 1.0),
    ],
    "village": [
        ("village name", 1.0), ("name of village", 1.0),
        ("revenue village", 1.0), ("village", 0.95), ("mauza", 1.0),
        ("mouza", 1.0), ("gram", 0.9), ("gaon", 0.9), ("grama", 0.9),
        ("ग्राम", 0.95), ("मौजा", 1.0), ("गाँव", 0.95), ("गाव", 0.9),
        ("கிராமம்", 1.0),
    ],
    "tehsil": [
        ("tehsil", 1.0), ("tahsil", 1.0), ("tehsil taluka", 1.0),
        ("taluka", 1.0), ("taluk", 1.0), ("taluq", 1.0), ("mandal", 0.9),
        ("firka", 0.9), ("anchal", 0.9), ("sub district", 0.9),
        ("तहसील", 1.0), ("तालुका", 1.0), ("வட்டம்", 1.0),
    ],
    "district": [
        ("district", 1.0), ("revenue district", 1.0), ("distt", 0.95),
        ("dist", 0.9), ("zila", 0.95), ("jilla", 0.95),
        ("जिला", 1.0), ("ज़िला", 1.0), ("மாவட்டம்", 1.0),
    ],
    "state": [
        ("state", 0.95), ("state name", 1.0), ("province", 0.9),
        ("राज्य", 1.0), ("மாநிலம்", 1.0),
    ],
    "pincode": [
        ("pin code", 1.0), ("pincode", 1.0), ("postal code", 1.0),
        ("post code", 1.0), ("zip code", 1.0), ("pin", 0.85),
        ("पिन कोड", 1.0), ("पिन", 0.85),
    ],
    "address": [
        ("address", 1.0), ("postal address", 1.0), ("address of owner", 1.0),
        ("owner address", 1.0), ("residing at", 1.0), ("resident of", 1.0),
        ("पता", 1.0), ("निवास", 0.9), ("முகவரி", 1.0),
    ],
    "land_classification": [
        ("land classification", 1.0), ("classification of land", 1.0),
        ("class of land", 1.0), ("land type", 1.0), ("type of land", 1.0),
        ("nature of land", 1.0), ("kind of land", 1.0), ("land use", 0.9),
        ("soil class", 1.0), ("classification", 0.9), ("land class", 1.0),
        ("भूमि वर्गीकरण", 1.0), ("भूमि का प्रकार", 1.0), ("भूमि प्रकार", 1.0),
        ("वर्गीकरण", 0.9), ("நில வகை", 1.0),
    ],
    "ownership_type": [
        ("ownership type", 1.0), ("type of ownership", 1.0),
        ("nature of ownership", 1.0), ("tenure type", 1.0),
        ("holding type", 1.0), ("ownership", 0.9), ("tenure", 0.85),
        ("स्वामित्व प्रकार", 1.0), ("स्वामित्व", 0.9), ("मालिकी", 0.9),
    ],
    "mutation_number": [
        ("mutation number", 1.0), ("mutation no", 1.0),
        ("dakhil kharij no", 1.0), ("dakhil kharij", 1.0),
        ("namantaran no", 1.0), ("mutation", 0.9),
        ("दाखिल खारिज नंबर", 1.0), ("दाखिल खारिज", 1.0), ("नामांतरण", 0.95),
    ],
    "mutation_date": [
        ("mutation date", 1.0), ("date of mutation", 1.0),
        ("दाखिल खारिज दिनांक", 1.0), ("नामांतरण दिनांक", 1.0),
    ],
    "registration_number": [
        ("registration number", 1.0), ("registration no", 1.0),
        ("regn number", 1.0), ("regn no", 1.0), ("reg no", 0.95),
        ("registry number", 1.0), ("registry no", 1.0),
        ("document number", 0.95), ("document no", 0.95),
        ("deed number", 1.0), ("deed no", 1.0), ("sale deed no", 1.0),
        ("registration", 0.85),
        ("पंजीकरण नंबर", 1.0), ("पंजीकरण", 0.9), ("रजिस्ट्री", 0.9),
    ],
    "registration_date": [
        ("registration date", 1.0), ("date of registration", 1.0),
        ("date of registry", 1.0), ("deed date", 1.0),
        ("पंजीकरण दिनांक", 1.0), ("पंजीकरण तिथि", 1.0),
    ],
    # Pseudo-field: a bare "Date:" token.  Never stored, but it terminates a
    # value and lets us attach the date to the identifier next to it.
    "_date": [("date", 0.6), ("dated", 0.7), ("dt", 0.6),
              ("दिनांक", 0.8), ("तिथि", 0.8)],
    # Pseudo-field: table header cells, so "Field | Value" headers do not
    # get mistaken for data. Also doubles as a boundary marker for columns
    # this schema doesn't model as a field (e.g. "Soil Type", "Remarks" on
    # a Tamil Nadu patta table) -- without registering *some* hit for them,
    # `_inline_candidate` has no boundary to stop at and happily swallows
    # an unmodeled header cell's text into whichever field's label sits
    # immediately to its left (e.g. "Father's Name" grabbing "Soil Remark").
    "_header": [("value", 0.6), ("particulars", 0.8), ("description", 0.7),
               ("details", 0.6), ("sl no", 0.8), ("s.no.", 0.5),
               ("soil type", 0.85), ("soil", 0.75),
               ("remarks", 0.85), ("remark", 0.8)],
}

PSEUDO_FIELDS = {"_date", "_header"}


def _merge_indic_labels() -> None:
    """Fold the Indic catalogue (Hindi, Bengali, Tamil, Telugu, Malayalam,
    Kannada, Gujarati, Marathi, Punjabi) into the alias lexicon."""
    for field, aliases in INDIC_LABELS.items():
        bucket = LABELS.setdefault(field, [])
        known = {a for a, _s in bucket}
        for alias, spec in aliases:
            if alias not in known:
                bucket.append((alias, spec))
                known.add(alias)


_merge_indic_labels()

# Pre-folded index: field -> [(folded_alias, alias, specificity, word_folds)]
ALIAS_INDEX = {
    field: sorted(((_fold(a), a, spec,
                    tuple(w for w in (_fold(x) for x in a.split()) if w))
                   for a, spec in aliases),
                  key=lambda t: -len(t[0]))
    for field, aliases in LABELS.items()
}
_ALIAS_LENGTHS = {len(f) for aliases in ALIAS_INDEX.values()
                  for f, _a, _s, _w in aliases}
_MIN_ALIAS_LEN = min(_ALIAS_LENGTHS)
_MAX_ALIAS_LEN = max(_ALIAS_LENGTHS)

FUZZY_THRESHOLD = 0.85          # >= 7 chars ("villaqe" -> "village")
FUZZY_THRESHOLD_MID = 0.87      # 5 chars
FUZZY_THRESHOLD_MID6 = 0.83     # 6 chars ("extont" -> "extent")
FUZZY_THRESHOLD_SHORT = 0.99    # <= 4 chars -> effectively exact

# ---------------------------------------------------------------------------
# 3. Entity patterns
# ---------------------------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
     "nov", "dec"], start=1)}

DATE_NUM_RE = re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\b")
DATE_ISO_RE = re.compile(r"\b(\d{4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\b")
DATE_TXT_RE = re.compile(
    r"\b(\d{1,2})\s*(?:st|nd|rd|th)?[\s\-,]+([A-Za-z]{3,9})[\s\-,]+(\d{2,4})\b")
DATE_TXT2_RE = re.compile(r"\b([A-Za-z]{3,9})[\s\-,]+(\d{1,2})(?:st|nd|rd|th)?[\s\-,]+(\d{2,4})\b")

PIN_RE = re.compile(r"\b([1-9]\d{2}\s?\d{3})\b")
NUM_RE = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")
# 123 | 123/4 | 123-A | 123/4A | 12/3-B | TS-45/2
# NOTE: uses \d (Unicode-aware) throughout, not [0-9] -- plot identifiers
# on Hindi/regional-script forms are frequently written with native-script
# digits (e.g. Devanagari "४७/२"), and \d matches those too. An earlier
# version mixed \d for the first number with [0-9] for the part after the
# separator, which silently truncated e.g. "४७/२" -> "४७".
PLOT_ID_RE = re.compile(
    r"\b(?:[A-Za-z]{1,4}[-/])?\d{1,6}[A-Za-z]{0,2}"
    r"(?:\s*[/-]\s*\d{1,5}[A-Za-z]{0,2}|\s*[/-]\s*[A-Za-z]{1,2}\b){0,3}")

AREA_UNIT_PATTERNS = [
    ("hectare", r"hect?ares?|hectr?es?|\bhect?\b|\bha\b|\bhe\b|हेक्टेयर|हेक्टर|"
                r"হেক্টর|ஹெக்டேர்|హెక్టార్|ഹെക്ടർ|ಹೆಕ್ಟೇರ್|હેક્ટર|ਹੈਕਟੇਅਰ"),
    ("acre", r"acres?|\bac\b|एकड़|एकड|ஏக்கர்|একর|ఎకరం|ఎకరాలు|ഏക്കർ|ಎಕರೆ|એકર|ਏਕੜ"),
    ("sq.ft", r"sq\.?\s*ft\b|sq\.?\s*feet|square\s*fe?e?t|sqft|वर्ग\s*फीट|"
              r"বর্গফুট|சதுர அடி|చదరపు అడుగులు|ચોરસ ફૂટ"),
    ("sq.m", r"sq\.?\s*m(?:tr|eters?|etres?)?\b|square\s*met(?:er|re)s?|वर्ग\s*मीटर"),
    ("sq.yd", r"sq\.?\s*y(?:d|ards?)\b|square\s*yards?|गज"),
    ("cent", r"cents?\b|சென்ட்|সেন্ট|సెంటు|സെന്റ്|ಸೆಂಟ್"),
    ("bigha", r"bighas?\b|बीघा"),
    ("biswa", r"bisw?as?\b|बिसवा"),
    ("guntha", r"gunthas?\b|guntas?\b|गुंठा"),
    ("kanal", r"kanals?\b|कनाल"),
    ("marla", r"marlas?\b|मरला"),
    ("ground", r"grounds?\b"),
    ("are", r"\bares?\b|आर"),
]

LAND_CLASS_HINTS = ["irrigated", "unirrigated", "dry", "wet", "nanjai",
                    "punjai", "agricultural", "agriculture", "residential",
                    "commercial", "barren", "waste", "forest", "orchard",
                    "garden", "industrial", "govt", "government",
                    "सिंचित", "असिंचित", "कृषि", "आवासीय", "बंजर"]
OWNERSHIP_HINTS = ["single", "joint", "individual", "co-owner", "coowner",
                   "government", "trust", "company", "institutional",
                   "एकल", "संयुक्त", "सरकारी", "व्यक्तिगत"]

# Office boiler-plate and enum values that must never become a name/place.
_STOP_VALUES = {
    "office", "government", "department", "form", "code", "revenue",
    "tehsildar", "collector", "signature", "seal", "date", "value",
    "कार्यालय", "शासन", "विभाग", "सरकार", "तहसीलदार", "पटवारी", "हस्ताक्षर",
    "मुहर", "एकल", "संयुक्त", "सिंचित", "असिंचित", "कृषि", "आवासीय",
    "single", "joint", "irrigated", "agricultural", "residential",
}

# Boiler-plate that must never become a value.
_JUNK_RE = re.compile(
    r"(?i)^(?:n/?a|nil|none|-+|not\s*available|value|particulars|details)$")
_HEADING_RE = re.compile(
    r"(?i)government\s+of|department\s+of|office\s+of|revenue\s+code|"
    r"form\s+no|prescribed|record\s+of\s+rights|extract|certificate")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _ensure_geometry(lines: list[dict]) -> list[dict]:
    """Guarantee every word has x/y/w/h/conf/page.

    Real Tesseract output always carries boxes, but the extractor is also fed
    from plain text (PDF text layers, tests, pasted OCR dumps).  Synthesising
    a fixed-width layout keeps the layout logic working instead of crashing.
    """
    out = []
    y = 0
    for idx, line in enumerate(lines):
        words = list(line.get("words") or
                     [{"text": t} for t in (line.get("text") or "").split()])
        if not words:
            continue
        needs = any("x" not in w or "y" not in w for w in words)
        fixed = []
        col = 0
        for w in words:
            w = dict(w)
            w.setdefault("conf", line.get("conf", 60.0))
            w.setdefault("page", line.get("page", 0))
            if needs:
                w["x"] = col * 10
                w["y"] = y
                w["w"] = max(len(w.get("text", "")), 1) * 10
                w["h"] = 26
                col += len(w.get("text", "")) + 1
            fixed.append(w)
        if needs:
            y += 38
        new_line = dict(line)
        new_line["words"] = fixed
        new_line.setdefault("text", " ".join(w.get("text", "") for w in fixed))
        new_line.setdefault("page", fixed[0].get("page", 0))
        out.append(new_line)
    return out


def _line_geom(line: dict) -> tuple[int, int, int, int]:
    ws = line.get("words") or []
    if not ws:
        return 0, 0, 0, 0
    x0 = min(w["x"] for w in ws)
    x1 = max(w["x"] + w["w"] for w in ws)
    y0 = min(w["y"] for w in ws)
    y1 = max(w["y"] + w["h"] for w in ws)
    return x0, y0, x1, y1


def _page_of(line: dict) -> int:
    ws = line.get("words") or []
    return line.get("page", ws[0].get("page", 0) if ws else 0)


# ---------------------------------------------------------------------------
# Label matching (steps 2 + 3)
# ---------------------------------------------------------------------------
class LabelHit:
    __slots__ = ("field", "start", "end", "score", "spec", "alias", "line",
                 "kind", "inline_rest")

    def __init__(self, field, start, end, score, spec, alias, line,
                 kind="exact", inline_rest=""):
        self.field, self.start, self.end = field, start, end
        self.score, self.spec, self.alias = score, spec, alias
        self.line, self.kind, self.inline_rest = line, kind, inline_rest

    @property
    def strength(self) -> float:
        """Combined label-match quality in [0, 1]."""
        return self.score * self.spec

    def __repr__(self):  # pragma: no cover - debugging aid
        return (f"<LabelHit {self.field} {self.alias!r} "
                f"{self.score:.2f}x{self.spec} toks {self.start}:{self.end}>")


def _match_alias(folded: str, tokens: tuple[str, ...] | None = None):
    """Return [(field, score, specificity, alias)] matching a folded window.

    Two comparison modes are combined:

    * whole-window similarity  ("ownername" vs "0wnername")
    * word-aligned similarity  ("ownr"+"nane" vs "owner"+"name") -- this is
      what saves labels where *each* word picked up its own OCR error.
    """
    out = []
    n = len(folded)
    if n < _MIN_ALIAS_LEN or n > _MAX_ALIAS_LEN + 6:
        return out
    for field, aliases in ALIAS_INDEX.items():
        best = None
        for afold, alias, spec, words in aliases:
            score = 0.0
            if folded == afold:
                score = 1.0
            else:
                # A *fuzzy* (non-exact) whole-window match against a
                # multi-word alias is only trusted when the candidate window
                # actually has that many tokens. Without this, a single
                # generic word close in raw characters to a specific
                # multi-word alias -- e.g. "number:" vs "sy number" (ratio
                # 0.86, clears FUZZY_THRESHOLD) or "document" vs
                # "document no" (0.89) -- can hijack a field from a
                # completely unrelated line (a reference/document number
                # elsewhere on the page getting read as the survey number).
                # An exact fold match is unaffected: that already means the
                # OCR text reproduced the alias verbatim (e.g. a single
                # glued token "SurveyNo" folding to the same string as
                # "survey no"), which is a real, safe match.
                window_ok = len(words) <= 1 or (tokens and len(tokens) == len(words))
                if window_ok and abs(len(afold) - n) <= max(2, int(len(afold) * 0.34)):
                    thr = (FUZZY_THRESHOLD_SHORT if len(afold) <= 4
                           else FUZZY_THRESHOLD_MID if len(afold) == 5
                           else FUZZY_THRESHOLD_MID6 if len(afold) == 6
                           else FUZZY_THRESHOLD)
                    r = _ratio(folded, afold)
                    if r >= thr:
                        score = r
                        if len(folded) - len(afold) >= 2:
                            # "tehsildar" vs "tehsil": a different word
                            score *= 0.78
                # word-aligned mode
                if (tokens and len(tokens) == len(words) and len(words) > 1
                        and score < 0.999):
                    per = [_ratio(t, w) for t, w in zip(tokens, words)]
                    if min(per) >= 0.70 and sum(per) / len(per) >= 0.80:
                        score = max(score, sum(per) / len(per) * 0.98)
            if score <= 0:
                continue
            cand = (field, score, spec, alias)
            if best is None or cand[1] * cand[2] > best[1] * best[2]:
                best = cand
        if best:
            out.append(best)
    return out


def _find_labels_in_line(line: dict) -> list[LabelHit]:
    """Exact + fuzzy label spotting inside a single OCR line."""
    toks = line.get("words") or []
    folded = [_fold(w["text"]) for w in toks]
    raw = [w["text"] for w in toks]
    hits: list[LabelHit] = []
    n = len(toks)

    def _window_ok(i, size):
        """A label window may not start/end on punctuation and may not end on
        a token containing digits -- otherwise a fuzzy match happily swallows
        the value ("Survey No : 88" -> label "survey no 88", value lost)."""
        if not folded[i] or not folded[i + size - 1]:
            return False
        last = raw[i + size - 1]
        if any(ch.isdigit() for ch in last):
            return False
        return True

    for size in (4, 3, 2, 1):
        for i in range(0, n - size + 1):
            if not _window_ok(i, size):
                continue
            window = "".join(folded[i:i + size])
            if len(window) < _MIN_ALIAS_LEN:
                continue
            for field, score, spec, alias in _match_alias(
                    window, tuple(folded[i:i + size])):
                hits.append(LabelHit(field, i, i + size, score, spec, alias,
                                     line, "exact" if score == 1.0 else "fuzzy"))

    # ---- glued label+value, e.g. "SurveyNo:123/4" or "District:Chennai" ----
    for i, tok in enumerate(raw):
        f = folded[i]
        if len(f) < 6 or not re.search(r"[A-Za-z\u0900-\u0DFF]", tok):
            continue
        if not re.search(r"[\d:]", tok):
            continue
        best_glued = None
        for cut in range(_MIN_ALIAS_LEN, min(len(f), _MAX_ALIAS_LEN + 2)):
            prefix = f[:cut]
            for field, score, spec, alias in _match_alias(prefix):
                # map the folded cut back to a raw offset
                kept, raw_cut = 0, len(tok)
                for pos, ch in enumerate(_nfkc(tok)):
                    if ch.isalnum():
                        kept += 1
                    if kept == cut:
                        raw_cut = pos + 1
                        break
                rest = _clean_text(tok[raw_cut:])
                if not rest:
                    continue
                cand = (score * spec, field, score, spec, alias, rest)
                if best_glued is None or cand[0] > best_glued[0]:
                    best_glued = cand
        if best_glued:
            _s, field, score, spec, alias, rest = best_glued
            hits.append(LabelHit(field, i, i + 1, score * 0.95, spec, alias,
                                 line, "glued", rest))

    # ---- overlap resolution: strongest match wins, longer span breaks
    # ties (so an exact 3-token "classification of land" beats a 4-token
    # fuzzy match that would have eaten the first value word).
    hits.sort(key=lambda h: (-(h.strength + 0.002 * (h.end - h.start)),
                             h.start))
    chosen: list[LabelHit] = []
    taken: set[int] = set()
    for h in hits:
        span = set(range(h.start, h.end))
        if span & taken:
            continue
        chosen.append(h)
        taken |= span
    chosen.sort(key=lambda h: h.start)
    # A "glued" hit that sits *after* a real label is almost always part of
    # the value itself (e.g. "Registration No  REG-2024/44"), so it must not
    # truncate that value.
    first_real = next((h.start for h in chosen if h.kind != "glued"), None)
    if first_real is not None:
        chosen = [h for h in chosen
                  if h.kind != "glued" or h.start <= first_real]
    return chosen


def _index_labels(lines: list[dict]) -> list[list[LabelHit]]:
    return [_find_labels_in_line(line) for line in lines]


def _is_label_only_line(line: dict, hits: list[LabelHit]) -> bool:
    toks = line.get("words") or []
    if not toks or not hits:
        return False
    covered = sum(h.end - h.start for h in hits)
    punct = sum(1 for w in toks if not _fold(w["text"]))
    return covered + punct >= len(toks)


# ---------------------------------------------------------------------------
# 5. Typed parsers
# ---------------------------------------------------------------------------
def _smart_case(text: str) -> str:
    if not text:
        return text
    if text.isupper() or text.islower():
        return " ".join(w[:1].upper() + w[1:].lower() for w in text.split())
    return text


def detect_unit(text: str) -> str:
    low = _nfkc(text).lower()
    for name, pattern in AREA_UNIT_PATTERNS:
        if re.search(pattern, low):
            return name
    return ""


_SPLIT_DECIMAL_RE = re.compile(r"\b(\d{1,3})\s+(\d{2})\s*(?=[A-Za-z\u0900-\u0DFF])")


def parse_area(text: str, require_unit: bool = False):
    """-> (value: float|None, unit: str, quality: float)."""
    text = _nfkc(text or "")
    if detect_unit(text):
        # "2 50 cents" -- Tesseract routinely drops the decimal point
        text = _SPLIT_DECIMAL_RE.sub(r"\1.\2 ", text)
    for m in NUM_RE.finditer(text):
        raw = m.group(0)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if value <= 0 or value >= 1e8:
            continue
        after = text[m.end():m.end() + 24]
        unit = detect_unit(after) or detect_unit(text[max(0, m.start() - 16):m.start()])
        if require_unit and not unit:
            continue
        quality = 1.0 if unit else 0.6
        return value, unit, quality
    return None, "", 0.0


def parse_date(text: str):
    """Recognise 12/05/2024, 12-05-2024, 12.05.2024, 12 May 2024,
    May 12 2024 and 2024-05-12 -> canonical DD/MM/YYYY."""
    text = _nfkc(text or "")

    def _ok(d, mo, y):
        if y < 100:
            y += 2000 if y < 50 else 1900
        if not (1 <= mo <= 12 and 1 <= d <= 31 and 1850 <= y <= 2100):
            return None
        return f"{d:02d}/{mo:02d}/{y:04d}"

    m = DATE_ISO_RE.search(text)
    if m:
        got = _ok(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if got:
            return got
    m = DATE_NUM_RE.search(text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        got = _ok(d, mo, y)
        if got is None and 1 <= d <= 12:      # tolerate MM/DD/YYYY
            got = _ok(mo, d, y)
        if got:
            return got
    m = DATE_TXT_RE.search(text)
    if m and m.group(2)[:3].lower() in MONTHS:
        got = _ok(int(m.group(1)), MONTHS[m.group(2)[:3].lower()],
                  int(m.group(3)))
        if got:
            return got
    m = DATE_TXT2_RE.search(text)
    if m and m.group(1)[:3].lower() in MONTHS:
        got = _ok(int(m.group(2)), MONTHS[m.group(1)[:3].lower()],
                  int(m.group(3)))
        if got:
            return got
    return None


def parse_plot_id(text: str):
    """-> (id, quality). Handles 123, 123/4, 123-A, 123/4A, S.No 123/4."""
    text = _nfkc(text or "")
    text = re.sub(r"(?i)^\s*(?:no|number|nos)\b[.:\s]*", "", text.strip())
    if not any(ch.isdigit() for ch in text):
        return "", 0.0
    m = PLOT_ID_RE.search(text)
    if not m:
        cleaned = _clean_text(text)
        return (cleaned[:40], 0.4) if cleaned else ("", 0.0)
    value = re.sub(r"\s*([/-])\s*", r"\1", m.group(0)).strip()
    quality = 1.0 if re.match(r"^\d", value) else 0.8
    return value, quality


_NAME_TOKEN_RE = re.compile(r"^[A-Za-z\u0900-\u0DFF][A-Za-z\u0900-\u0DFF.'’\-]*$")


def parse_name(text: str, max_tokens: int = 5):
    """-> (name, quality).

    Rejects numeric junk, office boiler-plate and the character soup a
    badly degraded scan produces: tokens are accepted only while they look
    like name tokens, so "ae, 4 es eS KAILASHCHAND" yields nothing rather
    than a fake value.
    """
    text = _clean_text(text or "")
    text = re.sub(r"(?i)^(?:shri|sri|smt|mr|mrs|ms|thiru|tmt)\.?\s+", "", text)
    text = re.sub(r"(?i)\b(?:s/o|w/o|d/o)\b.*$", "", text).strip()
    if not text or _JUNK_RE.match(text) or _HEADING_RE.search(text):
        return "", 0.0

    kept, dropped, singles = [], 0, 0
    for tok in text.split():
        tok = tok.strip(",;|")
        if not tok:
            continue
        if not _NAME_TOKEN_RE.match(tok) or tok.lower() in _STOP_VALUES:
            dropped += 1
            break
        if len(tok) == 1:
            singles += 1
            if singles > 1:
                dropped += 1
                break
        kept.append(tok)
        if len(kept) >= max_tokens:
            break
    if not kept:
        return "", 0.0
    value = " ".join(kept)
    if sum(ch.isalpha() for ch in value) < 3:
        return "", 0.0
    quality = 1.0 if not dropped else 0.7
    if len(kept) > 4:
        quality *= 0.8
    return _smart_case(value)[:120], quality


def parse_pin(text: str):
    m = PIN_RE.search(_nfkc(text or ""))
    if not m:
        return "", 0.0
    return m.group(1).replace(" ", ""), 1.0


def parse_free_text(text: str, hints=None, max_len=160):
    text = _clean_text(text or "")
    if not text or _JUNK_RE.match(text) or _HEADING_RE.search(text):
        return "", 0.0
    text = text[:max_len]
    quality = 0.75
    if hints and any(h in text.lower() for h in hints):
        quality = 1.0
    return _smart_case(text), quality


FIELD_TYPES = {
    "owner_name": "name", "father_name": "name",
    "survey_number": "id", "subdivision_number": "id",
    "khasra_number": "id", "khata_number": "id",
    "mutation_number": "docid", "registration_number": "docid",
    "plot_area": "area", "area_unit": "unit",
    "pincode": "pin", "address": "text",
    "village": "place", "tehsil": "place", "district": "place",
    "state": "place",
    "land_classification": "class", "ownership_type": "class",
    "mutation_date": "date", "registration_date": "date",
}


def parse_value(field: str, text: str):
    """Typed parse -> (value, quality, extra) where extra may hold a unit."""
    kind = FIELD_TYPES.get(field, "text")
    if kind == "name":
        v, q = parse_name(text)
        return v, q, {}
    if kind in ("id",):
        v, q = parse_plot_id(text)
        return v, q, {}
    if kind == "docid":
        raw = _clean_text(re.split(r"(?i)\b(?:date|dated|dt)\b|दिनांक", text)[0])
        raw = re.sub(r"(?i)^(?:no|number)\b[.:\s]*", "", raw)
        if not raw or _JUNK_RE.match(raw):
            return "", 0.0, {}
        return raw[:40], (1.0 if re.search(r"\d", raw) else 0.5), {}
    if kind == "area":
        v, unit, q = parse_area(text)
        return (f"{v:g}" if v is not None else ""), q, {"unit": unit}
    if kind == "unit":
        u = detect_unit(text)
        return u, (1.0 if u else 0.0), {}
    if kind == "pin":
        v, q = parse_pin(text)
        return v, q, {}
    if kind == "date":
        v = parse_date(text)
        return (v or ""), (1.0 if v else 0.0), {}
    if kind == "place":
        v, q = parse_name(text, max_tokens=4)
        return v[:120], q, {}
    if kind == "class":
        hints = (LAND_CLASS_HINTS if field == "land_classification"
                 else OWNERSHIP_HINTS)
        v, q = parse_free_text(text, hints, 120)
        return v, q, {}
    v, q = parse_free_text(text, None, 300)
    return v, q, {}


# ---------------------------------------------------------------------------
# 4. Contextual value location
# ---------------------------------------------------------------------------
def _tokens_text(words) -> str:
    return _clean_text(" ".join(w["text"] for w in words))


def _mean_conf(words, fallback=0.55) -> float:
    if not words:
        return fallback
    return mean(min(max(w.get("conf", 60.0), 0.0), 100.0) for w in words) / 100.0


def _inline_candidate(hit: LabelHit, hits: list[LabelHit]):
    """Value to the right of the label, cut at the next label on the line."""
    toks = hit.line.get("words") or []
    if hit.kind == "glued":
        w = toks[hit.start]
        return hit.inline_rest, [w], "inline (label and value joined)"
    nxt = len(toks)
    for other in hits:
        if other.start >= hit.end:
            nxt = min(nxt, other.start)
    words = toks[hit.end:nxt]
    text = _tokens_text(words)
    if not text:
        return "", [], ""
    return text, words, "inline value right of label"


def _column_candidate(hit: LabelHit, lines, line_hits, idx):
    """Table layout: value cell on the same row, further right, but OCR put
    it in a different line group."""
    lx0, ly0, lx1, ly1 = _line_geom(hit.line)
    toks = hit.line.get("words") or []
    if not toks:
        return "", [], ""
    label_right = toks[hit.end - 1]["x"] + toks[hit.end - 1]["w"]
    ymid = (ly0 + ly1) / 2.0
    height = max(ly1 - ly0, 1)
    page = _page_of(hit.line)
    best = None
    for j, other in enumerate(lines):
        if j == idx or _page_of(other) != page:
            continue
        ox0, oy0, ox1, oy1 = _line_geom(other)
        omid = (oy0 + oy1) / 2.0
        if abs(omid - ymid) > 0.6 * height:
            continue
        if ox0 < label_right:
            continue
        if _is_label_only_line(other, line_hits[j]):
            continue
        if best is None or ox0 < _line_geom(lines[best])[0]:
            best = j
    if best is None:
        return "", [], ""
    words = lines[best]["words"]
    return _tokens_text(words), words, "table column (same row, right cell)"


def _below_candidates(hit: LabelHit, lines, line_hits, idx):
    """Yields each plausible value line below the label, nearest first.

    A wrapped multi-column table header ("Survey No" / "Owner Name" / ...
    each splitting onto a second OCR line before the real data row even
    starts) means the line immediately below a label is sometimes still
    header text, not data -- see app docs / the real-scan regression this
    guards against. Yielding every candidate line (instead of only the
    first) lets the caller keep trying until one actually parses for the
    field's type, rather than committing to the first non-empty text.
    """
    toks = hit.line.get("words") or []
    if not toks:
        return
    lx0 = toks[hit.start]["x"]
    lx1 = toks[hit.end - 1]["x"] + toks[hit.end - 1]["w"]
    _a, ly0, _b, ly1 = _line_geom(hit.line)
    height = max(ly1 - ly0, 1)
    page = _page_of(hit.line)
    for j in range(idx + 1, min(idx + 4, len(lines))):
        other = lines[j]
        if _page_of(other) != page:
            break
        ox0, oy0, ox1, oy1 = _line_geom(other)
        if oy0 < ly1 - 0.3 * height:
            continue
        if oy0 - ly1 > 2.5 * height:
            break
        other_hits = line_hits[j]
        # A line that starts with a label belongs to the next field.
        if other_hits and other_hits[0].start == 0:
            return
        overlap = min(lx1, ox1) - max(lx0, ox0)
        if overlap <= 0 and not (ox0 <= lx1 and ox1 >= lx0):
            continue
        words = [w for w in other["words"]
                 if w["x"] + w["w"] > lx0 - 0.5 * height and w["x"] < lx1 + 4 * height]
        if not words:
            words = other["words"]
        text = _tokens_text(words)
        if text:
            yield text, words, "value on line below label"


def _below_candidate(hit: LabelHit, lines, line_hits, idx):
    """First candidate only -- kept for callers that don't need the retry
    loop (e.g. ad-hoc scripts/tests); extract_fields uses
    `_below_candidates` directly so it can keep trying lines below a
    wrapped header until one actually parses."""
    for cand in _below_candidates(hit, lines, line_hits, idx):
        return cand
    return "", [], ""


# ---------------------------------------------------------------------------
# 6. Candidate scoring
# ---------------------------------------------------------------------------
POSITION_WEIGHT = {
    "inline value right of label": 1.0,
    "inline (label and value joined)": 0.97,
    "table column (same row, right cell)": 0.96,
    "value on line below label": 0.9,
}


def _label_quality(hit: LabelHit) -> tuple[float, str]:
    if hit.kind == "exact" and hit.score == 1.0:
        return hit.spec, f"exact label '{hit.alias}'"
    return hit.score * hit.spec * 0.97, f"fuzzy label '{hit.alias}' ({hit.score:.2f})"


def _score(field, value, quality, hit, words, position):
    ocr_conf = _mean_conf(words)
    lq, lq_desc = _label_quality(hit)
    pos_w = POSITION_WEIGHT.get(position, 0.85)
    conf = ocr_conf * (0.55 + 0.45 * lq) * (0.55 + 0.45 * quality) * pos_w
    conf = max(0.05, min(0.99, conf))     # a value always keeps a floor score
    method = f"{lq_desc} + {position}"
    return round(conf, 3), method


# ---------------------------------------------------------------------------
# Contextual / pattern-only fallbacks (no label found)
# ---------------------------------------------------------------------------
def _known_places():
    from .validate import STATE_DISTRICTS
    states = list(STATE_DISTRICTS)
    districts = {d: s for s, ds in STATE_DISTRICTS.items() for d in ds}
    return states, districts


def _pattern_fallbacks(text: str, lines, line_hits, found: dict) -> dict:
    """Entity extraction that does not depend on a label being present."""
    out = {}
    low = text.lower()

    if not found.get("pincode"):
        v, q = parse_pin(text)
        if v:
            out["pincode"] = (v, 0.45, "pattern match: 6-digit PIN in text")

    states, districts = _known_places()
    if not found.get("state"):
        for st in states:
            if st.lower() in low:
                out["state"] = (st, 0.5,
                                "gazetteer match: state name found in text")
                break
    if not found.get("district"):
        for dist in districts:
            if re.search(r"\b" + re.escape(dist.lower()) + r"\b", low):
                out["district"] = (dist, 0.45,
                                   "gazetteer match: district name in text")
                break

    if not found.get("plot_area"):
        v, unit, q = parse_area(text, require_unit=True)
        if v is not None:
            out["plot_area"] = (f"{v:g}", 0.45,
                                "pattern match: number followed by an area unit")
            if unit and not found.get("area_unit"):
                out["area_unit"] = (unit, 0.45,
                                    "pattern match: area unit next to number")

    # dates that sit on the same line as their identifier
    for key, words in (("mutation_date", ("mutation", "दाखिल", "नामांतरण")),
                       ("registration_date", ("regist", "registry", "deed",
                                              "पंजीकरण", "रजिस्ट्री"))):
        if found.get(key):
            continue
        for line in lines:
            lt = line["text"].lower()
            if any(w in lt for w in words):
                d = parse_date(line["text"])
                if d:
                    out[key] = (d, round(max(0.05, _mean_conf(line["words"])
                                              * 0.8), 3),
                                "contextual: date on the same line as the "
                                "identifier")
                    break
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def extract_fields(ocr: dict, apply_learning: bool = True) -> dict:
    """Extract structured land-record fields from an OCR bundle.

    Returns ``{field: {value, confidence, source, method, needs_review,
    evidence}}`` for every field in ``FIELD_KEYS`` (empty values included so
    downstream stages always see the full schema).
    """
    from records.constants import REQUIRED_FIELDS

    threshold = getattr(settings, "OCR_CONFIDENCE_THRESHOLD", 0.75)
    try:
        lines = _ensure_geometry(list(ocr.get("lines") or []))
    except Exception:                     # malformed OCR bundle
        lines = []
    text = ocr.get("text") or "\n".join(l.get("text", "") for l in lines)

    result = {}
    if not lines:
        return _finalise({}, threshold, REQUIRED_FIELDS)

    line_hits = _index_labels(lines)

    # ---- collect every (field -> candidate) ------------------------------
    candidates: dict[str, list[tuple]] = {}
    date_context: list[tuple[int, LabelHit, str]] = []

    for idx, line in enumerate(lines):
        hits = line_hits[idx]
        for hit in hits:
            if hit.field == "_header":
                continue
            if hit.field == "_date":
                date_context.append((idx, hit, ""))
                continue

            value_text, words, position = _inline_candidate(hit, hits)
            if not value_text:
                value_text, words, position = _column_candidate(
                    hit, lines, line_hits, idx)
            if not value_text:
                # Try every plausible line below in turn (nearest first)
                # instead of committing to the first non-empty one -- a
                # wrapped multi-column header can put non-value text on the
                # very next line, and only a later line holds real data.
                value_text, words, position = "", [], ""
                for cand_text, cand_words, cand_pos in _below_candidates(
                        hit, lines, line_hits, idx):
                    cand_value, cand_quality, _ = parse_value(hit.field, cand_text)
                    if cand_value and cand_quality > 0:
                        value_text, words, position = cand_text, cand_words, cand_pos
                        break
            if not value_text:
                continue

            value, quality, extra = parse_value(hit.field, value_text)
            if not value or quality <= 0:
                continue
            conf, method = _score(hit.field, value, quality, hit, words,
                                  position)
            candidates.setdefault(hit.field, []).append(
                (conf, value, method, quality, hit, extra, line["text"]))

            if extra.get("unit"):
                unit_conf = round(conf * 0.98, 3)
                candidates.setdefault("area_unit", []).append(
                    (unit_conf, extra["unit"],
                     f"{method} (unit parsed from the same value)", 1.0, hit,
                     {}, line["text"]))

    # ---- bare "Date:" next to an identifier ------------------------------
    for idx, dhit, _ in date_context:
        hits = line_hits[idx]
        owner_field = None
        for other in hits:
            if other.end <= dhit.start and other.field in (
                    "mutation_number", "registration_number"):
                owner_field = other.field
        tail = " ".join(w["text"] for w in
                        (lines[idx].get("words") or [])[dhit.end:])
        d = parse_date(tail)
        if not d:
            continue
        target = ("mutation_date" if owner_field == "mutation_number"
                  else "registration_date" if owner_field == "registration_number"
                  else None)
        if target is None:
            continue
        words = (lines[idx].get("words") or [])[dhit.end:]
        conf = round(max(0.05, _mean_conf(words) * 0.92), 3)
        candidates.setdefault(target, []).append(
            (conf, d, "contextual: 'Date' label next to the "
                      f"{target.split('_')[0]} number", 1.0, dhit, {},
             lines[idx]["text"]))

    # ---- pick the best candidate per field -------------------------------
    for field, cands in candidates.items():
        cands.sort(key=lambda c: (c[0] * (0.6 + 0.4 * c[3])
                                  * (0.55 + 0.45 * c[4].strength),
                                  c[4].strength), reverse=True)
        conf, value, method, quality, hit, extra, evidence = cands[0]
        result[field] = {"value": value, "confidence": conf, "source": "ocr",
                         "method": method, "evidence": evidence[:180]}

    # ---- pattern-only fallbacks ------------------------------------------
    for field, (value, conf, method) in _pattern_fallbacks(
            text, lines, line_hits, result).items():
        if field in result and result[field]["value"]:
            continue
        result[field] = {"value": value, "confidence": conf, "source": "ocr",
                         "method": method, "evidence": ""}

    # area unit sanity: keep unit only when an area exists
    if result.get("area_unit") and not result.get("plot_area"):
        result.pop("area_unit", None)

    # ---- learned corrections (field-specific, similarity gated) ----------
    if apply_learning:
        for field in list(result):
            if field not in LEARNABLE_FIELDS or not result[field]["value"]:
                continue
            try:
                corrected = learn.apply_and_count(field, result[field]["value"])
            except Exception:            # DB unavailable -> never break OCR
                corrected = None
            if corrected and corrected != result[field]["value"]:
                result[field]["value"] = corrected
                result[field]["source"] = "learned"
                result[field]["method"] += (
                    " + verified-correction memory")
                result[field]["confidence"] = round(
                    min(0.95, result[field]["confidence"] + 0.10), 3)

    return _finalise(result, threshold, REQUIRED_FIELDS)


def _finalise(result: dict, threshold: float, required) -> dict:
    out = {}
    for field in FIELD_KEYS:
        item = result.get(field)
        if not item or not item.get("value"):
            out[field] = {"value": "", "confidence": 0.0, "source": "ocr",
                          "method": "not found in OCR text",
                          "needs_review": True,
                          "evidence": ""}
            continue
        conf = float(item.get("confidence") or 0.0)
        item["confidence"] = round(conf, 3)
        item["needs_review"] = bool(conf < threshold)
        item.setdefault("method", "")
        item.setdefault("evidence", "")
        item.setdefault("source", "ocr")
        out[field] = item
    # required-but-missing always needs review
    for field in required:
        if not out[field]["value"]:
            out[field]["needs_review"] = True
    return out
