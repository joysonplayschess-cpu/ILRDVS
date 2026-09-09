"""
Synthetic sample land-record generator.

Creates realistic "Record of Rights" style scans (all data is FICTIONAL and
used only for demonstration) in multiple quality grades - clean, dark,
faded, noisy, damaged - so the full enhancement + OCR + verification
pipeline can be experienced offline.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

PAGE_W, PAGE_H = 1240, 1754
FONT_ROOT = Path("/usr/share/fonts/truetype")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_ROOT / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def F_EN(size):  return _font("dejavu/DejaVuSans.ttf", size)
def F_ENB(size): return _font("dejavu/DejaVuSans-Bold.ttf", size)
def F_HI(size):  return _font("lohit-devanagari/Lohit-Devanagari.ttf", size)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
INK = (26, 34, 54)
LINE = (120, 130, 150)
PAPER = (251, 249, 243)


def _center_text(d, y, text, font, fill=INK):
    w = d.textlength(text, font=font)
    d.text(((PAGE_W - w) / 2, y), text, font=font, fill=fill)


def render_record(data: dict) -> Image.Image:
    """Render one Record-of-Rights form as an image."""
    rng = random.Random(data["filename"])
    img = Image.new("RGB", (PAGE_W, PAGE_H), PAPER)
    d = ImageDraw.Draw(img)
    hindi = data.get("script") == "devanagari"
    lab = F_HI(30) if hindi else F_EN(29)
    val = F_HI(32) if hindi else F_EN(30)

    # page frame + fake seal
    d.rectangle([50, 40, PAGE_W - 50, PAGE_H - 40], outline=LINE, width=3)
    d.ellipse([80, 70, 190, 180], outline=INK, width=3)
    d.ellipse([100, 90, 170, 160], outline=INK, width=2)
    d.text((116, 112), "GOI" if not hindi else "भा.स.", font=F_ENB(26), fill=INK)

    if hindi:
        _center_text(d, 76, f"{data['state_local']} शासन", F_HI(36))
        _center_text(d, 128, "राजस्व विभाग  -  तहसील कार्यालय", F_HI(28))
        _center_text(d, 178, "अधिकार अभिलेख (खतौनी)", F_HI(40))
        _center_text(d, 236, "RECORD OF RIGHTS (KHATAUNI)", F_ENB(24))
    else:
        _center_text(d, 76, f"GOVERNMENT OF {data['state'].upper()}", F_ENB(36))
        _center_text(d, 126, "REVENUE DEPARTMENT - OFFICE OF THE TEHSILDAR", F_EN(26))
        _center_text(d, 178, "RECORD OF RIGHTS (KHATAUNI)", F_ENB(40))
        _center_text(d, 236, "Form Prescribed under the State Land Revenue Code", F_EN(22))

    d.line([70, 286, PAGE_W - 70, 286], fill=INK, width=3)

    # meta line (extra OCR chatter, realistic registers have these)
    d.text((80, 302), f"Doc No: {data['doc_no']}   Register Page "
                      f"{rng.randint(3, 90)} of {rng.randint(100, 240)}",
           font=F_EN(22), fill=INK)

    # ---- field table -----------------------------------------------------
    rows = data["rows"]
    y0, rh = 350, 58
    x_l0, x_l1, x_r = 70, 520, PAGE_W - 70
    for i, (label, value) in enumerate(rows):
        y = y0 + i * rh
        d.line([x_l0, y, x_r, y], fill=LINE, width=2)
        d.text((90, y + 12), label, font=lab, fill=INK)
        d.text((x_l1 + 20, y + 10), value, font=val, fill=INK)
    yb = y0 + len(rows) * rh
    d.line([x_l0, yb, x_r, yb], fill=LINE, width=2)
    for x in (x_l0, x_l1, x_r):
        d.line([x, y0, x, yb], fill=LINE, width=2)

    # ---- footer ----------------------------------------------------------
    fy = yb + 60
    cert = ("Pramanit hai ki uparyukt pravisthi panjika ki satay prati hai."
            if hindi else
            "Certified that the above entries are a true copy of the record "
            "maintained in this office.")
    d.text((80, fy), cert, font=F_HI(24) if hindi else F_EN(22), fill=INK)
    d.text((80, fy + 60), f"Date of Issue: {data['issue_date']}", font=F_EN(22),
           fill=INK)
    d.text((80, fy + 104), f"Patwari Halka No. {rng.randint(3, 40)}", font=F_EN(22),
           fill=INK)
    d.text((PAGE_W - 420, fy + 104), "Sd/-  (A. K. Verma)", font=F_EN(24), fill=INK)
    d.text((PAGE_W - 420, fy + 140), "Signature of Lekhpal", font=F_EN(20), fill=INK)

    # fake barcode for authenticity
    bx, by = PAGE_W - 330, PAGE_H - 170
    x = bx
    for _ in range(46):
        bw = rng.choice([2, 2, 3, 4, 6])
        d.rectangle([x, by, x + bw, by + 64], fill=INK)
        x += bw + rng.choice([2, 3, 4])
        if x > bx + 240:
            break
    d.text((bx, by + 70), data["barcode"], font=F_EN(18), fill=INK)
    return img


# ---------------------------------------------------------------------------
# Degradations (simulate historical scan quality)
# ---------------------------------------------------------------------------
def degrade(img: Image.Image, quality: str, seed: int = 7) -> Image.Image:
    rng = random.Random(seed)
    if quality == "clean":
        return img.rotate(rng.uniform(-0.3, 0.3), resample=Image.BICUBIC,
                          fillcolor=PAPER)
    if quality == "dark":
        img = ImageEnhance.Brightness(img).enhance(0.52)
        img = ImageEnhance.Contrast(img).enhance(0.85)
    elif quality == "faded":
        img = ImageEnhance.Contrast(img).enhance(0.45)
        img = ImageEnhance.Brightness(img).enhance(1.22)
    elif quality == "noisy":
        arr = np.asarray(img).astype(np.int16)
        noise = np.random.default_rng(seed).normal(0, 16, arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        img = img.rotate(1.2, resample=Image.BICUBIC, fillcolor=PAPER)
    elif quality == "damaged":
        img = ImageEnhance.Contrast(img).enhance(0.55)
        img = ImageEnhance.Brightness(img).enhance(0.72)
        arr = np.asarray(img).astype(np.int16)
        noise = np.random.default_rng(seed).normal(0, 26, arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img, "RGBA")
        d.ellipse([620, 1180, 1030, 1560], fill=(160, 120, 60, 46))  # stain
        img = img.rotate(-1.8, resample=Image.BICUBIC, fillcolor=PAPER)
    # scanner speckles for everything non-clean
    d = ImageDraw.Draw(img)
    for _ in range(350 if quality != "damaged" else 900):
        x, y = rng.randrange(PAGE_W), rng.randrange(PAGE_H)
        d.point((x, y), fill=(rng.randrange(120),) * 3)
    return img


# ---------------------------------------------------------------------------
# Sample catalogue (ALL DATA FICTIONAL)
# ---------------------------------------------------------------------------
def _rows_en(d):
    return [
        ("District", d["district"]), ("Tehsil / Taluka", d["tehsil"]),
        ("Village / Mauza", d["village"]), ("State", d["state"]),
        ("Khata Number", d["khata"]), ("Khasra Number", d["khasra"]),
        ("Survey Number", d["survey"]),
        ("Owner Name", d["owner"]), ("Father's Name", d["father"]),
        ("Area of Plot", f"{d['area']} {d['unit']}"),
        ("Land Classification", d["land_class"]),
        ("Ownership Type", d["ownership"]),
        ("Mutation No.", f"{d['mutation']} Date: {d['mutation_date']}"),
        ("Registration No.", f"{d['reg']} Date: {d['reg_date']}"),
        ("Pin Code", d["pin"]),
    ]


def _rows_hi(d):
    return [
        ("जिला", d["district_hi"]), ("तहसील", d["tehsil_hi"]),
        ("ग्राम", d["village_hi"]), ("राज्य", d["state_hi"]),
        ("खाता नंबर", d["khata"]), ("खसरा नंबर", d["khasra"]),
        ("सर्वे नंबर", d["survey"]),
        ("मालिक का नाम", d["owner_hi"]), ("पिता का नाम", d["father_hi"]),
        ("रकबा", f"{d['area']} हेक्टेयर"),
        ("भूमि प्रकार", d["land_class_hi"]),
        ("मालिकी", d["ownership_hi"]),
        ("दाखिल खारिज नंबर", f"{d['mutation']} दिनांक: {d['mutation_date']}"),
        ("पंजीकरण नंबर", f"{d['reg']} दिनांक: {d['reg_date']}"),
        ("पिन कोड", d["pin"]),
    ]


SAMPLES = [
    dict(filename="UP_Varanasi_RoR_dark.pdf", quality="dark", fmt="pdf",
         script="latin", language="eng", doc_no="UP/VNS/2019/01871",
         barcode="UPVNS01871", state="Uttar Pradesh", district="Varanasi",
         tehsil="Pindra", village="Rampur Khurd", khata="0142",
         khasra="331/2", survey="331/2", owner="RAM KUMAR SINGH",
         father="SHYAM LAL SINGH", area="1.250", unit="Hectare",
         land_class="Agricultural - Irrigated", ownership="Single",
         mutation="M-2021-0187", mutation_date="14/03/2021",
         reg="REG-823144", reg_date="02/08/2019", pin="221206",
         issue_date="18/04/2022"),
    # deliberate duplicate of the plot above (rescan submitted twice)
    dict(filename="UP_Varanasi_rescan_clean.png", quality="clean", fmt="png",
         script="latin", language="eng", doc_no="UP/VNS/2019/01871-B",
         barcode="UPVNS01871B", state="Uttar Pradesh", district="Varanasi",
         tehsil="Pindra", village="Rampur Khurd", khata="0142",
         khasra="331/2", survey="331/2", owner="RAM KUMAR SINGH",
         father="SHYAM LAL SINGH", area="1.250", unit="Hectare",
         land_class="Agricultural - Irrigated", ownership="Single",
         mutation="M-2021-0187", mutation_date="14/03/2021",
         reg="REG-823144", reg_date="02/08/2019", pin="221206",
         issue_date="18/04/2022"),
    dict(filename="Bihar_Patna_faded.jpg", quality="faded", fmt="jpg",
         script="latin", language="eng", doc_no="BR/PAT/2020/00911",
         barcode="BRPAT00911", state="Bihar", district="Patna",
         tehsil="Maner", village="Bihta", khata="0059", khasra="78",
         survey="78", owner="ANITA DEVI", father="SURESH PRASAD",
         area="0.845", unit="Hectare", land_class="Agricultural",
         ownership="Joint", mutation="MUT-4471", mutation_date="09/01/2020",
         reg="D-221144", reg_date="17/06/2011", pin="801103",
         issue_date="05/05/2021"),
    dict(filename="MP_Bhopal_hindi.jpg", quality="clean", fmt="jpg",
         script="devanagari", language="hin", doc_no="MP/BPL/2021/00144",
         barcode="MPBPL00144", state="Madhya Pradesh", state_local="मध्य प्रदेश",
         district="Bhopal", district_hi="भोपाल", tehsil="Huzur",
         tehsil_hi="हुजूर", village="Kaliasot", village_hi="कलियासोत",
         khata="0214", khasra="47/2", survey="47/2",
         owner="RAM SINGH CHAUHAN", owner_hi="राम सिंह चौहान",
         father="MOHAN SINGH", father_hi="मोहन सिंह चौहान",
         area="2.100", unit="Hectare", land_class="Agricultural - Irrigated",
         land_class_hi="सिंचित कृषि भूमि", ownership="Single",
         ownership_hi="एकल", mutation="नामांतरण-118", mutation_date="22/07/2021",
         reg="पंजीकरण-55821", reg_date="11/11/2018", pin="462066",
         issue_date="02/02/2022"),
    dict(filename="Rajasthan_Jaipur_noisy.png", quality="noisy", fmt="png",
         script="latin", language="eng", doc_no="RJ/JAI/2018/03414",
         barcode="RJJAI03414", state="Rajasthan", district="Jaipur",
         tehsil="Amber", village="Kukas", khata="0311", khasra="1205/3",
         survey="1205/3", owner="KAILASH CHAND MEENA", father="GIRDHARI LAL",
         area="0.320", unit="Hectare", land_class="Barren / Uncultivable",
         ownership="Single", mutation="NM-2211", mutation_date="30/10/2018",
         reg="JPR-9912", reg_date="15/02/2017", pin="302028",
         issue_date="12/12/2019"),
    dict(filename="WB_Kolkata_clean.pdf", quality="clean", fmt="pdf",
         script="latin", language="eng", doc_no="WB/KOL/2022/00214",
         barcode="WBKOL00214", state="West Bengal", district="Kolkata",
         tehsil="Alipore", village="Behala", khata="0877", khasra="4411",
         survey="4411", owner="SOURAV MUKHERJEE", father="PRANAB MUKHERJEE",
         area="0.055", unit="Hectare", land_class="Non-Agricultural (Homestead)",
         ownership="Single", mutation="MS-2019-77", mutation_date="08/07/2019",
         reg="KOL-771201", reg_date="23/03/2015", pin="700060",
         issue_date="14/01/2023"),
    dict(filename="RJ_Sanganer_damaged.png", quality="damaged", fmt="png",
         script="devanagari", language="hin", doc_no="RJ/JAI/2016/11002",
         barcode="RJJAI11002", state="Rajasthan", state_local="राजस्थान",
         district="Jaipur", district_hi="जयपुर", tehsil="Sanganer",
         tehsil_hi="सांगानेर", village="Sanganer", village_hi="सांगानेर",
         khata="०१९८", khasra="४७/२", survey="४७/२",
         owner="GEETA DEVI", owner_hi="गीता देवी",
         father="HARI LAL", father_hi="पति श्री हरीलाल",
         area="0.410", unit="Hectare", land_class="Agricultural",
         land_class_hi="कृषि भूमि", ownership="Single", ownership_hi="एकल",
         mutation="५१/२०२२", mutation_date="12/02/2022",
         reg="जयपुर-8841", reg_date="09/09/2009", pin="303902",
         issue_date="21/03/2022"),
    dict(filename="Karnataka_Bengaluru_clean.png", quality="clean", fmt="png",
         script="latin", language="eng", doc_no="KA/BLR/2020/05510",
         barcode="KABLR05510", state="Karnataka", district="Bengaluru Urban",
         tehsil="Anekal", village="Sarjapur", khata="1102", khasra="92/1",
         survey="92/1", owner="LAKSHMI NARASIMHAN", father="RAGHAVAN IYER",
         area="0.162", unit="Hectare", land_class="Agricultural - Dry Land",
         ownership="Joint", mutation="MR H-330/2020", mutation_date="19/05/2020",
         reg="BLR-661205", reg_date="04/12/2012", pin="562125",
         issue_date="08/08/2021"),
    dict(filename="TN_Chennai_dark.jpg", quality="dark", fmt="jpg",
         script="latin", language="eng", doc_no="TN/CHE/2017/07008",
         barcode="TNCHE07008", state="Tamil Nadu", district="Chennai",
         tehsil="Guindy", village="Velachery", khata="2201", khasra="1183",
         survey="1183", owner="MURUGAN S", father="SUBRAMANIAN K",
         area="0.098", unit="Hectare", land_class="Non-Agricultural (Urban)",
         ownership="Single", mutation="PT-118", mutation_date="27/09/2018",
         reg="CHE-5520017", reg_date="13/10/2016", pin="600042",
         issue_date="16/02/2019"),
    dict(filename="Gujarat_Ahmedabad_faded.png", quality="faded", fmt="png",
         script="latin", language="eng", doc_no="GJ/AHD/2021/03071",
         barcode="GJAHD03071", state="Gujarat", district="Ahmedabad",
         tehsil="Daskroi", village="Bopal", khata="0661", khasra="312",
         survey="312", owner="HETAL BEN PATEL", father="MAHESH BHAI PATEL",
         area="0.710", unit="Hectare", land_class="Agricultural - Irrigated",
         ownership="Joint", mutation="VE-1901", mutation_date="11/04/2021",
         reg="AHD-33015", reg_date="25/01/2014", pin="380058",
         issue_date="29/09/2022"),
    dict(filename="Punjab_Ludhiana_clean.jpg", quality="clean", fmt="jpg",
         script="latin", language="eng", doc_no="PB/LDH/2019/01299",
         barcode="PBLDH01299", state="Punjab", district="Ludhiana",
         tehsil="Payal", village="Doraha", khata="0512", khasra="881/4",
         survey="881/4", owner="GURPREET SINGH", father="HARBHAJAN SINGH",
         area="1.620", unit="Hectare", land_class="Agricultural - Irrigated",
         ownership="Single", mutation="INT-512", mutation_date="05/03/2019",
         reg="LDH-90211", reg_date="18/05/2010", pin="141421",
         issue_date="24/06/2020"),
    # left deliberately UNPROCESSED by the seeder to show the lifecycle
    dict(filename="Telangana_Hyderabad_clean.png", quality="clean", fmt="png",
         script="latin", language="eng", doc_no="TS/HYD/2022/04810",
         barcode="TSHYD04810", state="Telangana", district="Hyderabad",
         tehsil="Serilingampally", village="Gachibowli", khata="1901",
         khasra="229", survey="229", owner="VENKATA RAO", father="SATYAM RAO",
         area="0.240", unit="Hectare", land_class="Non-Agricultural (Urban)",
         ownership="Single", mutation="TS-M-9012", mutation_date="02/05/2022",
         reg="HYD-8812005", reg_date="07/07/2017", pin="500032",
         issue_date="30/11/2022"),
]


def build(base_dir: Path) -> list[dict]:
    """Render all samples to disk; returns sample dicts with ``path`` added."""
    out_dir = Path(base_dir) / "seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for spec in SAMPLES:
        data = dict(spec)
        data.setdefault("state_local", data["state"])
        data.setdefault("state_hi", data["state_local"])
        data["rows"] = (_rows_hi(data) if data["script"] == "devanagari"
                        else _rows_en(data))
        img = degrade(render_record(data), data["quality"],
                      seed=len(data["filename"]))
        path = out_dir / data["filename"]
        if data["fmt"] == "pdf":
            img.convert("RGB").save(path, "PDF", resolution=150)
        elif data["fmt"] == "png":
            img.save(path)
        else:
            img.save(path, quality=88)
        data["path"] = path
        built.append(data)
    return built
