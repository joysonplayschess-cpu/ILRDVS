"""
Shared NLP regression corpus.

Each case is a realistic OCR *text dump* (including the mistakes real scans
produce) plus the fields a human would read off the document.  Used by
``records/tests.py`` and by ``tools/nlp_bench.py``.
"""

CASES = [
    # 1 --------------------------------------------------- clean document
    {
        "name": "clean_english_patta",
        "conf": 94,
        "text": """GOVERNMENT OF TAMIL NADU
DEPARTMENT OF SURVEY AND SETTLEMENT
CHITTA / PATTA EXTRACT

District : Chennai
Taluk : Sholinganallur
Village : Perumbakkam
Patta Number : 1183
Survey No : 123/4
Subdivision No : 2
Owner Name : Ramesh Kumar
Father's Name : Krishnan
Extent : 1.25 acres
Land Type : Wet Land (Nanjai)
Ownership Type : Single
Registration No : REG-2024-4471 Date : 12/05/2024
Pin Code : 600100""",
        "expect": {
            "district": "Chennai", "tehsil": "Sholinganallur",
            "village": "Perumbakkam", "khata_number": "1183",
            "survey_number": "123/4", "subdivision_number": "2",
            "owner_name": "Ramesh Kumar", "father_name": "Krishnan",
            "plot_area": "1.25", "area_unit": "acre",
            "ownership_type": "Single",
            "registration_number": "REG-2024-4471",
            "registration_date": "12/05/2024", "pincode": "600100",
        },
    },
    # 2 ------------------------------------------------- noisy OCR output
    {
        "name": "noisy_ocr_mistakes",
        "conf": 62,
        "text": """0ffice of the Tahsildar
0wner Nane : Rarnesh Kumar
Fathar Name : Krishnan
Survev No. 123/4A
Villaqe : Sholinganallur
Distrist : Chennai
Stale : Tamil Nadu
Extont : 2.50 cents
Reglstration No : 4471/2024 Dt. 12-05-2024""",
        "expect": {
            "owner_name": "Rarnesh Kumar", "father_name": "Krishnan",
            "survey_number": "123/4A", "village": "Sholinganallur",
            "district": "Chennai", "state": "Tamil Nadu",
            "plot_area": "2.5", "area_unit": "cent",
            "registration_number": "4471/2024",
            "registration_date": "12/05/2024",
        },
    },
    # 3 ------------------------------------------ label / value stacked
    {
        "name": "value_below_label",
        "conf": 90,
        "text": """Owner Name
Ramesh Kumar
Father Name
Krishnan
Survey Number
123/4
Village
Sholinganallur
District
Chennai
Area
1.25 acre""",
        "expect": {
            "owner_name": "Ramesh Kumar", "father_name": "Krishnan",
            "survey_number": "123/4", "village": "Sholinganallur",
            "district": "Chennai", "plot_area": "1.25", "area_unit": "acre",
        },
    },
    # 4 ------------------------------------- several fields on one line
    {
        "name": "multi_field_lines",
        "conf": 88,
        "text": """District: Chennai   Village: Sholinganallur   Taluk: Tambaram
Survey No: 45/2   Khata No: 771   Area: 0.50 hectare
Owner: Lakshmi Narayanan   Husband Name: Narayanan""",
        "expect": {
            "district": "Chennai", "village": "Sholinganallur",
            "tehsil": "Tambaram", "survey_number": "45/2",
            "khata_number": "771", "plot_area": "0.5",
            "area_unit": "hectare", "owner_name": "Lakshmi Narayanan",
            "father_name": "Narayanan",
        },
    },
    # 5 ------------------------------------------------ table document
    {
        "name": "table_layout",
        "conf": 91,
        "text": """Field                      Value
Khatedar                   Muthu Vel
S/o                        Arumugam
S.No.                      123/4A
Sub Division No            2B
Village                    Injambakkam
Taluk                      Sholinganallur
District                   Chengalpattu
State                      Tamil Nadu
Extent                     1250 sq.ft
Classification of Land     Dry Land
Patta No                   4471""",
        "expect": {
            "owner_name": "Muthu Vel", "father_name": "Arumugam",
            "survey_number": "123/4A", "subdivision_number": "2B",
            "village": "Injambakkam", "tehsil": "Sholinganallur",
            "district": "Chengalpattu", "state": "Tamil Nadu",
            "plot_area": "1250", "area_unit": "sq.ft",
            "land_classification": "Dry Land", "khata_number": "4471",
        },
    },
    # 6 -------------------------------------------------- missing fields
    {
        "name": "missing_fields",
        "conf": 89,
        "text": """Village : Kelambakkam
Survey No : 88
Owner : Suresh Babu""",
        "expect": {
            "village": "Kelambakkam", "survey_number": "88",
            "owner_name": "Suresh Babu",
            "father_name": "", "khasra_number": "", "mutation_number": "",
        },
    },
    # 7 ----------------------------------------- label spelling variants
    {
        "name": "label_synonyms_and_typos",
        "conf": 70,
        "text": """Ownr Name : Anitha Devi
Land Owner : Anitha Devi
Patta Holder : Anitha Devi
Khatedar : Anitha Devi
Ownar Name : Anitha Devi
Fathar's Name : Devaraj
Villaqe : Medavakkam
Dist : Chennai""",
        "expect": {
            "owner_name": "Anitha Devi", "father_name": "Devaraj",
            "village": "Medavakkam", "district": "Chennai",
        },
    },
    # 8 ------------------------------------------ survey number formats
    {
        "name": "survey_formats_1",
        "conf": 92,
        "text": "Survey No: 123/4\nVillage: Adyar",
        "expect": {"survey_number": "123/4", "village": "Adyar"},
    },
    {
        "name": "survey_formats_2",
        "conf": 92,
        "text": "S.No. 123-A\nVillage: Adyar",
        "expect": {"survey_number": "123-A"},
    },
    {
        "name": "survey_formats_3",
        "conf": 92,
        "text": "Survey Number 123/4A\nSub Division No: 5",
        "expect": {"survey_number": "123/4A", "subdivision_number": "5"},
    },
    {
        "name": "survey_formats_4",
        "conf": 92,
        "text": "SurveyNo:123\nVillage: Adyar",
        "expect": {"survey_number": "123"},
    },
    {
        # Regression: PLOT_ID_RE used [0-9] (ASCII-only) after the '/'
        # separator while using \d (Unicode-aware) before it, so a
        # Devanagari-numeral plot id like "४७/२" got silently truncated to
        # "४७". Hindi/regional-script land records routinely use
        # native-script digits for khasra/khata/survey numbers.
        "name": "devanagari_numeral_plot_id",
        "conf": 92,
        "text": "खसरा नंबर: ४७/२\nखाता नंबर: ०१९८\nसर्वे नंबर: ४७/२",
        "expect": {"khasra_number": "४७/२", "khata_number": "०१९८",
                   "survey_number": "४७/२"},
    },
    # 9 --------------------------------------------------- area formats
    {"name": "area_acre", "conf": 92,
     "text": "Extent : 1.25 acre", "expect": {"plot_area": "1.25", "area_unit": "acre"}},
    {"name": "area_acres", "conf": 92,
     "text": "Area : 1.25 acres", "expect": {"plot_area": "1.25", "area_unit": "acre"}},
    {"name": "area_hectare", "conf": 92,
     "text": "Area of Plot : 0.50 hectare", "expect": {"plot_area": "0.5", "area_unit": "hectare"}},
    {"name": "area_sqft_dot", "conf": 92,
     "text": "Extent : 1250 sq.ft", "expect": {"plot_area": "1250", "area_unit": "sq.ft"}},
    {"name": "area_sqft_space", "conf": 92,
     "text": "Extent : 1250 sq ft", "expect": {"plot_area": "1250", "area_unit": "sq.ft"}},
    {"name": "area_cents", "conf": 92,
     "text": "Extent : 2.50 cents", "expect": {"plot_area": "2.5", "area_unit": "cent"}},
    # 10 -------------------------------------------------- date formats
    {"name": "date_slash", "conf": 92,
     "text": "Registration No : R-99 Date : 12/05/2024",
     "expect": {"registration_date": "12/05/2024", "registration_number": "R-99"}},
    {"name": "date_dash", "conf": 92,
     "text": "Registration Date : 12-05-2024",
     "expect": {"registration_date": "12/05/2024"}},
    {"name": "date_dot", "conf": 92,
     "text": "Date of Registration : 12.05.2024",
     "expect": {"registration_date": "12/05/2024"}},
    {"name": "date_text", "conf": 92,
     "text": "Mutation No : M-77 Mutation Date : 12 May 2024",
     "expect": {"mutation_date": "12/05/2024", "mutation_number": "M-77"}},
    # 11 ------------------------------------------------- Hindi RoR form
    {
        "name": "hindi_form",
        "conf": 80,
        "text": """भू-अभिलेख खतौनी
जिला : भोपाल
तहसील : हुजूर
ग्राम : कलियासोत
खाता नंबर : 124
खसरा नंबर : 47/2
मालिक का नाम : राजेश शर्मा
पिता का नाम : मोहन लाल शर्मा
क्षेत्रफल : 0.85 हेक्टेयर
भूमि का प्रकार : सिंचित कृषि भूमि
पिन कोड : 462066""",
        "expect": {
            "district": "भोपाल", "tehsil": "हुजूर", "village": "कलियासोत",
            "khata_number": "124", "khasra_number": "47/2",
            "owner_name": "राजेश शर्मा", "father_name": "मोहन लाल शर्मा",
            "plot_area": "0.85", "area_unit": "hectare",
            "pincode": "462066",
        },
    },
    # 12 ------------------------------------ address + adjacent-field bleed
    {
        "name": "no_field_bleed",
        "conf": 92,
        "text": """District: Chennai
Village: Sholinganallur
Address: No 12, 2nd Main Road, Sholinganallur
Owner Name: Ramesh Kumar
Father Name: Krishnan""",
        "expect": {
            "district": "Chennai", "village": "Sholinganallur",
            "owner_name": "Ramesh Kumar", "father_name": "Krishnan",
            "address": "No 12, 2nd Main Road, Sholinganallur",
        },
    },
    # 13 ------------------------------------------- Tamil patta extract
    {
        "name": "tamil_patta",
        "conf": 86,
        "text": """தமிழ்நாடு அரசு
மாவட்டம் : சென்னை
வட்டம் : சோழிங்கநல்லூர்
கிராமம் : பெரும்பாக்கம்
பட்டா எண் : 1183
புல எண் : 123/4
பட்டாதாரர் பெயர் : ரமேஷ் குமார்
தந்தையின் பெயர் : கிருஷ்ணன்
பரப்பளவு : 1.25 ஏக்கர்
நில வகை : நன்செய்""",
        "expect": {
            "district": "சென்னை", "tehsil": "சோழிங்கநல்லூர்",
            "village": "பெரும்பாக்கம்", "khata_number": "1183",
            "survey_number": "123/4", "owner_name": "ரமேஷ் குமார்",
            "father_name": "கிருஷ்ணன்", "plot_area": "1.25",
            "area_unit": "acre", "land_classification": "நன்செய்",
        },
    },
    # 14 ------------------------------------------------- Bengali record
    {
        "name": "bengali_record",
        "conf": 84,
        "text": """জেলা : কলকাতা
মৌজা : বেহালা
খাতা নম্বর : 4411
জরিপ নম্বর : 88/2
মালিকের নাম : সৌরভ মুখার্জি
পিতার নাম : রঞ্জন মুখার্জি
জমির পরিমাণ : 0.50 হেক্টর""",
        "expect": {
            "district": "কলকাতা", "village": "বেহালা",
            "khata_number": "4411", "survey_number": "88/2",
            "owner_name": "সৌরভ মুখার্জি", "father_name": "রঞ্জন মুখার্জি",
            "plot_area": "0.5", "area_unit": "hectare",
        },
    },
    # 15 -------------------------------------------------- Telugu record
    {
        "name": "telugu_record",
        "conf": 83,
        "text": """జిల్లా : హైదరాబాద్
తాలూకా : శేరిలింగంపల్లి
గ్రామం : గచ్చిబౌలి
సర్వే నంబర్ : 47/2
యజమాని పేరు : రవి కుమార్
తండ్రి పేరు : వెంకటేష్
విస్తీర్ణం : 2.50 ఎకరం""",
        "expect": {
            "district": "హైదరాబాద్", "tehsil": "శేరిలింగంపల్లి",
            "village": "గచ్చిబౌలి", "survey_number": "47/2",
            "owner_name": "రవి కుమార్", "father_name": "వెంకటేష్",
            "plot_area": "2.5", "area_unit": "acre",
        },
    },
    # 16 ----------------------------------------------- empty OCR result
    {"name": "empty_ocr", "conf": 0, "text": "", "expect": {}},
    # 14 ---------------------------------------------- garbage OCR result
    {"name": "garbage_ocr", "conf": 18,
     "text": "~~ ,,, ||| ?? 8*# ....\n### %%%",
     "expect": {}},
]
