"""Smoke + unit tests for ILRDVS."""
import io

import numpy as np

from django.test import Client, TestCase
from django.urls import reverse

from records.models import Document, LandRecord, UserProfile
from records.pipeline import extract, preprocess, validate
from django.contrib.auth.models import User


class OCREngineFallbackTests(TestCase):
    """settings.OCR_ENGINE_PRIORITY = ['bhashini', 'tesseract', 'paddleocr']
    must fall through cleanly at every stage."""

    def _blank_img(self):
        img = np.full((60, 200), 255, dtype=np.uint8)
        return img

    def test_bhashini_unconfigured_falls_through_to_tesseract(self):
        from django.test import override_settings
        from records.pipeline import ocr
        with override_settings(BHASHINI_USER_ID="", BHASHINI_API_KEY=""):
            result = ocr.ocr_image(self._blank_img(), lang="eng")
        self.assertTrue(result["engine"].startswith("Tesseract"))
        self.assertIn("bhashini: skipped", result["engine_attempts"][0])

    def test_bhashini_configured_but_failing_falls_through(self):
        from unittest.mock import patch
        from django.test import override_settings
        from records.pipeline import ocr
        from records.pipeline.bhashini_ocr import BhashiniOCR
        with override_settings(BHASHINI_USER_ID="u", BHASHINI_API_KEY="k"):
            with patch.object(BhashiniOCR, "recognize", return_value=None):
                result = ocr.run_ocr(self._blank_img(), lang="eng")
        self.assertTrue(result["engine"].startswith("Tesseract"))
        self.assertIn("bhashini: skipped", result["engine_attempts"][0])

    def test_paddle_missing_does_not_break_pipeline(self):
        """With paddleocr not installed (the default in this environment),
        it must never be the reason a document fails -- Tesseract, ahead of
        it in priority, should already have produced a result."""
        from records.pipeline import ocr
        result = ocr.ocr_image(self._blank_img(), lang="eng")
        self.assertIn("engine", result)
        self.assertIn("word_count", result)

    def test_priority_order_is_bhashini_then_tesseract_then_paddleocr(self):
        from django.conf import settings
        self.assertEqual(settings.OCR_ENGINE_PRIORITY,
                         ["bhashini", "tesseract", "paddle"])


class GammaCorrectionTests(TestCase):
    def test_auto_gamma_brightens_dark_scans(self):
        dark = np.full((200, 200), 90, dtype=np.uint8)      # dark scan
        g = preprocess.auto_gamma_value(dark)
        self.assertGreater(g, 1.0)                          # should brighten
        out = preprocess.apply_gamma(dark, g)
        self.assertGreater(float(np.mean(out)), float(np.mean(dark)))

    def test_auto_gamma_darkens_washed_out_scans(self):
        pale = np.full((200, 200), 235, dtype=np.uint8)     # faded scan
        g = preprocess.auto_gamma_value(pale)
        self.assertLess(g, 1.0)
        out = preprocess.apply_gamma(pale, g)
        self.assertLess(float(np.mean(out)), float(np.mean(pale)))

    def test_manual_gamma_lut(self):
        img = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
        out = preprocess.apply_gamma(img, 2.2)
        self.assertEqual(out.shape, img.shape)
        self.assertGreaterEqual(int(out[0, 128]), int(img[0, 128]))  # brightened

    def test_preprocess_pipeline_marks_steps(self):
        rng = np.random.default_rng(0)
        page = rng.integers(120, 255, (300, 300, 3), dtype=np.uint8)
        res = preprocess.preprocess_image(page, gamma="auto")
        self.assertIn("gamma_used", res["info"])
        names = [s["name"] for s in res["info"]["steps"]]
        self.assertIn("Gamma correction", names)
        self.assertIn("Binarize", names)


class ExtractionTests(TestCase):
    OCR = {
        "text": ("District Varanasi\nTehsil / Taluka Pindra\n"
                 "Village / Mauza Rampur Khurd\nKhasra Number 331/2\n"
                 "Owner Name RAM KUMAR SINGH\nArea of Plot 1.250 Hectare\n"
                 "Mutation No. M-2021-0187 Date: 14/03/2021\nPin Code 221206"),
        "lines": [
            {"id": (0, 0, i), "text": t, "words":
                [{"text": w, "conf": 96.0} for w in t.split()]}
            for i, t in enumerate([
                "District Varanasi", "Tehsil / Taluka Pindra",
                "Village / Mauza Rampur Khurd", "Khasra Number 331/2",
                "Owner Name RAM KUMAR SINGH", "Area of Plot 1.250 Hectare",
                "Mutation No. M-2021-0187 Date: 14/03/2021",
                "Pin Code 221206"])
        ],
    }

    def test_field_extraction(self):
        f = extract.extract_fields(self.OCR, apply_learning=False)
        self.assertEqual(f["district"]["value"], "Varanasi")
        self.assertEqual(f["tehsil"]["value"], "Pindra")
        self.assertEqual(f["village"]["value"], "Rampur Khurd")
        self.assertEqual(f["khasra_number"]["value"], "331/2")
        self.assertEqual(f["owner_name"]["value"], "Ram Kumar Singh")
        self.assertEqual(f["plot_area"]["value"], "1.25")
        self.assertEqual(f["area_unit"]["value"], "hectare")
        self.assertEqual(f["mutation_number"]["value"], "M-2021-0187")
        self.assertEqual(f["mutation_date"]["value"], "14/03/2021")
        self.assertEqual(f["pincode"]["value"], "221206")

    def test_full_schema_returned(self):
        from records.constants import FIELD_KEYS
        f = extract.extract_fields({"text": "", "lines": []},
                                   apply_learning=False)
        self.assertEqual(set(f.keys()), set(FIELD_KEYS))


class ValidationTests(TestCase):
    def _record(self, **kw):
        r = LandRecord(pk=999, **{**dict(
            owner_name="Ram Kumar Singh", survey_number="331/2",
            village="Rampur", district="Varanasi", state="Uttar Pradesh",
            plot_area=1.25, area_unit="hectare", pincode="221206"), **kw})
        return r

    def test_clean_record_has_no_errors(self):
        issues = validate.validate_record(self._record())
        self.assertNotIn("error", [i["severity"] for i in issues])

    def test_missing_required_is_error(self):
        issues = validate.validate_record(self._record(owner_name=""))
        self.assertTrue(any(i["severity"] == "error" and
                            i["field_name"] == "owner_name" for i in issues))

    def test_bad_pincode(self):
        issues = validate.validate_record(self._record(pincode="1234"))
        self.assertTrue(any(i["rule_code"] == "PINCODE_FORMAT"
                            for i in issues))

    def test_implausible_area(self):
        issues = validate.validate_record(
            self._record(plot_area=900000, area_unit="hectare"))
        self.assertTrue(any(i["rule_code"] == "AREA_RANGE"
                            for i in issues))

    def test_state_district_mismatch_warns(self):
        issues = validate.validate_record(
            self._record(district="Chennai", state="Gujarat"))
        self.assertTrue(any(i["rule_code"] == "STATE_DISTRICT"
                            for i in issues))


class ViewAndApiTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("op", password="x")
        UserProfile.objects.update_or_create(user=self.user,
                                             defaults={"role": "OPERATOR"})

    def test_login_required(self):
        self.assertEqual(self.client.get("/").status_code, 302)

    def test_dashboard_ok(self):
        self.client.login(username="op", password="x")
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_api_requires_auth(self):
        self.assertEqual(self.client.get("/api/v1/stats/").status_code, 401)

    def test_api_with_key(self):
        resp = self.client.get("/api/v1/stats/", HTTP_X_API_KEY="lr-demo-key-2026")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("documents", resp.json())

    def test_upload_and_process_via_api(self):
        from PIL import Image, ImageDraw, ImageFont
        from django.conf import settings as djsettings
        djsettings.OCR_MAX_PAGES = 3
        img = Image.new("RGB", (900, 400), "white")
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except OSError:                                    # pragma: no cover
            self.skipTest("DejaVu font unavailable")
        d.text((40, 40), "RECORD OF RIGHTS", font=f, fill="black")
        d.text((40, 100), "District Varanasi", font=f, fill="black")
        d.text((40, 150), "Village Rampur Test", font=f, fill="black")
        d.text((40, 200), "Owner Name TEST PERSON", font=f, fill="black")
        d.text((40, 250), "Survey Number 51/1", font=f, fill="black")
        d.text((40, 300), "Area of Plot 0.500 Hectare", font=f, fill="black")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)
        resp = self.client.post("/api/v1/documents/", {"file": buf},
                                HTTP_X_API_KEY="lr-demo-key-2026")
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertEqual(data["status"], "PROCESSED")
        rec = LandRecord.objects.get(document_id=data["id"])
        self.assertEqual(rec.owner_name, "Test Person")
        self.assertEqual(rec.survey_number, "51/1")
        self.assertAlmostEqual(rec.plot_area, 0.5)

    def test_field_schema_endpoint(self):
        resp = self.client.get("/api/v1/schema/fields/",
                               HTTP_X_API_KEY="lr-demo-key-2026")
        self.assertEqual(resp.status_code, 200)
        names = [f["name"] for f in resp.json()["fields"]]
        self.assertIn("survey_number", names)


# ---------------------------------------------------------------------------
# NLP regression corpus: clean / noisy / stacked / table / multi-field /
# missing / typo'd / survey+area+date formats / Hindi / empty / garbage
# ---------------------------------------------------------------------------
class NlpCorpusTests(TestCase):
    def test_corpus(self):
        from records.pipeline import ocr_text
        from records.pipeline.nlp_corpus import CASES

        failures = []
        for case in CASES:
            bundle = ocr_text.bundle_from_text(case["text"],
                                               conf=case["conf"] or 1)
            got = extract.extract_fields(bundle, apply_learning=False)
            for field, want in case["expect"].items():
                actual = got[field]["value"]
                if actual.strip().lower() != str(want).strip().lower():
                    failures.append(f"{case['name']}.{field}: "
                                    f"want {want!r} got {actual!r}")
            if not case["expect"]:      # empty / garbage input
                filled = {k: v["value"] for k, v in got.items() if v["value"]}
                if filled:
                    failures.append(f"{case['name']}: hallucinated {filled}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_every_field_has_confidence_and_method(self):
        from records.pipeline import ocr_text
        bundle = ocr_text.bundle_from_text(
            "Owner Name : Ramesh Kumar\nSurvey No : 123/4", conf=93)
        got = extract.extract_fields(bundle, apply_learning=False)
        self.assertGreater(got["owner_name"]["confidence"], 0.7)
        self.assertIn("label", got["owner_name"]["method"])
        self.assertFalse(got["owner_name"]["needs_review"])
        self.assertEqual(got["district"]["value"], "")
        self.assertTrue(got["district"]["needs_review"])

    def test_low_confidence_is_flagged_for_review(self):
        from records.pipeline import ocr_text
        bundle = ocr_text.bundle_from_text("Ownr Nane : Ramesh Kumar",
                                           conf=41)
        got = extract.extract_fields(bundle, apply_learning=False)
        self.assertEqual(got["owner_name"]["value"], "Ramesh Kumar")
        self.assertLess(got["owner_name"]["confidence"], 0.75)
        self.assertTrue(got["owner_name"]["needs_review"])

    def test_no_bleed_between_adjacent_fields(self):
        from records.pipeline import ocr_text
        bundle = ocr_text.bundle_from_text(
            "District: Chennai\nVillage: Sholinganallur", conf=90)
        got = extract.extract_fields(bundle, apply_learning=False)
        self.assertEqual(got["district"]["value"], "Chennai")
        self.assertEqual(got["village"]["value"], "Sholinganallur")

    def test_survives_malformed_ocr_bundles(self):
        for bundle in ({}, {"lines": None, "text": None},
                       {"lines": [{"text": "Owner Name Ram"}], "text": ""},
                       {"lines": [{"words": []}], "text": ""}):
            got = extract.extract_fields(bundle, apply_learning=False)
            self.assertEqual(len(got), len(__import__(
                "records.constants", fromlist=["x"]).FIELD_KEYS))


class LearningMemoryTests(TestCase):
    def test_identifiers_are_not_learnable(self):
        from records.pipeline import learn
        self.assertIsNone(
            learn.record_correction("survey_number", "123/4", "123/5"))
        self.assertIsNone(
            learn.record_correction("pincode", "600100", "600101"))

    def test_unrelated_replacement_is_not_generalised(self):
        from records.pipeline import learn
        self.assertIsNone(
            learn.record_correction("owner_name", "Ram Kumar", "Sita Devi"))

    def test_ocr_repair_is_remembered_and_reapplied(self):
        from records.pipeline import learn, ocr_text
        self.assertIsNotNone(
            learn.record_correction("village", "Daskrol", "Daskroi"))
        bundle = ocr_text.bundle_from_text("Village : Daskrol", conf=88)
        got = extract.extract_fields(bundle, apply_learning=True)
        self.assertEqual(got["village"]["value"], "Daskroi")
        self.assertEqual(got["village"]["source"], "learned")
        self.assertIn("verified-correction memory", got["village"]["method"])


class EndToEndPipelineTests(TestCase):
    """Scan -> gamma -> OCR -> NLP -> validation -> DB -> UI -> correction
    -> re-processing, exercised through the real views."""

    def _scan(self, village="Kelambakam"):
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (1000, 520), "white")
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        except OSError:                                    # pragma: no cover
            self.skipTest("DejaVu font unavailable")
        rows = ["RECORD OF RIGHTS", "District : Chennai",
                f"Village : {village}", "Owner Name : Ramesh Kumar",
                "Father Name : Krishnan", "Survey No : 123/4",
                "Extent : 1.25 acres", "Pin Code : 600100"]
        for i, line in enumerate(rows):
            d.text((40, 30 + i * 56), line, font=f, fill="black")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)
        return buf

    def setUp(self):
        self.admin = User.objects.create_user("adm", password="x",
                                              is_staff=True)
        UserProfile.objects.update_or_create(user=self.admin,
                                             defaults={"role": "ADMIN"})
        self.client.login(username="adm", password="x")

    def test_full_flow_upload_verify_correct_reprocess(self):
        from records.models import FieldExtraction, LearnedCorrection
        from django.core.files.uploadedfile import SimpleUploadedFile

        # --- upload through the real web form ---------------------------
        upload = SimpleUploadedFile("scan.png", self._scan().read(),
                                    content_type="image/png")
        resp = self.client.post(reverse("upload"),
                                {"files": upload, "language": "eng",
                                 "gamma_mode": "auto"})
        self.assertIn(resp.status_code, (302, 200))
        doc = Document.objects.latest("id")
        self.assertEqual(doc.status, "PROCESSED")
        record = doc.record

        # --- OCR text really reached the NLP stage ----------------------
        self.assertIn("Ramesh", doc.ocr_text)
        self.assertEqual(record.owner_name, "Ramesh Kumar")
        self.assertEqual(record.survey_number, "123/4")
        self.assertEqual(record.district, "Chennai")
        self.assertAlmostEqual(record.plot_area, 1.25)
        self.assertEqual(record.area_unit, "acre")
        self.assertEqual(record.pincode, "600100")

        # --- per-field provenance stored --------------------------------
        rows = {f.field_name: f for f in record.fields.all()}
        self.assertTrue(rows["owner_name"].method)
        self.assertGreater(rows["owner_name"].confidence, 0.6)

        # --- the UI shows the real extracted values ---------------------
        page = self.client.get(reverse("document_detail", args=[doc.pk]))
        self.assertContains(page, "Ramesh Kumar")
        self.assertContains(page, "inline value right of label")
        self.assertEqual(self.client.get(reverse("dashboard")).status_code, 200)
        self.assertEqual(
            self.client.get(reverse("verify_record", args=[record.pk]))
            .status_code, 200)

        # --- human correction (village OCR'd from the scan) -------------
        payload = {k: (getattr(record, k) or "") for k, *_ in
                   __import__("records.constants",
                              fromlist=["x"]).FIELD_DEFS}
        payload["plot_area"] = "1.25"
        payload["village"] = "Kelambakkam"
        payload["note"] = "corrected village spelling"
        self.client.post(reverse("verify_record", args=[record.pk]), payload)
        record.refresh_from_db()
        self.assertEqual(record.village, "Kelambakkam")
        self.assertEqual(record.status, "VERIFIED")
        self.assertTrue(FieldExtraction.objects.filter(
            record=record, field_name="village", source="manual").exists())

        # --- the correction is memorised for the next document ----------
        if record.fields.get(field_name="village").value.lower() != "kelambakkam":
            self.assertTrue(LearnedCorrection.objects.filter(
                field_name="village").exists())
            upload2 = SimpleUploadedFile("scan2.png", self._scan().read(),
                                         content_type="image/png")
            self.client.post(reverse("upload"),
                             {"files": upload2, "language": "eng",
                              "gamma_mode": "auto"})
            doc2 = Document.objects.latest("id")
            self.assertEqual(doc2.record.village, "Kelambakkam")

        # --- re-processing keeps working --------------------------------
        resp = self.client.post(reverse("document_reprocess", args=[doc.pk]),
                                {"gamma_mode": "auto"})
        doc.refresh_from_db()
        self.assertEqual(doc.status, "PROCESSED")

    def _two_page_pdf(self):
        import fitz
        from PIL import Image
        page1 = Image.open(self._scan())
        page2 = Image.open(self._scan(village="Kelambakam"))
        buf1, buf2 = io.BytesIO(), io.BytesIO()
        page1.save(buf1, "PNG")
        page2.save(buf2, "PNG")
        doc = fitz.open()
        for b in (buf1, buf2):
            b.seek(0)
            rect = fitz.Rect(0, 0, 1000, 520)
            p = doc.new_page(width=1000, height=520)
            p.insert_image(rect, stream=b.read())
        out = io.BytesIO()
        doc.save(out)
        doc.close()
        out.seek(0)
        return out

    def test_multi_page_verify_is_a_single_review(self):
        """A document with N pages must still be ONE record needing ONE
        approval - the verify console just lets you flip between page
        previews, it must not require repeating the review per page."""
        from django.core.files.uploadedfile import SimpleUploadedFile

        upload = SimpleUploadedFile("two_page.pdf", self._two_page_pdf().read(),
                                    content_type="application/pdf")
        resp = self.client.post(reverse("upload"),
                                {"files": upload, "language": "eng",
                                 "gamma_mode": "auto"})
        self.assertIn(resp.status_code, (302, 200))
        doc = Document.objects.latest("id")
        self.assertEqual(doc.status, "PROCESSED")
        self.assertEqual(doc.page_count, 2)
        self.assertEqual(doc.processed_page_count, 2)
        self.assertEqual(doc.pages.count(), 2)
        # every page's preview must actually differ (regression guard for
        # "enhanced images just show page 1 repeated") -- compare bytes.
        images = [p.processed_image.read() for p in doc.pages.all()]
        self.assertNotEqual(images[0], images[1])

        # Exactly one LandRecord and one pending review for the whole doc.
        self.assertEqual(LandRecord.objects.filter(document=doc).count(), 1)
        record = doc.record

        page = self.client.get(reverse("verify_record", args=[record.pk]))
        self.assertEqual(page.status_code, 200)
        html = page.content.decode()
        self.assertEqual(html.count('class="verify-layout"'), 1,
                         "one verify form must cover the whole document, "
                         "not one per page")
        import re
        self.assertEqual(len(re.findall(r'\bpage-tab\b', html)), 2,
                         "expected a tab per page")
        # only page 1's block is visible on first load; page 2 starts hidden
        self.assertEqual(html.count('class="page-block'), 2)
        self.assertRegex(html, r'class="page-block"\s+data-page="1"')
        self.assertRegex(html, r'class="page-block hidden"\s+data-page="2"')

        # approving once resolves the whole document, not just one page.
        payload = {k: (getattr(record, k) or "") for k, *_ in
                   __import__("records.constants",
                              fromlist=["x"]).FIELD_DEFS}
        payload["plot_area"] = str(record.plot_area or "1.25")
        payload["note"] = "single approval covers both pages"
        self.client.post(reverse("verify_record", args=[record.pk]), payload)
        record.refresh_from_db()
        self.assertEqual(record.status, "VERIFIED")
        self.assertEqual(LandRecord.objects.filter(document=doc).count(), 1)

    def test_page_count_reports_true_total_when_capped(self):
        """A source PDF with more pages than OCR_MAX_PAGES must report its
        real page count, not silently shrink to whatever was processed
        (previously `page_count = len(pages)` used the *capped* list)."""
        from django.conf import settings as djsettings
        from django.core.files.uploadedfile import SimpleUploadedFile

        old_cap = djsettings.OCR_MAX_PAGES
        djsettings.OCR_MAX_PAGES = 1
        try:
            upload = SimpleUploadedFile(
                "two_page_capped.pdf", self._two_page_pdf().read(),
                content_type="application/pdf")
            self.client.post(reverse("upload"),
                             {"files": upload, "language": "eng",
                              "gamma_mode": "auto"})
            doc = Document.objects.latest("id")
            self.assertEqual(doc.status, "PROCESSED")
            self.assertEqual(doc.page_count, 2,           # true source total
                             "page_count must reflect the source file, "
                             "not the OCR_MAX_PAGES cap")
            self.assertEqual(doc.processed_page_count, 1)  # capped at 1
            self.assertEqual(doc.pages.count(), 1)
        finally:
            djsettings.OCR_MAX_PAGES = old_cap

    def test_unreadable_scan_does_not_crash(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        from PIL import Image
        blank = Image.new("RGB", (300, 200), "white")
        buf = io.BytesIO()
        blank.save(buf, "PNG")
        buf.seek(0)
        self.client.post(reverse("upload"),
                         {"files": SimpleUploadedFile("blank.png", buf.read(),
                                                      content_type="image/png"),
                          "language": "eng", "gamma_mode": "auto"})
        doc = Document.objects.latest("id")
        self.assertEqual(doc.status, "PROCESSED")
        self.assertEqual(doc.record.owner_name, "")
        self.assertTrue(doc.record.issues.filter(severity="error").exists())
        self.assertEqual(
            self.client.get(reverse("document_detail",
                                    args=[doc.pk])).status_code, 200)

    def test_invalid_pdf_fails_gracefully(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        self.client.post(reverse("upload"),
                         {"files": SimpleUploadedFile(
                             "broken.pdf", b"not really a pdf",
                             content_type="application/pdf"),
                          "language": "eng", "gamma_mode": "auto"})
        doc = Document.objects.latest("id")
        self.assertEqual(doc.status, "FAILED")
        self.assertTrue(doc.error_message)
        self.assertEqual(
            self.client.get(reverse("document_detail",
                                    args=[doc.pk])).status_code, 200)
