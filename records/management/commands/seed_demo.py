"""
Seed the platform with demo users + a realistic corpus of synthetic land
records (clean / dark / faded / noisy / damaged, English + Hindi, PDF +
images), then runs the real AI pipeline over them.

    python manage.py seed_demo                    # idempotent, 1 verify pass
    python manage.py seed_demo --reset             # wipe and rebuild
    python manage.py seed_demo --reset --train-passes 10
        # re-run the reprocess -> verify-with-correct-values cycle 10x for
        # the demo corpus, so the verified-correction memory
        # (records.pipeline.learn) is well seeded before a demo. This does
        # NOT retrain any OCR/NLP model -- see learn.py's own docstring --
        # it just gives the deterministic correction cache real repeated
        # examples to learn from, the same way a verification officer
        # correcting the same kind of OCR misread over and over would.
"""
from django.contrib.auth.models import User
from django.core.files import File
from django.core.management.base import BaseCommand
from django.utils import timezone

from records.models import (AuditLog, Document, FieldExtraction, LandRecord,
                            LearnedCorrection, UserProfile, ValidationIssue)
from records.pipeline import sample_docs
from records.pipeline.service import (apply_verification, process_document)

USERS = [
    ("admin", "admin123", "ADMIN", "ILRDVS Central Cell", True),
    ("operator1", "operate123", "OPERATOR", "District Digitization Cell", False),
    ("verifier1", "verify123", "VERIFIER", "Tehsil Record Room", False),
    ("viewer1", "view123", "VIEWER", "Citizen Services", False),
]

VERIFY_LIST = ["UP_Varanasi_RoR_dark.pdf", "Bihar_Patna_faded.jpg",
               "MP_Bhopal_hindi.jpg", "Rajasthan_Jaipur_noisy.png",
               "WB_Kolkata_clean.pdf", "Karnataka_Bengaluru_clean.png",
               "TN_Chennai_dark.jpg", "Punjab_Ludhiana_clean.jpg"]
SKIP_PROCESS = {"Telangana_Hyderabad_clean.png"}   # stays 'Uploaded'


class Command(BaseCommand):
    help = "Seed demo users and a synthetic land-record corpus."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true")
        parser.add_argument(
            "--train-passes", type=int, default=1,
            help="how many times to run reprocess+verify-with-correct-"
                 "values over the demo corpus, to seed the correction "
                 "memory (records.pipeline.learn). Default 1.")

    def handle(self, *args, **opts):
        if opts["reset"]:
            self.stdout.write("Wiping existing demo data ...")
            for m in (Document, LandRecord, FieldExtraction,
                      ValidationIssue, LearnedCorrection, AuditLog):
                m.objects.all().delete()
        if Document.objects.exists():
            self.stdout.write("Database already seeded - use --reset to rebuild.")
            return

        # ---------------- users --------------------------------------
        for username, pwd, role, org, is_super in USERS:
            user, created = User.objects.get_or_create(username=username)
            user.set_password(pwd)
            user.is_staff = True
            user.is_superuser = is_super
            user.save()
            UserProfile.objects.update_or_create(
                user=user, defaults={"role": role, "organization": org})
        self.stdout.write(self.style.SUCCESS("Users: admin/admin123, "
                                             "operator1/operate123, "
                                             "verifier1/verify123, "
                                             "viewer1/view123"))
        verifier = User.objects.get(username="verifier1")
        operator = User.objects.get(username="operator1")

        # ---------------- documents ----------------------------------
        from django.conf import settings
        self.stdout.write("Rendering synthetic scans ...")
        samples = sample_docs.build(settings.MEDIA_ROOT)
        specs = {s["filename"]: s for s in samples}

        self.stdout.write("Running OCR pipeline on the corpus ...")
        docs = {}
        for spec in samples:
            with open(spec["path"], "rb") as fh:
                doc = Document.objects.create(
                    original_name=spec["filename"],
                    language=spec["language"],
                    state_name=spec["state"],
                    district_name=spec["district"],
                    uploaded_by=operator,
                    file_size=spec["path"].stat().st_size)
                doc.file.save(spec["filename"], File(fh), save=True)
            docs[spec["filename"]] = (doc, spec)
            if spec["filename"] in SKIP_PROCESS:
                continue
            record = process_document(doc, gamma_mode="auto",
                                      lang=spec["language"])
            ok = "ok" if record else "FAILED"
            conf = f" conf={record.overall_confidence:.2f}" if record else ""
            self.stdout.write(f"  - {spec['filename']}: {ok}{conf}")

        # ---------------- reject the duplicate rescan first -----------
        from records.pipeline.service import reject_record
        dup_doc, _dup_spec = docs["UP_Varanasi_rescan_clean.png"]
        if getattr(dup_doc, "record", None):
            reject_record(dup_doc.record, verifier,
                          "Duplicate rescan of plot 331/2, Rampur Khurd "
                          "(already digitized from doc #1).")

        # ---------------- human verification of a subset --------------
        passes = max(1, opts["train_passes"])
        self.stdout.write(f"Verifying a subset as verifier1 "
                          f"({passes} pass(es), seeding correction memory) ...")
        for pass_no in range(1, passes + 1):
            for name in VERIFY_LIST:
                doc, spec = docs[name]
                if pass_no > 1:
                    # re-run OCR/NLP so this pass exercises the *real*
                    # extraction again (and can pick up an already-learned
                    # correction) rather than just re-saving old values.
                    process_document(doc, gamma_mode="auto",
                                     lang=spec["language"])
                record = getattr(doc, "record", None)
                if record is None:
                    continue
                cleaned = {
                    "owner_name": spec["owner"], "father_name": spec["father"],
                    "survey_number": spec["survey"], "khasra_number": spec["khasra"],
                    "khata_number": spec["khata"], "plot_area": spec["area"],
                    "area_unit": spec["unit"].lower(), "village": spec["village"],
                    "tehsil": spec["tehsil"], "district": spec["district"],
                    "state": spec["state"], "pincode": spec["pin"],
                    "land_classification": spec["land_class"],
                    "ownership_type": spec["ownership"],
                    "mutation_number": spec["mutation"],
                    "mutation_date": spec["mutation_date"],
                    "registration_number": spec["reg"],
                    "registration_date": spec["reg_date"],
                }
                apply_verification(record, cleaned, verifier,
                                   note=f"Seeded demo verification "
                                        f"(pass {pass_no}/{passes})")
            if passes > 1:
                self.stdout.write(
                    f"  pass {pass_no}/{passes}: "
                    f"learned={LearnedCorrection.objects.count()} correction(s)")

        AuditLog.objects.create(
            user_label="system", action="SEED",
            object_type="System", object_id="-",
            detail=f"Demo corpus seeded at {timezone.now():%Y-%m-%d %H:%M}")

        n = Document.objects.count()
        self.stdout.write(self.style.SUCCESS(
            f"Done. {n} documents | "
            f"verified={LandRecord.objects.filter(status='VERIFIED').count()} | "
            f"pending={LandRecord.objects.filter(status='PENDING').count()} | "
            f"learned={LearnedCorrection.objects.count()}"))
