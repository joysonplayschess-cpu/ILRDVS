# Generated for the page-count fix: page_count now always holds the TRUE
# (uncapped) number of pages in the source file; processed_page_count is
# the new field recording how many of those were actually enhanced/OCR'd
# (<= settings.OCR_MAX_PAGES). See records/pipeline/preprocess.load_pages
# and records/pipeline/service.process_document.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('records', '0004_documentpage'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='processed_page_count',
            field=models.PositiveIntegerField(default=1),
        ),
    ]
