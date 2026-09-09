#!/usr/bin/env bash
# Swap in the tessdata_best language models for Tesseract, which are
# measurably more accurate than the distro-packaged ones (especially for
# Indic scripts on faded/noisy/damaged scans) at the cost of being slower
# per page. Safe to re-run.
#
#   ./scripts/install_tessdata_best.sh                      # eng + hin
#   ./scripts/install_tessdata_best.sh hin ben tam tel kan guj mar pan mal
#
# Requires network access and `curl`. Needs root (or run under sudo) to
# write into the system tessdata directory unless TESSDATA_PREFIX is set
# to a writable location Tesseract already searches.
set -euo pipefail

LANGS=("$@")
if [ ${#LANGS[@]} -eq 0 ]; then
  LANGS=(eng hin)
fi

# Autodetect the tessdata directory Tesseract is actually using.
if [ -n "${TESSDATA_PREFIX:-}" ]; then
  TESSDATA_DIR="$TESSDATA_PREFIX"
else
  TESSDATA_DIR="$(find /usr/share/tesseract-ocr -maxdepth 2 -type d -name tessdata 2>/dev/null | head -n1)"
fi
if [ -z "${TESSDATA_DIR:-}" ] || [ ! -d "$TESSDATA_DIR" ]; then
  echo "Could not find a tessdata directory. Install tesseract-ocr first," >&2
  echo "or set TESSDATA_PREFIX to point at one." >&2
  exit 1
fi
echo "Installing tessdata_best models into: $TESSDATA_DIR"

BASE_URL="https://github.com/tesseract-ocr/tessdata_best/raw/main"
for lang in "${LANGS[@]}"; do
  url="${BASE_URL}/${lang}.traineddata"
  dest="${TESSDATA_DIR}/${lang}.traineddata"
  echo "  - ${lang}: ${url} -> ${dest}"
  curl -fL --retry 3 -o "${dest}.tmp" "$url"
  mv "${dest}.tmp" "$dest"
done

echo "Done. No settings.py or code changes are needed -- pipeline/ocr.py"
echo "picks up the new models automatically on the next OCR call."
