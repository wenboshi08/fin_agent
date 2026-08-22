#!/usr/bin/env bash
# Package the FinAgent project into a zip for Google Colab.
#
# Usage:
#   bash scripts/export_colab_zip.sh            # exclude Dataset (small zip)
#   bash scripts/export_colab_zip.sh --with-data  # include Dataset (~68MB)
#
# Output: finagent_colab.zip in repo root.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INCLUDE_DATA="${1:-}"

EXCLUDE=(
  ".venv"
  ".tools"
  "chroma_stores"
  "results"
  "__pycache__"
  ".pytest_cache"
  "Dataset"   # added back only with --with-data
)

ARGS=()
for item in "${EXCLUDE[@]}"; do
  ARGS+=( "-x" "$item/*" "-x" "*/$item/*" )
done

if [[ "$INCLUDE_DATA" == "--with-data" ]]; then
  echo "==> Including Dataset/ in the zip"
  # Remove Dataset from exclude list by rebuilding args
  ARGS=()
  for item in "${EXCLUDE[@]}"; do
    if [[ "$item" != "Dataset" ]]; then
      ARGS+=( "-x" "$item/*" )
    fi
  done
fi

rm -f finagent_colab.zip

echo "==> Creating finagent_colab.zip ..."
zip -r finagent_colab.zip . "${ARGS[@]}" -x "*.git/*" -x ".gitignore" -x "*.DS_Store"

echo "==> Done: $(pwd)/finagent_colab.zip"
ls -lh finagent_colab.zip
