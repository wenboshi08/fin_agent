#!/usr/bin/env bash
# Download ICAIF-24 Finance RAG Challenge data from Kaggle.
#
# Prerequisites:
#   1. A Kaggle account, accept competition rules:
#      https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge
#   2. Kaggle API credentials at ~/.kaggle/kaggle.json
#      (username + key from https://www.kaggle.com/settings -> API)
#
# Usage:
#   bash scripts/download_data.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="$ROOT/Dataset"
TMP_DIR="${TMPDIR:-/tmp}/finagent-kaggle"

echo "==> Preparing directories"
mkdir -p "$DATASET_DIR" "$TMP_DIR"

LOCAL_KAGGLE="$ROOT/.tools/bin/kaggle"
if [[ -x "$LOCAL_KAGGLE" ]]; then
  KAGGLE_CMD="$LOCAL_KAGGLE"
elif command -v kaggle >/dev/null 2>&1; then
  KAGGLE_CMD="kaggle"
else
  echo "==> kaggle CLI not found. Installing via uv into $ROOT/.tools ..."
  mkdir -p "$ROOT/.tools/uv" "$ROOT/.tools/bin"
  UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/uvcache}" \
    UV_TOOL_DIR="$ROOT/.tools/uv" \
    UV_TOOL_BIN_DIR="$ROOT/.tools/bin" \
    uv tool install kaggle
  KAGGLE_CMD="$LOCAL_KAGGLE"
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" && ! -f "$HOME/.kaggle/access_token" ]]; then
  echo ""
  echo "ERROR: No Kaggle credentials found."
  echo ""
  echo "Option A — API token file (recommended):"
  echo "  1. Go to https://www.kaggle.com/settings -> API"
  echo "  2. Click 'Create New Token' -> downloads kaggle.json"
  echo "  3. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
  echo "  4. chmod 600 ~/.kaggle/kaggle.json"
  echo ""
  echo "Option B — Access token:"
  echo "  1. Copy your access token to ~/.kaggle/access_token"
  echo "  2. chmod 600 ~/.kaggle/access_token"
  echo ""
  exit 1
fi

echo "==> Downloading competition data (this may take a while)"
"$KAGGLE_CMD" competitions download -c icaif-24-finance-rag-challenge -p "$TMP_DIR"

ZIP_FILE="$TMP_DIR/icaif-24-finance-rag-challenge.zip"
if [[ ! -f "$ZIP_FILE" ]]; then
  echo "ERROR: expected zip not found at $ZIP_FILE"
  exit 1
fi

echo "==> Extracting into $DATASET_DIR"
unzip -o "$ZIP_FILE" -d "$DATASET_DIR"

echo "==> Cleaning up zip"
rm -f "$ZIP_FILE"

echo ""
echo "Done. Files are under:"
echo "  $DATASET_DIR"
echo ""
echo "Next: python main.py --dataset financebench"
