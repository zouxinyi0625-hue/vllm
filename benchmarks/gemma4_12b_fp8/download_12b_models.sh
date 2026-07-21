#!/usr/bin/env bash
# Download Gemma4 12B target and assistant models for this benchmark scaffold.
#
# Usage:
#   bash download_12b_models.sh [/tmp/models/gemma4_12b]
#
# Optional:
#   export HF_TOKEN=...   # if the environment needs authenticated HF access
set -euo pipefail

ROOT=${1:-/tmp/models/gemma4_12b}
TARGET_REPO=${GEMMA4_12B_TARGET_REPO:-google/gemma-4-12B-it}
ASSISTANT_REPO=${GEMMA4_12B_ASSISTANT_REPO:-google/gemma-4-12B-it-assistant}

mkdir -p "$ROOT"

echo "=== Download Gemma4 12B models ==="
echo "Root          : $ROOT"
echo "Target repo   : $TARGET_REPO"
echo "Assistant repo: $ASSISTANT_REPO"
echo ""

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' command not found. Install Hugging Face Hub CLI first." >&2
  echo "Example: python3 -m pip install huggingface_hub[cli] --break-system-packages" >&2
  exit 1
fi

hf download "$TARGET_REPO" \
  --local-dir "$ROOT/model"

hf download "$ASSISTANT_REPO" \
  --local-dir "$ROOT/assistant"

echo ""
echo "=== Done ==="
echo "Target model   : $ROOT/model"
echo "Assistant model: $ROOT/assistant"
echo ""
echo "Recommended env for running this scaffold:"
echo "export GEMMA4_MODEL_PATH=$ROOT/model"
echo "export GEMMA4_ASSISTANT_MODEL_PATH=$ROOT/assistant"
