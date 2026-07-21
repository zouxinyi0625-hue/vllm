#!/usr/bin/env bash
# Download the official RedHatAI EAGLE-3 speculator (draft) for the 26B-A4B
# target, so benchmarks load it from local disk instead of hitting the HF Hub
# at serve time (faster, no rate limits, reproducible).
#
# Usage:
#   bash download_eagle3_speculator.sh [/tmp/models/gemma4/eagle3_speculator]
#
# Optional:
#   export HF_TOKEN=...   # STRONGLY recommended: unauthenticated LFS downloads
#                         # get rate-limited and the 1.86GB model.safetensors
#                         # can be silently skipped (leaving only ~9KB of
#                         # config/metadata). See the size check below.
set -euo pipefail

DEST=${1:-/tmp/models/gemma4/eagle3_speculator}
SPECULATOR_REPO=${EAGLE3_SPECULATOR_REPO:-RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3}
# The real weights file must be at least this many bytes (repo has 1.86 GB).
MIN_WEIGHTS_BYTES=${MIN_WEIGHTS_BYTES:-1000000000}

mkdir -p "$DEST"

echo "=== Download Gemma4 26B-A4B EAGLE-3 speculator ==="
echo "Speculator repo: $SPECULATOR_REPO"
echo "Local dir      : $DEST"
echo "HF_TOKEN       : ${HF_TOKEN:+set}${HF_TOKEN:-<unset - LFS may be rate-limited!>}"
echo ""

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' command not found. Install Hugging Face Hub CLI first." >&2
  echo "Example: python3 -m pip install huggingface_hub[cli] --break-system-packages" >&2
  exit 1
fi

# Retry a few times: LFS fetches can fail/partial under rate limiting.
attempt=0
max_attempts=3
while :; do
  attempt=$((attempt + 1))
  echo "--- Download attempt ${attempt}/${max_attempts} ---"
  if hf download "$SPECULATOR_REPO" --local-dir "$DEST"; then
    break
  fi
  if [[ "$attempt" -ge "$max_attempts" ]]; then
    echo "ERROR: hf download failed after ${max_attempts} attempts." >&2
    exit 1
  fi
  echo "Download attempt failed; retrying in 5s..." >&2
  sleep 5
done

# --- Verify the LFS weights actually landed (not just metadata) ---
WEIGHTS="$DEST/model.safetensors"
if [[ ! -f "$WEIGHTS" ]]; then
  echo "" >&2
  echo "ERROR: $WEIGHTS is missing. Only metadata was downloaded." >&2
  echo "This usually means the LFS weights were skipped/rate-limited." >&2
  echo "Fix: export HF_TOKEN=<your token> and re-run this script." >&2
  exit 1
fi

# Portable file size (Linux stat -c, macOS/BSD stat -f).
WEIGHTS_BYTES=$(stat -c%s "$WEIGHTS" 2>/dev/null || stat -f%z "$WEIGHTS" 2>/dev/null || echo 0)
if [[ "$WEIGHTS_BYTES" -lt "$MIN_WEIGHTS_BYTES" ]]; then
  echo "" >&2
  echo "ERROR: $WEIGHTS is only ${WEIGHTS_BYTES} bytes (expected >= ${MIN_WEIGHTS_BYTES})." >&2
  echo "The LFS weights were not fully downloaded (likely rate-limited)." >&2
  echo "Fix: export HF_TOKEN=<your token> and re-run this script." >&2
  exit 1
fi

echo ""
echo "=== Done ==="
echo "EAGLE-3 speculator: $DEST"
echo "Weights size      : $(( WEIGHTS_BYTES / 1024 / 1024 )) MB (model.safetensors)"
echo ""
echo "Point the benchmark at the local copy (overrides the config default):"
echo "export GEMMA4_SPECULATOR_MODEL=$DEST"
echo ""
echo "Then run, e.g.:"
echo "  export GEMMA4_MODEL_PATH=/tmp/models/gemma4/text_only   # 26B target"
echo "  bash run_online_align.sh configs/26b_eagle3.json --max-concurrency none --num-prompts 1000"
