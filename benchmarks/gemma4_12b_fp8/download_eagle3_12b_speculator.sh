#!/usr/bin/env bash
# Download the DeepSeek release EAGLE-3 speculator (draft) for the Gemma4-12B
# dense target, so benchmarks load it from local disk instead of hitting the HF
# Hub at serve time (faster, no rate limits, reproducible).
#
# Draft repo: deepseek-ai/eagle3_gemma4_12b_ttt7 (trained with TTT length 7).
# Pairs with target google/gemma-4-12B-it (dense). Config: configs/12b_eagle3.json.
#
# Usage:
#   bash download_eagle3_12b_speculator.sh [/tmp/models/gemma4_12b/eagle3_speculator]
#
# Optional:
#   export HF_TOKEN=...   # STRONGLY recommended: unauthenticated LFS downloads
#                         # get rate-limited and large *.safetensors shards can be
#                         # silently skipped (leaving only KB of config/metadata).
#                         # The size check below fails loudly if that happens.
set -euo pipefail

DEST=${1:-/tmp/models/gemma4_12b/eagle3_speculator}
SPECULATOR_REPO=${EAGLE3_12B_SPECULATOR_REPO:-deepseek-ai/eagle3_gemma4_12b_ttt7}
# Total weight bytes across all *.safetensors must be at least this. An EAGLE-3
# draft for a 12B model is a single small decoder layer (hundreds of MB); 100 MB
# is a conservative floor that still catches a metadata-only (KB) partial pull.
MIN_WEIGHTS_BYTES=${MIN_WEIGHTS_BYTES:-100000000}

mkdir -p "$DEST"

echo "=== Download Gemma4-12B EAGLE-3 speculator (DeepSeek release) ==="
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
# The DeepSeek repo's exact shard filename is unknown ahead of time, so we sum
# every *.safetensors under DEST rather than assuming a fixed name.
shopt -s nullglob
WEIGHT_FILES=("$DEST"/*.safetensors "$DEST"/**/*.safetensors)
shopt -u nullglob
if [[ ${#WEIGHT_FILES[@]} -eq 0 ]]; then
  echo "" >&2
  echo "ERROR: no *.safetensors found under $DEST. Only metadata was downloaded." >&2
  echo "This usually means the LFS weights were skipped/rate-limited." >&2
  echo "Fix: export HF_TOKEN=<your token> and re-run this script." >&2
  exit 1
fi

TOTAL_BYTES=0
for f in "${WEIGHT_FILES[@]}"; do
  sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
  TOTAL_BYTES=$((TOTAL_BYTES + sz))
done

if [[ "$TOTAL_BYTES" -lt "$MIN_WEIGHTS_BYTES" ]]; then
  echo "" >&2
  echo "ERROR: total weights are only ${TOTAL_BYTES} bytes (expected >= ${MIN_WEIGHTS_BYTES})." >&2
  echo "The LFS weights were not fully downloaded (likely rate-limited)." >&2
  echo "Fix: export HF_TOKEN=<your token> and re-run this script." >&2
  exit 1
fi

echo ""
echo "=== Done ==="
echo "EAGLE-3 speculator: $DEST"
echo "Weights total     : $(( TOTAL_BYTES / 1024 / 1024 )) MB across ${#WEIGHT_FILES[@]} shard(s)"
echo ""
echo "Point the benchmark at the local copy (overrides the config default):"
echo "export GEMMA4_SPECULATOR_MODEL=$DEST"
echo ""
echo "Then run, e.g.:"
echo "  export GEMMA4_MODEL_PATH=/tmp/models/gemma4_12b/model   # 12B dense target"
echo "  bash run_online_align.sh configs/12b_eagle3.json --max-concurrency none --num-prompts 1000"
