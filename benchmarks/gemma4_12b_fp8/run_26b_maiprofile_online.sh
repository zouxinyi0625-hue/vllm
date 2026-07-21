#!/usr/bin/env bash
# 26B-A4B per-layer MAI Profile ONLINE benchmark — MTP + no-MTP.
#
# Thin wrapper over run_maiprofile_online.sh (the same driver used for 12B):
# point it at the local 26B model dir and the 26B configs. Nothing else changes.
#
# Runs two passes on the 5 MAI Profile eval layers:
#   1) MTP     (configs/26b_e011_mtp.json)    -> per-layer accept rate + tok/s
#   2) no-MTP  (configs/26b_e011_no_mtp.json) -> per-layer dense tok/s baseline
#
# Assumes the model is ALREADY downloaded to $MODEL_ROOT (no download here):
#   $MODEL_ROOT/text_only   (target)
#   $MODEL_ROOT/assistant   (MTP draft)
#
# TP=1: 26B-A4B inference fits on one GPU (tp>2 was only for hidden-states).
#
# USAGE (server, vLLM venv):
#   bash run_26b_maiprofile_online.sh
#   # only the dense baseline:
#   CONFIGS=configs/26b_e011_no_mtp.json bash run_26b_maiprofile_online.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_ROOT="${MODEL_ROOT:-/tmp/models/gemma4}"

# Point the shared driver at the local 26B model. Use the generic
# GEMMA4_MODEL_PATH / GEMMA4_ASSISTANT_MODEL_PATH — these are the HIGHEST-priority
# overrides in serve_align.sh, so they beat any stale 12B/26B env from a previous
# run (which is what silently loaded the wrong 12B target before). One set of
# vars, no path mixing.
# Clear legacy/conflicting vars first so nothing can override the generic ones.
unset GEMMA4_26B_TEXT_ONLY_MODEL_PATH GEMMA4_26B_ASSISTANT_MODEL_PATH _ModelDataPath_ GEMMA4_SPECULATOR_MODEL
export GEMMA4_MODEL_PATH="${MODEL_ROOT}/text_only"
export GEMMA4_ASSISTANT_MODEL_PATH="${MODEL_ROOT}/assistant"
export GEMMA4_TOKENIZER="${MODEL_ROOT}/text_only"   # gated tokenizer -> use local dir

# 26B-A4B on one GPU. Unlimited concurrency to match the baseline runs.
export TP_SIZE="${TP_SIZE:-1}"
export NUM_PROMPTS="${NUM_PROMPTS:-200}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-none}"
# EVAL_DIR / LAYERS fall through to the driver's defaults (5 short layers from
# the Azure ML mount). Override either via env if needed.

CONFIGS="${CONFIGS:-configs/26b_e011_mtp.json configs/26b_e011_no_mtp.json}"

for cfg in $CONFIGS; do
  base="$(basename "$cfg" .json)"
  echo ""
  echo "========================================================"
  echo " 26B per-layer online: ${base}"
  echo "========================================================"
  CONFIG="$cfg" SUMMARY_JSON="maiprofile_${base}_online_summary.json" \
    bash run_maiprofile_online.sh || echo "[WARN] ${base} run failed; continuing"
done
