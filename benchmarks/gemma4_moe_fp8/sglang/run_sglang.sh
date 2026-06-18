#!/usr/bin/env bash
# run_sglang.sh — Launch SGLang ablation experiments.
#
# Usage:
#   ./run_sglang.sh S001                    # single experiment
#   ./run_sglang.sh S001,S002,S003          # comma-separated list
#   ./run_sglang.sh --all                   # all available experiments
#   ./run_sglang.sh S001 --scenario sc1 --reps 2
#
# Environment variables (override before calling):
#   GEMMA4_MODEL_PATH              full model path
#   GEMMA4_TEXT_ONLY_MODEL_PATH    text-only (vision stripped) checkpoint
#   GEMMA4_ASSISTANT_MODEL_PATH    EAGLE3 draft model for speculative decoding

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Default model paths
# ---------------------------------------------------------------------------
: "${GEMMA4_MODEL_PATH:=google/gemma-4-26B-A4B-it}"
: "${GEMMA4_TEXT_ONLY_MODEL_PATH:=${GEMMA4_MODEL_PATH}-text-only}"
: "${GEMMA4_ASSISTANT_MODEL_PATH:=${GEMMA4_MODEL_PATH}-assistant}"
export GEMMA4_MODEL_PATH GEMMA4_TEXT_ONLY_MODEL_PATH GEMMA4_ASSISTANT_MODEL_PATH

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
EXP_IDS=""
SCENARIO="sc1"
REPS=2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            EXP_IDS="S001,S002,S003,S004,S005,S006,S007,S008,S009,S010,S011,S012,S013,S014"
            shift ;;
        --baseline)
            EXP_IDS="S001,S002"
            shift ;;
        --best)
            EXP_IDS="S003,S004"
            shift ;;
        --scenario)
            SCENARIO="$2"; shift 2 ;;
        --reps)
            REPS="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            EXP_IDS="$1"; shift ;;
    esac
done

if [[ -z "$EXP_IDS" ]]; then
    echo "Usage: $0 <EXP_IDS|--all|--baseline|--best> [--scenario sc1] [--reps 2]" >&2
    exit 1
fi

echo "=================================================="
echo "  SGLang Ablation benchmark (A100 80GB)"
echo "  Experiments : $EXP_IDS"
echo "  Scenario    : $SCENARIO"
echo "  Reps        : $REPS"
echo "  $(date)"
echo "=================================================="
echo ""
echo "  GEMMA4_MODEL_PATH=$GEMMA4_MODEL_PATH"
echo "  GEMMA4_TEXT_ONLY_MODEL_PATH=$GEMMA4_TEXT_ONLY_MODEL_PATH"
echo "  GEMMA4_ASSISTANT_MODEL_PATH=$GEMMA4_ASSISTANT_MODEL_PATH"
echo ""

python3 sglang/bench_ablation_sglang.py \
    --exp "$EXP_IDS" \
    --scenario "$SCENARIO" \
    --reps "$REPS"
