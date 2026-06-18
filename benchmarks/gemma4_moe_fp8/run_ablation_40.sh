#!/usr/bin/env bash
# run_ablation_40.sh — Ablation experiments for 40GB A100 (gpu_mem=0.45).
#
# Skips BF16 experiments (E001, E003, E014, E015) that exceed 40GB VRAM.
# All FP8 experiments run with gpu_memory_utilization=0.45.
# Experiment IDs are kept consistent with the full run_ablation.sh.
#
# Usage:
#   ./run_ablation_40.sh --all --scenario sc1 --reps 5
#   ./run_ablation_40.sh E002,E004,E006 --reps 3

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Default model paths (override via env)
# ---------------------------------------------------------------------------
: "${GEMMA4_MODEL_PATH:=google/gemma-4-26B-A4B-it}"
: "${GEMMA4_TEXT_ONLY_MODEL_PATH:=${GEMMA4_MODEL_PATH}-text-only}"
: "${GEMMA4_ASSISTANT_MODEL_PATH:=google/gemma-4-26B-A4B-it-assistant}"
export GEMMA4_MODEL_PATH GEMMA4_TEXT_ONLY_MODEL_PATH GEMMA4_ASSISTANT_MODEL_PATH

echo "  GEMMA4_MODEL_PATH           = ${GEMMA4_MODEL_PATH}"
echo "  GEMMA4_TEXT_ONLY_MODEL_PATH = ${GEMMA4_TEXT_ONLY_MODEL_PATH}"
echo "  GEMMA4_ASSISTANT_MODEL_PATH = ${GEMMA4_ASSISTANT_MODEL_PATH}"

# ---------------------------------------------------------------------------
# 40GB A100 override: force gpu_memory_utilization=0.45
# ---------------------------------------------------------------------------
export ABLATION_GPU_MEM_OVERRIDE="0.45"

# ---------------------------------------------------------------------------
# BF16 experiments to skip (will OOM on 40GB with gpu_mem=0.45)
# ---------------------------------------------------------------------------
SKIP_EXPS=("E001" "E003" "E014" "E015")

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
EXP_IDS=""
SCENARIO="sc1"
REPS=2
RUN_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)          RUN_ALL=1; shift ;;
    --scenario)     SCENARIO="$2"; shift 2 ;;
    --reps)         REPS="$2"; shift 2 ;;
    --list)
      python3 bench_ablation.py --exp E002 --list
      exit 0
      ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)              EXP_IDS="$1"; shift ;;
  esac
done

if [[ $RUN_ALL -eq 1 ]]; then
  # All 15 experiments minus the BF16 ones that won't fit
  EXP_IDS="E002,E004,E005,E006,E007,E008,E009,E010,E011,E012,E013"
fi

if [[ -z "$EXP_IDS" ]]; then
  echo "ERROR: no experiment ID provided. Use --all or specify e.g. E002"
  exit 1
fi

# ---------------------------------------------------------------------------
# Per-experiment environment variable table (same as run_ablation.sh)
# ---------------------------------------------------------------------------
set_env_for_exp() {
  local exp="$1"
  export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
  export VLLM_USE_FLASHINFER_SAMPLER="0"

  case "$exp" in
    E001|E003|E014|E015)
      export VLLM_USE_FLASHINFER_MOE_FP8="0"
      ;;
    E002|E004|E005|E006|E007|E008|E009|E010|E011|E012|E013)
      COMPUTE_CAP=$(python3 -c "import torch; cc=torch.cuda.get_device_capability(); print(cc[0]*10+cc[1])" 2>/dev/null || echo "0")
      if [[ "$COMPUTE_CAP" -ge 90 ]]; then
        export VLLM_USE_FLASHINFER_MOE_FP8="1"
        echo "  → H100 detected (sm_${COMPUTE_CAP}): enabling VLLM_USE_FLASHINFER_MOE_FP8=1"
      else
        export VLLM_USE_FLASHINFER_MOE_FP8="0"
        echo "  → non-H100 (sm_${COMPUTE_CAP}): VLLM_USE_FLASHINFER_MOE_FP8=0 (Marlin FP8 MoE fallback)"
      fi
      ;;
    *)
      export VLLM_USE_FLASHINFER_MOE_FP8="0"
      echo "WARNING: unknown exp '$exp' — using safe defaults"
      ;;
  esac

  echo "  ENV: VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND  VLLM_USE_FLASHINFER_MOE_FP8=$VLLM_USE_FLASHINFER_MOE_FP8  VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
}

# ---------------------------------------------------------------------------
# Helper: check if experiment should be skipped
# ---------------------------------------------------------------------------
should_skip() {
  local exp="$1"
  for skip in "${SKIP_EXPS[@]}"; do
    if [[ "$exp" == "$skip" ]]; then
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
IFS=',' read -ra EXPS <<< "$EXP_IDS"

echo "=================================================="
echo "  Ablation benchmark (40GB A100, gpu_mem=0.45)"
echo "  Experiments : ${EXP_IDS}"
echo "  Skipping    : ${SKIP_EXPS[*]} (BF16, exceeds VRAM)"
echo "  Scenario    : ${SCENARIO}"
echo "  Reps        : ${REPS}"
echo "  $(date)"
echo "=================================================="

FAILED=()
SKIPPED=()
for EXP in "${EXPS[@]}"; do
  EXP="$(echo "$EXP" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"

  if should_skip "$EXP"; then
    echo ""
    echo ">>> SKIPPING ${EXP} (BF16 — exceeds 40GB VRAM at gpu_mem=0.45)"
    SKIPPED+=("$EXP")
    continue
  fi

  echo ""
  echo ">>> Setting up environment for ${EXP}"
  set_env_for_exp "$EXP"

  echo ">>> Launching bench_ablation.py --exp ${EXP} --scenario ${SCENARIO} --reps ${REPS} --gpu-mem 0.45"
  if python3 bench_ablation.py \
      --exp "${EXP}" \
      --scenario "${SCENARIO}" \
      --reps "${REPS}" \
      --gpu-mem 0.45; then
    echo ">>> ${EXP} COMPLETED"
  else
    echo ">>> ${EXP} FAILED (exit $?)"
    FAILED+=("$EXP")
  fi
done

echo ""
echo "=================================================="
echo "  Ablation run finished (40GB A100)"
echo "  Results: ablation_results/all_runs.csv"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
  echo "  SKIPPED experiments: ${SKIPPED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  FAILED experiments: ${FAILED[*]}"
  exit 1
fi
echo "  All experiments PASSED"
echo "=================================================="
