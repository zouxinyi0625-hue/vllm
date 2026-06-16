#!/bin/bash
# Profile Gemma4 MoE E011 config with Nsight Systems
# Usage: bash profile_e011.sh [prompts] [output_dir]
set -e

PROMPTS=${1:-200}
OUTPUT_DIR=${2:-/tmp/gemma4_profiles}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/autobench/configs/baseline_e011.json"

mkdir -p "$OUTPUT_DIR"

echo "=== Gemma4 MoE E011 Profiling ==="
echo "Config: $CONFIG"
echo "Prompts: $PROMPTS"
echo "Output: $OUTPUT_DIR"
echo ""

# --- Method 1: Nsight Systems (recommended for GPU kernel breakdown) ---
echo "[1/3] Running Nsight Systems profile..."
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "${OUTPUT_DIR}/gemma4_e011_nsys" \
  python "${SCRIPT_DIR}/autobench/run_experiment.py" \
  --config "$CONFIG" \
  --prompts "$PROMPTS" \
  --reps 1

echo "Nsight report: ${OUTPUT_DIR}/gemma4_e011_nsys.nsys-rep"
echo "View with: nsys-ui ${OUTPUT_DIR}/gemma4_e011_nsys.nsys-rep"
echo ""

# --- Method 2: MoE padding diagnostics ---
echo "[2/3] Running MoE padding diagnostic (50 prompts)..."
VLLM_MOE_PADDING_LOG=1 python "${SCRIPT_DIR}/autobench/run_experiment.py" \
  --config "$CONFIG" \
  --prompts 50 \
  --reps 1 2>&1 | grep "moe_padding" | tail -20

echo ""
echo "=== Profiling complete ==="
echo ""
echo "Quick stats command:"
echo "  nsys stats ${OUTPUT_DIR}/gemma4_e011_nsys.nsys-rep --report cuda_gpu_kern_sum"
