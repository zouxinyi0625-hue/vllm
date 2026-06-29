#!/bin/bash
# Full offline + online benchmark pass for Qwen3.6-35B-A3B FP8
# Mirrors gemma4_moe_fp8/run.bench.fullpass.sh
set -xe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${QWEN_MODEL_PATH:-Qwen/Qwen3.6-35B-A3B}"
QUANT="fp8"
KV_DTYPE=""  # auto
MNS_SWEEP="64,128,256,512"
REPS=3
CHUNK_SIZE=200

echo "============================================"
echo "  Qwen3.6-35B-A3B FP8 Full Benchmark Pass"
echo "  Model: $MODEL"
echo "  Quantization: $QUANT"
echo "============================================"

# --- Offline throughput (sc1) ---
echo ""
echo ">>> Offline throughput - scenario sc1 <<<"
python3 bench_offline.py \
  --scenario sc1 \
  --model "$MODEL" \
  --quantization "$QUANT" \
  --max-num-seqs "$MNS_SWEEP" \
  --reps "$REPS" \
  --chunk-size "$CHUNK_SIZE" \
  --output-dir bench_results/offline_sc1

# --- Offline throughput (sc2) ---
echo ""
echo ">>> Offline throughput - scenario sc2 <<<"
python3 bench_offline.py \
  --scenario sc2 \
  --model "$MODEL" \
  --quantization "$QUANT" \
  --max-num-seqs "$MNS_SWEEP" \
  --reps "$REPS" \
  --chunk-size "$CHUNK_SIZE" \
  --output-dir bench_results/offline_sc2

echo ""
echo "=== Offline benchmarks complete ==="
echo "Results in: bench_results/"
