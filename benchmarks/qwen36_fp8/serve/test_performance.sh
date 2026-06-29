#!/bin/bash
# Performance test for Qwen3.6 FP8 serve config.
# Runs vllm bench serve at multiple request rates and concurrency levels.
# Aligned with gemma4_moe_fp8/serve/test_performance.sh
#
# Usage:
#   bash test_performance.sh --base-url http://host:port [--num-prompts N] [--model NAME]
#   bash test_performance.sh [HOST] [PORT] [NUM_PROMPTS] [MODEL_NAME]
#
# Prerequisites: server must be running (serve.sh)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/bench_results"
mkdir -p "$OUTPUT_DIR"
RESULT_FILE="${OUTPUT_DIR}/result_$(date +%Y%m%d_%H%M%S).txt"

BASE_URL=""
NUM_PROMPTS=1000
MODEL_NAME=qwen36

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    *)
      # Legacy positional: HOST PORT NUM_PROMPTS MODEL_NAME
      if [[ -z "$_POS1" ]]; then _POS1="$1"
      elif [[ -z "$_POS2" ]]; then _POS2="$1"
      elif [[ -z "$_POS3" ]]; then NUM_PROMPTS="$1"
      elif [[ -z "$_POS4" ]]; then MODEL_NAME="$1"
      fi
      shift ;;
  esac
done

# Build BASE_URL from positional args if --base-url not provided
if [[ -z "$BASE_URL" ]]; then
  HOST=${_POS1:-localhost}
  PORT=${_POS2:-8000}
  BASE_URL="http://${HOST}:${PORT}"
fi

DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/../datasets/sc1_delta_v2.jsonl}"

# Tokenizer path
TOKENIZER_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3.6-35B-A3B-FP8}"

echo "=== Qwen3.6 FP8 Online Serving Benchmark ==="
echo "  Target: ${BASE_URL}"
echo "  Model: ${MODEL_NAME}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Prompts: ${NUM_PROMPTS}"
echo ""

# --- Wait for server ready ---
echo "Waiting for server to be ready..."
MAX_WAIT=600
WAITED=0
until curl -s "${BASE_URL}/health" > /dev/null 2>&1; do
  sleep 5
  WAITED=$((WAITED + 5))
  if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo "ERROR: Server not ready after ${MAX_WAIT}s"
    exit 1
  fi
done
echo "Server ready (waited ${WAITED}s)."
echo ""

# --- Throughput test (request_rate=inf) ---
echo "========================================"
echo "  TEST 1: Max throughput (rate=inf, no concurrency limit)"
echo "========================================"
COMMON_ARGS=(
  --backend openai-chat
  --base-url "$BASE_URL"
  --endpoint /v1/chat/completions
  --model "$MODEL_NAME"
  --tokenizer "$TOKENIZER_PATH"
  --dataset-name custom
  --dataset-path "$DATASET_PATH"
  --num-prompts "$NUM_PROMPTS"
  --output-len 8192
  --request-rate inf
)

vllm bench serve "${COMMON_ARGS[@]}" \
  2>&1 | tee "$RESULT_FILE"

echo ""

# --- Max-concurrency sweep ---
for CONC in 32 64 128; do
  echo "========================================"
  echo "  TEST: rate=inf, max-concurrency=${CONC}"
  echo "========================================"
  vllm bench serve "${COMMON_ARGS[@]}" \
    --max-concurrency "$CONC" \
    2>&1 | tee -a "$RESULT_FILE"
  echo ""
done

echo "=== All tests complete ==="
echo "Results saved to: ${RESULT_FILE}"
