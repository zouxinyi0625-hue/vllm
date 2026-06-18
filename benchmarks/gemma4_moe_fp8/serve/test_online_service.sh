#!/bin/bash
# Online service benchmark — test remote endpoint with same params as local test_performance.sh
# Compares remote (fabric router) vs local serving performance.
#
# Usage:
#   bash test_online_service.sh [--num-prompts N]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/bench_results"
mkdir -p "$OUTPUT_DIR"
RESULT_FILE="${OUTPUT_DIR}/result_online_$(date +%Y%m%d_%H%M%S).txt"

NUM_PROMPTS=1000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    *) shift ;;
  esac
done

BASE_URL="https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/dlis-coreranker.chrona-gemma4"
MODEL_NAME=gemma4
TOKENIZER="google/gemma-4-26B-A4B-it"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/../datasets/sc1_delta_v2.jsonl}"

echo "=== Online Service Benchmark ==="
echo "  Target: ${BASE_URL}"
echo "  Model: ${MODEL_NAME}"
echo "  Tokenizer: ${TOKENIZER}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Prompts: ${NUM_PROMPTS}"
echo "  Max concurrency: 32"
echo "  Results: ${RESULT_FILE}"
echo ""

COMMON_ARGS=(
  --backend openai-chat
  --base-url "$BASE_URL"
  --endpoint /v1/chat/completions
  --model "$MODEL_NAME"
  --tokenizer "$TOKENIZER"
  --dataset-name custom
  --dataset-path "$DATASET_PATH"
  --num-prompts "$NUM_PROMPTS"
  --output-len 8192
  --request-rate inf
)

echo "========================================"
echo "  TEST: rate=inf, max-concurrency=32"
echo "========================================"
vllm bench serve "${COMMON_ARGS[@]}" \
  --max-concurrency 32 \
  2>&1 | tee "$RESULT_FILE"

echo ""
echo "=== Done. Results saved to: ${RESULT_FILE} ==="
