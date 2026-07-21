#!/usr/bin/env bash
# Benchmark a running vLLM server with the unchanged sc1_delta_v2 dataset.
set -euo pipefail
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SCRIPT_PATH")"

CONFIG="configs/26b_e011_mtp.json"
HOST="localhost"
PORT="8100"
BASE_URL=""
NUM_PROMPTS=""
MAX_CONCURRENCY="32"
REQUEST_RATE="inf"
OUTPUT_DIR="online_results"
DATASET_OVERRIDE=""
RESULT_TAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
    --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --dataset-path) DATASET_OVERRIDE="$2"; shift 2 ;;
    --result-tag) RESULT_TAG="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,80p' "$SCRIPT_PATH"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

read_json() {
  local expr="$1"
  python3 - "$CONFIG" "$expr" <<'PY'
import json, sys
path, expr = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    d = json.load(f)
cur = d
for part in expr.split('.'):
    if part == '':
        continue
    cur = cur.get(part, "") if isinstance(cur, dict) else ""
print(cur if cur is not None else "")
PY
}

NAME=$(read_json name)
MODEL_NAME=$(read_json served_model_name)
TOKENIZER=$(read_json tokenizer)
DATASET_PATH=$(read_json dataset_path)
DEFAULT_NUM_PROMPTS=$(read_json num_prompts)
MAX_TOKENS=$(read_json max_tokens)

# --dataset-path overrides the config's dataset_path (e.g. per-layer MAI
# Profile prompt files). Env GEMMA4_DATASET_PATH is honored too; the CLI wins.
if [[ -n "${GEMMA4_DATASET_PATH:-}" ]]; then
  DATASET_PATH="$GEMMA4_DATASET_PATH"
fi
if [[ -n "$DATASET_OVERRIDE" ]]; then
  DATASET_PATH="$DATASET_OVERRIDE"
fi

NUM_PROMPTS=${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}
if [[ -n "${GEMMA4_TOKENIZER:-}" ]]; then
  TOKENIZER="$GEMMA4_TOKENIZER"
fi
DATASET_ABS=$(python3 - "$DATASET_PATH" <<'PY'
from pathlib import Path
import sys
script_dir = Path.cwd()
p = Path(sys.argv[1])
print(p if p.is_absolute() else (script_dir / p).resolve())
PY
)

case "${MAX_CONCURRENCY,,}" in
  none|inf|infinite|unlimited|0)
    MAX_CONCURRENCY_LABEL="none"
    ;;
  *)
    MAX_CONCURRENCY_LABEL="$MAX_CONCURRENCY"
    EXTRA_ARGS+=(--max-concurrency "$MAX_CONCURRENCY")
    ;;
esac

mkdir -p "$OUTPUT_DIR"
if [[ -n "$RESULT_TAG" ]]; then
  RESULT_FILE="${OUTPUT_DIR}/${NAME}_${RESULT_TAG}_online_$(date +%Y%m%d_%H%M%S).txt"
else
  RESULT_FILE="${OUTPUT_DIR}/${NAME}_online_$(date +%Y%m%d_%H%M%S).txt"
fi

if [[ -z "$BASE_URL" ]]; then
  BASE_URL="http://${HOST}:${PORT}"
fi

echo "=== Online alignment benchmark ==="
echo "Config         : $CONFIG ($NAME)"
echo "Base URL       : $BASE_URL"
echo "Model          : $MODEL_NAME"
echo "Tokenizer      : $TOKENIZER"
echo "Dataset        : $DATASET_ABS"
echo "Prompts        : $NUM_PROMPTS"
echo "Max tokens     : $MAX_TOKENS"
echo "Request rate   : $REQUEST_RATE"
echo "Max concurrency: $MAX_CONCURRENCY_LABEL"
echo "Result         : $RESULT_FILE"
echo ""

vllm bench serve \
  --backend openai-chat \
  --base-url "$BASE_URL" \
  --endpoint /v1/chat/completions \
  --model "$MODEL_NAME" \
  --tokenizer "$TOKENIZER" \
  --dataset-name custom \
  --dataset-path "$DATASET_ABS" \
  --num-prompts "$NUM_PROMPTS" \
  --output-len "$MAX_TOKENS" \
  --request-rate "$REQUEST_RATE" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RESULT_FILE"
