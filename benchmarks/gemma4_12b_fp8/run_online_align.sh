#!/usr/bin/env bash
# End-to-end online alignment benchmark: start vLLM server, wait for readiness,
# run vLLM bench serve, then stop the server.
#
# Usage:
#   bash run_online_align.sh configs/12b_e011_mtp.json --max-concurrency none
#   bash run_online_align.sh configs/12b_e011_mtp.json --max-concurrency 32 --num-prompts 1000
set -euo pipefail
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SCRIPT_PATH")"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,90p' "$SCRIPT_PATH"
  exit 0
fi

CONFIG=${1:-configs/12b_e011_mtp.json}
shift || true

PORT=8100
HOST=localhost
MAX_CONCURRENCY=32
NUM_PROMPTS=""
REQUEST_RATE=inf
OUTPUT_DIR=online_results
SERVER_LOG_DIR=server_logs
DATASET_OVERRIDE=""
RESULT_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
    --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
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

mkdir -p "$SERVER_LOG_DIR"
SERVER_LOG="${SERVER_LOG_DIR}/$(basename "$CONFIG" .json)_server_$(date +%Y%m%d_%H%M%S).log"

echo "=== Online alignment end-to-end ==="
echo "Config         : $CONFIG"
echo "Host/port      : $HOST:$PORT"
echo "Max concurrency: $MAX_CONCURRENCY"
echo "Request rate   : $REQUEST_RATE"
echo "Server log     : $SERVER_LOG"
echo ""

bash serve_align.sh "$CONFIG" "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "Stopping server PID=$SERVER_PID"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY_URL="http://${HOST}:${PORT}/v1/models"
# EAGLE-3 / large FP8 models need to load two models plus compile CUDA graphs,
# which can take several minutes. Default to 600s; override with READY_TIMEOUT.
READY_TIMEOUT=${READY_TIMEOUT:-600}
echo "Waiting for server readiness: $READY_URL (timeout ${READY_TIMEOUT}s)"
for i in $(seq 1 "$READY_TIMEOUT"); do
  if curl -fsS "$READY_URL" >/dev/null 2>&1; then
    echo "Server ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: server exited before readiness. Last 120 log lines:" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
  if [[ "$i" == "$READY_TIMEOUT" ]]; then
    echo "ERROR: server not ready after ${READY_TIMEOUT}s. Last 120 log lines:" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
done

BENCH_ARGS=(
  --host "$HOST"
  --port "$PORT"
  --config "$CONFIG"
  --request-rate "$REQUEST_RATE"
  --max-concurrency "$MAX_CONCURRENCY"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$NUM_PROMPTS" ]]; then
  BENCH_ARGS+=(--num-prompts "$NUM_PROMPTS")
fi
if [[ -n "$DATASET_OVERRIDE" ]]; then
  BENCH_ARGS+=(--dataset-path "$DATASET_OVERRIDE")
fi
if [[ -n "$RESULT_TAG" ]]; then
  BENCH_ARGS+=(--result-tag "$RESULT_TAG")
fi

bash bench_online_align.sh "${BENCH_ARGS[@]}"
