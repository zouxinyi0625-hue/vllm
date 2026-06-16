#!/bin/bash
# End-to-end run: start server + run performance test
# Usage: bash run_e011_test.sh [PORT] [TP_SIZE]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${1:-8100}
TP_SIZE=${2:-1}
NUM_PROMPTS=${3:-1000}
GPU_UTIL=${4:-0.45}

# --- Environment ---
export _ModelDataPath_=/scratch/azureml/cr/j/3516b4b376e7447d9bb012f326e5b81b/exe/wd/vllm-msn/models

echo "=== Gemma4 E011 End-to-End Test ==="
echo "  Models: ${_ModelDataPath_}"
echo "  Port: ${PORT}, TP: ${TP_SIZE}, GPU_UTIL: ${GPU_UTIL}"
echo ""

# --- Start server in background ---
echo "Starting server..."
bash "${SCRIPT_DIR}/serve_e011.sh" "$PORT" "$TP_SIZE" 24576 "$GPU_UTIL" &
SERVER_PID=$!

# Cleanup on exit
trap "echo 'Stopping server (PID=$SERVER_PID)...'; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null" EXIT

# --- Run benchmark ---
bash "${SCRIPT_DIR}/test_performance.sh" localhost "$PORT" "$NUM_PROMPTS"

echo ""
echo "=== Done. Server PID=$SERVER_PID will be stopped. ==="
