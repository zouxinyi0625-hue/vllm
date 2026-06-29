#!/bin/bash
# Basic Qwen3.6-35B-A3B FP8 serve script (no speculative decoding)
set -xe

# --- 1. Positional Arguments (with defaults) ---
PORT=${1:-8000}
TP_SIZE=${2:-1}
MAX_LEN=${3:-24576}
GPU_UTIL=${4:-0.95}
DTYPE=${5:-"auto"}
SERVED_NAME=${6:-"qwen36"}

# --- 2. Environment Setup ---
if [[ -z "${_ModelDataPath_}" ]]; then
  echo "Assuming local environment"
  model_dir="$(pwd)/INPUT_model_dir"
else
  echo "Using _ModelDataPath_: ${_ModelDataPath_}"
  model_dir="${_ModelDataPath_}/model"
fi

# --- 3. Model Path Resolution ---
model="Qwen/Qwen3.6-35B-A3B-FP8"
[[ -d "$model_dir" ]] && model="$model_dir"

# --- 4. Execute vLLM ---
echo "Starting server on port $PORT..."
echo "Serving Model As: $SERVED_NAME"

vllm serve "$model" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype "$DTYPE" \
  --kv-cache-dtype auto \
  --trust-remote-code \
  --language-model-only \
  --async-scheduling \
  --no-enable-log-requests
