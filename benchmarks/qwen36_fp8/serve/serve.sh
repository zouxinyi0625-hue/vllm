#!/bin/bash
# Serve Qwen3.6-35B-A3B with FP8 quantization
# Config aligned with gemma4_moe_fp8/serve/serve_e011.sh
set -xe

# --- 1. Positional Arguments ---
PORT=${1:-8000}
TP_SIZE=${2:-1}
MAX_LEN=${3:-24576}
GPU_UTIL=${4:-0.95}
SERVED_NAME=${5:-"qwen36"}
MAX_NUM_SEQS=${6:-128}
MAX_BATCHED_TOKENS=${7:-16384}

# --- 2. Model Path Resolution ---
if [[ -z "${_ModelDataPath_}" ]]; then
  echo "Assuming local environment"
  model_dir="$(pwd)/INPUT_model_dir"
else
  model_dir="${_ModelDataPath_}"
fi

model="${QWEN_MODEL_PATH:-${model_dir}/model}"

# Fallback to HuggingFace if local path doesn't exist
[[ ! -d "$model" ]] && model="Qwen/Qwen3.6-35B-A3B-FP8"

# --- 3. Environment ---
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

# A100 (sm_80): FlashInfer FP8 MoE requires Hopper; use Marlin fallback
GPU_ARCH=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0)[0])" 2>/dev/null || echo "8")
if [[ "$GPU_ARCH" -lt 9 ]]; then
  export VLLM_USE_FLASHINFER_MOE_FP8=${VLLM_USE_FLASHINFER_MOE_FP8:-0}
  export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
fi

# --- 4. Build vllm serve command ---
echo "Starting vLLM server (Qwen3.6 FP8)..."
echo "  Model: $model"
echo "  Port: $PORT, TP: $TP_SIZE, MaxLen: $MAX_LEN, GPU_UTIL: $GPU_UTIL"
echo "  MaxNumSeqs: $MAX_NUM_SEQS, MaxBatchedTokens: $MAX_BATCHED_TOKENS"

vllm serve "$model" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --dtype auto \
  --kv-cache-dtype auto \
  --trust-remote-code \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --async-scheduling \
  --no-enable-log-requests \
  --language-model-only
