#!/usr/bin/env bash
# Serve a Gemma4 alignment config through vLLM OpenAI-compatible API.
# Usage: bash serve_align.sh configs/26b_e011_mtp.json [PORT]
set -euo pipefail
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SCRIPT_PATH")"

CONFIG=${1:-configs/26b_e011_mtp.json}
PORT=${2:-8100}
TP_SIZE=${TP_SIZE:-1}

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
MODEL=$(read_json model)
ASSISTANT=$(read_json assistant_model)
SERVED_MODEL_NAME=$(read_json served_model_name)
MAX_MODEL_LEN=$(read_json max_model_len)
MAX_NUM_SEQS=$(read_json max_num_seqs)
MAX_BATCHED_TOKENS=$(read_json max_num_batched_tokens)
GPU_UTIL=$(read_json gpu_memory_utilization)
QUANTIZATION=$(read_json quantization)
KV_CACHE_DTYPE=$(read_json kv_cache_dtype)
SPEC_TOKENS=$(read_json spec_tokens)
SPEC_METHOD=$(read_json spec_method)
SPECULATOR_MODEL=$(read_json speculator_model)
ASYNC_SCHED=$(read_json async_scheduling)
ENABLE_PREFIX_CACHING=$(read_json enable_prefix_caching)

# Environment vars must be set before vLLM starts.
export VLLM_USE_FLASHINFER_MOE_FP8=${VLLM_USE_FLASHINFER_MOE_FP8:-$(read_json env.VLLM_USE_FLASHINFER_MOE_FP8)}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-$(read_json env.VLLM_USE_FLASHINFER_SAMPLER)}
export VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-$(read_json env.VLLM_MOE_USE_DEEP_GEMM)}

# Local/server overrides without editing JSON. Generic variables are preferred;
# 26B-specific variables are kept for backward compatibility with the alignment run.
if [[ -n "${GEMMA4_MODEL_PATH:-}" ]]; then
  MODEL="$GEMMA4_MODEL_PATH"
elif [[ -n "${GEMMA4_26B_TEXT_ONLY_MODEL_PATH:-}" ]]; then
  MODEL="$GEMMA4_26B_TEXT_ONLY_MODEL_PATH"
elif [[ -n "${_ModelDataPath_:-}" && -d "${_ModelDataPath_}/model" ]]; then
  MODEL="${_ModelDataPath_}/model"
elif [[ -n "${_ModelDataPath_:-}" && -d "${_ModelDataPath_}/text_only" ]]; then
  MODEL="${_ModelDataPath_}/text_only"
fi

if [[ -n "${GEMMA4_ASSISTANT_MODEL_PATH:-}" ]]; then
  ASSISTANT="$GEMMA4_ASSISTANT_MODEL_PATH"
elif [[ -n "${GEMMA4_26B_ASSISTANT_MODEL_PATH:-}" ]]; then
  ASSISTANT="$GEMMA4_26B_ASSISTANT_MODEL_PATH"
elif [[ -n "${_ModelDataPath_:-}" && -d "${_ModelDataPath_}/assistant" ]]; then
  ASSISTANT="${_ModelDataPath_}/assistant"
fi

ARGS=(
  vllm serve "$MODEL"
  --served-model-name "${SERVED_MODEL_NAME:-gemma4}"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --dtype auto
  --trust-remote-code
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --no-enable-log-requests
)

if [[ -n "$QUANTIZATION" && "$QUANTIZATION" != "None" ]]; then
  ARGS+=(--quantization "$QUANTIZATION")
fi
if [[ -n "$KV_CACHE_DTYPE" && "$KV_CACHE_DTYPE" != "auto" ]]; then
  ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
else
  ARGS+=(--kv-cache-dtype auto)
fi
if [[ "$ASYNC_SCHED" == "True" || "$ASYNC_SCHED" == "true" ]]; then
  ARGS+=(--async-scheduling)
fi
if [[ "$ENABLE_PREFIX_CACHING" == "False" || "$ENABLE_PREFIX_CACHING" == "false" ]]; then
  ARGS+=(--no-enable-prefix-caching)
fi
if [[ -n "${GEMMA4_SPECULATOR_MODEL:-}" ]]; then
  SPECULATOR_MODEL="$GEMMA4_SPECULATOR_MODEL"
fi

# Speculative decoding wiring.
# - eagle3 / dspark: self-contained speculator loaded via --speculative-config
#   JSON (method carried through from the config). DSpark auto-normalizes
#   target_layer_ids / block_size from the draft; num_speculative_tokens comes
#   from spec_tokens (e.g. 7 for dspark_gemma4_12b_block7).
# - default (MTP): assistant draft loaded via --spec-model / --spec-tokens.
if [[ "${SPEC_METHOD}" == "eagle3" || "${SPEC_METHOD}" == "dspark" ]]; then
  if [[ "${SPEC_TOKENS:-0}" != "0" && -n "${SPECULATOR_MODEL:-}" ]]; then
    SPEC_CONFIG_JSON=$(python3 - "$SPECULATOR_MODEL" "$SPEC_TOKENS" "$SPEC_METHOD" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
    "method": sys.argv[3],
}))
PY
)
    ARGS+=(--speculative-config "$SPEC_CONFIG_JSON")
  fi
elif [[ "${SPEC_TOKENS:-0}" != "0" && -n "${ASSISTANT:-}" ]]; then
  ARGS+=(--spec-model "$ASSISTANT" --spec-tokens "$SPEC_TOKENS")
fi

echo "=== vLLM serve alignment ==="
echo "Config       : $CONFIG ($NAME)"
echo "Model        : $MODEL"
echo "Assistant    : ${ASSISTANT:-DISABLED}"
echo "Spec method  : ${SPEC_METHOD:-mtp}"
echo "Speculator   : ${SPECULATOR_MODEL:-N/A}"
echo "Port         : $PORT"
echo "TP size      : $TP_SIZE"
echo "Spec tokens  : ${SPEC_TOKENS:-0}"
echo "Max seqs     : $MAX_NUM_SEQS"
echo "Max batched  : $MAX_BATCHED_TOKENS"
echo "GPU util     : $GPU_UTIL"
echo "Env          : VLLM_USE_FLASHINFER_MOE_FP8=$VLLM_USE_FLASHINFER_MOE_FP8 VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER VLLM_MOE_USE_DEEP_GEMM=$VLLM_MOE_USE_DEEP_GEMM"
echo "Command      : ${ARGS[*]}"
echo ""

exec "${ARGS[@]}"
