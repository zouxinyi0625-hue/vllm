#!/usr/bin/env bash
# Per-layer ONLINE benchmark of ANY Gemma4-12B config on the MAI Profile eval
# layers, in one command. Works for both the MTP draft config and the no-MTP
# dense baseline (and any other config) — pick via CONFIG=.
#
# WHY ONLINE (not offline): the acceptance rate / length / per-position numbers
# are only emitted by `vllm bench serve` (online). The offline path runs with
# stats disabled and cannot report them. So per-layer accept rate = online only.
# (A dense/no-MTP config simply has no Speculative Decoding block; the summary
#  shows accept rate = n/a for it, which is expected — you compare tok/s.)
#
# WHAT YOU GET
#   - Per layer: acceptance rate, acceptance length, per-position acceptance
#     (blank for dense/no-MTP), output tok/s (all parsed from bench stdout).
#   - A summary table + a JSON file (SUMMARY_JSON, default depends on config).
#
# PIPELINE
#   1. convert_maiprofile_eval_to_sc1.py: DSpark eval -> sc1 {"prompt":...} files.
#   2. serve_align.sh: start the vLLM server once for the chosen CONFIG.
#   3. bench_online_align.sh --dataset-path <layer> --result-tag <layer>: per layer.
#   4. parse each online_results/*_<layer>_*.txt for the metrics; summarize.
#
# RUN ON THE SERVER (GPU + Azure ML mount), inside the vLLM venv.
#
# USAGE
#   # Google MTP draft (default config):
#   bash run_maiprofile_online.sh
#
#   # no-MTP dense baseline (use a distinct SUMMARY_JSON so it doesn't clobber
#   # the MTP summary; accept rate will be n/a, compare the tok/s):
#   CONFIG=configs/12b_e011_no_mtp.json \
#     SUMMARY_JSON=maiprofile_dense_online_summary.json \
#     bash run_maiprofile_online.sh
#
#   # explicit eval-datasets dir; subset of layers; more prompts:
#   EVAL_DIR=/path/to/eval_datasets LAYERS="layer1_actual,layer3_seasonality" \
#     NUM_PROMPTS=200 bash run_maiprofile_online.sh
#
#   # unlimited concurrency (max throughput, matches the RESULTS.md baseline runs):
#   MAX_CONCURRENCY=none bash run_maiprofile_online.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Config (override via env) ----
CONFIG="${CONFIG:-configs/12b_e011_mtp.json}"      # default: Google MTP (target + assistant)
LAYERS="${LAYERS:-short}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-none}"          # 'none' = unlimited (baseline style)
PORT="${PORT:-8100}"
HOST="${HOST:-localhost}"
PROMPTS_DIR="${PROMPTS_DIR:-maiprofile_bench_prompts}"
OUTPUT_DIR="${OUTPUT_DIR:-online_results}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-server_logs}"
# Default summary name derives from the config, so MTP vs dense don't clobber
# each other even if you forget to set SUMMARY_JSON.
_CFG_BASE="$(basename "$CONFIG" .json)"
SUMMARY_JSON="${SUMMARY_JSON:-maiprofile_${_CFG_BASE}_online_summary.json}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
EVAL_DIR="${EVAL_DIR:-}"

# Model overrides passed to serve_align.sh. Only export when the user actually
# sets them — otherwise the config's own model/assistant paths are used. This
# matters for 26B configs (which carry HF repo names in the JSON): a blanket
# 12B default here would silently override the 26B target.
if [[ -n "${GEMMA4_MODEL_PATH:-}" ]]; then
  export GEMMA4_MODEL_PATH
fi
if [[ -n "${GEMMA4_ASSISTANT_MODEL_PATH:-}" ]]; then
  export GEMMA4_ASSISTANT_MODEL_PATH
fi
# TP size for the served model. 26B-A4B MoE needs tp>=2; 12B dense fits on tp=1.
# Passed through to serve_align.sh.
export TP_SIZE="${TP_SIZE:-1}"

echo "============================================="
echo " MAI Profile per-layer online benchmark — ${_CFG_BASE} (ONLINE)"
echo "============================================="
echo "  config       : $CONFIG"
echo "  layers       : $LAYERS"
echo "  prompts/layer: $NUM_PROMPTS   max_concurrency: $MAX_CONCURRENCY"
echo "  tp size      : $TP_SIZE"
echo "  target       : ${GEMMA4_MODEL_PATH:-<from config>}"
echo "  assistant    : ${GEMMA4_ASSISTANT_MODEL_PATH:-<from config>}"
echo ""

# ---- Step 1: convert eval datasets -> sc1 per-layer prompt files ----
echo "===== Step 1/4: convert eval datasets to benchmark format ====="
CONV_ARGS=(--output-dir "$PROMPTS_DIR" --layers "$LAYERS")
if [[ -n "$EVAL_DIR" ]]; then
  CONV_ARGS+=(--input-dir "$EVAL_DIR")
fi
python3 convert_maiprofile_eval_to_sc1.py "${CONV_ARGS[@]}"
echo ""

mapfile -t LAYER_FILES < <(ls "$PROMPTS_DIR"/sc1_maiprofile_*.jsonl 2>/dev/null | grep -v '_all\.jsonl$' || true)
if [[ ${#LAYER_FILES[@]} -eq 0 ]]; then
  echo "[FATAL] no per-layer prompt files in $PROMPTS_DIR" >&2
  exit 1
fi

# ---- Step 2: start the vLLM server ONCE ----
echo "===== Step 2/4: start vLLM server (loads the config's model(s) once) ====="
mkdir -p "$SERVER_LOG_DIR" "$OUTPUT_DIR"

# Pre-clean: kill any stale vLLM/serve_align holding the GPU or port 8100 from a
# previous run. A leftover server would OOM this one (or squat the port), which
# shows up downstream as "ConnectionRefused" when the bench client can't reach it.
echo "  pre-clean: killing any stale vllm serve / serve_align ..."
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "serve_align.sh" 2>/dev/null || true
# also free the port if something still holds it
if command -v fuser >/dev/null 2>&1; then fuser -k "${PORT}/tcp" 2>/dev/null || true; fi
sleep 5

SERVER_LOG="${SERVER_LOG_DIR}/$(basename "$CONFIG" .json)_maiprofile_server_$(date +%Y%m%d_%H%M%S).log"
# setsid: run serve in its OWN process group so we can kill the WHOLE tree
# (bash wrapper + vllm children). Killing just the bash PID leaves vllm orphaned
# and holding GPU memory — the root cause of a failed second run.
setsid bash serve_align.sh "$CONFIG" "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
SERVER_PGID="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
echo "  server PID: $SERVER_PID  PGID: ${SERVER_PGID:-?}  log: $SERVER_LOG"

cleanup() {
  echo ""
  echo "Stopping server (PID=$SERVER_PID PGID=${SERVER_PGID:-?})"
  # Kill the whole process group (bash wrapper + vllm workers).
  if [[ -n "${SERVER_PGID:-}" ]]; then
    kill -TERM "-${SERVER_PGID}" 2>/dev/null || true
    sleep 5
    kill -KILL "-${SERVER_PGID}" 2>/dev/null || true
  fi
  kill -9 "$SERVER_PID" 2>/dev/null || true
  # Belt-and-suspenders: sweep any vllm still bound to this model/port.
  pkill -9 -f "vllm serve" 2>/dev/null || true
  if command -v fuser >/dev/null 2>&1; then fuser -k "${PORT}/tcp" 2>/dev/null || true; fi
  wait "$SERVER_PID" 2>/dev/null || true
  sleep 3   # let the GPU free before any next invocation
}
trap cleanup EXIT

READY_URL="http://${HOST}:${PORT}/v1/models"
echo "  waiting for readiness: $READY_URL (timeout ${READY_TIMEOUT}s)"
for i in $(seq 1 "$READY_TIMEOUT"); do
  if curl -fsS "$READY_URL" >/dev/null 2>&1; then
    echo "  server ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[FATAL] server exited before readiness. Last 60 log lines:" >&2
    tail -60 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 1
  if [[ "$i" == "$READY_TIMEOUT" ]]; then
    echo "[FATAL] server not ready after ${READY_TIMEOUT}s. Last 60 log lines:" >&2
    tail -60 "$SERVER_LOG" >&2 || true
    exit 1
  fi
done
echo ""

# ---- Step 3: benchmark each layer against the live server ----
echo "===== Step 3/4: per-layer online benchmark ====="
declare -a LAYER_NAMES RESULT_FILES
for f in "${LAYER_FILES[@]}"; do
  base="$(basename "$f" .jsonl)"          # sc1_maiprofile_<layer>
  layer="${base#sc1_maiprofile_}"
  echo ""
  echo "--- Layer: $layer  ($f) ---"
  before="$(ls -1 "$OUTPUT_DIR"/*_"${layer}"_online_*.txt 2>/dev/null | sort || true)"
  bash bench_online_align.sh \
    --config "$CONFIG" \
    --host "$HOST" --port "$PORT" \
    --dataset-path "$f" \
    --result-tag "$layer" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --output-dir "$OUTPUT_DIR"
  after="$(ls -1 "$OUTPUT_DIR"/*_"${layer}"_online_*.txt 2>/dev/null | sort || true)"
  newfile="$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | tail -1)"
  [[ -z "$newfile" ]] && newfile="$(ls -t "$OUTPUT_DIR"/*_"${layer}"_online_*.txt 2>/dev/null | head -1)"
  LAYER_NAMES+=("$layer")
  RESULT_FILES+=("$newfile")
  echo "  result: $newfile"
done

# ---- Step 4: parse metrics + summary ----
echo ""
echo "===== Step 4/4: summary ====="
python3 - "$SUMMARY_JSON" "${#LAYER_NAMES[@]}" "${LAYER_NAMES[@]}" "${RESULT_FILES[@]}" <<'PY'
import json, re, sys

summary_path = sys.argv[1]
n = int(sys.argv[2])
names = sys.argv[3:3 + n]
files = sys.argv[3 + n:3 + 2 * n]


def parse_bench_txt(path):
    """Pull metrics out of `vllm bench serve` stdout.

    Exact field names from vllm/benchmarks/serve.py:1080-1106:
      --------------- Speculative Decoding ---------------
      Acceptance rate (%):                     80.58
      Acceptance length:                       5.03
      Drafts:                                  284215
      Draft tokens:                            1421075
      Accepted tokens:                         1145159
      Per-position acceptance (%):
        Position 0:                            93.61
        Position 1:                            87.08
    Plus the general throughput block:
      Output token throughput (tok/s):         1133.69
      Total Token throughput (tok/s):          4361.09
      Request throughput (req/s):              0.74
    """
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except Exception as e:
        return {"error": f"read failed: {e}"}

    def grab(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    out = {
        "output_tps": grab(r"Output token throughput \(tok/s\):\s*([\d.]+)"),
        "total_tps": grab(r"Total [Tt]oken throughput \(tok/s\):\s*([\d.]+)"),
        "request_throughput": grab(r"Request throughput \(req/s\):\s*([\d.]+)"),
        "acceptance_rate": grab(r"Acceptance rate \(%\):\s*([\d.]+)"),
        "acceptance_length": grab(r"Acceptance length:\s*([\d.]+)"),
        "num_drafts": grab(r"^Drafts:\s*([\d.]+)"),
        "num_draft_tokens": grab(r"Draft tokens:\s*([\d.]+)"),
        "num_accepted_tokens": grab(r"Accepted tokens:\s*([\d.]+)"),
    }
    # Per-position: "  Position N:   xx.xx"
    perpos = {}
    for m in re.finditer(r"Position\s*(\d+):\s*([\d.]+)", text, re.IGNORECASE):
        perpos[int(m.group(1))] = float(m.group(2))
    out["per_position_acceptance"] = {k: perpos[k] for k in sorted(perpos)} or None
    return out


rows = []
for layer, path in zip(names, files):
    if not path:
        rows.append({"layer": layer, "error": "no result file found"})
        continue
    m = parse_bench_txt(path)
    m["layer"] = layer
    m["result_file"] = path
    rows.append(m)

# Table
print()
hdr = f"{'layer':30s} {'accept_rate%':>12s} {'accept_len':>11s} {'out_tok/s':>10s}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    if "error" in r:
        print(f"{r['layer']:30s}  ERROR: {r['error']}")
        continue
    def s(v, f="{:.2f}"):
        return f.format(v) if isinstance(v, (int, float)) else "n/a"
    print(f"{r['layer']:30s} {s(r.get('acceptance_rate')):>12s} "
          f"{s(r.get('acceptance_length')):>11s} {s(r.get('output_tps'), '{:.1f}'):>10s}")
print()

with open(summary_path, "w", encoding="utf-8") as fh:
    json.dump({"num_layers": len(rows), "per_layer": rows}, fh, indent=2, ensure_ascii=False)
print(f"Saved summary: {summary_path}")

# Per-position detail
print("\nPer-position acceptance (%) by layer:")
any_pp = False
for r in rows:
    pp = r.get("per_position_acceptance")
    if pp:
        any_pp = True
        print(f"  {r['layer']:30s} " + "  ".join(f"pos{k}={v}" for k, v in pp.items()))
if not any_pp:
    print("  (none parsed — check the raw .txt files; field names may differ in your vllm version)")
PY

echo ""
echo "Done. Per-layer raw logs in $OUTPUT_DIR/*_<layer>_online_*.txt"
echo "Summary JSON: $SUMMARY_JSON"
