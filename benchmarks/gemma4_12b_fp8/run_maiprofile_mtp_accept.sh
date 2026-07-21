#!/usr/bin/env bash
# Per-layer MAI Profile acceptance-rate + aggregate offline throughput for the
# Google MTP model (Gemma4-12B assistant), in ONE command.
#
# WHAT YOU GET
#   - Per layer: acceptance_rate, acceptance_length, per-position acceptance,
#     output tok/s.
#   - Aggregate: one offline throughput number = total output tokens / total
#     wall time across all layers (the correct way to "add up" independent
#     offline runs; a plain sum of per-run tok/s would be meaningless).
#   - A single summary table (printed + saved to maiprofile_mtp_summary.json).
#
# HOW IT WORKS
#   1. Convert each DSpark eval file maiprofile_<layer>.jsonl -> sc1-style
#      {"prompt": ...} via convert_maiprofile_eval_to_sc1.py.
#   2. For each layer, run bench_offline_align.py with configs/12b_e011_mtp.json
#      and --dataset-path pointing at that layer's prompts.
#   3. Parse each run's saved offline_results/*.json for spec metrics + timing,
#      and roll up an aggregate throughput.
#
# RUN ON THE SERVER (GPU + Azure ML mount visible), inside the vLLM venv.
#
# USAGE
#   # defaults: read the 5 short layers from $AZURE_ML_INPUT_msndni, 200 prompts,
#   # 1 rep (eval-style), Google MTP config.
#   bash run_maiprofile_mtp_accept.sh
#
#   # explicit eval-datasets dir (folder holding maiprofile_<layer>.jsonl):
#   EVAL_DIR=/path/to/eval_datasets bash run_maiprofile_mtp_accept.sh
#
#   # subset of layers / more prompts / more reps:
#   LAYERS="layer1_actual,layer3_seasonality" NUM_PROMPTS=200 REPS=1 \
#     bash run_maiprofile_mtp_accept.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Config (override via env) ----
CONFIG="${CONFIG:-configs/12b_e011_mtp.json}"      # Google MTP (target+assistant)
LAYERS="${LAYERS:-short}"                            # 'short' = 5 default layers
NUM_PROMPTS="${NUM_PROMPTS:-200}"                    # eval sets are 200/layer
REPS="${REPS:-1}"                                    # 1 rep = eval-style
PROMPTS_DIR="${PROMPTS_DIR:-maiprofile_bench_prompts}"
SUMMARY_JSON="${SUMMARY_JSON:-maiprofile_mtp_summary.json}"
# EVAL_DIR: dir containing maiprofile_<layer>.jsonl. If unset, the converter
# falls back to $AZURE_ML_INPUT_msndni/.../short_layers/eval_datasets.
EVAL_DIR="${EVAL_DIR:-}"

# Model path overrides (passed through to bench_offline_align.py). Set these if
# your models are not already baked into the config's paths.
#   GEMMA4_MODEL_PATH, GEMMA4_ASSISTANT_MODEL_PATH
export GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-/tmp/models/gemma4_12b/model}"
export GEMMA4_ASSISTANT_MODEL_PATH="${GEMMA4_ASSISTANT_MODEL_PATH:-/tmp/models/gemma4_12b/assistant}"

echo "============================================="
echo " MAI Profile per-layer accept rate — Google MTP"
echo "============================================="
echo "  config      : $CONFIG"
echo "  layers      : $LAYERS"
echo "  prompts/layer: $NUM_PROMPTS   reps: $REPS"
echo "  target      : $GEMMA4_MODEL_PATH"
echo "  assistant   : $GEMMA4_ASSISTANT_MODEL_PATH"
echo ""

# ---- Step 1: convert eval datasets -> sc1 per-layer prompt files ----
echo "===== Step 1/3: convert eval datasets to benchmark format ====="
CONV_ARGS=(--output-dir "$PROMPTS_DIR" --layers "$LAYERS")
if [[ -n "$EVAL_DIR" ]]; then
  CONV_ARGS+=(--input-dir "$EVAL_DIR")
fi
python3 convert_maiprofile_eval_to_sc1.py "${CONV_ARGS[@]}"
echo ""

# Resolve the layer list actually produced (files present).
mapfile -t LAYER_FILES < <(ls "$PROMPTS_DIR"/sc1_maiprofile_*.jsonl 2>/dev/null | grep -v '_all\.jsonl$' || true)
if [[ ${#LAYER_FILES[@]} -eq 0 ]]; then
  echo "[FATAL] no per-layer prompt files in $PROMPTS_DIR" >&2
  exit 1
fi

# ---- Step 2: per-layer offline benchmark ----
echo "===== Step 2/3: offline benchmark per layer ====="
# Snapshot existing result files so we can find each new one deterministically.
mkdir -p offline_results
declare -a LAYER_NAMES RESULT_FILES
for f in "${LAYER_FILES[@]}"; do
  base="$(basename "$f" .jsonl)"                 # sc1_maiprofile_<layer>
  layer="${base#sc1_maiprofile_}"
  echo ""
  echo "--- Layer: $layer  ($f) ---"
  before="$(ls -1 offline_results/*.json 2>/dev/null | sort || true)"
  python3 bench_offline_align.py \
    --config "$CONFIG" \
    --dataset-path "$f" \
    --prompts "$NUM_PROMPTS" \
    --reps "$REPS"
  after="$(ls -1 offline_results/*.json 2>/dev/null | sort || true)"
  # Newest result file for this layer = set difference (fallback: newest mtime).
  newfile="$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | tail -1)"
  if [[ -z "$newfile" ]]; then
    newfile="$(ls -t offline_results/*.json 2>/dev/null | head -1)"
  fi
  LAYER_NAMES+=("$layer")
  RESULT_FILES+=("$newfile")
  echo "  result: $newfile"
done

# ---- Step 3: aggregate + summary table ----
echo ""
echo "===== Step 3/3: summary ====="
python3 - "$SUMMARY_JSON" "${#LAYER_NAMES[@]}" "${LAYER_NAMES[@]}" "${RESULT_FILES[@]}" <<'PY'
import json, sys

summary_path = sys.argv[1]
n = int(sys.argv[2])
names = sys.argv[3:3 + n]
files = sys.argv[3 + n:3 + 2 * n]

rows = []
tot_out_tokens = 0.0
tot_time = 0.0
for layer, path in zip(names, files):
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
    except Exception as e:
        rows.append({"layer": layer, "error": f"read failed: {e}", "file": path})
        continue
    spec = r.get("spec_metrics") or {}
    # Sum output tokens and wall time across reps for a wall-clock throughput.
    out_tokens = sum(row.get("output_tokens_total", 0) for row in r.get("rows", []))
    elapsed = sum(row.get("elapsed_time", 0.0) for row in r.get("rows", []))
    tot_out_tokens += out_tokens
    tot_time += elapsed
    rows.append({
        "layer": layer,
        "acceptance_rate": spec.get("acceptance_rate"),
        "acceptance_length": spec.get("acceptance_length"),
        "per_position_acceptance": spec.get("per_position_acceptance"),
        "mean_output_tps": r.get("mean_output_tps"),
        "output_tokens_total": out_tokens,
        "elapsed_time_total": round(elapsed, 2),
        "num_prompts": r.get("num_prompts"),
        "result_file": path,
    })

agg_tps = round(tot_out_tokens / tot_time, 2) if tot_time else 0.0

# Pretty table
print()
print(f"{'layer':30s} {'accept_rate%':>12s} {'accept_len':>11s} {'out_tok/s':>10s} {'prompts':>8s}")
print("-" * 76)
for row in rows:
    if "error" in row:
        print(f"{row['layer']:30s}  ERROR: {row['error']}")
        continue
    ar = row["acceptance_rate"]
    al = row["acceptance_length"]
    tps = row["mean_output_tps"]
    npr = row["num_prompts"]
    ar_s = f"{ar:.2f}" if isinstance(ar, (int, float)) else "n/a"
    al_s = f"{al:.2f}" if isinstance(al, (int, float)) else "n/a"
    tps_s = f"{tps:.1f}" if isinstance(tps, (int, float)) else "n/a"
    print(f"{row['layer']:30s} {ar_s:>12s} {al_s:>11s} {tps_s:>10s} {str(npr):>8s}")
print("-" * 76)
print(f"{'AGGREGATE (all layers)':30s} {'':>12s} {'':>11s} {agg_tps:>10.1f}")
print()
print(f"Aggregate offline throughput = {tot_out_tokens:.0f} output tokens / "
      f"{tot_time:.1f}s = {agg_tps} tok/s")
print()

out = {
    "config": None,
    "num_layers": len(rows),
    "per_layer": rows,
    "aggregate_output_tokens": tot_out_tokens,
    "aggregate_wall_time_s": round(tot_time, 2),
    "aggregate_offline_output_tps": agg_tps,
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"Saved summary: {summary_path}")

# Per-position acceptance detail (below the main table).
print("\nPer-position acceptance (%) by layer:")
for row in rows:
    if "error" in row:
        continue
    pp = row.get("per_position_acceptance") or {}
    if pp:
        pp_s = "  ".join(f"pos{k}={v}" for k, v in pp.items())
        print(f"  {row['layer']:30s} {pp_s}")
PY

echo ""
echo "Done. Summary JSON: $SUMMARY_JSON"
