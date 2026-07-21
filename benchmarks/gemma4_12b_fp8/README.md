# Gemma4 12B FP8 Benchmark Scaffold

This directory is the starting point for Gemma4 12B dense FP8 benchmarking.

For the first round, it intentionally keeps the **current 26B Gemma4 MoE E011 alignment configs** so the server can reproduce the known 26B numbers before switching to 12B dense.

## Scope

- Reuse the existing production-shaped dataset from `../gemma4_moe_fp8/datasets/sc1_delta_v2.jsonl`.
- Keep two 26B reference configs:
  - `26b_e011_mtp`: E011, FP8 + CUDA graphs + MTP k=5 + text-only model.
  - `26b_e011_no_mtp`: same as E011 but speculative decoding disabled.
- Provide both offline and online/server scripts.
- Do **not** change the dataset or sampling settings during alignment.

## Expected 26B reference numbers

From `../gemma4_moe_fp8/RESULTS.md`:

| Environment | Config | Output tok/s | Notes |
|---|---|---:|---|
| A100 80GB offline | E011 MTP | ~2020 | `gpu_memory_utilization=0.95` |
| A100 80GB simulated 40GB offline | E011 MTP | ~1255 | `gpu_memory_utilization=0.45` |
| A100 40GB DLIS online | E011 MTP | ~1293 | max concurrency 32 |
| A100 80GB offline | E011 no MTP | ~1465 | E012 isolate, no speculative decoding |

Use these as sanity checks before running 12B dense experiments.

## MTP gain summary

The same production-shaped dataset shows very different MTP gains for 26B-A4B MoE vs 12B dense.

| Model / mode | no-MTP output tok/s | MTP output tok/s | MTP output uplift | no-MTP total tok/s | MTP total tok/s | MTP total uplift |
|---|---:|---:|---:|---:|---:|---:|
| 26B-A4B offline | 1474.46 | 2023.06 | **+37.2%** | 6430.13 | 8723.79 | **+35.7%** |
| 12B dense offline | 880.20 | 972.89 | **+10.5%** | 3684.50 | 4391.49 | **+19.2%** |

Key interpretation:

- MTP helps both models, but it helps 26B-A4B much more.
- 26B-A4B is MoE: total parameters are ~26B, but active compute per token is much closer to A4B-level.
- 12B is dense: every token activates the full 12B model, so target verification remains compute-heavy even with MTP.
- 12B MTP acceptance is healthy, so the smaller uplift is likely dominated by dense target compute/backend cost rather than poor draft quality.

## Directory layout

```text
gemma4_12b_fp8/
├── README.md
├── configs/
│   ├── 26b_e011_mtp.json
│   ├── 26b_e011_no_mtp.json
│   ├── 12b_e011_mtp.json
│   └── 12b_e011_no_mtp.json
├── download_12b_models.sh
├── run_offline_align.sh
├── bench_offline_align.py
├── serve_align.sh
├── bench_online_align.sh
├── run_online_align.sh
└── RESULTS.md
```

## Runtime image requirement

For 12B dense experiments, use the newer unified Gemma4 image:

```text
vllm_gemma4:6
```

This image is based on:

```text
vllm/vllm-openai:gemma4-unified
```

The older image used for the first 26B alignment work:

```text
vllm_gemma4:3
```

cannot run the 12B checkpoints because the 12B target/assistant use newer `gemma4_unified` / `gemma4_unified_assistant` configs and require the unified Gemma4 support path.

## 12B model download

Default model root:

```text
/tmp/models/gemma4_12b
```

Download target and assistant:

```bash
cd benchmarks/gemma4_12b_fp8
bash download_12b_models.sh /tmp/models/gemma4_12b
```

Equivalent explicit commands:

```bash
hf download google/gemma-4-12B-it \
  --local-dir /tmp/models/gemma4_12b/model

hf download google/gemma-4-12B-it-assistant \
  --local-dir /tmp/models/gemma4_12b/assistant
```

Optional environment overrides:

```bash
export GEMMA4_MODEL_PATH=/tmp/models/gemma4_12b/model
export GEMMA4_ASSISTANT_MODEL_PATH=/tmp/models/gemma4_12b/assistant
```

The checked-in 12B configs already point to these default paths, so the exports are only needed if you place models elsewhere.

## 12B offline runs

Run 12B MTP:

```bash
cd benchmarks/gemma4_12b_fp8
bash run_offline_align.sh configs/12b_e011_mtp.json --reps 1 --prompts 1000
```

Run 12B no-MTP:

```bash
bash run_offline_align.sh configs/12b_e011_no_mtp.json --reps 1 --prompts 1000
```

## 12B online runs

One-shot online MTP with unlimited client concurrency:

```bash
cd benchmarks/gemma4_12b_fp8
bash run_online_align.sh configs/12b_e011_mtp.json --max-concurrency none --num-prompts 1000
```

One-shot online MTP with max concurrency 32:

```bash
bash run_online_align.sh configs/12b_e011_mtp.json --max-concurrency 32 --num-prompts 1000
```

Manual two-shell mode is still supported:

```bash
# Shell 1
bash serve_align.sh configs/12b_e011_mtp.json 8100

# Shell 2
bash bench_online_align.sh --port 8100 --config configs/12b_e011_mtp.json --num-prompts 1000 --max-concurrency none
```

Use no-MTP online by replacing the config path with `configs/12b_e011_no_mtp.json`.

## Dataset prerequisite

This scaffold deliberately reuses the old dataset path:

```text
benchmarks/gemma4_moe_fp8/datasets/sc1_delta_v2.jsonl
```

The dataset is not committed in this clone. On the GPU server, restore or generate the same file before running. The previous result file documents the source as:

```text
https://www.cosmos09.osdinfra.net/cosmos/MSN.DnI/shares/users/zxy/data/mai_profile/sc1_delta_v2.jsonl?property=info
```

Keep the file content unchanged for the first alignment run.

## Offline alignment

Run E011 with MTP:

```bash
cd benchmarks/gemma4_12b_fp8
bash run_offline_align.sh configs/26b_e011_mtp.json --reps 2 --prompts 1000
```

Run E011 without MTP:

```bash
cd benchmarks/gemma4_12b_fp8
bash run_offline_align.sh configs/26b_e011_no_mtp.json --reps 2 --prompts 1000
```

Outputs are written under:

```text
benchmarks/gemma4_12b_fp8/offline_results/
```

## Online alignment

Start server with E011 MTP:

```bash
cd benchmarks/gemma4_12b_fp8
bash serve_align.sh configs/26b_e011_mtp.json 8100
```

In another shell, benchmark the server:

```bash
cd benchmarks/gemma4_12b_fp8
bash bench_online_align.sh --port 8100 --config configs/26b_e011_mtp.json --num-prompts 1000 --max-concurrency 32
```

Start server without MTP:

```bash
cd benchmarks/gemma4_12b_fp8
bash serve_align.sh configs/26b_e011_no_mtp.json 8100
```

Then benchmark similarly:

```bash
bash bench_online_align.sh --port 8100 --config configs/26b_e011_no_mtp.json --num-prompts 1000 --max-concurrency 32
```

## Model path overrides

The config files contain Hugging Face defaults, but server runs usually use local model mounts.
Override with environment variables when needed:

```bash
export GEMMA4_26B_TEXT_ONLY_MODEL_PATH=/path/to/gemma4/text_only
export GEMMA4_26B_ASSISTANT_MODEL_PATH=/path/to/gemma4/assistant
```

For Azure/DLIS-style model mounts, `serve_align.sh` also looks for:

```text
${_ModelDataPath_}/text_only
${_ModelDataPath_}/assistant
```

## Next step for 12B dense

After the 26B alignment is reproduced, add 12B dense configs beside the 26B configs, keeping dataset/sampling fixed first. Only change the model path and model-specific options initially so the result is comparable.

## EAGLE-3 speculator benchmarking (draft-model exploration)

`configs/26b_eagle3.json` benchmarks the official RedHatAI EAGLE-3 speculator
against the MTP baseline, using the **same FP8 text-only target** so the only
variable is the draft model / spec method. This is line 1 (Step A) of the
draft-model roadmap (`DRAFT_MODEL_ROADMAP.md`).

Key difference from the MTP configs: EAGLE-3 sets `spec_method: "eagle3"` and
`speculator_model`, and `serve_align.sh` loads it via vLLM
`--speculative-config '{"model": ..., "num_speculative_tokens": N, "method": "eagle3"}'`
instead of `--spec-model` / `--spec-tokens` (which the MTP path still uses).

**Step 0: download the EAGLE-3 draft locally** (avoids hitting the HF Hub at
serve time). The config points `speculator_model` at
`/tmp/models/gemma4/eagle3_speculator` by default:

```bash
bash download_eagle3_speculator.sh /tmp/models/gemma4/eagle3_speculator
```

Run it end-to-end (server + bench, unlimited concurrency, same as the MTP
online run):

```bash
# Point the target at the 26B model. GEMMA4_MODEL_PATH is the generic,
# highest-priority override (use it to avoid a stale 12B _ModelDataPath_ or
# GEMMA4_MODEL_PATH silently winning):
export GEMMA4_MODEL_PATH=/tmp/models/gemma4/text_only
# The EAGLE-3 draft is read from the config's speculator_model (local path).
# Override it only if you downloaded it elsewhere:
export GEMMA4_SPECULATOR_MODEL=/tmp/models/gemma4/eagle3_speculator
# If the FP8 target + EAGLE-3 draft combo fails to load, fall back to the
# official bf16 target the checkpoint was validated against:
#   edit "model" in the config to google/gemma-4-26B-A4B-it and drop quantization

bash run_online_align.sh configs/26b_eagle3.json --max-concurrency none --num-prompts 1000
```

> **⚠️ Target/draft must match sizes.** The `RedHatAI/...26B-A4B...eagle3`
> draft only works with the **26B** target (`/tmp/models/gemma4/text_only`).
> Pointing `GEMMA4_..._MODEL_PATH` at a 12B target while using the 26B draft
> will fail (hidden size / vocab mismatch). For 12B you need a 12B eagle3 draft.

> **⏱ Startup can take several minutes.** EAGLE-3 loads two models and compiles
> CUDA graphs, so readiness easily exceeds the old 180s. The wait now defaults
> to 600s; override with `READY_TIMEOUT=900 bash run_online_align.sh ...` if
> needed.

The `vllm bench serve` output already prints the metrics we care about:
Output token throughput (tok/s), Acceptance rate (%), Acceptance length, and
Per-position acceptance (%).

### Offline EAGLE-3 run

The offline path (`bench_offline_align.py`) supports EAGLE-3 too, matching the
MTP offline baseline (`26b_e011_mtp`, offline 2023 tok/s). It reads the same
`spec_method: eagle3` config and now also collects spec-decode metrics
(acceptance rate/length + per-position) via `llm.get_metrics()`:

```bash
export GEMMA4_26B_TEXT_ONLY_MODEL_PATH=/path/to/gemma4/text_only
python bench_offline_align.py --config configs/26b_eagle3.json --reps 1 --prompts 1000
```

Results are saved to `offline_results/26b_eagle3_<ts>.json` (includes a
`spec_metrics` block). Both `--spec-model` (MTP) and `--speculative-config`
(EAGLE-3) offline paths are wired through `build_llm_kwargs`.

Then compare against the MTP baseline and check the +10~20% throughput goal:

```bash
python compare_spec_results.py --auto online_results/
# or explicitly:
python compare_spec_results.py \
  --baseline 'online_results/26b_e011_mtp_online_*.txt' \
  --candidate 'online_results/26b_eagle3_online_*.txt'
```

`compare_spec_results.py` is read-only: it parses existing result files and
prints a side-by-side table with per-metric deltas and a MET/PARTIAL/REGRESSION
verdict on output throughput. It does not run any benchmark itself.

### Fairness note

The MTP baseline (`26b_e011_mtp`, online 2113.78 tok/s, accept_len 5.03) and
this EAGLE-3 run share the same FP8 text-only target, dataset
(`sc1_delta_v2.jsonl`), 1000 prompts, output-len 8192, and unlimited
concurrency. `spec_tokens=5` matches the MTP baseline's `k=5` for an
apples-to-apples acceptance-length comparison (the RedHatAI card's example uses
3; we override to 5).

