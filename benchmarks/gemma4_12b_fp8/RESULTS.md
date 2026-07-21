# Gemma4 12B FP8 Benchmark Scaffold — 26B Alignment Results

This file records the 26B Gemma4 MoE alignment runs used to validate the new `benchmarks/gemma4_12b_fp8/` scaffold before switching to 12B dense FP8 experiments.

## Environment

- Date: 2026-07-07
- Branch: `feat/gemma4-12b-fp8-bench`
- Runtime image note:
  - 26B alignment originally worked on `vllm_gemma4:3`.
  - 12B dense runs require the newer `vllm_gemma4:6` image, based on `vllm/vllm-openai:gemma4-unified`.
  - `vllm_gemma4:3` cannot run the 12B checkpoints because they use newer `gemma4_unified` / `gemma4_unified_assistant` configs.
- Server path used by user:
  ```text
  /scratch/azureml/cr/j/574c7ade511e4bb08b1db2772cdac624/exe/wd/vllm-msn/benchmarks/gemma4_12b_fp8
  ```
- Model paths:
  ```bash
  export GEMMA4_26B_TEXT_ONLY_MODEL_PATH=/tmp/models/gemma4/text_only
  export GEMMA4_26B_ASSISTANT_MODEL_PATH=/tmp/models/gemma4/assistant
  export _ModelDataPath_=/tmp/models/gemma4
  ```
- Dataset:
  ```text
  ../gemma4_moe_fp8/datasets/sc1_delta_v2.jsonl
  ```
- Dataset shape from run output:
  - Requests: 1000
  - Total input tokens: 4,371,285
  - Avg input tokens/request: ~4,371

## Configs

### `26b_e011_mtp`

- Model: `/tmp/models/gemma4/text_only`
- Assistant: `/tmp/models/gemma4/assistant`
- FP8 weights
- CUDA graphs (`enforce_eager=false`)
- MTP speculative decoding with `spec_tokens=5`
- `max_model_len=24576`
- `max_num_seqs=128`
- `max_num_batched_tokens=16384`
- `gpu_memory_utilization=0.95`
- `async_scheduling=true`
- Prefix caching enabled

### `26b_e011_no_mtp`

Same as `26b_e011_mtp`, but speculative decoding disabled:

```json
"assistant_model": "",
"spec_tokens": 0
```

## Offline Results

Commands:

```bash
bash run_offline_align.sh configs/26b_e011_mtp.json --reps 1 --prompts 1000
bash run_offline_align.sh configs/26b_e011_no_mtp.json --reps 1 --prompts 1000
```

Output files on server:

```text
offline_results/26b_e011_mtp_20260707_060751.json
offline_results/26b_e011_no_mtp_20260707_062601.json
```

| Config | Reps | Output tok/s | Total tok/s | Notes |
|---|---:|---:|---:|---|
| `26b_e011_mtp` | 1 | **2023.06** | 8723.79 | Matches previous E011 baseline (~2020) |
| `26b_e011_no_mtp` | 1 | **1474.46** | 6430.13 | Matches previous E012/no-MTP baseline (~1465). Run 2026-07-07. |
| `26b_eagle3` | 1 | **888.44** | 3893.85 | 2026-07-08. Official RedHatAI EAGLE-3 draft + FP8 text-only target. **Slower than no-MTP** — draft appears to net-negative under FP8 target (low acceptance suspected; offline can't read accept metrics, `disable_log_stats=True`). Needs same-env no-MTP re-run to rule out environment drift. |

### EAGLE-3 offline note (2026-07-08)

The official EAGLE-3 speculator ran at **888 tok/s offline**, *below* the no-MTP
baseline (1474 from 2026-07-07) and far below MTP (2023). Config diff confirms
`26b_eagle3` is identical to `26b_e011_no_mtp` except the spec fields
(`spec_method=eagle3`, `spec_tokens=5`, `speculator_model`), so the slowdown is
the draft itself, not the scaffold. Likely causes: A100 has no native FP8
(Marlin dequant path), 5 draft tokens + 3 aux-hidden layers per step, and a
bf16-trained draft mismatched to the FP8 target → low acceptance → wasted draft
compute (net-negative speculative decoding). Offline `LLM()` sets
`disable_log_stats=True`, so acceptance rate/length must come from an **online**
run. Before drawing conclusions, re-run `26b_e011_no_mtp` offline in the *same*
environment/day to rule out environment drift vs the 07-07 baseline.

### Offline alignment conclusion

Both offline alignment points reproduce the previous 26B results within normal run-to-run noise. This validates:

1. Dataset path/content is aligned.
2. `text_only` model path is correct.
3. MTP assistant path is correct.
4. E011 config is faithfully reproduced by the new scaffold.
5. No-MTP baseline is available for future 12B dense comparison.

## Online Result — MTP, A100 80GB, Unlimited Concurrency

The user requested an online MTP vLLM bench with unlimited client concurrency on the 80GB setup.

Server startup:

```bash
bash serve_align.sh configs/26b_e011_mtp.json 8100
```

Bench command semantics:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8100 \
  --endpoint /v1/chat/completions \
  --model gemma4 \
  --tokenizer google/gemma-4-26B-A4B-it \
  --dataset-name custom \
  --dataset-path ../gemma4_moe_fp8/datasets/sc1_delta_v2.jsonl \
  --num-prompts 1000 \
  --output-len 8192 \
  --request-rate inf
```

Note: `--max-concurrency` was intentionally omitted, so vLLM reports:

```text
Maximum request concurrency: None
Peak concurrent requests: 1000
```

### Summary table

| Metric | Value |
|---|---:|
| Successful requests | 1000 |
| Failed requests | 0 |
| Benchmark duration (s) | 675.55 |
| Request throughput (req/s) | 1.48 |
| Total input tokens | 4,371,285 |
| Total generated tokens | 1,427,960 |
| Output token throughput (tok/s) | **2113.78** |
| Peak output token throughput (tok/s) | 1025.00 |
| Total token throughput (tok/s) | **8584.49** |
| Peak concurrent requests | 1000 |

### Latency

| Metric | Mean | Median | P99 |
|---|---:|---:|---:|
| TTFT (ms) | 313,680.33 | 318,780.79 | 617,809.77 |
| TPOT excl. first token (ms) | 58.09 | 56.45 | 122.39 |
| ITL (ms) | 270.74 | 155.00 | 1,155.56 |

High TTFT is expected for unlimited concurrency because all 1000 requests enter the queue at once. The throughput result is the key signal for this run.

### Speculative decoding metrics

| Metric | Value |
|---|---:|
| Acceptance rate (%) | 80.58 |
| Acceptance length | 5.03 |
| Drafts | 284,215 |
| Draft tokens | 1,421,075 |
| Accepted tokens | 1,145,159 |

Per-position acceptance:

| Draft position | Acceptance (%) |
|---:|---:|
| 0 | 93.61 |
| 1 | 87.08 |
| 2 | 80.45 |
| 3 | 74.23 |
| 4 | 67.54 |

## Online Result — EAGLE-3, A100 80GB, Unlimited Concurrency (2026-07-08)

Official RedHatAI EAGLE-3 draft (`RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3`)
+ FP8 text-only target, same dataset/concurrency/`spec_tokens=5` as the MTP run.
Config is identical to `26b_e011_no_mtp` except the spec fields, so this is an
apples-to-apples swap of MTP-assistant → EAGLE-3 draft.

| Metric | EAGLE-3 | MTP baseline | vs MTP |
|---|---:|---:|---:|
| Output tok/s | **931.46** | 2113.78 | **−56%** |
| Total tok/s | 3777.37 | 8584.49 | −56% |
| **Acceptance rate (%)** | **9.75** | 80.58 | −70.8 pts |
| **Acceptance length** | **1.49** | 5.03 | −3.54 |
| Drafts | 961,132 | 284,215 | — |
| Draft tokens | 4,805,660 | 1,421,075 | — |
| Accepted tokens | 468,551 | 1,145,159 | — |
| Mean TTFT (ms) | 619,992 | 313,680 | worse |
| Mean TPOT (ms) | 110.23 | 58.09 | worse |

Per-position acceptance (EAGLE-3):

| Draft position | Acceptance (%) |
|---:|---:|
| 0 | 31.31 |
| 1 | 10.56 |
| 2 | 3.90 |
| 3 | 1.85 |
| 4 | 1.13 |

### Conclusion — official EAGLE-3 draft does NOT work for this deployment

The official EAGLE-3 draft achieves only **9.75% acceptance / 1.49 accept
length** on the FP8 26B-A4B target, vs MTP's 80.58% / 5.03. Even position 0 is
only 31% (MTP: 93.6%). This is net-negative speculative decoding: output
throughput drops to **931 tok/s, ~56% below MTP (2113) and below even the
no-MTP baseline (~1474/2023 offline / online)**. The three runs corroborate:

| Setup | offline tok/s | online tok/s | online accept rate |
|---|---:|---:|---:|
| MTP (baseline) | 2023 | 2113 | 80.58% |
| no-MTP (pure target) | ~1474 | 1474 | — |
| **EAGLE-3 (official)** | **888** | **931** | **9.75%** |

Likely causes: the draft was trained on bf16 `gemma-4-26B-A4B-it` (generic
Magpie/UltraChat data), mismatched to the FP8 target and to the MAI Profile
distribution; A100 lacks native FP8 (Marlin dequant). **This validates the
project's core thesis: a generic off-the-shelf draft is not enough — we must
train our own draft (EAGLE-3 and/or the MTP assistant) on MAI Profile data.**

## 12B Online Result — MTP, A100 80GB, Unlimited Concurrency

This run used the `vllm_gemma4:6` / `vllm/vllm-openai:gemma4-unified` image and the 12B MTP config:

```text
configs/12b_e011_mtp.json
```

Bench command characteristics from vLLM output:

- Backend: `openai-chat`
- Base URL: `http://localhost:8100`
- Served model: `gemma4-12b`
- Tokenizer: `/tmp/models/gemma4_12b/model`
- Dataset: unchanged `sc1_delta_v2.jsonl`
- Output length cap: 8192
- Request rate: `inf`
- Max concurrency: `None` / unlimited
- Peak concurrent requests: 1000

### Summary table

| Metric | Value |
|---|---:|
| Successful requests | 1000 |
| Failed requests | 0 |
| Benchmark duration (s) | 1354.43 |
| Request throughput (req/s) | 0.74 |
| Total input tokens | 4,371,285 |
| Total generated tokens | 1,535,504 |
| Output token throughput (tok/s) | **1133.69** |
| Peak output token throughput (tok/s) | 642.00 |
| Total token throughput (tok/s) | **4361.09** |
| Peak concurrent requests | 1000 |

### Latency

| Metric | Mean | Median | P99 |
|---|---:|---:|---:|
| TTFT (ms) | 612,495.63 | 606,979.80 | 1,252,569.32 |
| TPOT excl. first token (ms) | 114.34 | 108.77 | 273.50 |
| ITL (ms) | 509.81 | 249.82 | 3,273.51 |

High TTFT is expected for unlimited concurrency because all 1000 requests are queued at once.

### Speculative decoding metrics

| Metric | Value |
|---|---:|
| Acceptance rate (%) | 80.80 |
| Acceptance length | 5.04 |
| Drafts | 304,790 |
| Draft tokens | 1,523,950 |
| Accepted tokens | 1,231,329 |

Per-position acceptance:

| Draft position | Acceptance (%) |
|---:|---:|
| 0 | 93.97 |
| 1 | 87.12 |
| 2 | 80.75 |
| 3 | 74.55 |
| 4 | 67.61 |

### Initial interpretation

The 12B dense MTP run is much slower than 26B-A4B MTP on this A100 80GB setup:

| Mode | Output tok/s | Total tok/s |
|---|---:|---:|
| 26B-A4B online MTP unlimited | 2113.78 | 8584.49 |
| 12B dense online MTP unlimited | 1133.69 | 4361.09 |

This is plausible because 26B-A4B is MoE with much lower active compute per token than a 12B dense model. The 12B MTP acceptance metrics are healthy and similar to 26B, so the throughput gap is likely dominated by target-model compute/backend path rather than poor speculative acceptance.

## 12B Offline Result — MTP, A100 80GB

The user provided the final `12b_e011_mtp` offline result from the JSON output:

```text
ts: 2026-07-07T09:24:35.703203+00:00
reps: 1
num_prompts: 1000
engine_build_time: 282.9
mean_output_tps: 972.89
stdev_output_tps: 0
mean_total_tps: 4391.49
```

Per-rep metrics:

| Metric | Value |
|---|---:|
| Elapsed time (s) | 1274.876 |
| Request throughput (req/s) | 0.7844 |
| Prompt tokens total | 4,358,285 |
| Output tokens total | 1,240,319 |
| Total tokens | 5,598,604 |
| Prompt tok/s | 3418.59 |
| Output tok/s | **972.89** |
| Total tok/s | **4391.49** |
| Mean output length | 1240.32 |
| Output length p50 | 1154 |
| Output length p90 | 2166 |
| Output length max | 8192 |
| Finish stop | 999 |
| Finish length | 1 |
| Finish other | 0 |

## 12B Offline Result — no MTP, A100 80GB

The user ran 12B no-MTP offline with the same dataset and E011-like settings, disabling speculative decoding:

```bash
env \
  -u GEMMA4_26B_TEXT_ONLY_MODEL_PATH \
  -u GEMMA4_26B_ASSISTANT_MODEL_PATH \
  -u GEMMA4_ASSISTANT_MODEL_PATH \
  -u _ModelDataPath_ \
  GEMMA4_MODEL_PATH=/tmp/models/gemma4_12b/model \
  bash run_offline_align.sh configs/12b_e011_no_mtp.json --reps 1 --prompts 1000
```

Run log summary:

```text
Processed prompts: 1000/1000 [25:46<00:00, 1.55s/it, est. speed input: 2818.97 toks/s, output: 884.82 toks/s]
elapsed=1554.2s req/s=0.643 output_tps=880.2 total_tps=3684.5 out_len_mean=1367.97
```

| Metric | Value |
|---|---:|
| Successful requests | 1000 |
| Elapsed time (s) | 1554.2 |
| Request throughput (req/s) | 0.643 |
| Estimated input tok/s from progress | 2818.97 |
| Estimated output tok/s from progress | 884.82 |
| Final output tok/s | **880.2** |
| Final total tok/s | **3684.5** |
| Mean output length | 1367.97 |

### MTP vs no-MTP comparison for 12B

| Mode | Output tok/s | Total tok/s | Notes |
|---|---:|---:|---|
| 12B offline MTP | **972.89** | **4391.49** | Offline `vllm.LLM`, speculative decoding enabled |
| 12B offline no-MTP | 880.2 | 3684.5 | Offline `vllm.LLM`, no speculative decoding |
| 12B online MTP unlimited | 1133.69 | 4361.09 | OpenAI server path, unlimited queueing |

Offline apples-to-apples MTP uplift:

| Metric | Uplift |
|---|---:|
| Output tok/s | **+10.5%** |
| Total tok/s | **+19.2%** |

MTP is beneficial for 12B on this workload, but the gain is much smaller than the 26B-A4B E011 stack-up and the absolute 12B dense throughput remains lower than 26B-A4B.

## 12B Per-Layer MAI Profile Result — MTP, A100 80GB, Unlimited Concurrency (2026-07-09)

Google MTP assistant (`12b_e011_mtp`, FP8 12B dense target + assistant, `spec_tokens=5`)
benchmarked **per MAI Profile eval layer** to see how acceptance varies by task
type. Each layer = 200 eval prompts (the DSpark eval split), converted from the
`maiprofile_<layer>.jsonl` eval datasets to the sc1 `{"prompt": ...}` format via
`convert_maiprofile_eval_to_sc1.py`, then run with `run_maiprofile_online.sh`
(one server load, all five layers benchmarked against it). Unlimited concurrency,
`temperature=0.7`, `output-len` cap 8192. Metrics parsed from `vllm bench serve`.

| Layer | Accept rate (%) | Accept length | Output tok/s | Prompts |
|---|---:|---:|---:|---:|
| **layer3_seasonality** | **97.91** | **5.90** | **2687.77** | 200 |
| layer1_actual | 67.30 | 4.37 | 965.93 | 200 |
| layer4_commercial_preference | 61.88 | 4.09 | 1350.43 | 200 |
| layer2_temporal | 50.78 | 3.54 | 893.21 | 200 |
| layer1_intent | 49.27 | 3.46 | 1100.44 | 200 |
| **Aggregate (all 5, wall-clock)** | **69.28** | **4.46** | **1382.32** | 1000 |

Aggregate = 563,530 output tokens / 407.67 s summed across the five sequential
per-layer runs; aggregate accept rate/length are token-weighted across all
layers (437,565 accepted / 631,625 draft tokens; 126,325 drafts).

Per-position acceptance (%):

| Layer | pos0 | pos1 | pos2 | pos3 | pos4 |
|---|---:|---:|---:|---:|---:|
| layer3_seasonality | 99.68 | 99.07 | 98.17 | 97.20 | 95.45 |
| layer1_actual | 84.63 | 73.51 | 65.71 | 59.19 | 53.47 |
| layer4_commercial_preference | 83.97 | 70.03 | 59.94 | 50.86 | 44.62 |
| layer2_temporal | 77.24 | 59.16 | 47.52 | 38.52 | 31.49 |
| layer1_intent | 73.97 | 56.42 | 45.49 | 38.22 | 32.27 |

### Interpretation

- **Acceptance is strongly layer-dependent** (49–98%). `layer3_seasonality` is
  near-saturated (97.9% / accept_len 5.90 / 2688 tok/s) — its outputs are short
  and highly patterned, so the MTP draft nails them. This matches the DSpark
  finding that seasonality is the easiest layer (DSpark accept_len 5.88 there).
- The free-form layers (`intent`, `temporal`, `actual`, `commercial`) sit at
  49–67% / accept_len 3.5–4.4 — open-ended generation is harder to draft, same
  ordering DSpark saw.
- **tok/s does not track accept rate directly** (e.g. `layer1_intent` has the
  lowest accept rate but not the lowest tok/s) because output length per prompt
  differs a lot by layer (seasonality generates 206k tokens from 200 prompts vs
  intent's 60k). Throughput is jointly set by acceptance *and* output length.
- This is the **MTP baseline** the trained drafts (Phase 1 EAGLE-3/DSpark on
  dense-12B) must beat, per layer. Pair each layer with the no-MTP dense run
  (`configs/12b_e011_no_mtp.json`, same driver) for the per-layer speedup ratio.

## 12B Per-Layer MAI Profile Result — no-MTP dense baseline + MTP speedup (2026-07-09)

Same per-layer setup as above, config `12b_e011_no_mtp` (FP8 12B dense target,
speculative decoding disabled). No acceptance metrics — dense has no draft — so
this is the pure-target throughput baseline. Pairing it with the MTP run gives
the **per-layer MTP speedup** (MTP tok/s ÷ dense tok/s).

| Layer | dense tok/s | MTP tok/s | MTP speedup | MTP accept rate |
|---|---:|---:|---:|---:|
| layer3_seasonality | 1515.58 | 2687.77 | **1.77×** | 97.91% |
| layer4_commercial_preference | 1094.08 | 1350.43 | 1.23× | 61.88% |
| layer2_temporal | 757.79 | 893.21 | 1.18× | 50.78% |
| layer1_actual | 920.06 | 965.93 | 1.05× | 67.30% |
| layer1_intent | 1068.85 | 1100.44 | 1.03× | 49.27% |
| **Aggregate (wall-clock)** | **1105.97** | **1382.32** | **1.25×** | 69.28% |

Aggregate dense = 561,810 output tokens / 507.98 s across the five sequential
per-layer runs.

### Interpretation

- **MTP speedup tracks acceptance, as expected.** `layer3_seasonality` (97.9%
  accept) gets a **1.77× speedup**; the harder free-form layers get much less
  (1.03–1.23×). Overall wall-clock speedup is **1.25×**.
- **`layer1_actual` is the notable outlier**: 67.3% accept rate (2nd highest)
  but only **1.05× speedup** — lower than `commercial` (61.9% accept → 1.23×) and
  `temporal` (50.8% accept → 1.18×). Acceptance rate alone does not determine
  end-to-end speedup: `layer1_actual` generates relatively few tokens per prompt
  (~486 avg) so its runs are dominated by prefill/TTFT rather than decode, muting
  the decode-side MTP win. Layers with longer generations (seasonality 206k
  tokens, commercial 126k) convert acceptance into throughput more effectively.
- Practical takeaway: **the Phase-1 trained drafts should target the low-speedup
  free-form layers** (intent, temporal, actual) — seasonality is already near the
  ceiling under MTP, so there is little headroom to beat there.

## 26B-A4B Per-Layer MAI Profile Result — MTP, A100 80GB, Unlimited Concurrency (2026-07-10)

The final-target model. Google MTP assistant on **26B-A4B MoE** (config
`26b_e011_mtp`, FP8 text-only target + assistant, `spec_tokens=5`, TP=1),
benchmarked per MAI Profile eval layer with the same driver as 12B
(`run_26b_maiprofile_online.sh` → `run_maiprofile_online.sh`), 200 prompts/layer,
unlimited concurrency, `temperature=0.7`. Metrics parsed from `vllm bench serve`.

> Model wiring verified correct this run: target `/tmp/models/gemma4/text_only`,
> assistant `/tmp/models/gemma4/assistant`, no speculator. (An earlier attempt
> silently loaded the 12B target + 26B assistant via stale env, causing a CUDA
> gather index-out-of-bounds; fixed by using the generic `GEMMA4_MODEL_PATH`.)

| Layer | Accept rate (%) | Accept length | Output tok/s | Prompts |
|---|---:|---:|---:|---:|
| **layer3_seasonality** | **99.03** | **5.95** | **5003.34** | 200 |
| layer1_actual | 75.45 | 4.77 | 1704.66 | 200 |
| layer4_commercial_preference | 69.10 | 4.46 | 2483.02 | 200 |
| layer1_intent | 59.34 | 3.97 | 2341.30 | 200 |
| layer2_temporal | 54.50 | 3.72 | 1955.60 | 200 |
| **Aggregate (all 5, wall-clock)** | **75.87** | **4.79** | **2678.67** | 1000 |

Aggregate = 523,761 output tokens / 195.53 s; token-weighted across layers
(415,123 accepted / 547,120 draft tokens; 109,424 drafts).

Per-position acceptance (%):

| Layer | pos0 | pos1 | pos2 | pos3 | pos4 |
|---|---:|---:|---:|---:|---:|
| layer3_seasonality | 99.84 | 99.64 | 99.23 | 98.77 | 97.67 |
| layer1_actual | 89.09 | 80.75 | 74.52 | 68.97 | 63.90 |
| layer4_commercial_preference | 85.89 | 75.76 | 67.82 | 61.12 | 54.92 |
| layer2_temporal | 78.34 | 62.26 | 51.74 | 43.42 | 36.73 |
| layer1_intent | 80.75 | 66.24 | 56.81 | 49.78 | 43.14 |

### 26B-A4B vs 12B MTP (same layers, same spec_tokens=5)

| Layer | 12B accept_len | 26B accept_len | 12B tok/s | 26B tok/s | tok/s ratio |
|---|---:|---:|---:|---:|---:|
| layer3_seasonality | 5.90 | 5.95 | 2687.8 | 5003.3 | 1.86× |
| layer1_actual | 4.37 | 4.77 | 965.9 | 1704.7 | 1.76× |
| layer4_commercial_preference | 4.09 | 4.46 | 1350.4 | 2483.0 | 1.84× |
| layer2_temporal | 3.54 | 3.72 | 893.2 | 1955.6 | 2.19× |
| layer1_intent | 3.46 | 3.97 | 1100.4 | 2341.3 | 2.13× |
| **Aggregate** | **4.46** | **4.79** | **1382.3** | **2678.7** | **1.94×** |

### Interpretation

- **26B-A4B MTP beats 12B MTP on every layer** — higher acceptance length on all
  5 layers and ~1.8–2.2× the throughput. Aggregate accept_len 4.79 vs 12B 4.46,
  aggregate tok/s 2679 vs 1382 (**1.94×**). The MoE's larger effective capacity
  makes the MTP draft both more accurate and faster in absolute terms.
- **Same layer ordering as 12B**: seasonality saturates (99% accept / 5.95
  accept_len / 5003 tok/s), free-form layers trail (intent/temporal lowest).
- This aggregate (accept rate 75.87%, accept_len 4.79) is close to the earlier
  26B sc1_delta MTP baseline (80.58% / 5.03) — the MAI Profile layer mix is a bit
  harder than sc1_delta, driven by the low-acceptance free-form layers.
- **This is the per-layer 26B MTP baseline the trained drafts must beat.** Pair
  with the 26B no-MTP dense run (`configs/26b_e011_no_mtp.json`, same driver) for
  the per-layer speedup ratio.

## 26B-A4B Per-Layer MAI Profile Result — no-MTP dense baseline + MTP speedup (2026-07-10)

Same per-layer setup, config `26b_e011_no_mtp` (FP8 26B-A4B text-only target,
speculative decoding disabled, TP=1). No acceptance metrics — dense has no draft.
Pairing with the 26B MTP run gives the **per-layer MTP speedup** on the final
target model.

| Layer | dense tok/s | MTP tok/s | MTP speedup | MTP accept rate |
|---|---:|---:|---:|---:|
| layer3_seasonality | 2227.81 | 5003.34 | **2.25×** | 99.03% |
| layer4_commercial_preference | 1782.48 | 2483.02 | 1.39× | 69.10% |
| layer1_intent | 1711.87 | 2341.30 | 1.37× | 59.34% |
| layer2_temporal | 1492.74 | 1955.60 | 1.31× | 54.50% |
| layer1_actual | 1346.87 | 1704.66 | 1.27× | 75.45% |
| **Aggregate (wall-clock)** | **1766.15** | **2678.67** | **1.52×** | 75.87% |

Aggregate dense = 523,505 output tokens / 296.41 s across the five per-layer runs.

### Interpretation

- **MTP speedup on 26B-A4B is 1.52× aggregate** (vs 1.25× on 12B) — MTP helps the
  MoE more than the dense 12B. `layer3_seasonality` reaches **2.25×** (99% accept),
  free-form layers 1.27–1.39×.
- **26B-A4B no-MTP is already much faster than 12B dense** (aggregate 1766 vs 1106
  tok/s) because the 26B-A4B MoE activates only ~4B params/token (vs the full 12B
  dense); MTP then stacks another 1.52× on top, so 26B MTP (2679 tok/s) is
  **~1.9× the 12B MTP** throughput.
- **`layer1_actual` is again the low-speedup outlier** (1.27× despite 75% accept)
  — same short-generation/TTFT-bound effect seen on 12B: its runs don't spend
  enough time in decode for the acceptance win to dominate.
- Practical takeaway unchanged: **trained drafts should target the free-form
  layers** (actual, temporal, intent, commercial) where MTP speedup is 1.27–1.39×;
  seasonality is already near the ceiling at 2.25×.

## Alignment vs Previous Results

Previous `gemma4_moe_fp8/RESULTS.md` headline numbers:

| Mode | Historical output tok/s | New scaffold output tok/s | Status |
|---|---:|---:|---|
| Offline E011 MTP, A100 80GB | ~2020 | 2023.06 | Reproduced |
| Offline no-MTP, A100 80GB | ~1465 | 1474.46 | Reproduced |
| Online E011 MTP, A100 80GB unlimited | Not previously recorded in this exact 80GB/unlimited form | 2113.78 | New alignment point |

The online 80GB unlimited throughput is slightly above offline E011 in output tok/s, while total tok/s is close. This is acceptable as an online serving alignment point; latency metrics should be interpreted with the unlimited-concurrency queueing behavior in mind.

## Next Step

Add 12B dense FP8 configs beside the 26B alignment configs. Keep dataset, sampling, prompt count, and measurement scripts unchanged for the first 12B run so differences are attributable to the model/config rather than benchmark drift.
