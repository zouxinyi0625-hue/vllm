# SGLang Benchmark — Gemma 4 MoE (A100 80GB)

SGLang counterpart to the vLLM ablation benchmark (`../bench_ablation.py`).
Same datasets, prompts, sampling params, and CSV format for direct comparison.

## Prerequisites

```bash
pip install "sglang[all]" transformers
```

## Quick Start

```bash
cd benchmarks/gemma4_moe_fp8

# BF16 baseline (compare vLLM E001)
./sglang/run_sglang.sh S001 --scenario sc1 --reps 1

# All experiments
./sglang/run_sglang.sh --all --scenario sc1 --reps 2
```

## Experiment Matrix

### Group A: Baselines

| ID | Description | vLLM comparison |
|---|---|---|
| S001 | BF16, no CUDA graphs | E001 (639 tok/s your machine) |
| S002 | BF16 + CUDA graphs | ~E004 minus FP8 |

### Group B: + Speculative Decoding (EAGLE3)

| ID | Description | vLLM comparison |
|---|---|---|
| S003 | BF16 + CG + EAGLE3 k=5 | ~E005 minus FP8 |
| S004 | BF16 + CG + EAGLE3 k=5 + text-only **(best candidate)** | ~E006 minus FP8 |
| S005 | BF16 + CG + text-only (no spec) | ~E012 minus FP8 |

### Group C: Batch size sweep (from S004)

| ID | mns |
|---|---:|
| S006 | 64 |
| S004 | 128 (control) |
| S007 | 192 |
| S008 | 256 |

### Group D: gpu_mem sweep (from S004)

| ID | gpu_mem |
|---|---:|
| S009 | 0.80 |
| S004 | 0.90 (control) |
| S010 | 0.95 |

### Group E: Isolation

| ID | Description |
|---|---|
| S011 | BF16 text-only, no CG, no spec (worst case) |
| S012 | No CG at optimal (isolates CUDA graph contribution) |

## Key Differences from vLLM

| | vLLM | SGLang |
|---|---|---|
| Eager mode | `enforce_eager=True` | `cuda_graph_max_bs=0` |
| CUDA graphs | `enforce_eager=False` | `cuda_graph_max_bs=256` |
| Speculative | MTP (native) | EAGLE3 (`Gemma4AssistantForCausalLM`) |
| Attention | TRITON_ATTN (forced) | FlashInfer (default) |
| FP8 on A100 | Marlin FP8 (software) | Not recommended |
| Prefix cache | `enable_prefix_caching=True` | Radix attention (always on) |

## Models Needed

| Path | What | How to get |
|---|---|---|
| `GEMMA4_MODEL_PATH` | Full model | `google/gemma-4-26B-A4B-it` (HF) |
| `GEMMA4_TEXT_ONLY_MODEL_PATH` | Vision stripped | Run `examples/create_text_only_model.py` |
| `GEMMA4_ASSISTANT_MODEL_PATH` | EAGLE3 draft model | See below |

### EAGLE3 Draft Model

SGLang uses `Gemma4AssistantForCausalLM` for speculative decoding.
Check if the model has built-in MTP heads:

```bash
python3 -c "
import json
from huggingface_hub import hf_hub_download
cfg = json.load(open(hf_hub_download('google/gemma-4-26B-A4B-it', 'config.json')))
for k,v in cfg.items():
    if 'spec' in k.lower() or 'mtp' in k.lower() or 'assistant' in k.lower():
        print(f'{k}: {v}')
"
```

If the model has MTP config, you can point `GEMMA4_ASSISTANT_MODEL_PATH` to the
same model path. Otherwise, skip speculative experiments (S003, S004, S006-S010, S012).

## Results

Output: `sglang_results/all_runs.csv` + per-run JSON files.

Compare with vLLM:
```bash
echo "=== vLLM ===" && cat ablation_results/all_runs.csv | cut -d, -f2,3,21 | column -t -s,
echo "=== SGLang ===" && cat sglang_results/all_runs.csv | cut -d, -f2,3,21 | column -t -s,
```
