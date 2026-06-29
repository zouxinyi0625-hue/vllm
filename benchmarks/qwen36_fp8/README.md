# Qwen3.6-35B-A3B FP8 Benchmark Suite

Benchmark configs for **Qwen/Qwen3.6-35B-A3B** with FP8 quantization.
Uses the same dataset and methodology as `gemma4_moe_fp8`.

## Key Differences from Gemma4 Config

| Parameter | Gemma4 | Qwen3.6 |
|-----------|--------|---------|
| Model | google/gemma-4-26B-A4B-it | Qwen/Qwen3.6-35B-A3B |
| TP Size | 1 | 8 |
| Max Model Len | 24576 | 262144 |
| Reasoning | N/A | Disabled (no reasoning parser) |
| Multimodal | text_only | --language-model-only |
| Speculative | MTP k=5 assistant model | None |

## Files

- `serve/serve.sh` — Start vLLM server with FP8
- `serve/test_performance.sh` — Online serving benchmark (rate sweep)
- `serve/Dockerfile` — Container image for AzureML
- `bench_offline.py` — Offline throughput benchmark driver
- `prep_dataset.py` — Dataset preparation (tokenize + filter with Qwen tokenizer)
- `run.qwen36.sh` — Simple serve script
- `run.bench.fullpass.sh` — Full offline benchmark pass

## Quick Start

```bash
# 1. Start server
bash serve/serve.sh 8000 8 262144 0.95

# 2. Run online benchmark
bash serve/test_performance.sh --base-url http://localhost:8000

# 3. Or run offline benchmark
python3 bench_offline.py --scenario sc1 --model Qwen/Qwen3.6-35B-A3B \
  --quantization fp8 --max-num-seqs 64,128,256
```

## Dataset

Uses the same datasets as gemma4_moe_fp8 (`datasets/sc1_delta_v2.jsonl`, `datasets/sc2_personal_v2.jsonl`).
Symlink or copy from `../gemma4_moe_fp8/datasets/`.
