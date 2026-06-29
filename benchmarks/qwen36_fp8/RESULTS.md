# Qwen3.6-35B-A3B-FP8 Benchmark Results

- **Model**: Qwen/Qwen3.6-35B-A3B-FP8 (Mixture of Experts, pre-quantized FP8)
- **Hardware**: NVIDIA A100 80GB
- **Dataset**: sc1_delta_v2.jsonl, 1000 prompts
- **vLLM version**: 0.21.1rc1.dev270+g6cbe448ee

---

## Summary — Output Token Throughput

| Config | Label | output tok/s | total tok/s | vs Baseline | 対標 Gemma4 | Gemma4 結果 |
|--------|-------|:---:|:---:|:---:|---|:---:|
| baseline | FP8 + CG | 1704.77 | 4560.37 | 1.00× | E004 | 1432.7 |
| mtp5 | FP8 + CG + MTP k=5 | 1997.39 | 5321.22 | 1.17× | E005 | 1967.0 |
| mtp5_text_only | FP8 + CG + MTP k=5 + text_only | 2001.07 | 5306.29 | 1.17× | E006/E011 | 2017.3/2020.1 |

---

## Experiment Configs

| Config | quantization | CUDA Graphs | MTP k | language_model_only | gpu_mem | max_num_seqs |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline.json | FP8 (model) | ✓ | 0 | ✗ | 0.95 | 128 |
| mtp5.json | FP8 (model) | ✓ | 5 | ✗ | 0.95 | 128 |
| mtp5_text_only.json | FP8 (model) | ✓ | 5 | ✓ | 0.95 | 128 |

---

## Notes

- Reasoning (thinking) verified disabled via `enable_thinking=False` in chat template
- `language_model_only` has minimal impact on Qwen3.6 (~2 tok/s difference vs mtp5)
- MTP k=5 provides ~17% speedup over baseline
- Qwen3.6 baseline is ~19% faster than Gemma4 E004, but with MTP both converge to ~2000 tok/s
