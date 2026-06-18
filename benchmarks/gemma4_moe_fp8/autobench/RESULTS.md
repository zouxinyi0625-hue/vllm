# Autobench Results — Gemma4 26B MoE FP8 on A100 80GB

## Baseline
- **E011**: 2020 output tok/s, 8540 total tok/s (fp8 | CG | MTP-k5 | mns=128 | mnbt=16384)

## Best Result
- **humming-grouped + async-sched**: 2054.02 output tok/s (+1.7% vs baseline)

## Round 1: One-at-a-time exploration (2026-06-08)

| Config | output_tps | total_tps | stdev | vs baseline |
|--------|-----------|-----------|-------|-------------|
| **humming-grouped** | **2030.29** | 8870.83 | 21.01 | **+0.5%** |
| baseline (re-run) | 2025.02 | 8872.53 | 5.57 | +0.2% |
| mnbt=24576 | 2005.95 | 8686.84 | 23.82 | -0.7% |
| mnbt=8192 | 2002.93 | 8732.70 | 34.94 | -0.8% |
| mns=256 | 1997.02 | 8646.74 | 22.29 | -1.1% |
| mml=16384 | 1996.15 | 8803.30 | 31.15 | -1.2% |
| MTP-k3 | 1984.48 | 8730.40 | 47.74 | -1.8% |
| mnbt=32768 | 1982.39 | 8648.52 | 7.22 | -1.9% |
| mns=96 | 1955.14 | 8514.48 | 28.27 | -3.2% |
| humming-indexed | (see round 2) | — | — | — |
| moe=cutlass | (not in this table) | — | — | — |
| moe=triton | CRASH | — | — | sm_89+ |
| moe=humming | CRASH | — | — | not in FP8 oracle |
| MTP-k9 | CRASH | — | — | — |

## Round 2: Combinations around humming-grouped (2026-06-09)

| Config | output_tps | total_tps | stdev | vs baseline |
|--------|-----------|-----------|-------|-------------|
| humming-grouped + MTP-k3 | 2025.13 | 8841.67 | 2.82 | +0.3% |
| humming-grouped + mml=16384 + mnbt=24576 | 2011.43 | 8908.26 | 25.65 | -0.4% |
| humming-grouped + mnbt=24576 | 2007.38 | 8766.04 | 13.42 | -0.6% |
| humming-indexed + mns=192 | 2000.07 | 8681.59 | 29.89 | -1.0% |
| humming-grouped + mml=16384 | 1994.79 | 8848.11 | 25.35 | -1.2% |
| humming-grouped + mnbt=8192 | 1993.39 | 8739.34 | 64.18 | -1.3% |
| humming-indexed | 1985.73 | 8692.10 | 12.39 | -1.7% |
| humming-grouped + MTP-k7 | 1978.15 | 8684.54 | 2.60 | -2.1% |
| humming-grouped + mns=192 + mnbt=24576 | 1976.36 | 8637.61 | 28.56 | -2.2% |
| humming-grouped + mns=192 | 1961.06 | 8576.98 | 10.97 | -2.9% |
| humming-grouped + mns=192 + mnbt=32768 | 1960.07 | 8545.59 | 3.67 | -3.0% |
| humming-grouped + mns=64 | 1869.49 | 8275.30 | 4.48 | -7.5% |

## Crashes (DO NOT retry)

| Config | Root Cause |
|--------|-----------|
| moe_backend=triton | Triton FP8 needs sm_89+ (A100 is sm_80) |
| moe_backend=humming | Not in FP8 oracle mapping (only MXFP4) |
| moe_backend=deep_gemm | Requires sm_90+ (Hopper/Blackwell) |
| MTP-k9 | Unknown (likely OOM or kernel issue) |
| kv_cache_dtype=fp8_e4m3 | Triton fp8e4nv codegen needs sm_89+ |
| max_num_batched_tokens=32768 | OOM (round 1), runs but slow (round 2 w/ humming) |

## Round 3: Individual feature exploration (2026-06-09)

| Config | output_tps | total_tps | stdev | vs baseline |
|--------|-----------|-----------|-------|-------------|
| **async-sched** | **2041.08** | 8901.59 | 0.0 | **+1.0%** |
| humming-grouped (re-run) | 2031.63 | 8759.47 | 18.79 | +0.6% |
| marlin-forced | 1997.09 | 8769.21 | 0.0 | -1.1% |
| f16acc (humming-grouped) | 1996.37 | 8756.44 | 0.0 | -1.2% |
| moe=marlin | 1990.87 | 8750.68 | 0.0 | -1.4% |
| no-prefix-cache | 1905.15 | 8299.38 | 0.0 | -5.7% |
| kv-fp8 | CRASH | — | — | sm_89+ needed |

## Round 4: Combo of winners (2026-06-09)

| Config | output_tps | total_tps | stdev | vs baseline |
|--------|-----------|-----------|-------|-------------|
| **humming-grouped + async-sched** | **2054.02** | 8995.19 | 0.0 | **+1.7%** |

## Key Findings

1. **humming-grouped + async-sched is the new best** at 2054 tok/s (+1.7% over baseline).
2. **async-sched alone is +1.0%**, humming-grouped alone is +0.5% — they stack.
3. **Default batch params (mns=128, mnbt=16384) are near-optimal** — all deviations perform worse.
4. **MTP-k5 is the sweet spot** — k3 is comparable, k7 is worse, k9 crashes.
5. **mns=192/256 hurts throughput** — more concurrent sequences doesn't help at this scale.
6. **mnbt=24576/32768 doesn't help** — extra batched tokens adds overhead without benefit.
7. **mml=16384 is slightly worse** — shorter context doesn't improve throughput for our workload.
8. **humming-indexed is consistently worse than grouped** (~1.7-2% slower).
9. **Prefix caching helps** — disabling it loses 5.7%.
10. **f16acc, marlin-forced, moe=marlin** — all slightly worse than baseline, not worth it.

## Round 5: vLLM 0.22.1 + CUDA 13.0 image comparison (2026-06-09)

Old image: vLLM 0.21.0 + CUDA 12.9 | New image: vLLM 0.22.1 + CUDA 13.0

| Config | Old output_tps | New output_tps | Old total_tps | New total_tps | stdev (old/new) |
|--------|----------------|----------------|---------------|---------------|-----------------|
| baseline (fp8 CG MTP-k5 mns=128 mnbt=16384) | 2016.54 | 2010.58 | 8701.09 | 8928.23 | 9.88 / 36.62 |
| humming-grouped + async-sched | 2021.67 | 2028.30 | 8844.86 | 8852.40 | 0.72 / 0.08 |

### Findings

- **output_tps unchanged**: generation speed differs by <1% between images, within noise.
- **total_tps ~2-3% higher on new image**: indicates prefill speedup from CUDA 13.0 / vLLM 0.22.1 optimizations.
- **humming-grouped + async-sched gain is unstable**: only +5~18 tok/s over baseline, within stdev — not statistically significant.
- **New image is safe to adopt**: no regression in output_tps, slight prefill improvement, marginally higher QPS.

## Observation

The A100 80GB is close to its throughput ceiling for this model.
Best config (humming-grouped + async-sched) gives +1.7% over baseline.
Meaningful further gains (>5%) likely require:
- Hardware upgrade (H100/H200 for DeepGEMM, FlashInfer FP8, fp8 KV cache)
- Model-level changes (different quantization, pruning)
- Multi-GPU (TP=2) with larger batch sizes
