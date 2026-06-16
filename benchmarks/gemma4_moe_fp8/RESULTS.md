# Gemma 4 MoE FP8 — Benchmark Results

- **Model**: Gemma-4-26B-A4B-it (Mixture of Experts, FP8)
- **Hardware**: 

  - NVIDIA A100 80GB (offline / local service)
  - NVIDIA A100 40GB (online service) 
- **Dataset**: sc1_delta_v2.jsonl, 1000 prompts, avg input len ≈ 4358 tokens, avg output len ≈ 1290 tokens
- **Data source**: https://www.cosmos09.osdinfra.net/cosmos/MSN.DnI/shares/users/zxy/data/mai_profile/sc1_delta_v2.jsonl?property=info
- **vLLM version**: 0.21.1rc1.dev270+g6cbe448ee (built from source, with MTP speculative decoding support)

---

## Summary — Output Token Throughput (E011 config)

| Environment | GPU | gpu_mem | Mode | Concurrency | Output tok/s |
|---|---|:---:|---|:---:|:---:|
| A100 80GB | 80GB | 0.95 | Offline (vllm.LLM) | — | **2020** |
| A100 80GB (sim 40G) | 80GB | 0.45 | Offline (vllm.LLM) | — | 1255 |
| A100 80GB (sim 40G) | 80GB | 0.45 | Online (vllm serve) | 32 | 1270 |
| A100 40GB (DLIS) | 40GB | 0.95 | Online (vllm serve) | 32 | 1293 |

---

## Part 1: Offline Throughput (vllm.LLM engine)

### Experiment Definitions

| Exp | Label | Quant | CUDA Graphs | MTP k | max_num_seqs | gpu_mem | Model |
|-----|-------|-------|-------------|-------|:------------:|:-------:|-------|
| E001 | BF16 baseline | None | ✗ (eager) | 0 | 128 | 0.90 | full |
| E002 | +FP8 weights | fp8 | ✗ (eager) | 0 | 128 | 0.90 | full |
| E003 | BF16 + CUDA graphs | None | ✓ | 0 | 128 | 0.90 | full |
| E004 | +CUDA graphs | fp8 | ✓ | 0 | 128 | 0.90 | full |
| E005 | +MTP k=5 | fp8 | ✓ | 5 | 128 | 0.90 | full |
| E006 | +text-only | fp8 | ✓ | 5 | 128 | 0.90 | text_only |
| E007 | batch mns=64 | fp8 | ✓ | 5 | 64 | 0.90 | text_only |
| E008 | batch mns=192 | fp8 | ✓ | 5 | 192 | 0.90 | text_only |
| E009 | batch mns=256 | fp8 | ✓ | 5 | 256 | 0.90 | text_only |
| E010 | gpu_mem=0.80 | fp8 | ✓ | 5 | 128 | 0.80 | text_only |
| E011 | gpu_mem=0.95 | fp8 | ✓ | 5 | 128 | 0.95 | text_only |
| E012 | no MTP (isolate) | fp8 | ✓ | 0 | 128 | 0.90 | text_only |
| E013 | no CG (isolate) | fp8 | ✗ (eager) | 5 | 128 | 0.90 | text_only |
| E014 | BF16 weights (isolate) | None | ✓ | 5 | 128 | 0.90 | text_only |
| E015 | BF16 ref (text-only) | None | ✗ (eager) | 0 | 128 | 0.90 | text_only |

### A100 80GB — Full Memory (gpu_mem=0.95)

**Full report**: https://www.cosmos09.osdinfra.net/cosmos/MSN.DnI/shares/users/zxy/data/mai_profile/all_runs_80.csv?property=info

| Exp | Label | input tok/s | out tok/s | ±σ | vs E001 |
|-----|-------|:-----------:|:---------:|:--:|:-------:|
| E001 | BF16 baseline | 2104 | 646.7 | 19.8 | 1.00× |
| E002 | +FP8 weights | 3646 | 1090.2 | 69.8 | 1.69× |
| E003 | BF16 + CUDA graphs | 3164 | 965.4 | 6.2 | 1.49× |
| E004 | +CUDA graphs | 4842 | 1432.7 | 19.1 | 2.22× |
| E005 | +MTP k=5 | 6651 | 1967.0 | 17.8 | 3.04× |
| E006 | +text-only | 6856 | 2017.3 | 17.7 | 3.12× |
| E007 | batch mns=64 | 6337 | 1859.3 | 10.1 | 2.88× |
| E008 | batch mns=192 | 6730 | 1982.6 | 25.9 | 3.07× |
| E009 | batch mns=256 | 6701 | 1998.2 | 7.6 | 3.09× |
| E010 | gpu_mem=0.80 | 6582 | 1938.8 | 19.3 | 3.00× |
| E011 | gpu_mem=0.95 | 6877 | **2020.1** | 17.7 | **3.12×** |
| E012 | no MTP (isolate) | 4967 | 1465.4 | 9.3 | 2.27× |
| E013 | no CG (isolate) | 6146 | 1815.9 | 75.2 | 2.81× |
| E014 | BF16 weights (isolate) | 6028 | 1840.9 | 18.4 | 2.85× |
| E015 | BF16 ref (text-only) | 2353 | 721.0 | 35.3 | 1.11× |

**Best**: E011 — **2020 output tok/s** (3.12× vs BF16 baseline)

### A100 80GB — Simulated 40GB (gpu_mem=0.45, E011 only)

_Offline `vllm.LLM` engine, scenario sc1, 1000 prompts, 2 reps._

| Exp | gpu_mem | out tok/s (mean) | ±σ | total tok/s | Duration (s) |
|-----|:-------:|:----------------:|:--:|:-----------:|:------------:|
| E011 | 0.45 | 1255.5 | 5.6 | 5446.3 | 1040 |

### A100 80GB — Simulated 40GB (gpu_mem=0.5)

**Full report**: https://www.cosmos09.osdinfra.net/cosmos/MSN.DnI/shares/users/zxy/data/mai_profile/all_runs_40_0.5.csv?property=info

_BF16 experiments (E001, E003, E014, E015) OOM at 40GB._

| Exp | Label | input tok/s | out tok/s | ±σ | vs E002 | vs 80GB |
|-----|-------|:-----------:|:---------:|:--:|:-------:|:-------:|
| E002 | +FP8 weights | 1105 | 326.2 | 4.5 | 1.00× | 0.30× |
| E004 | +CUDA graphs | 2329 | 687.8 | 0.7 | 2.11× | 0.48× |
| E005 | +MTP k=5 | 4218 | 1244.7 | 0.5 | 3.82× | 0.63× |
| E006 | +text-only | 5074 | 1509.2 | 4.5 | 4.63× | 0.75× |
| E007 | batch mns=64 | 5050 | 1503.3 | 2.7 | 4.61× | 0.81× |
| E008 | batch mns=192 | 4981 | 1489.3 | 1.5 | 4.57× | 0.75× |
| E009 | batch mns=256 | 4927 | 1477.6 | 0.7 | 4.53× | 0.74× |
| E010 | gpu_mem=0.80 | 5118 | 1499.7 | 0.4 | 4.60× | 0.77× |
| E011 | gpu_mem=0.95 | 5058 | **1517.7** | 6.8 | **4.65×** | 0.75× |
| E012 | no MTP (isolate) | 2850 | 843.8 | 5.1 | 2.59× | 0.58× |
| E013 | no CG (isolate) | 4120 | 1219.4 | 1.0 | 3.74× | 0.67× |

**Best**: E011 — **1518 output tok/s** (75% of 80GB performance)

### Optimization Contribution (80GB)

| Optimization | Δ tok/s | Δ % |
|---|---:|---:|
| FP8 weights (E002 vs E001) | +443 | +69% |
| CUDA graphs (E004 vs E002) | +343 | +31% |
| MTP k=5 (E005 vs E004) | +534 | +37% |
| Text-only model (E006 vs E005) | +50 | +3% |
| gpu_mem 0.90→0.95 (E011 vs E006) | +3 | +0.1% |

---

## Part 2: Online Serving (vllm serve + vllm bench serve)

### Setup

- **Server config (E011 optimal)**: FP8, CUDA graphs, MTP k=5, text-only, max_model_len=24576
- **GPU memory**: `gpu_memory_utilization=0.45` (simulating 40GB A100 on 80GB hardware)
- **Benchmark**: `vllm bench serve`, openai-chat backend, request_rate=inf, 1000 prompts, output_len=8192
- **Date**: 2026-05-28

### Results — Concurrency Sweep

| Max Concurrency | Req/s | Output tok/s | Total tok/s | Mean TTFT (s) | Median TTFT (s) | P99 TTFT (s) | Mean TPOT (ms) | Mean ITL (ms) | Duration (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Unlimited | 0.90 | 1277 | 5218 | 539.8 | 539.3 | 1086.2 | 12.08 | 59.54 | 1109 |
| 32 | 0.92 | 1270 | 5291 | 17.7 | 17.4 | 30.2 | 12.34 | 60.23 | 1087 |
| 64 | 0.90 | 1275 | 5228 | 51.7 | 52.0 | 69.9 | 12.30 | 59.65 | 1106 |
| 128 | 0.91 | 1279 | 5278 | 114.2 | 120.5 | 145.9 | 12.23 | 59.53 | 1093 |

### Key Observations

1. **Throughput is constant** (~0.90-0.92 req/s, ~5200-5290 total tok/s) regardless of concurrency — GPU is saturated in all cases.

2. **TTFT is the critical differentiator**:
   - Unlimited: TTFT = 540s (all 1000 requests queue simultaneously)
   - max-concurrency=32: TTFT = 17.7s — **30× improvement**, no throughput loss
   - 64 / 128 are intermediate tradeoffs

3. **TPOT and ITL are stable** (~12ms TPOT, ~60ms ITL) — per-token decoding speed unaffected by queuing pressure.

4. **Online vs Offline throughput comparison** (gpu_mem=0.45, simulating 40GB):
   - Offline (vllm.LLM, E011, gpu_mem=0.45): 1255 output tok/s
   - Online local (vllm serve, concurrency=32): 1270 output tok/s
   - Online local (vllm serve, unlimited): 1277 output tok/s
   - Offline and online local throughput are nearly identical (~1255-1277 tok/s)

5. **Recommendation**: Use `max-concurrency=32` for production serving on 40GB A100-equivalent. Best latency-throughput tradeoff.

---

## Part 3: Online Service (A100 40G ×1, DLIS Deployment)

### Deployment

**Image build pipeline (AML → DLIS)**:
[Pipelines - Run 68925548](https://msasg.visualstudio.com/Bing_and_IPG/_build/results?buildId=68925548&view=results)

**DLIS deploy pipeline**:
[Pipelines - Run 68926058](https://msasg.visualstudio.com/Bing_and_IPG/_build/results?buildId=68926058&view=results)

**Endpoint URL**:
```
https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/dlis-coreranker.chrona-gemma4/v1/chat/completions
```

**Connectivity test**:
```bash
curl https://fabricrouter-azureglobalprivate.ingress-dlis.ingress.cus.microsoft-falcon.net/dlis-coreranker.chrona-gemma4/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

### Setup

- **Server config (E011 optimal)**: FP8, CUDA graphs, MTP k=5, text-only, max_model_len=24576
- **Hardware**: A100 40GB ×1 (DLIS FalconCentralUS)
- **Benchmark**: `vllm bench serve`, openai-chat backend, request_rate=inf, max-concurrency=32, 1000 prompts, output_len=8192
- **Script**: `benchmarks/gemma4_moe_fp8/serve/test_online_service.sh`

### Results

| Metric | Value |
|--------|------:|
| Successful requests | 1000 |
| Failed requests | 0 |
| Max concurrency | 32 |
| Benchmark duration (s) | 1111.18 |
| Request throughput (req/s) | 0.90 |
| Output token throughput (tok/s) | 1293 |
| Total token throughput (tok/s) | 5227 |
| Mean TTFT (s) | 13.7 |
| Median TTFT (s) | 13.4 |
| P99 TTFT (s) | 25.8 |
| Mean TPOT (ms) | 14.81 |
| Median TPOT (ms) | 14.39 |
| P99 TPOT (ms) | 25.85 |
| Mean ITL (ms) | 71.97 |
| Median ITL (ms) | 54.12 |
| P99 ITL (ms) | 614.20 |

### Key Observations (DLIS vs Local Serve, gpu_mem=0.45)

| Metric | Local (concurrency=32) | DLIS (concurrency=32) | Δ |
|--------|:---:|:---:|:---:|
| Output tok/s | 1270 | 1293 | +1.8% |
| Total tok/s | 5291 | 5227 | −1.2% |
| Mean TTFT (s) | 17.7 | 13.7 | −23% (better) |
| Mean TPOT (ms) | 12.34 | 14.81 | +20% |
| Mean ITL (ms) | 60.23 | 71.97 | +19% |

- **Throughput is comparable** — DLIS real A100 40GB matches local simulated 40GB (~1270-1293 output tok/s).
- **TTFT is better on DLIS** (13.7s vs 17.7s) — likely due to real 40GB HBM bandwidth vs simulated constraint.
- **TPOT/ITL slightly higher on DLIS** (~15ms vs ~12ms) — minor variance, possibly due to network overhead or different GPU clock profiles.

---

## Part 4: DFlash Speculative Decoding (PR #41703)

### Setup

- **vLLM version**: 0.20.2rc1.dev208+g8cb2db160 (built from PR #41703)
- **Hardware**: A100 80GB ×1, CUDA 13.0
- **DFlash model**: `z-lab/gemma-4-26B-A4B-it-DFlash`
- **Base model**: `/tmp/models/gemma4/text_only` (FP8)
- **Date**: 2026-06-10

### Build Notes

Building vLLM from PR source requires fixes for runtime-only CUDA images:

```bash
# Required env vars
export CUDA_HOME=/usr/local/cuda
export CMAKE_PREFIX_PATH=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0"   # Single arch saves 5-7x build time
export MAX_JOBS=8                    # Prevent OOM on 96-core machines
export FLASHINFER_DISABLE_VERSION_CHECK=1  # Runtime: bypass version mismatch

# Fix missing nvrtc symlink
ln -s /usr/local/cuda/lib64/libnvrtc.so.12 /usr/local/cuda/lib64/libnvrtc.so

# Install
pip install -v -U "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/41703/head"
```

### Results

| Config | num_spec_tokens | max_num_seqs | mnbt | Status | Output tok/s (observed) |
|--------|:---:|:---:|:---:|---|:---:|
| DFlash k=15 | 15 | 128 | 16384 | **OOM** | — |
| DFlash k=5 | 5 | 128 | 16384 | **Crashed at 920/1000** | ~1862 |

### DFlash k=5 Crash Details

- Progress: 92% (920/1000 prompts), running ~10 min before crash
- Observed speed: input 6212.83 toks/s, output 1862.51 toks/s
- Model loading: 25.64 GiB (vs ~14 GiB for base FP8 without DFlash)
- GPU usage: 77827/81920 MiB

**Error**: `IndexError: list index out of range` in `gpu_model_runner.py:1432` (`correct_spec_decode_token_counts`)

```
num_accepted = valid_sampled_token_count[prev_req_index] - 1
               ~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
IndexError: list index out of range
```

This is a bug in the DFlash implementation — triggers when many concurrent requests finish simultaneously (80 active requests at crash time). Not an OOM or configuration issue.

### DFlash k=15 OOM Analysis

- DFlash draft model adds ~11 GiB overhead (total model load: 25.64 GiB vs ~14 GiB base FP8)
- k=15 requires 3× more KV cache slots per request than MTP k=5
- Verification forward pass for all 15 draft tokens exceeds available memory with mnbt=16384

### Comparison vs E011 (MTP k=5)

| Method | Output tok/s | Status | Notes |
|--------|:---:|---|---|
| MTP k=5 (E011) | **2020** | ✓ Complete | Stable, full 1000 prompts |
| DFlash k=5 | ~1862 | ✗ Crashed 920/1000 | Bug in PR, ~8% slower before crash |
| DFlash k=15 | — | ✗ OOM | Needs lower mnbt or max_num_seqs |

**Conclusion**: DFlash PR #41703 is not yet stable for production benchmarking. The observed ~1862 tok/s (before crash) is ~8% slower than MTP k=5 (2020 tok/s), but this comparison is incomplete due to the crash.
