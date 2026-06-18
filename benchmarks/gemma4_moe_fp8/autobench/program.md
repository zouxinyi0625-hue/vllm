# Autobench: Gemma4 MoE Serving Optimization — Agent Instructions

You are a Claude Code agent optimizing vLLM serving throughput for **Gemma 4 27B MoE (FP8)** on **NVIDIA A100 80GB**. Your goal: push `output_tps` beyond the current best of **2020 tok/s**.

---

## How It Works

```
You (local Claude Code)                    AML Singularity (A100 GPU)
├── Read results.tsv                       ├── git clone vllm-msn
├── Analyze: what worked, what didn't      ├── cp mount data → local
├── Decide next configs (1-4 per round)    ├── python run_experiment.py
├── python agent_loop.py --round           ├── cp results → mount
├── Wait for jobs to complete              └── exit
├── Read new results
└── Repeat
```

**You don't run GPU code directly.** You submit AML jobs that do the heavy lifting.

---

## Workflow

### 1. Check status

```bash
cd benchmarks/gemma4_moe_fp8/autobench
python agent_loop.py --status
```

Or read results directly:
```bash
cat results.tsv
```

### 2. Suggest next configs

```bash
python agent_loop.py --suggest --jobs 4
```

Review the suggestions. Override if you have a better hypothesis.

### 3. Submit a round

```bash
python agent_loop.py --round --jobs 4 --prompts 200 --reps 2
```

This will:
- Generate 4 configs based on history
- Submit 4 parallel AML jobs
- Poll until all complete (~15-25 min total)
- New results appear in results.tsv

### 4. Or submit a specific config

```bash
# Write your config
cat > /tmp/my_config.json << 'EOF'
{
    "max_num_seqs": 192,
    "max_num_batched_tokens": 24576,
    "max_model_len": 24576,
    "spec_tokens": 7,
    "quantization": "fp8",
    "kv_cache_dtype": "auto",
    "enforce_eager": false,
    "gpu_memory_utilization": 0.95,
    "VLLM_MOE_USE_DEEP_GEMM": "1",
    "VLLM_USE_FUSED_MOE_GROUPED_TOPK": "1",
    "VLLM_FLASHINFER_MOE_BACKEND": "throughput",
    "VLLM_HUMMING_MOE_GEMM_TYPE": "grouped"
}
EOF

python agent_loop.py --submit /tmp/my_config.json
```

### 5. Full auto (multiple rounds)

```bash
python agent_loop.py --auto --rounds 5 --jobs 4
```

---

## Search Strategy

### Priority order (most likely to yield improvement):

1. **MoE kernel backends** (unexplored territory):
   - `VLLM_MOE_USE_DEEP_GEMM`: "0" vs "1"
   - `VLLM_HUMMING_MOE_GEMM_TYPE`: "indexed" vs "grouped" vs "auto"
   - `VLLM_FLASHINFER_MOE_BACKEND`: "throughput" vs "latency"
   - `VLLM_USE_FUSED_MOE_GROUPED_TOPK`: "0" vs "1"

2. **MTP depth**: k=7 or k=9 (more speculative tokens, if acceptance rate holds)

3. **Batch/scheduling tuning** (around the current optimum):
   - `max_num_seqs`: 96, 128, 192 (fine-grained near sweet spot)
   - `max_num_batched_tokens`: 24576, 32768 (larger prefill batches)
   - `max_model_len`: 16384 (smaller → more KV cache slots)

4. **Combinations**: Take individual winners and combine them.

### What NOT to search:
- `gpu_memory_utilization` < 0.95 (always worse)
- `quantization`: null/bf16 (always slower)
- `enforce_eager`: true (CUDA graphs always help)
- `kv_cache_dtype`: "fp8_e4m3" (crashes on A100)

---

## Constraints (A100 sm_80)

- `VLLM_USE_FLASHINFER_MOE_FP8` must be "0" (requires Hopper)
- `kv_cache_dtype` must be "auto" (fp8 KV not supported)
- `max_num_batched_tokens` must be >= `max_num_seqs`
- `spec_tokens > 0` requires `quantization: "fp8"` (assistant model is FP8)

---

## Reading Results

Results are in `results.tsv` (tab-separated):
```
run_id    output_tps    total_tps    stdev    status    config_summary
```

Detailed per-run JSONs in `run_results/` include full config and metrics.

For deeper analysis:
```bash
python analyze.py
```

---

## Key Decisions

- **Keep/improve**: If output_tps > 2020, the config is valuable. Explore variations of it.
- **Discard**: If output_tps < 2020, note what hurt and avoid similar configs.
- **Crash**: Usually OOM (reduce max_num_seqs) or unsupported backend (skip it).
- **Diminishing returns**: Even 2100 (+4%) is a meaningful improvement for production.

---

## Data & Model Paths

- Input data: `$MOUNT/shares/users/zxy/autobench/data/sc1_delta_v2.jsonl` (pre-existing)
- Results: `$MOUNT/shares/users/zxy/autobench/results/`
- Model (on AML): `${_ModelDataPath_}/text_only` and `${_ModelDataPath_}/assistant`
- Code: Cloned from GitHub `zxy_dev` branch

---

## Important Notes

- AML jobs take ~10-15 min to start + ~5 min to run. Each round is ~20 min wall clock.
- Submit 4 jobs in parallel to maximize throughput of exploration.
- Always push code to GitHub before submitting jobs (they clone from HEAD of zxy_dev).
- Results append to the same TSV on mount — no data loss between rounds.
