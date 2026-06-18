# Autobench — vLLM Serving Parameter Optimizer

Automated parameter search for maximizing vLLM serving throughput on a single GPU.
Currently configured for **Gemma 4 26B MoE (FP8) on A100 80GB**.

## Quick Start

### 1. Run a single experiment on GPU

```bash
cd benchmarks/gemma4_moe_fp8/autobench

# Set model paths (or use defaults: google/gemma-4-27b-it-text-only)
export GEMMA4_TEXT_ONLY_MODEL_PATH=/path/to/text_only
export GEMMA4_ASSISTANT_MODEL_PATH=/path/to/assistant

# Clear stale env vars
unset VLLM_HUMMING_MOE_GEMM_TYPE
unset VLLM_HUMMING_USE_F16_ACCUM
unset VLLM_TEST_FORCE_FP8_MARLIN

# Run with a config JSON
cat > /tmp/my_config.json << 'EOF'
{"quantization":"fp8","kv_cache_dtype":"auto","enforce_eager":false,"gpu_memory_utilization":0.95,"max_num_seqs":128,"max_num_batched_tokens":16384,"max_model_len":24576,"spec_tokens":5,"async_scheduling":true,"VLLM_HUMMING_MOE_GEMM_TYPE":"grouped"}
EOF
python run_experiment.py --config /tmp/my_config.json --prompts 1000 --reps 1
```

Results are appended to `results_all.tsv` (on mount, downloaded locally) and detailed JSON saved to `run_results/`.

### 2. Submit to AML (remote GPU)

```bash
# Single config
python submit_job.py --config configs/baseline_e011.json

# Multiple configs in one job
python submit_job.py --config configs/a.json configs/b.json configs/c.json

# Dry-run (preview command without submitting)
python submit_job.py --config configs/a.json --dry-run
```

### 3. Agent-driven loop (auto-explore)

```bash
# Check current results
python agent_loop.py --status

# Get AI-suggested next configs
python agent_loop.py --suggest --jobs 4

# Submit a round (generate + submit + poll)
python agent_loop.py --round --jobs 4 --branch zxy_dev

# Fully automatic: multiple rounds
python agent_loop.py --auto --rounds 5 --jobs 4
```

## Writing Config JSONs

A config JSON contains vLLM engine parameters and environment variables:

```json
{
    "quantization": "fp8",
    "kv_cache_dtype": "auto",
    "enforce_eager": false,
    "gpu_memory_utilization": 0.95,
    "max_num_seqs": 128,
    "max_num_batched_tokens": 16384,
    "max_model_len": 24576,
    "spec_tokens": 5,
    "async_scheduling": true,
    "VLLM_HUMMING_MOE_GEMM_TYPE": "grouped"
}
```

### Searchable parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| `max_num_seqs` | 64, 96, 128, 192, 256 | Max concurrent sequences |
| `max_num_batched_tokens` | 8192, 16384, 24576 | Must be >= max_num_seqs |
| `max_model_len` | 16384, 24576 | Context window |
| `spec_tokens` | 3, 4, 5, 7, 9 | MTP speculative decoding depth |
| `moe_backend` | auto, cutlass, marlin | MoE kernel |
| `async_scheduling` | true, false | Async scheduler |
| `enable_prefix_caching` | true, false | Prefix cache |
| `enable_chunked_prefill` | true, false | Chunked prefill |
| `VLLM_HUMMING_MOE_GEMM_TYPE` | indexed, grouped | MoE GEMM variant |
| `VLLM_HUMMING_USE_F16_ACCUM` | 0, 1 | FP16 accumulation |
| `VLLM_TEST_FORCE_FP8_MARLIN` | 0, 1 | Force Marlin FP8 kernel |

### Known crashes on A100 (do not use)

| Config | Reason |
|--------|--------|
| `moe_backend: "triton"` | Triton FP8 needs sm_89+ |
| `moe_backend: "humming"` | Not in FP8 oracle (MXFP4 only) |
| `moe_backend: "deep_gemm"` | Needs sm_90+ |
| `kv_cache_dtype: "fp8_e4m3"` | Triton fp8e4nv needs sm_89+ |
| `max_num_batched_tokens: 32768` | OOM on 80GB |

## Analyzing Results

```bash
# Print ranked results and parameter importance
python analyze.py

# Analyze from a specific TSV
python analyze.py --tsv /path/to/results.tsv

# Detailed analysis from per-run JSONs
python analyze.py --json-dir run_results/
```

## File Structure

```
autobench/
├── run_experiment.py    # Runs one benchmark on GPU (the workhorse)
├── agent_loop.py        # Local orchestrator: suggest/submit/poll
├── submit_job.py        # Submit AML Singularity jobs
├── analyze.py           # Results analysis and ranking
├── config_space.py      # Parameter space, validation, config utilities
├── check_env.py         # GPU environment diagnostics
├── configs/             # Saved config JSONs
├── results_all.tsv     # Cumulative results from all AML jobs (append-only)
├── run_results/         # Detailed per-run JSON outputs
├── RESULTS.md           # Human-readable findings summary
└── program.md           # Claude Code agent instructions
```

## Agentic Mode (Claude Code drives the loop)

Let Claude Code autonomously explore the parameter space:

1. Open Claude Code in this repo
2. Tell it: "run autobench, find configs that beat 2041 output_tps"
3. Claude Code will:
   - Read `results_all.tsv` to understand what's been tried
   - Generate promising configs based on `config_space.py`
   - Submit AML jobs via `submit_job.py`
   - Poll until jobs complete
   - Analyze results, decide next round
   - Repeat until improvement plateaus

Prerequisites:
- Azure CLI authenticated (`az login`)
- `azure-ai-ml` SDK installed
- Git branch pushed (AML jobs clone from remote HEAD)

The agent instructions are in `program.md` — Claude Code reads this to understand
the search strategy, constraints, and how to interpret results.

## Manual Mode (you run on GPU, paste results back)

For quick one-off experiments or debugging:

```bash
# On GPU node
unset VLLM_HUMMING_MOE_GEMM_TYPE
unset VLLM_HUMMING_USE_F16_ACCUM
unset VLLM_TEST_FORCE_FP8_MARLIN

echo '{"quantization":"fp8",...}' > /tmp/cfg.json
python run_experiment.py --config /tmp/cfg.json --prompts 1000 --reps 1
```

Then paste the output_tps/total_tps back to Claude Code for analysis and next suggestions.

## Tips

- Always `unset` env vars before running a new config — vLLM reads them at import time.
- Use `--prompts 200 --reps 2` for quick screening, `--prompts 1000 --reps 1` for final numbers.
- Combine individual winners (e.g., async-sched + humming-grouped) to find synergies.
- Check `results_all.tsv` before running — don't re-run configs that already have data.
