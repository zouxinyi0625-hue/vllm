#!/usr/bin/env python3
"""Ablation-study benchmark for Gemma 4 26B MoE — SGLang version.

Mirror of ../bench_ablation.py but using sglang.Engine for offline inference.
Produces results in the same CSV format so they can be directly compared.

Target: A100 80GB single card (same as vLLM experiments).
BF16 full model fits on 80GB. For 40GB, use quantized variants.

Usage:
    python3 bench_ablation_sglang.py --exp S001 --scenario sc1 --reps 1
    python3 bench_ablation_sglang.py --exp S001,S002,S003 --scenario sc1 --reps 2
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_DIR = Path("sglang_results")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR / "all_runs.csv"

CSV_FIELDS = [
    "ts", "exp_id", "label", "scenario",
    "quantization", "kv_cache_dtype", "attention_backend",
    "enforce_eager", "mtp", "mtp_k",
    "max_num_seqs", "gpu_memory_utilization", "model_variant",
    "num_prompts", "output_len_cap", "max_model_len", "max_num_batched_tokens",
    "rep", "seed",
    "elapsed_time",
    "requests_per_second",
    "prompt_tokens_total", "output_tokens_total", "total_tokens",
    "prompt_tps", "output_tps", "total_tps",
    "out_len_mean", "out_len_stdev", "out_len_p50", "out_len_p90", "out_len_max",
    "finish_stop", "finish_length", "finish_other",
]

# ---------------------------------------------------------------------------
# Scenario definitions (same as vLLM bench_ablation.py)
# ---------------------------------------------------------------------------
SCENARIOS = {
    "sc1": dict(
        dataset="datasets/sc1_delta_v2.jsonl",
        num_prompts=1000,
        output_len=8192,
        max_model_len=24576,
        max_num_batched_tokens=16384,
    ),
    "sc2": dict(
        dataset="datasets/sc2_personal_v2.jsonl",
        num_prompts=500,
        output_len=8192,
        max_model_len=49152,
        max_num_batched_tokens=16384,
    ),
}

# ---------------------------------------------------------------------------
# Model paths (same env vars as vLLM version)
# ---------------------------------------------------------------------------
MODEL_BASE = os.environ.get("GEMMA4_MODEL_PATH", "google/gemma-4-26B-A4B-it")
MODEL_TEXT_ONLY = os.environ.get(
    "GEMMA4_TEXT_ONLY_MODEL_PATH",
    MODEL_BASE + "-text-only",
)
# SGLang uses EAGLE3 with Gemma4AssistantForCausalLM for speculative decoding
MODEL_ASSISTANT = os.environ.get(
    "GEMMA4_ASSISTANT_MODEL_PATH",
    MODEL_BASE + "-assistant",
)

# ---------------------------------------------------------------------------
# Experiment matrix — A100 80GB, matches vLLM ablation settings
#
# SGLang optimization levers:
#   - CUDA graphs: cuda_graph_max_bs (default enabled, set 0 to disable)
#   - Speculative decoding: EAGLE3 with Gemma4AssistantForCausalLM
#   - Quantization: fp8 on H100, bf16/awq on A100
#   - No enforce_eager concept — use cuda_graph_max_bs=0 for eager equivalent
#
# Note: SGLang does not have explicit "enforce_eager". Instead:
#   - eager mode = cuda_graph_max_bs=0 (no CUDA graphs captured)
#   - graph mode = cuda_graph_max_bs=256 (default, captures graphs up to bs=256)
# ---------------------------------------------------------------------------
EXPERIMENTS: dict[str, dict] = {
    # === Group A: Baselines (match vLLM E001, E002) ===
    "S001": dict(
        label="BF16 baseline (no CUDA graphs) — compare vLLM E001",
        quantization=None,
        cuda_graphs=False,
        speculative=False, spec_k=0,
        max_num_seqs=128,
        gpu_memory_utilization=0.90,
        model_variant="full",
    ),
    "S002": dict(
        label="BF16 + CUDA graphs — compare vLLM E001 with CG",
        quantization=None,
        cuda_graphs=True,
        speculative=False, spec_k=0,
        max_num_seqs=128,
        gpu_memory_utilization=0.90,
        model_variant="full",
    ),
    "S003": dict(
        label="BF16 + CG + EAGLE3 spec k=5 — compare vLLM E005 (minus FP8)",
        quantization=None,
        cuda_graphs=True,
        speculative=True, spec_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.90,
        model_variant="full",
    ),
    # === Group B: text-only model ===
    "S004": dict(
        label="BF16 + CG + EAGLE3 k=5 + text-only — best candidate",
        quantization=None,
        cuda_graphs=True,
        speculative=True, spec_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.90,
        model_variant="text_only",
    ),
    "S005": dict(
        label="BF16 + CG + text-only (no spec) — isolate spec contribution",
        quantization=None,
        cuda_graphs=True,
        speculative=False, spec_k=0,
        max_num_seqs=128,
        gpu_memory_utilization=0.90,
        model_variant="text_only",
    ),
    # === Group C: Batch size sweep (from S004) ===
    "S006": dict(
        label="batch sweep: mns=64",
        quantization=None,
        cuda_graphs=True,
        speculative=True, spec_k=5,
        max_num_seqs=64,
        gpu_memory_utilization=0.90,
        model_variant="text_only",
    ),
    "S007": dict(
        label="batch sweep: mns=192",
        quantization=None,
        cuda_graphs=True,
        speculative=True, spec_k=5,
        max_num_seqs=192,
        gpu_memory_utilization=0.90,
        model_variant="text_only",
    ),
    "S008": dict(
        label="batch sweep: mns=256",
        quantization=None,
        cuda_graphs=True,
        speculative=True, spec_k=5,
        max_num_seqs=256,
        gpu_memory_utilization=0.90,
        model_variant="text_only",
    ),
    # === Group D: gpu_mem sweep (from S004) ===
    "S009": dict(
        label="gpu_mem sweep: 0.80",
        quantization=None,
        cuda_graphs=True,
        speculative=True, spec_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.80,
        model_variant="text_only",
    ),
    "S010": dict(
        label="gpu_mem sweep: 0.95",
        quantization=None,
        cuda_graphs=True,
        speculative=True, spec_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.95,
        model_variant="text_only",
    ),
    # === Group E: Isolation experiments ===
    "S011": dict(
        label="BF16 text-only no CG no spec (worst case) — compare vLLM E015",
        quantization=None,
        cuda_graphs=False,
        speculative=False, spec_k=0,
        max_num_seqs=128,
        gpu_memory_utilization=0.90,
        model_variant="text_only",
    ),
    "S012": dict(
        label="no CUDA graphs at optimal (isolates CG) — compare vLLM E013",
        quantization=None,
        cuda_graphs=False,
        speculative=True, spec_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.90,
        model_variant="text_only",
    ),
    # === Group F: W8A8 quantization ===
    "S013": dict(
        label="W8A8_INT8 + CUDA graphs — weight+activation quantization",
        quantization="w8a8_int8",
        cuda_graphs=True,
        speculative=False, spec_k=0,
        max_num_seqs=128,
        gpu_memory_utilization=0.90,
        model_variant="full",
    ),
    "S014": dict(
        label="W8A8_INT8 + CG + NEXTN spec k=5",
        quantization="w8a8_int8",
        cuda_graphs=True,
        speculative=True, spec_k=5,
        max_num_seqs=128,
        gpu_memory_utilization=0.90,
        model_variant="full",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_csv_header():
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(row: dict):
    with CSV_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


def load_prompts(dataset_path: str, n: int) -> list[str]:
    prompts: list[str] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            if len(prompts) >= n:
                break
    if not prompts:
        raise FileNotFoundError(
            f"Dataset empty or not found: {dataset_path}\n"
            "Run prep_dataset.py first to generate the JSONL datasets."
        )
    return prompts


def render_chat(tok, raw_prompts: list[str]) -> list[str]:
    out = []
    for p in raw_prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out.append(text)
    return out


def percentile(sorted_vals: list, q: float):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    exp_id: str,
    exp_cfg: dict,
    scenario: str,
    sc_cfg: dict,
    reps: int,
) -> list[dict]:
    """Build SGLang engine for one experiment config, run `reps` generations."""
    import sglang as sgl
    from transformers import AutoTokenizer

    # Resolve model path based on variant
    variant = exp_cfg["model_variant"]
    if variant == "text_only":
        model = MODEL_TEXT_ONLY
    else:
        model = MODEL_BASE

    print(
        f"\n{'='*70}\n"
        f"  [SGLang] Experiment : {exp_id}  ({exp_cfg['label']})\n"
        f"  Scenario   : {scenario}  ({sc_cfg['num_prompts']} prompts, "
        f"max_model_len={sc_cfg['max_model_len']})\n"
        f"  Model      : {model}\n"
        f"  quantization={exp_cfg['quantization']}  "
        f"cuda_graphs={exp_cfg['cuda_graphs']}\n"
        f"  max_num_seqs={exp_cfg['max_num_seqs']}  "
        f"gpu_mem={exp_cfg['gpu_memory_utilization']}  "
        f"speculative={exp_cfg['speculative']} k={exp_cfg['spec_k']}\n"
        f"{'='*70}",
        flush=True,
    )

    # Load tokenizer and prepare prompts
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    raw_prompts = load_prompts(sc_cfg["dataset"], sc_cfg["num_prompts"])
    prompts = render_chat(tok, raw_prompts)
    print(f"loaded {len(prompts)} prompts from {sc_cfg['dataset']}", flush=True)

    # Build SGLang Engine kwargs
    # SGLang mem_fraction_static is the fraction of *remaining* memory (after model)
    # allocated to KV cache. Unlike vLLM's gpu_memory_utilization (total budget),
    # setting this too high starves CUDA graphs. Use 0.80 to leave room for graphs.
    mem_frac = min(exp_cfg["gpu_memory_utilization"], 0.80)

    # torch_compile conflicts with CUDA graphs on large models — compiled kernels
    # consume too much memory leaving no room for graph capture beyond bs=1.
    # Only enable torch_compile when CUDA graphs are disabled.
    use_torch_compile = not exp_cfg["cuda_graphs"]

    engine_kwargs: dict = dict(
        model_path=model,
        trust_remote_code=True,
        context_length=sc_cfg["max_model_len"],
        mem_fraction_static=mem_frac,
        enable_torch_compile=use_torch_compile,
        # Scheduling: use longest-prefix-match for shared chat template prefixes
        schedule_policy="lpm",
    )

    # Quantization
    if exp_cfg["quantization"]:
        engine_kwargs["quantization"] = exp_cfg["quantization"]

    # Max running requests (equivalent to vLLM max_num_seqs)
    engine_kwargs["max_running_requests"] = exp_cfg["max_num_seqs"]

    # Chunked prefill size (equivalent to vLLM max_num_batched_tokens)
    engine_kwargs["chunked_prefill_size"] = sc_cfg["max_num_batched_tokens"]

    # CUDA graphs control
    # SGLang default: CUDA graphs enabled.
    # Use disable_cuda_graph=True to disable (equivalent to vLLM enforce_eager=True).
    if not exp_cfg["cuda_graphs"]:
        engine_kwargs["disable_cuda_graph"] = True

    # Speculative decoding (NEXTN for Gemma 4 — matches official config)
    if exp_cfg["speculative"]:
        engine_kwargs["speculative_algorithm"] = "NEXTN"
        engine_kwargs["speculative_draft_model_path"] = MODEL_ASSISTANT
        engine_kwargs["speculative_num_steps"] = exp_cfg["spec_k"]
        engine_kwargs["speculative_eagle_topk"] = 1
        engine_kwargs["speculative_num_draft_tokens"] = 6

    print(f"Engine kwargs: {engine_kwargs}", flush=True)

    t_engine = time.time()
    engine = sgl.Engine(**engine_kwargs)
    print(f"engine built in {time.time()-t_engine:.1f}s", flush=True)

    # Diagnostic: print engine internal config to verify optimizations
    try:
        server_args = getattr(engine, "server_args", None)
        if server_args:
            print(f"[DIAG] server_args.enable_torch_compile = {getattr(server_args, 'enable_torch_compile', 'N/A')}", flush=True)
            print(f"[DIAG] server_args.disable_cuda_graph = {getattr(server_args, 'disable_cuda_graph', 'N/A')}", flush=True)
            print(f"[DIAG] server_args.chunked_prefill_size = {getattr(server_args, 'chunked_prefill_size', 'N/A')}", flush=True)
            print(f"[DIAG] server_args.max_running_requests = {getattr(server_args, 'max_running_requests', 'N/A')}", flush=True)
            print(f"[DIAG] server_args.mem_fraction_static = {getattr(server_args, 'mem_fraction_static', 'N/A')}", flush=True)
            print(f"[DIAG] server_args.schedule_policy = {getattr(server_args, 'schedule_policy', 'N/A')}", flush=True)
            print(f"[DIAG] server_args.attention_backend = {getattr(server_args, 'attention_backend', 'N/A')}", flush=True)
            print(f"[DIAG] server_args.cuda_graph_max_bs = {getattr(server_args, 'cuda_graph_max_bs', 'N/A')}", flush=True)
        else:
            print("[DIAG] server_args not found, trying engine attributes...", flush=True)
            for attr in dir(engine):
                if any(k in attr.lower() for k in ['compile', 'cuda', 'chunk', 'schedule', 'attention', 'running']):
                    print(f"[DIAG] engine.{attr} = {getattr(engine, attr, 'N/A')}", flush=True)
    except Exception as diag_err:
        print(f"[DIAG] diagnostic failed: {diag_err}", flush=True)

    rows: list[dict] = []
    for rep in range(1, reps + 1):
        seed = rep
        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_new_tokens": sc_cfg["output_len"],
            "ignore_eos": False,
            "sampling_seed": seed,
        }

        tag = f"{exp_id}_{scenario}_rep{rep}"
        print(f"\n--- RUN {tag} seed={seed} ---", flush=True)
        t0 = time.time()
        outputs = engine.generate(prompts, sampling_params)
        elapsed = time.time() - t0

        # Parse outputs
        out_lens: list[int] = []
        prompt_total = 0
        output_total = 0
        finish_counts = {"stop": 0, "length": 0, "other": 0}

        for o in outputs:
            meta = o.get("meta_info", {})

            # Token counts from meta_info
            p_toks = meta.get("prompt_tokens", 0)
            c_toks = meta.get("completion_tokens", 0)
            prompt_total += p_toks
            output_total += c_toks
            out_lens.append(c_toks)

            # Finish reason
            fr = meta.get("finish_reason", {})
            if isinstance(fr, dict):
                reason = fr.get("type", "other").lower()
            elif isinstance(fr, str):
                reason = fr.lower()
            else:
                reason = "other"

            if reason == "stop":
                finish_counts["stop"] += 1
            elif reason in ("length", "max_new_tokens"):
                finish_counts["length"] += 1
            else:
                finish_counts["other"] += 1

        total = prompt_total + output_total
        out_lens_sorted = sorted(out_lens)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "exp_id": exp_id,
            "label": exp_cfg["label"],
            "scenario": scenario,
            "quantization": str(exp_cfg["quantization"]),
            "kv_cache_dtype": "auto",
            "attention_backend": "sglang_default",
            "enforce_eager": not exp_cfg["cuda_graphs"],
            "mtp": exp_cfg["speculative"],
            "mtp_k": exp_cfg["spec_k"],
            "max_num_seqs": exp_cfg["max_num_seqs"],
            "gpu_memory_utilization": exp_cfg["gpu_memory_utilization"],
            "model_variant": exp_cfg["model_variant"],
            "num_prompts": len(prompts),
            "output_len_cap": sc_cfg["output_len"],
            "max_model_len": sc_cfg["max_model_len"],
            "max_num_batched_tokens": sc_cfg["max_num_batched_tokens"],
            "rep": rep,
            "seed": seed,
            "elapsed_time": round(elapsed, 3),
            "requests_per_second": round(len(prompts) / elapsed, 4),
            "prompt_tokens_total": prompt_total,
            "output_tokens_total": output_total,
            "total_tokens": total,
            "prompt_tps": round(prompt_total / elapsed, 2),
            "output_tps": round(output_total / elapsed, 2),
            "total_tps": round(total / elapsed, 2),
            "out_len_mean": round(statistics.mean(out_lens), 2) if out_lens else None,
            "out_len_stdev": round(statistics.stdev(out_lens), 2) if len(out_lens) > 1 else 0.0,
            "out_len_p50": int(percentile(out_lens_sorted, 0.5)) if out_lens else None,
            "out_len_p90": int(percentile(out_lens_sorted, 0.9)) if out_lens else None,
            "out_len_max": max(out_lens) if out_lens else None,
            "finish_stop": finish_counts["stop"],
            "finish_length": finish_counts["length"],
            "finish_other": finish_counts["other"],
        }
        append_csv_row(row)
        rows.append(row)

        per_run = OUT_DIR / f"{tag}.json"
        with per_run.open("w") as f:
            json.dump({**row, "out_lens": out_lens}, f, indent=2)

        print(
            f"  elapsed={elapsed:.1f}s  req/s={row['requests_per_second']:.3f}  "
            f"out_tok/s={row['output_tps']:.0f}  total_tok/s={row['total_tps']:.0f}  "
            f"out_len(mean±sd)={row['out_len_mean']}±{row['out_len_stdev']}  "
            f"finish=stop:{finish_counts['stop']}/len:{finish_counts['length']}",
            flush=True,
        )

    _summarize(exp_id, exp_cfg["label"], scenario, rows)

    engine.shutdown()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return rows


def _summarize(exp_id: str, label: str, scenario: str, rows: list[dict]):
    if not rows:
        print(f"[SUMMARY] {exp_id} {scenario}: no successful runs", flush=True)
        return

    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    el_m, el_s = m("elapsed_time")
    o_m, o_s = m("output_tps")
    t_m, t_s = m("total_tps")
    ol_m, _ = m("out_len_mean")
    print(
        f"\n[SUMMARY] {exp_id} | {label}\n"
        f"  scenario={scenario}  reps={len(rows)}\n"
        f"  elapsed_time   : {el_m:.2f} ± {el_s:.2f} s\n"
        f"  output tokens/s: {o_m:.2f} ± {o_s:.2f}\n"
        f"  total tokens/s : {t_m:.2f} ± {t_s:.2f}\n"
        f"  mean out_len   : {ol_m:.1f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run ablation experiments via SGLang Engine offline benchmark."
    )
    ap.add_argument(
        "--exp", required=True,
        help="Experiment ID(s), comma-separated. E.g. S001 or S001,S002,S003",
    )
    ap.add_argument(
        "--scenario", default="sc1", choices=list(SCENARIOS.keys()),
        help="Dataset scenario (default: sc1)",
    )
    ap.add_argument("--reps", type=int, default=2,
                    help="Repetitions per experiment (default: 2)")
    ap.add_argument("--list", action="store_true",
                    help="Print the experiment matrix and exit")
    args = ap.parse_args()

    if args.list:
        print(f"{'ID':<6}  {'label'}")
        print("-" * 70)
        for eid, ecfg in EXPERIMENTS.items():
            print(f"{eid:<6}  {ecfg['label']}")
        return 0

    ensure_csv_header()
    sc_cfg = SCENARIOS[args.scenario]
    exp_ids = [x.strip() for x in args.exp.split(",")]

    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"ERROR: unknown experiment ID '{exp_id}'. "
                  f"Valid: {list(EXPERIMENTS.keys())}", file=sys.stderr)
            return 1
        exp_cfg = EXPERIMENTS[exp_id]

        try:
            run_experiment(
                exp_id=exp_id,
                exp_cfg=exp_cfg,
                scenario=args.scenario,
                sc_cfg=sc_cfg,
                reps=args.reps,
            )
        except Exception as e:
            print(f"!!! experiment {exp_id} FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
