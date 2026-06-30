#!/usr/bin/env python3
"""SGLang DFlash benchmark for Qwen3.6-35B-A3B-FP8.

Uses sglang.Engine offline with DFlash speculative decoding.
Produces results in the same CSV format as the vLLM autobench for comparison.

Usage:
    python bench_sglang_dflash.py --exp D001 --scenario sc1 --reps 1
    python bench_sglang_dflash.py --exp D001,D002,D003 --scenario sc1 --reps 2
    python bench_sglang_dflash.py --list
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
    "spec_algorithm", "spec_draft_model", "spec_block_size", "spec_tokens",
    "max_num_seqs", "gpu_memory_utilization",
    "num_prompts", "output_len_cap", "max_model_len",
    "rep", "seed",
    "elapsed_time",
    "requests_per_second",
    "prompt_tokens_total", "output_tokens_total", "total_tokens",
    "prompt_tps", "output_tps", "total_tps",
    "out_len_mean", "out_len_stdev", "out_len_p50", "out_len_p90", "out_len_max",
    "finish_stop", "finish_length", "finish_other",
]

# ---------------------------------------------------------------------------
# Scenario definitions (same dataset as vLLM autobench)
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
# Model paths
# ---------------------------------------------------------------------------
MODEL = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3.6-35B-A3B-FP8")
DFLASH_DRAFT = os.environ.get("QWEN_DFLASH_MODEL_PATH", "z-lab/Qwen3.6-35B-A3B-DFlash")

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
EXPERIMENTS: dict[str, dict] = {
    "D001": dict(
        label="Baseline: CG, no spec (SGLang)",
        speculative=False,
        spec_algorithm=None,
        spec_block_size=0,
        spec_tokens=0,
        cuda_graphs=True,
        max_num_seqs=128,
        gpu_memory_utilization=0.95,
    ),
    "D002": dict(
        label="DFlash block_size=8 (対標 HF card default)",
        speculative=True,
        spec_algorithm="DFLASH",
        spec_block_size=8,
        spec_tokens=8,
        cuda_graphs=True,
        max_num_seqs=128,
        gpu_memory_utilization=0.95,
    ),
    "D003": dict(
        label="DFlash block_size=16 (longer accept, better single-conc)",
        speculative=True,
        spec_algorithm="DFLASH",
        spec_block_size=16,
        spec_tokens=16,
        cuda_graphs=True,
        max_num_seqs=128,
        gpu_memory_utilization=0.95,
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
    full = {k: row.get(k, "") for k in CSV_FIELDS}
    with CSV_PATH.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(full)


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
        raise FileNotFoundError(f"Dataset empty or not found: {dataset_path}")
    return prompts


def render_chat(tok, raw_prompts: list[str]) -> list[str]:
    out = []
    for p in raw_prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
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
    import sglang as sgl
    from transformers import AutoTokenizer

    print(
        f"\n{'='*70}\n"
        f"  [SGLang] Experiment : {exp_id}  ({exp_cfg['label']})\n"
        f"  Scenario   : {scenario}  ({sc_cfg['num_prompts']} prompts)\n"
        f"  Model      : {MODEL}\n"
        f"  DFlash     : {DFLASH_DRAFT if exp_cfg['speculative'] else 'DISABLED'}\n"
        f"  block_size={exp_cfg['spec_block_size']}  "
        f"cuda_graphs={exp_cfg['cuda_graphs']}  "
        f"gpu_mem={exp_cfg['gpu_memory_utilization']}\n"
        f"{'='*70}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    raw_prompts = load_prompts(sc_cfg["dataset"], sc_cfg["num_prompts"])
    prompts = render_chat(tok, raw_prompts)
    print(f"loaded {len(prompts)} prompts from {sc_cfg['dataset']}", flush=True)

    mem_frac = min(exp_cfg["gpu_memory_utilization"], 0.85)

    engine_kwargs: dict = dict(
        model_path=MODEL,
        trust_remote_code=True,
        context_length=sc_cfg["max_model_len"],
        mem_fraction_static=mem_frac,
        max_running_requests=exp_cfg["max_num_seqs"],
        chunked_prefill_size=sc_cfg["max_num_batched_tokens"],
        schedule_policy="lpm",
    )

    if not exp_cfg["cuda_graphs"]:
        engine_kwargs["disable_cuda_graph"] = True

    if exp_cfg["speculative"]:
        engine_kwargs["speculative_algorithm"] = exp_cfg["spec_algorithm"]
        engine_kwargs["speculative_draft_model_path"] = DFLASH_DRAFT
        engine_kwargs["speculative_dflash_block_size"] = exp_cfg["spec_block_size"]
        engine_kwargs["speculative_draft_attention_backend"] = "fa4"

    print(f"Engine kwargs: {engine_kwargs}", flush=True)

    t_engine = time.time()
    engine = sgl.Engine(**engine_kwargs)
    print(f"engine built in {time.time()-t_engine:.1f}s", flush=True)

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

        out_lens: list[int] = []
        prompt_total = 0
        output_total = 0
        finish_counts = {"stop": 0, "length": 0, "other": 0}

        for o in outputs:
            meta = o.get("meta_info", {})
            p_toks = meta.get("prompt_tokens", 0)
            c_toks = meta.get("completion_tokens", 0)
            prompt_total += p_toks
            output_total += c_toks
            out_lens.append(c_toks)

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
            "spec_algorithm": exp_cfg["spec_algorithm"] or "",
            "spec_draft_model": DFLASH_DRAFT if exp_cfg["speculative"] else "",
            "spec_block_size": exp_cfg["spec_block_size"],
            "spec_tokens": exp_cfg["spec_tokens"],
            "max_num_seqs": exp_cfg["max_num_seqs"],
            "gpu_memory_utilization": exp_cfg["gpu_memory_utilization"],
            "num_prompts": len(prompts),
            "output_len_cap": sc_cfg["output_len"],
            "max_model_len": sc_cfg["max_model_len"],
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
        description="Run SGLang DFlash benchmark for Qwen3.6-35B-A3B-FP8."
    )
    ap.add_argument(
        "--exp", required=True,
        help="Experiment ID(s), comma-separated. E.g. D001 or D001,D002,D003",
    )
    ap.add_argument(
        "--scenario", default="sc1", choices=list(SCENARIOS.keys()),
        help="Dataset scenario (default: sc1)",
    )
    ap.add_argument("--reps", type=int, default=1,
                    help="Repetitions per experiment (default: 1)")
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
