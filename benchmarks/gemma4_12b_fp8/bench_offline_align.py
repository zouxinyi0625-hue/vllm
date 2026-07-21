#!/usr/bin/env python3
"""Offline alignment benchmark for Gemma4 12B FP8 work.

The initial configs intentionally reproduce the existing 26B Gemma4 MoE E011
and no-MTP baselines using the unchanged sc1_delta_v2 dataset.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "offline_results"


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def load_config(path: str) -> dict[str, Any]:
    cfg_path = resolve_path(path)
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def apply_env(cfg: dict[str, Any]) -> None:
    for key, value in cfg.get("env", {}).items():
        os.environ[str(key)] = str(value)


def override_model_paths(cfg: dict[str, Any]) -> None:
    """Allow server/GPU nodes to override local model paths without editing JSON.

    Generic variables are preferred for new 12B/26B configs:
      - GEMMA4_MODEL_PATH
      - GEMMA4_ASSISTANT_MODEL_PATH

    Legacy 26B-specific variables are still supported for old alignment commands:
      - GEMMA4_26B_TEXT_ONLY_MODEL_PATH
      - GEMMA4_26B_ASSISTANT_MODEL_PATH
    """
    if os.environ.get("GEMMA4_MODEL_PATH"):
        cfg["model"] = os.environ["GEMMA4_MODEL_PATH"]
    elif os.environ.get("GEMMA4_26B_TEXT_ONLY_MODEL_PATH"):
        cfg["model"] = os.environ["GEMMA4_26B_TEXT_ONLY_MODEL_PATH"]

    if os.environ.get("GEMMA4_ASSISTANT_MODEL_PATH"):
        cfg["assistant_model"] = os.environ["GEMMA4_ASSISTANT_MODEL_PATH"]
    elif os.environ.get("GEMMA4_26B_ASSISTANT_MODEL_PATH"):
        cfg["assistant_model"] = os.environ["GEMMA4_26B_ASSISTANT_MODEL_PATH"]

    # EAGLE-3 (and other --speculative-config methods) use a self-contained
    # speculator model, overridable without editing JSON.
    if os.environ.get("GEMMA4_SPECULATOR_MODEL"):
        cfg["speculator_model"] = os.environ["GEMMA4_SPECULATOR_MODEL"]

    # Dataset override (e.g. per-layer MAI Profile prompt files) without editing
    # JSON. Also settable via --dataset-path (which takes precedence, applied in
    # main()). Lets one config sweep many prompt files.
    if os.environ.get("GEMMA4_DATASET_PATH"):
        cfg["dataset_path"] = os.environ["GEMMA4_DATASET_PATH"]


def load_prompts(dataset_path: Path, n: int) -> list[str]:
    prompts: list[str] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "prompt" not in d:
                raise ValueError(f"Expected JSONL records with 'prompt', got keys={list(d)}")
            prompts.append(d["prompt"])
            if len(prompts) >= n:
                break
    if not prompts:
        raise FileNotFoundError(f"Dataset empty or missing: {dataset_path}")
    return prompts


def render_chat(tokenizer, raw_prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in raw_prompts
    ]


def percentile(sorted_vals: list[int], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def build_llm_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "trust_remote_code": True,
        "max_model_len": cfg.get("max_model_len", 24576),
        "max_num_seqs": cfg.get("max_num_seqs", 128),
        "max_num_batched_tokens": cfg.get("max_num_batched_tokens", 16384),
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.95),
        "enforce_eager": cfg.get("enforce_eager", False),
        "seed": 0,
    }
    if cfg.get("quantization"):
        kwargs["quantization"] = cfg["quantization"]
    if cfg.get("kv_cache_dtype", "auto") != "auto":
        kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    if "enable_prefix_caching" in cfg:
        kwargs["enable_prefix_caching"] = cfg["enable_prefix_caching"]
    if "enable_chunked_prefill" in cfg:
        kwargs["enable_chunked_prefill"] = cfg["enable_chunked_prefill"]
    if "async_scheduling" in cfg:
        kwargs["async_scheduling"] = cfg["async_scheduling"]

    spec_tokens = int(cfg.get("spec_tokens", 0) or 0)
    spec_method = cfg.get("spec_method")
    assistant_model = cfg.get("assistant_model")
    speculator_model = cfg.get("speculator_model")
    if spec_tokens > 0:
        if spec_method == "eagle3":
            # EAGLE-3 uses a self-contained speculator via speculative_config,
            # mirroring the serve --speculative-config JSON path.
            if not speculator_model:
                raise ValueError(
                    "spec_method=eagle3 requires 'speculator_model' (empty)"
                )
            kwargs["speculative_config"] = {
                "model": speculator_model,
                "num_speculative_tokens": spec_tokens,
                "method": "eagle3",
            }
        else:
            # Default: MTP assistant draft via spec_model / spec_tokens.
            if not assistant_model:
                raise ValueError("spec_tokens > 0 but assistant_model is empty")
            kwargs["spec_model"] = assistant_model
            kwargs["spec_tokens"] = spec_tokens
    return kwargs


def collect_spec_metrics(llm) -> dict[str, Any] | None:
    """Extract speculative-decoding metrics from an offline LLM engine.

    Mirrors what ``vllm bench serve`` reports for the online path, using the
    same vllm:spec_decode_* counters but read via ``llm.get_metrics()`` instead
    of scraping the Prometheus endpoint. Returns None if spec decoding is off.

    Metric names (see vllm/v1/metrics/reader.py):
      - vllm:spec_decode_num_drafts
      - vllm:spec_decode_num_draft_tokens
      - vllm:spec_decode_num_accepted_tokens
      - vllm:spec_decode_num_accepted_tokens_per_pos  (labeled by position)
    """
    try:
        metrics = llm.get_metrics()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"WARN: get_metrics() failed: {exc}", flush=True)
        return None

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted = 0
    accepted_per_pos: dict[int, int] = {}
    found = False

    for m in metrics:
        name = getattr(m, "name", "")
        if "spec_decode" not in name:
            continue
        found = True
        if name == "vllm:spec_decode_num_drafts":
            num_drafts += int(getattr(m, "value", 0) or 0)
        elif name == "vllm:spec_decode_num_draft_tokens":
            num_draft_tokens += int(getattr(m, "value", 0) or 0)
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            # Vector metric: .values is a plain list[int], index == draft
            # position, value == accepted-token count at that position.
            values = getattr(m, "values", None)
            if values is not None:
                for pos, v in enumerate(values):
                    accepted_per_pos[pos] = accepted_per_pos.get(pos, 0) + int(v)
                    num_accepted += int(v)

    if not found:
        return None

    acceptance_rate = (
        (num_accepted / num_draft_tokens * 100.0) if num_draft_tokens else 0.0
    )
    # Acceptance length = mean accepted tokens per draft + 1 (the always-verified
    # target token), matching the online "Acceptance length" definition.
    acceptance_length = (num_accepted / num_drafts + 1.0) if num_drafts else 0.0
    per_pos_rate = {}
    if num_drafts:
        for pos, cnt in sorted(accepted_per_pos.items()):
            per_pos_rate[pos] = round(cnt / num_drafts * 100.0, 2)

    return {
        "acceptance_rate": round(acceptance_rate, 2),
        "acceptance_length": round(acceptance_length, 2),
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted,
        "per_position_acceptance": per_pos_rate,
    }


def run(cfg: dict[str, Any], reps: int, num_prompts: int | None) -> dict[str, Any]:
    # Import after env vars are applied.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    dataset_path = resolve_path(cfg["dataset_path"])
    n = num_prompts or int(cfg.get("num_prompts", 1000))

    tokenizer_name = cfg.get("tokenizer") or cfg["model"]
    print(f"Config      : {cfg.get('name')} ({cfg.get('_config_path')})", flush=True)
    print(f"Model       : {cfg['model']}", flush=True)
    print(f"Assistant   : {cfg.get('assistant_model') or 'DISABLED'}", flush=True)
    print(f"Tokenizer   : {tokenizer_name}", flush=True)
    print(f"Dataset     : {dataset_path}", flush=True)
    print(f"Prompts     : {n}, reps={reps}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    raw_prompts = load_prompts(dataset_path, n)
    prompts = render_chat(tokenizer, raw_prompts)

    llm_kwargs = build_llm_kwargs(cfg)
    print("LLM kwargs  :", json.dumps({k: str(v) for k, v in llm_kwargs.items()}, indent=2), flush=True)

    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    engine_build_time = time.time() - t_engine
    print(f"Engine built in {engine_build_time:.1f}s", flush=True)

    rows: list[dict[str, Any]] = []
    for rep in range(1, reps + 1):
        sampling = SamplingParams(
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.95)),
            max_tokens=int(cfg.get("max_tokens", 8192)),
            seed=rep,
            ignore_eos=False,
        )
        print(f"\n--- Rep {rep}/{reps} ---", flush=True)
        t0 = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        elapsed = time.time() - t0

        prompt_total = 0
        output_total = 0
        out_lens: list[int] = []
        finish_counts = {"stop": 0, "length": 0, "other": 0}
        for output in outputs:
            prompt_total += len(output.prompt_token_ids)
            for comp in output.outputs:
                out_len = len(comp.token_ids)
                output_total += out_len
                out_lens.append(out_len)
                reason = (comp.finish_reason or "other").lower()
                if reason in finish_counts:
                    finish_counts[reason] += 1
                else:
                    finish_counts["other"] += 1

        out_lens_sorted = sorted(out_lens)
        row = {
            "rep": rep,
            "elapsed_time": round(elapsed, 3),
            "requests_per_second": round(len(prompts) / elapsed, 4),
            "prompt_tokens_total": prompt_total,
            "output_tokens_total": output_total,
            "total_tokens": prompt_total + output_total,
            "prompt_tps": round(prompt_total / elapsed, 2),
            "output_tps": round(output_total / elapsed, 2),
            "total_tps": round((prompt_total + output_total) / elapsed, 2),
            "out_len_mean": round(statistics.mean(out_lens), 2),
            "out_len_p50": int(percentile(out_lens_sorted, 0.5)),
            "out_len_p90": int(percentile(out_lens_sorted, 0.9)),
            "out_len_max": max(out_lens) if out_lens else 0,
            "finish_stop": finish_counts["stop"],
            "finish_length": finish_counts["length"],
            "finish_other": finish_counts["other"],
        }
        print(
            f"elapsed={elapsed:.1f}s req/s={row['requests_per_second']:.3f} "
            f"output_tps={row['output_tps']:.1f} total_tps={row['total_tps']:.1f} "
            f"out_len_mean={row['out_len_mean']}",
            flush=True,
        )
        rows.append(row)

    # Collect speculative-decoding metrics before tearing down the engine.
    spec_metrics = collect_spec_metrics(llm)
    if spec_metrics:
        print(
            f"\nSpec decode: accept_rate={spec_metrics['acceptance_rate']}% "
            f"accept_len={spec_metrics['acceptance_length']} "
            f"drafts={spec_metrics['num_drafts']}",
            flush=True,
        )
        pp = spec_metrics.get("per_position_acceptance", {})
        if pp:
            print("  per-position acceptance (%):", flush=True)
            for pos, rate in pp.items():
                print(f"    position {pos}: {rate}", flush=True)

    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    output_tps = [r["output_tps"] for r in rows]
    total_tps = [r["total_tps"] for r in rows]
    result = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "reps": reps,
        "num_prompts": len(prompts),
        "engine_build_time": round(engine_build_time, 1),
        "mean_output_tps": round(statistics.mean(output_tps), 2),
        "stdev_output_tps": round(statistics.stdev(output_tps), 2) if len(output_tps) > 1 else 0.0,
        "mean_total_tps": round(statistics.mean(total_tps), 2),
        "spec_metrics": spec_metrics,
        "rows": rows,
    }
    return result


def save_result(result: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    name = result["config"].get("name", "config")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{name}_{run_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved result: {out_path}", flush=True)
    print(
        f"SUMMARY {name}: output_tps={result['mean_output_tps']} ± {result['stdev_output_tps']} "
        f"total_tps={result['mean_total_tps']}",
        flush=True,
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline Gemma4 alignment benchmark")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--prompts", type=int, default=None)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override the config's dataset_path (e.g. a per-layer MAI Profile "
        "prompt file). Takes precedence over GEMMA4_DATASET_PATH and the JSON.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_env(cfg)
    override_model_paths(cfg)
    # --dataset-path wins over env/JSON so a driver can sweep many layers.
    if args.dataset_path:
        cfg["dataset_path"] = args.dataset_path
    result = run(cfg, reps=args.reps, num_prompts=args.prompts)
    save_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
