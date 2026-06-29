"""Parameter space definition for Qwen3.6-35B-A3B FP8 serving optimization on A100 80GB.

Aligned with gemma4_moe_fp8/autobench/config_space.py but adapted for Qwen3.6:
  - No speculative decoding (no assistant model)
  - language_model_only=True (no multimodal)
  - No reasoning parser (pure generation benchmark)

Fixed params (always use, not searchable):
  - quantization: "fp8"
  - enforce_eager: False (CUDA graphs on)
  - gpu_memory_utilization: 0.95
  - kv_cache_dtype: "auto"
  - tensor_parallel_size: 1

Fixed env vars (A100 sm_80 constraints):
  - VLLM_USE_FLASHINFER_MOE_FP8=0 (requires sm_90+)
  - VLLM_USE_FLASHINFER_SAMPLER=0
  - VLLM_MOE_USE_DEEP_GEMM=0 (requires sm_90+)
"""
from __future__ import annotations

import random

PARAM_SPACE = {
    "max_num_seqs": [64, 96, 128, 192, 256],
    "max_num_batched_tokens": [8192, 16384, 24576],
    "max_model_len": [16384, 24576],
    "moe_backend": ["auto", "cutlass", "marlin"],
    "VLLM_HUMMING_MOE_GEMM_TYPE": ["indexed", "grouped"],
    "VLLM_HUMMING_USE_F16_ACCUM": ["0", "1"],
    "VLLM_TEST_FORCE_FP8_MARLIN": ["0", "1"],
    "enable_prefix_caching": [True, False],
    "enable_chunked_prefill": [True, False],
    "async_scheduling": [True, False],
}

FIXED_PARAMS = {
    "kv_cache_dtype": "auto",
    "enforce_eager": False,
    "gpu_memory_utilization": 0.95,
}

ENV_VAR_PARAMS = {
    "VLLM_HUMMING_MOE_GEMM_TYPE",
    "VLLM_HUMMING_USE_F16_ACCUM",
    "VLLM_TEST_FORCE_FP8_MARLIN",
}

A100_FIXED_ENV = {
    "VLLM_USE_FLASHINFER_MOE_FP8": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_MOE_USE_DEEP_GEMM": "0",
}

BASELINE_CONFIG = {
    **FIXED_PARAMS,
    "max_num_seqs": 128,
    "max_num_batched_tokens": 16384,
    "max_model_len": 24576,
}


def validate_config(cfg: dict) -> tuple[bool, str]:
    mns = cfg.get("max_num_seqs", 128)
    mnbt = cfg.get("max_num_batched_tokens", 16384)
    if mnbt < mns:
        return False, f"max_num_batched_tokens ({mnbt}) < max_num_seqs ({mns})"

    mml = cfg.get("max_model_len", 24576)
    if mml < 8192:
        return False, f"max_model_len ({mml}) too small for 8192-token outputs"

    moe_be = cfg.get("moe_backend", "auto")
    if moe_be == "triton":
        return False, "moe_backend=triton crashes on A100 (Triton FP8 needs sm_89+)"
    if moe_be == "humming":
        return False, "moe_backend=humming crashes (not in FP8 oracle, only MXFP4)"
    if moe_be == "deep_gemm":
        return False, "moe_backend=deep_gemm crashes on A100 (needs sm_90+)"

    if mnbt >= 32768:
        return False, "max_num_batched_tokens=32768 OOMs on A100 80GB"

    if cfg.get("kv_cache_dtype") == "fp8_e4m3":
        return False, "fp8_e4m3 KV cache crashes on A100 (Triton fp8e4nv needs sm_89+)"

    return True, ""


def config_to_env_vars(cfg: dict) -> dict[str, str]:
    env = dict(A100_FIXED_ENV)
    for key in ENV_VAR_PARAMS:
        if key in cfg:
            env[key] = str(cfg[key])
    return env


def config_to_llm_kwargs(cfg: dict, scenario_cfg: dict) -> dict:
    kwargs = {
        "trust_remote_code": True,
        "max_model_len": cfg.get("max_model_len", 24576),
        "max_num_seqs": cfg.get("max_num_seqs", 128),
        "max_num_batched_tokens": cfg.get("max_num_batched_tokens",
                                          scenario_cfg.get("max_num_batched_tokens", 16384)),
        "gpu_memory_utilization": cfg.get("gpu_memory_utilization", 0.95),
        "enforce_eager": cfg.get("enforce_eager", False),
        "seed": 0,
    }
    if cfg.get("quantization"):
        kwargs["quantization"] = cfg["quantization"]
    if cfg.get("kv_cache_dtype", "auto") != "auto":
        kwargs["kv_cache_dtype"] = cfg["kv_cache_dtype"]
    moe_backend = cfg.get("moe_backend")
    if moe_backend and moe_backend != "auto":
        kwargs["moe_backend"] = moe_backend
    if "enable_prefix_caching" in cfg:
        kwargs["enable_prefix_caching"] = cfg["enable_prefix_caching"]
    if "enable_chunked_prefill" in cfg:
        kwargs["enable_chunked_prefill"] = cfg["enable_chunked_prefill"]
    if "async_scheduling" in cfg:
        kwargs["async_scheduling"] = cfg["async_scheduling"]
    if cfg.get("attention_backend"):
        kwargs["attention_backend"] = cfg["attention_backend"]
    return kwargs


def config_summary(cfg: dict) -> str:
    parts = []
    q = cfg.get("quantization", "bf16") or "bf16"
    parts.append(q)
    if not cfg.get("enforce_eager", True):
        parts.append("CG")
    parts.append(f"mns={cfg.get('max_num_seqs', 128)}")
    parts.append(f"mnbt={cfg.get('max_num_batched_tokens', 16384)}")
    mml = cfg.get("max_model_len", 24576)
    if mml != 24576:
        parts.append(f"mml={mml}")
    moe_be = cfg.get("moe_backend")
    if moe_be and moe_be != "auto":
        parts.append(f"moe={moe_be}")
    humming = cfg.get("VLLM_HUMMING_MOE_GEMM_TYPE")
    if humming is not None:
        parts.append(f"humming-{humming}")
    if cfg.get("VLLM_HUMMING_USE_F16_ACCUM") == "1":
        parts.append("f16acc")
    if cfg.get("VLLM_TEST_FORCE_FP8_MARLIN") == "1":
        parts.append("marlin-forced")
    if cfg.get("enable_prefix_caching") is False:
        parts.append("no-prefix-cache")
    if cfg.get("enable_chunked_prefill") is False:
        parts.append("no-chunked-pf")
    if cfg.get("kv_cache_dtype", "auto") == "fp8_e4m3":
        parts.append("kv-fp8")
    if cfg.get("async_scheduling") is True:
        parts.append("async-sched")
    return " | ".join(parts)


def parse_summary(summary: str) -> dict:
    cfg = dict(FIXED_PARAMS)
    cfg["max_num_seqs"] = 128
    cfg["max_num_batched_tokens"] = 16384
    cfg["max_model_len"] = 24576

    parts = [p.strip() for p in summary.split("|")]
    for part in parts:
        if part == "fp8":
            cfg["quantization"] = "fp8"
        elif part == "bf16":
            cfg["quantization"] = None
        elif part == "CG":
            cfg["enforce_eager"] = False
        elif part.startswith("mns="):
            cfg["max_num_seqs"] = int(part[4:])
        elif part.startswith("mnbt="):
            cfg["max_num_batched_tokens"] = int(part[5:])
        elif part.startswith("mml="):
            cfg["max_model_len"] = int(part[4:])
        elif part.startswith("moe="):
            cfg["moe_backend"] = part[4:]
        elif part.startswith("humming-"):
            cfg["VLLM_HUMMING_MOE_GEMM_TYPE"] = part[8:]
        elif part == "f16acc":
            cfg["VLLM_HUMMING_USE_F16_ACCUM"] = "1"
        elif part == "marlin-forced":
            cfg["VLLM_TEST_FORCE_FP8_MARLIN"] = "1"
        elif part == "no-prefix-cache":
            cfg["enable_prefix_caching"] = False
        elif part == "no-chunked-pf":
            cfg["enable_chunked_prefill"] = False
        elif part == "kv-fp8":
            cfg["kv_cache_dtype"] = "fp8_e4m3"
        elif part == "async-sched":
            cfg["async_scheduling"] = True
    return cfg


def generate_next_configs(history: list[dict], n: int = 4) -> list[dict]:
    if not history:
        return _initial_exploration(n)

    valid_runs = [h for h in history if h.get("status") == "ok" and h.get("output_tps", 0) > 0]
    if not valid_runs:
        return _initial_exploration(n)

    best = max(valid_runs, key=lambda h: h["output_tps"])
    best_cfg = best.get("config") or parse_summary(best.get("config_summary", ""))

    tried_summaries = {h.get("config_summary", "") for h in history}
    candidates = []

    for param, values in PARAM_SPACE.items():
        for val in values:
            if val == best_cfg.get(param):
                continue
            candidate = {**best_cfg, **FIXED_PARAMS, param: val}
            valid, _ = validate_config(candidate)
            if valid and config_summary(candidate) not in tried_summaries:
                candidates.append(candidate)

    if len(candidates) <= n:
        return candidates

    random.shuffle(candidates)
    return candidates[:n]


def _initial_exploration(n: int) -> list[dict]:
    configs = []
    variations = [
        {"VLLM_HUMMING_MOE_GEMM_TYPE": "grouped"},
        {"async_scheduling": True},
        {"VLLM_TEST_FORCE_FP8_MARLIN": "1"},
        {"VLLM_HUMMING_MOE_GEMM_TYPE": "grouped", "VLLM_HUMMING_USE_F16_ACCUM": "1"},
        {"enable_prefix_caching": False},
        {"enable_chunked_prefill": False},
        {"moe_backend": "marlin"},
        {"max_num_seqs": 192},
        {"max_num_seqs": 256},
        {"max_num_batched_tokens": 24576},
        {"VLLM_HUMMING_MOE_GEMM_TYPE": "indexed"},
        {"moe_backend": "cutlass"},
        {"max_num_seqs": 96},
    ]
    for v in variations[:n]:
        cfg = {**BASELINE_CONFIG, **v}
        valid, _ = validate_config(cfg)
        if valid:
            configs.append(cfg)
    return configs
