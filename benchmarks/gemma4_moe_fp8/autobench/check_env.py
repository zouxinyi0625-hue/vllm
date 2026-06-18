#!/usr/bin/env python3
"""Environment check script — runs on AML GPU to verify configuration.

Saves a diagnostic report to mount so you can inspect without SSH.

Usage (in AML job or interactively):
    python check_env.py
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def check_env():
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
    }

    # Python & vLLM version
    report["python_version"] = sys.version
    try:
        import vllm
        report["vllm_version"] = vllm.__version__
    except ImportError:
        report["vllm_version"] = "NOT INSTALLED"

    # CUDA info
    try:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_version"] = torch.version.cuda
            report["gpu_count"] = torch.cuda.device_count()
            report["gpu_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            report["compute_capability"] = f"sm_{cap[0]}{cap[1]}"
            report["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
    except ImportError:
        report["torch_version"] = "NOT INSTALLED"

    # vLLM env vars
    vllm_env_vars = {}
    for key, val in sorted(os.environ.items()):
        if key.startswith("VLLM_"):
            vllm_env_vars[key] = val
    report["vllm_env_vars"] = vllm_env_vars

    # Model paths
    report["model_text_only"] = os.environ.get("GEMMA4_TEXT_ONLY_MODEL_PATH", "NOT SET")
    report["model_assistant"] = os.environ.get("GEMMA4_ASSISTANT_MODEL_PATH", "NOT SET")

    # Check model directories exist
    for key in ["GEMMA4_TEXT_ONLY_MODEL_PATH", "GEMMA4_ASSISTANT_MODEL_PATH"]:
        path = os.environ.get(key, "")
        if path:
            report[f"{key}_exists"] = os.path.isdir(path)
            if os.path.isdir(path):
                files = os.listdir(path)
                report[f"{key}_files"] = files[:20]

    # Check available MoE backends
    try:
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
            Fp8MoeBackend,
            _get_priority_backends,
        )
        report["fp8_moe_backends"] = [b.value for b in Fp8MoeBackend]
    except Exception as e:
        report["fp8_moe_backends_error"] = str(e)

    # Check what --moe-backend options exist
    try:
        from vllm.config.kernel import KernelConfig
        moe_field = KernelConfig.model_fields.get("moe_backend")
        if moe_field and moe_field.description:
            report["moe_backend_description"] = moe_field.description[:500]
    except Exception as e:
        report["moe_backend_info_error"] = str(e)

    # huggingface CLI version
    try:
        result = subprocess.run(["hf", "--version"], capture_output=True, text=True)
        report["hf_cli_version"] = result.stdout.strip()
    except Exception:
        report["hf_cli_version"] = "NOT AVAILABLE"

    # Disk space
    try:
        result = subprocess.run(["df", "-h", "/tmp"], capture_output=True, text=True)
        report["disk_space_tmp"] = result.stdout.strip()
    except Exception:
        pass

    # Print and save
    print(json.dumps(report, indent=2))

    # Save to mount if available
    mount = os.environ.get("AZURE_ML_INPUT_msndni", "")
    if mount:
        out_dir = Path(mount) / "shares/users/zxy/autobench/results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"env_check_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with out_file.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {out_file}")
    else:
        # Save locally
        out_file = Path("env_check.json")
        with out_file.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {out_file}")

    return report


if __name__ == "__main__":
    check_env()
