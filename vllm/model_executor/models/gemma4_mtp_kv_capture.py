# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP shared-KV capture for online speculative-decoding training.

TorchSpec trains a Gemma4 MTP draft that, at deploy time, shares the target's
K/V from the last non-shared sliding layer (L28) and full layer (L29) via
vLLM's KV-sharing mechanism. To generate training data with vLLM (so the draft
learns exactly the deploy-time distribution), we must capture those same K/V
tensors during the target forward and hand them to the Mooncake connector.

This module is a lightweight, OFF-BY-DEFAULT side channel:

  * Enabled only when env ``VLLM_GEMMA4_MTP_CAPTURE_KV=1`` is set (TorchSpec's
    VllmEngine exports it when running Gemma4 MTP online data-gen).
  * Gemma4Attention.forward, for the target's last non-shared sliding/full
    layers, calls ``record(layer_type, k, v)`` with the post-RoPE + post-norm
    K/V -- byte-for-byte the same tensors the deploy-time draft reads, and the
    same representation the HF training path stores
    (transformers modeling_gemma4.py:1266-1276).
  * The captured tensors are read out per forward by the connector / proposer
    glue and stored to Mooncake alongside last_hidden.

When disabled, ``record`` is a no-op with negligible overhead (one env check
cached at import) and vLLM's numerics are completely untouched.
"""

from __future__ import annotations

import os
import threading

import torch

_local = threading.local()


def is_enabled() -> bool:
    # Read live each call (NOT an import-time snapshot): vLLM's import graph /
    # worker fork order can import this module before the env is set, which
    # would freeze a stale False. Live read is cheap and robust.
    return os.environ.get("VLLM_GEMMA4_MTP_CAPTURE_KV", "0") == "1"


def _buf() -> dict:
    b = getattr(_local, "buf", None)
    if b is None:
        b = {}
        _local.buf = b
    return b


def reset() -> None:
    """Clear the capture buffer at the start of a target forward."""
    if is_enabled():
        _buf().clear()


def record(layer_type: str, k: torch.Tensor, v: torch.Tensor) -> None:
    """Record post-RoPE/post-norm K/V for a shared-source layer.

    Args:
        layer_type: "sliding_attention" or "full_attention".
        k, v: shape [num_tokens, num_kv_heads * head_dim] (as computed in
            Gemma4Attention.forward right before self.attn(q, k, v)).
    """
    if not is_enabled():
        return
    # Detach + clone so later in-place ops (if any) can't corrupt the capture.
    _buf()[layer_type] = (k.detach(), v.detach())
    # Diagnostic dump path also fires here so it works on ANY inference path
    # (bench / deploy Gemma4Proposer), not only the extract_hidden_states
    # training proposer that calls take(). Only dump once BOTH layer types are
    # present (so the file has sliding + full together).
    buf = _buf()
    if "sliding_attention" in buf and "full_attention" in buf:
        _maybe_dump(dict(buf))


def take() -> dict:
    """Return and clear the current forward's captured K/V.

    Returns:
        dict mapping layer_type -> (k, v). Empty if nothing was captured.
    """
    if not is_enabled():
        return {}
    b = _buf()
    out = dict(b)
    b.clear()
    _maybe_dump(out)
    return out


# Diagnostic: dump the first non-empty capture to disk for HF-vs-vLLM shared_kv
# verification. Set VLLM_GEMMA4_MTP_DUMP_DIR to enable. Dumps once per process.
_dumped = {"done": False}


def _maybe_dump(captured: dict) -> None:
    dump_dir = os.environ.get("VLLM_GEMMA4_MTP_DUMP_DIR")
    if not dump_dir or _dumped["done"] or not captured:
        return
    try:
        payload = {}
        for ltype, (k, v) in captured.items():
            payload[f"{ltype}_k"] = k.detach().float().cpu()
            payload[f"{ltype}_v"] = v.detach().float().cpu()
        os.makedirs(dump_dir, exist_ok=True)
        fp = os.path.join(dump_dir, "vllm_shared_kv.pt")
        torch.save(payload, fp)
        shapes = {kk: tuple(vv.shape) for kk, vv in payload.items()}
        print(f"[MTP-KV-DUMP] wrote {fp} shapes={shapes}", flush=True)
        _dumped["done"] = True
    except Exception as e:
        print(f"[MTP-KV-DUMP] failed: {e}", flush=True)
