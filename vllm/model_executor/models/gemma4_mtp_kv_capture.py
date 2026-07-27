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

_ENABLED = os.environ.get("VLLM_GEMMA4_MTP_CAPTURE_KV", "0") == "1"

# Per-thread capture buffer: {layer_type: (k, v)} for the CURRENT forward.
# vLLM worker runs the model forward on a single thread; thread-local keeps
# TP workers / multiple engines from clobbering each other.
_local = threading.local()


def is_enabled() -> bool:
    return _ENABLED


def _buf() -> dict:
    b = getattr(_local, "buf", None)
    if b is None:
        b = {}
        _local.buf = b
    return b


def reset() -> None:
    """Clear the capture buffer at the start of a target forward."""
    if _ENABLED:
        _buf().clear()


def record(layer_type: str, k: torch.Tensor, v: torch.Tensor) -> None:
    """Record post-RoPE/post-norm K/V for a shared-source layer.

    Args:
        layer_type: "sliding_attention" or "full_attention".
        k, v: shape [num_tokens, num_kv_heads * head_dim] (as computed in
            Gemma4Attention.forward right before self.attn(q, k, v)).
    """
    if not _ENABLED:
        return
    # Detach + clone so later in-place ops (if any) can't corrupt the capture.
    _buf()[layer_type] = (k.detach(), v.detach())


def take() -> dict:
    """Return and clear the current forward's captured K/V.

    Returns:
        dict mapping layer_type -> (k, v). Empty if nothing was captured.
    """
    if not _ENABLED:
        return {}
    b = _buf()
    out = dict(b)
    b.clear()
    return out
