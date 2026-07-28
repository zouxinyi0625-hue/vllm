# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP shared-KV capture for online speculative-decoding training.

TorchSpec trains a Gemma4 MTP draft that, at deploy time, shares the target's
K/V from the last non-shared sliding layer (L28) and full layer (L29) via
vLLM's KV-sharing mechanism. To generate training data with vLLM (so the draft
learns exactly the deploy-time distribution), we must capture those same K/V
tensors during the target forward.

FULLGRAPH CONSTRAINT: vLLM torch.compiles the model forward in *fullgraph* mode,
which forbids graph breaks. A plain Python function with side effects (dict
write / file dump) triggers a graph break and errors ("Skip calling
torch.compiler.disable()d function"). Even @torch._dynamo.disable is rejected.

SOLUTION: expose capture as a custom torch op (torch.library). Custom ops are
opaque to Dynamo (a legal single node in the fullgraph), so their Python body --
including copying the tensors aside and dumping to disk -- runs normally without
breaking the graph. This mirrors how vLLM itself moves data out of compiled
forwards (unified_kv_cache_update is a custom op).

OFF-BY-DEFAULT: enabled only when env VLLM_GEMMA4_MTP_CAPTURE_KV=1. Diagnostic
disk dump when VLLM_GEMMA4_MTP_DUMP_DIR is also set. When disabled, record() is
a cheap no-op and vLLM numerics are untouched.
"""

from __future__ import annotations

import os
import threading

import torch

_local = threading.local()
_dumped = {"done": False}


def is_enabled() -> bool:
    # Live env read (NOT an import-time snapshot): vLLM's import graph / worker
    # fork order can import this module before the env is set.
    return os.environ.get("VLLM_GEMMA4_MTP_CAPTURE_KV", "0") == "1"


def _buf() -> dict:
    b = getattr(_local, "buf", None)
    if b is None:
        b = {}
        _local.buf = b
    return b


def reset() -> None:
    """Clear the per-forward capture buffer."""
    if is_enabled():
        _buf().clear()


def take() -> dict:
    """Return and clear the current forward's captured K/V (training proposer)."""
    if not is_enabled():
        return {}
    b = _buf()
    out = dict(b)
    b.clear()
    return out


def _maybe_dump() -> None:
    dump_dir = os.environ.get("VLLM_GEMMA4_MTP_DUMP_DIR")
    if not dump_dir or _dumped["done"]:
        return
    buf = _buf()
    if "sliding_attention" not in buf or "full_attention" not in buf:
        return
    try:
        payload = {}
        for ltype, (k, v) in buf.items():
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


# ---- Custom op: opaque to Dynamo, so its Python body is fullgraph-safe. ----
# layer_kind: 0 = sliding_attention, 1 = full_attention.
_KIND = {0: "sliding_attention", 1: "full_attention"}

try:
    _LIB = torch.library.Library("gemma4_mtp", "FRAGMENT")
    _LIB.define("record(int layer_kind, Tensor k, Tensor v) -> ()")

    def _record_impl(layer_kind: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if not is_enabled():
            return
        ltype = _KIND.get(int(layer_kind))
        if ltype is None:
            return
        _buf()[ltype] = (k.detach().clone(), v.detach().clone())
        _maybe_dump()

    def _record_meta(layer_kind: int, k: torch.Tensor, v: torch.Tensor) -> None:
        return None

    _LIB.impl("record", _record_impl, "CompositeExplicitAutograd")
    _LIB.impl("record", _record_meta, "Meta")
    _OP_OK = True
except Exception as _e:  # pragma: no cover - defensive
    _OP_OK = False
    print(f"[MTP-KV] custom op registration failed: {_e}", flush=True)


def record(layer_type: str, k: torch.Tensor, v: torch.Tensor) -> None:
    """Capture entry called from Gemma4Attention.forward (inside fullgraph).

    Dispatches to the custom op so Dynamo sees one opaque node (no graph break).
    """
    kind = 0 if layer_type == "sliding_attention" else 1
    torch.ops.gemma4_mtp.record(kind, k, v)
