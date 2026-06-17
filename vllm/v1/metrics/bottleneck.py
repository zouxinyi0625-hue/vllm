# SPDX-License-Identifier: Apache-2.0
"""
Lightweight bottleneck profiler for agentic optimization.

Enable via VLLM_BOTTLENECK_PROFILE=1. Outputs a JSON summary every
VLLM_BOTTLENECK_PROFILE_INTERVAL steps (default 100) to the path
specified by VLLM_BOTTLENECK_PROFILE_OUTPUT (default /tmp/vllm_bottleneck.json).

Sub-component breakdown (attention/MoE/MLP) is enabled when
VLLM_BOTTLENECK_PROFILE_DETAIL=1 (default 0). Hooks only fire during
eager execution (warmup/profile_run/capture); during CUDA graph replay,
calibrated ratios from the eager phase are applied to model_forward_ms.

The summary is designed to be consumed by agentic_vllm's analyze command
to decide which optimization strategy to apply next.
"""

import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

_ENABLED = os.environ.get("VLLM_BOTTLENECK_PROFILE", "0") == "1"
_DETAIL = os.environ.get("VLLM_BOTTLENECK_PROFILE_DETAIL", "0") == "1"
_INTERVAL = int(os.environ.get("VLLM_BOTTLENECK_PROFILE_INTERVAL", "100"))
_OUTPUT_PATH = os.environ.get("VLLM_BOTTLENECK_PROFILE_OUTPUT",
                              "/tmp/vllm_bottleneck.json")



@dataclass
class StepTimings:
    scheduler_ms: float = 0.0
    model_forward_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_step_ms: float = 0.0

    # Sub-component breakdown (populated when VLLM_BOTTLENECK_PROFILE_DETAIL=1)
    attention_ms: float = 0.0
    moe_ms: float = 0.0
    mlp_ms: float = 0.0

    num_scheduled_tokens: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_running_reqs: int = 0
    batch_size: int = 0
    kv_cache_usage: float = 0.0


@dataclass
class BottleneckSummary:
    num_steps: int = 0
    avg_step_ms: float = 0.0
    avg_scheduler_ms: float = 0.0
    avg_model_forward_ms: float = 0.0
    avg_preprocess_ms: float = 0.0
    avg_postprocess_ms: float = 0.0

    scheduler_pct: float = 0.0
    model_forward_pct: float = 0.0
    preprocess_pct: float = 0.0
    postprocess_pct: float = 0.0

    # Sub-component breakdown (when detail enabled)
    avg_attention_ms: float = 0.0
    avg_moe_ms: float = 0.0
    avg_mlp_ms: float = 0.0
    attention_pct: float = 0.0
    moe_pct: float = 0.0
    mlp_pct: float = 0.0

    avg_batch_size: float = 0.0
    avg_num_scheduled_tokens: float = 0.0
    avg_prefill_tokens: float = 0.0
    avg_decode_tokens: float = 0.0
    prefill_ratio: float = 0.0
    avg_kv_cache_usage: float = 0.0

    tokens_per_second: float = 0.0
    bottleneck: str = ""


class LayerEventAccumulator:
    """Direct CUDA event accumulator for model sub-components.

    Instead of using nn.Module forward hooks (which don't fire under
    torch.compile or CUDA graph replay), this class provides explicit
    record_start/record_end methods called directly from model forward code.

    CUDA events recorded during graph capture are automatically re-recorded
    during replay, so elapsed_time() returns valid measurements even in
    CUDA graph mode.
    """

    def __init__(self):
        import torch
        self._torch = torch
        self._attention_events: list[tuple] = []
        self._moe_events: list[tuple] = []
        self._mlp_events: list[tuple] = []
        self._active = False
        self._pending_starts: dict[str, "torch.cuda.Event"] = {}

    def activate(self):
        """Enable event recording."""
        self._active = True

    def record_start(self, category: str):
        """Record a start event for a sub-component. Call before the op."""
        if not self._active:
            return
        start = self._torch.cuda.Event(enable_timing=True)
        start.record()
        self._pending_starts[category] = start

    def record_end(self, category: str):
        """Record an end event for a sub-component. Call after the op."""
        if not self._active:
            return
        start = self._pending_starts.pop(category, None)
        if start is None:
            return
        end = self._torch.cuda.Event(enable_timing=True)
        end.record()
        event_list = getattr(self, f"_{category}_events")
        event_list.append((start, end))

    def begin_step(self):
        self._attention_events.clear()
        self._moe_events.clear()
        self._mlp_events.clear()
        self._pending_starts.clear()

    def collect_step_ms(self) -> tuple[float, float, float]:
        """Synchronize GPU and return (attention_ms, moe_ms, mlp_ms).

        CUDA events recorded during graph capture are automatically
        re-recorded during graph replay, so this works in all modes.
        """
        if not self._active:
            return 0.0, 0.0, 0.0

        has_events = bool(
            self._attention_events or self._moe_events or self._mlp_events
        )
        if not has_events:
            return 0.0, 0.0, 0.0

        self._torch.cuda.synchronize()
        attn_ms = sum(s.elapsed_time(e) for s, e in self._attention_events)
        moe_ms = sum(s.elapsed_time(e) for s, e in self._moe_events)
        mlp_ms = sum(s.elapsed_time(e) for s, e in self._mlp_events)
        return attn_ms, moe_ms, mlp_ms

    def deactivate(self):
        self._active = False


class BottleneckProfiler:
    def __init__(self):
        self.enabled = _ENABLED
        self.detail = _DETAIL and _ENABLED
        self.interval = _INTERVAL
        self.output_path = Path(_OUTPUT_PATH)
        self._history: deque[StepTimings] = deque(maxlen=_INTERVAL)
        self._step_count = 0
        self._current: StepTimings | None = None
        self._timer_stack: dict[str, float] = {}
        self._layer_events: LayerEventAccumulator | None = None

    def register_model_hooks(self, model: "nn.Module"):
        """Initialize the layer event accumulator for sub-component timing.

        The accumulator is activated immediately. Model forward code calls
        record_start/record_end directly via get_layer_events().
        """
        if not self.detail:
            return
        self._layer_events = LayerEventAccumulator()
        self._layer_events.activate()

    def begin_step(self):
        if not self.enabled:
            return
        self._current = StepTimings()
        self._timer_stack["step"] = time.perf_counter()
        if self._layer_events:
            self._layer_events.begin_step()

    def begin_phase(self, name: str):
        if not self.enabled:
            return
        self._timer_stack[name] = time.perf_counter()

    def end_phase(self, name: str):
        if not self.enabled or name not in self._timer_stack:
            return
        elapsed_ms = (time.perf_counter() - self._timer_stack.pop(name)) * 1000
        if self._current is None:
            return
        if name == "scheduler":
            self._current.scheduler_ms = elapsed_ms
        elif name == "model_forward":
            self._current.model_forward_ms = elapsed_ms
            if self._layer_events:
                attn, moe, mlp = self._layer_events.collect_step_ms()
                self._current.attention_ms = attn
                self._current.moe_ms = moe
                self._current.mlp_ms = mlp
        elif name == "preprocess":
            self._current.preprocess_ms = elapsed_ms
        elif name == "postprocess":
            self._current.postprocess_ms = elapsed_ms

    def record_batch_info(self, num_scheduled_tokens: int,
                          num_prefill_tokens: int, num_decode_tokens: int,
                          num_running_reqs: int, batch_size: int,
                          kv_cache_usage: float):
        if not self.enabled or self._current is None:
            return
        self._current.num_scheduled_tokens = num_scheduled_tokens
        self._current.num_prefill_tokens = num_prefill_tokens
        self._current.num_decode_tokens = num_decode_tokens
        self._current.num_running_reqs = num_running_reqs
        self._current.batch_size = batch_size
        self._current.kv_cache_usage = kv_cache_usage

    def end_step(self):
        if not self.enabled or self._current is None:
            return
        self._current.total_step_ms = (
            time.perf_counter() - self._timer_stack.pop("step", time.perf_counter())
        ) * 1000
        self._history.append(self._current)
        self._current = None
        self._step_count += 1

        if self._step_count % self.interval == 0:
            self._flush()

    def _flush(self):
        if not self._history:
            return
        n = len(self._history)
        total_step = sum(s.total_step_ms for s in self._history)
        total_sched = sum(s.scheduler_ms for s in self._history)
        total_fwd = sum(s.model_forward_ms for s in self._history)
        total_pre = sum(s.preprocess_ms for s in self._history)
        total_post = sum(s.postprocess_ms for s in self._history)
        total_tokens = sum(s.num_scheduled_tokens for s in self._history)
        total_prefill = sum(s.num_prefill_tokens for s in self._history)
        total_decode = sum(s.num_decode_tokens for s in self._history)

        # Sub-component totals
        total_attn = sum(s.attention_ms for s in self._history)
        total_moe = sum(s.moe_ms for s in self._history)
        total_mlp = sum(s.mlp_ms for s in self._history)

        avg_step = total_step / n
        avg_sched = total_sched / n
        avg_fwd = total_fwd / n
        avg_pre = total_pre / n
        avg_post = total_post / n
        avg_attn = total_attn / n
        avg_moe = total_moe / n
        avg_mlp = total_mlp / n

        summary = BottleneckSummary(
            num_steps=n,
            avg_step_ms=round(avg_step, 3),
            avg_scheduler_ms=round(avg_sched, 3),
            avg_model_forward_ms=round(avg_fwd, 3),
            avg_preprocess_ms=round(avg_pre, 3),
            avg_postprocess_ms=round(avg_post, 3),
            scheduler_pct=round(avg_sched / avg_step * 100, 1) if avg_step > 0 else 0,
            model_forward_pct=round(avg_fwd / avg_step * 100, 1) if avg_step > 0 else 0,
            preprocess_pct=round(avg_pre / avg_step * 100, 1) if avg_step > 0 else 0,
            postprocess_pct=round(avg_post / avg_step * 100, 1) if avg_step > 0 else 0,
            avg_attention_ms=round(avg_attn, 3),
            avg_moe_ms=round(avg_moe, 3),
            avg_mlp_ms=round(avg_mlp, 3),
            attention_pct=round(avg_attn / avg_fwd * 100, 1) if avg_fwd > 0 else 0,
            moe_pct=round(avg_moe / avg_fwd * 100, 1) if avg_fwd > 0 else 0,
            mlp_pct=round(avg_mlp / avg_fwd * 100, 1) if avg_fwd > 0 else 0,
            avg_batch_size=round(sum(s.batch_size for s in self._history) / n, 1),
            avg_num_scheduled_tokens=round(total_tokens / n, 1),
            avg_prefill_tokens=round(total_prefill / n, 1),
            avg_decode_tokens=round(total_decode / n, 1),
            prefill_ratio=round(total_prefill / total_tokens, 3) if total_tokens > 0 else 0,
            avg_kv_cache_usage=round(
                sum(s.kv_cache_usage for s in self._history) / n, 3),
            tokens_per_second=round(total_tokens / (total_step / 1000), 1) if total_step > 0 else 0,
        )

        # Determine bottleneck
        phases = {
            "model_forward": avg_fwd,
            "scheduler": avg_sched,
            "preprocess": avg_pre,
            "postprocess": avg_post,
        }
        summary.bottleneck = max(phases, key=phases.get)  # type: ignore

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)

    def get_latest_summary(self) -> BottleneckSummary | None:
        if not self._history:
            return None
        self._flush()
        return None


_PROFILER = BottleneckProfiler()


def get_bottleneck_profiler() -> BottleneckProfiler:
    return _PROFILER


def get_layer_events() -> LayerEventAccumulator | None:
    """Get the layer event accumulator for direct sub-component timing.

    Called from model forward code (e.g., Gemma4DecoderLayer.forward()).
    Returns None if detail profiling is not enabled.
    """
    return _PROFILER._layer_events
