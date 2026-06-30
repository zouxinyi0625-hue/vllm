#!/usr/bin/env python3
"""Inspect Gemma4-26B-A4B model architecture and print detailed structure.

Outputs:
  1. Full config parameters
  2. Layer type distribution (sliding vs full attention)
  3. KV sharing (YOCO) structure
  4. RoPE configuration per layer type
  5. MoE parameters
  6. ASCII architecture diagram

Usage:
    python inspect_gemma4_arch.py
    python inspect_gemma4_arch.py --model google/gemma-4-26B-A4B-it
    python inspect_gemma4_arch.py --model /path/to/local/model
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--assistant", default=None,
                    help="Path to assistant/MTP model (optional)")
    args = ap.parse_args()

    from transformers import AutoConfig

    print("=" * 80)
    print(f"  Gemma4 Architecture Inspector")
    print(f"  Model: {args.model}")
    print("=" * 80)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # =========================================================================
    # 1. Core Parameters
    # =========================================================================
    print("\n" + "=" * 80)
    print("  1. CORE MODEL PARAMETERS")
    print("=" * 80)

    core_fields = [
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads",
        "head_dim", "global_head_dim",
        "num_global_key_value_heads",
        "vocab_size", "max_position_embeddings",
        "rms_norm_eps", "hidden_activation",
        "tie_word_embeddings",
    ]
    for f in core_fields:
        val = getattr(config, f, "N/A")
        print(f"  {f:40s} = {val}")

    # =========================================================================
    # 2. MoE Parameters
    # =========================================================================
    print("\n" + "=" * 80)
    print("  2. MIXTURE OF EXPERTS (MoE)")
    print("=" * 80)

    moe_fields = [
        "num_experts", "top_k_experts", "num_shared_experts",
        "moe_intermediate_size", "expert_intermediate_size",
        "enable_moe_block", "use_second_mlp_block",
    ]
    for f in moe_fields:
        val = getattr(config, f, "N/A")
        print(f"  {f:40s} = {val}")

    # Compute active params estimate
    num_experts = getattr(config, "num_experts", 0)
    top_k = getattr(config, "top_k_experts", 0)
    hidden = getattr(config, "hidden_size", 0)
    moe_inter = getattr(config, "moe_intermediate_size", None) or getattr(config, "expert_intermediate_size", 0)
    if num_experts and top_k and moe_inter:
        expert_params = 3 * hidden * moe_inter  # gate + up + down
        total_expert_params = expert_params * num_experts
        active_expert_params = expert_params * top_k
        print(f"\n  --- Derived ---")
        print(f"  {'params per expert (gate+up+down)':40s} = {expert_params / 1e6:.1f}M")
        print(f"  {'total expert params (all experts)':40s} = {total_expert_params / 1e9:.2f}B")
        print(f"  {'active expert params (top-k)':40s} = {active_expert_params / 1e6:.1f}M")

    # =========================================================================
    # 3. Attention Architecture
    # =========================================================================
    print("\n" + "=" * 80)
    print("  3. ATTENTION ARCHITECTURE")
    print("=" * 80)

    attn_fields = [
        "sliding_window", "attention_bias",
        "attn_logit_softcapping", "final_logit_softcapping",
        "attention_k_eq_v",
    ]
    for f in attn_fields:
        val = getattr(config, f, "N/A")
        print(f"  {f:40s} = {val}")

    # =========================================================================
    # 4. Layer Types Distribution
    # =========================================================================
    print("\n" + "=" * 80)
    print("  4. LAYER TYPES DISTRIBUTION")
    print("=" * 80)

    layer_types = getattr(config, "layer_types", [])
    if layer_types:
        print(f"  Total layers: {len(layer_types)}")
        print(f"  Distribution: {dict(Counter(layer_types))}")
        print()

        # Print pattern
        print("  Layer pattern (S=sliding, F=full):")
        pattern = ""
        for i, lt in enumerate(layer_types):
            if lt == "sliding_attention":
                pattern += "S"
            elif lt == "full_attention":
                pattern += "F"
            else:
                pattern += "?"
            if (i + 1) % 10 == 0:
                pattern += "|"
        print(f"  {pattern}")
        print()

        # Print detailed per-layer
        print("  Per-layer detail:")
        for i, lt in enumerate(layer_types):
            marker = "  "
            if i == 0:
                marker = ">>"
            print(f"  {marker} Layer {i:2d}: {lt}")
    else:
        print("  layer_types not found in config")

    # =========================================================================
    # 5. KV Sharing (YOCO)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  5. KV SHARING (YOCO - You Only Cache Once)")
    print("=" * 80)

    kv_fields = [
        "num_kv_shared_layers",
        "kv_sharing_fast_prefill",
    ]
    for f in kv_fields:
        val = getattr(config, f, "N/A")
        print(f"  {f:40s} = {val}")

    num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
    num_layers = getattr(config, "num_hidden_layers", 0)
    if num_kv_shared and num_layers:
        self_layers = num_layers - num_kv_shared
        print(f"\n  --- Structure ---")
        print(f"  Self-decoder layers (own KV cache):  0 .. {self_layers - 1}  ({self_layers} layers)")
        print(f"  Cross-decoder layers (shared KV):    {self_layers} .. {num_layers - 1}  ({num_kv_shared} layers)")
        print(f"  KV cache savings: {num_kv_shared}/{num_layers} layers share KV = {num_kv_shared/num_layers*100:.0f}% less KV storage")

    # =========================================================================
    # 6. RoPE Configuration
    # =========================================================================
    print("\n" + "=" * 80)
    print("  6. RoPE (Rotary Position Embedding) CONFIGURATION")
    print("=" * 80)

    rope_fields = [
        "rope_theta", "rope_scaling", "rope_local_base_freq",
        "partial_rotary_factor",
    ]
    for f in rope_fields:
        val = getattr(config, f, "N/A")
        if isinstance(val, dict):
            print(f"  {f}:")
            print(f"    {json.dumps(val, indent=4, default=str)}")
        else:
            print(f"  {f:40s} = {val}")

    rope_params = getattr(config, "rope_parameters", None)
    if rope_params:
        print(f"\n  rope_parameters (per layer type):")
        print(f"    {json.dumps(rope_params, indent=4, default=str)}")

    # =========================================================================
    # 7. Special Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("  7. SPECIAL FEATURES")
    print("=" * 80)

    special_fields = [
        "hidden_size_per_layer_input", "vocab_size_per_layer_input",
        "use_double_wide_mlp",
        "use_ordered_embeddings", "num_centroids",
        "centroid_intermediate_top_k",
    ]
    for f in special_fields:
        val = getattr(config, f, "N/A")
        print(f"  {f:40s} = {val}")

    # =========================================================================
    # 8. MTP / Speculative Decoding
    # =========================================================================
    print("\n" + "=" * 80)
    print("  8. MTP / SPECULATIVE DECODING CONFIG")
    print("=" * 80)

    mtp_fields = [
        "num_nextn_predict_layers", "mtp_num_hidden_layers",
        "speculator_model_id", "assistant_model_id",
    ]
    for f in mtp_fields:
        val = getattr(config, f, "N/A")
        print(f"  {f:40s} = {val}")

    # =========================================================================
    # 9. Assistant Model (if provided)
    # =========================================================================
    if args.assistant:
        print("\n" + "=" * 80)
        print("  9. ASSISTANT (MTP DRAFT) MODEL")
        print("=" * 80)
        try:
            ast_config = AutoConfig.from_pretrained(args.assistant, trust_remote_code=True)
            ast_fields = [
                "num_hidden_layers", "hidden_size", "num_attention_heads",
                "num_key_value_heads", "head_dim", "intermediate_size",
            ]
            for f in ast_fields:
                val = getattr(ast_config, f, "N/A")
                print(f"  {f:40s} = {val}")
        except Exception as e:
            print(f"  Failed to load assistant config: {e}")

    # =========================================================================
    # 10. ASCII Architecture Diagram
    # =========================================================================
    print("\n" + "=" * 80)
    print("  10. ARCHITECTURE DIAGRAM")
    print("=" * 80)

    head_dim = getattr(config, "head_dim", 256)
    global_head_dim = getattr(config, "global_head_dim", 512)
    n_heads = getattr(config, "num_attention_heads", 0)
    n_kv_heads = getattr(config, "num_key_value_heads", 0)
    n_global_kv_heads = getattr(config, "num_global_key_value_heads", 0)
    inter_size = getattr(config, "intermediate_size", 0)
    sw = getattr(config, "sliding_window", 0)

    diagram = f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Gemma4-26B-A4B Architecture                       │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  Input Tokens                                                       │
    │       │                                                             │
    │       ▼                                                             │
    │  ┌─────────────────┐                                                │
    │  │   Embedding      │  vocab_size={getattr(config, 'vocab_size', '?')}
    │  │                  │  hidden_size={hidden}
    │  └────────┬────────┘                                                │
    │           │                                                         │
    │           ▼                                                         │
    │  ╔═══════════════════════════════════════════════════════════════╗   │
    │  ║  DECODER LAYERS  (x{num_layers} total)                        ║   │
    │  ╠═══════════════════════════════════════════════════════════════╣   │
    │  ║                                                               ║   │
    │  ║  ┌─── Self-Decoder (layers 0..{self_layers-1 if num_kv_shared else num_layers-1}) ──────────────────┐  ║   │
    │  ║  │  Own KV cache per layer                          │  ║   │
    │  ║  │                                                  │  ║   │
    │  ║  │  SLIDING ATTENTION layers (head_dim={head_dim}):  │  ║   │
    │  ║  │    Q heads: {n_heads}, KV heads: {n_kv_heads}              │  ║   │
    │  ║  │    Window: {sw} tokens                      │  ║   │
    │  ║  │    RoPE: local base freq                         │  ║   │
    │  ║  │    Backend: TRITON_ATTN (forced)                 │  ║   │
    │  ║  │                                                  │  ║   │
    │  ║  │  FULL ATTENTION layers (head_dim={global_head_dim}):       │  ║   │
    │  ║  │    Q heads: {n_heads}, KV heads: {n_global_kv_heads or n_kv_heads}              │  ║   │
    │  ║  │    K=V: {getattr(config, 'attention_k_eq_v', False)}                            │  ║   │
    │  ║  │    RoPE: global base freq                        │  ║   │
    │  ║  │    Backend: TRITON_ATTN (forced, 512>FA limit)   │  ║   │
    │  ║  └──────────────────────────────────────────────────┘  ║   │
    │  ║                                                               ║   │"""

    if num_kv_shared:
        diagram += f"""
    │  ║  ┌─── Cross-Decoder (layers {self_layers}..{num_layers-1}) ─────────────┐  ║   │
    │  ║  │  SHARED KV cache (from self-decoder layers)      │  ║   │
    │  ║  │  Double-wide MLP: {getattr(config, 'use_double_wide_mlp', 'N/A')}                │  ║   │
    │  ║  └──────────────────────────────────────────────────┘  ║   │"""

    diagram += f"""
    │  ║                                                               ║   │
    │  ║  Each layer contains:                                         ║   │
    │  ║  ┌──────────────────────────────────────────────────────┐     ║   │
    │  ║  │  RMSNorm → Attention (Q/K/V norms) → RMSNorm → FFN  │     ║   │
    │  ║  │                                                      │     ║   │
    │  ║  │  FFN = MLP (inter_size={inter_size})           │     ║   │
    │  ║  │      + MoE ({num_experts} experts, top-{top_k} active)     │     ║   │
    │  ║  │        (moe_inter={moe_inter})              │     ║   │
    │  ║  │                                                      │     ║   │
    │  ║  │  + Per-Layer Embedding (if enabled)                  │     ║   │
    │  ║  │  + Layer Scalar (per-layer learned scale)            │     ║   │
    │  ║  └──────────────────────────────────────────────────────┘     ║   │
    │  ║                                                               ║   │
    │  ╚═══════════════════════════════════════════════════════════════╝   │
    │           │                                                         │
    │           ▼                                                         │
    │  ┌─────────────────┐                                                │
    │  │   RMSNorm        │                                                │
    │  │   LM Head        │  → logits (softcap={getattr(config, 'final_logit_softcapping', 'N/A')})
    │  └─────────────────┘                                                │
    │                                                                     │
    ├─────────────────────────────────────────────────────────────────────┤
    │  MTP Speculative Decoding (Assistant Model):                        │
    │    - Q-only draft layers (K/V shared from target)                   │
    │    - Pre/Post projection to bridge hidden_size mismatch             │
    │    - Optional centroids masking for sparse vocab computation         │
    └─────────────────────────────────────────────────────────────────────┘
"""
    print(diagram)

    # =========================================================================
    # 11. Raw config dump
    # =========================================================================
    print("\n" + "=" * 80)
    print("  11. FULL CONFIG (raw)")
    print("=" * 80)
    try:
        config_dict = config.to_dict()
        # Remove very long lists for readability (but keep layer_types)
        for k, v in sorted(config_dict.items()):
            if isinstance(v, list) and len(v) > 20 and k != "layer_types":
                print(f"  {k}: [{v[0]}, {v[1]}, ... ({len(v)} items)]")
            elif isinstance(v, dict):
                print(f"  {k}: {json.dumps(v, indent=4, default=str)}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"  Failed to dump config: {e}")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
