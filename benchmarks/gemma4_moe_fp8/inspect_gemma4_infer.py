#!/usr/bin/env python3
"""Deep inspection of Gemma4-26B-A4B inference flow.

Loads the actual model and prints:
  1. Complete layer-by-layer module structure with parameter shapes
  2. Forward pass execution order per layer
  3. MoE routing & expert dispatch details
  4. MTP (speculative decoding) integration points
  5. KV sharing structure
  6. Detailed architecture diagram

Usage:
    # Inspect model structure (loads weights, needs GPU)
    python inspect_gemma4_infer.py --model /path/to/text_only

    # Config-only mode (no GPU needed, just prints structure from config)
    python inspect_gemma4_infer.py --model google/gemma-4-26B-A4B-it --config-only

    # With MTP assistant model
    python inspect_gemma4_infer.py --model /path/to/text_only --assistant /path/to/assistant
"""
from __future__ import annotations
import argparse
import sys
import json
from collections import OrderedDict


def print_section(title):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}\n")


def format_size(numel):
    if numel >= 1e9:
        return f"{numel/1e9:.2f}B"
    elif numel >= 1e6:
        return f"{numel/1e6:.2f}M"
    elif numel >= 1e3:
        return f"{numel/1e3:.1f}K"
    return str(numel)


def inspect_config(config):
    """Print the text model config structure."""
    # Gemma4 is multimodal, text config is nested
    text_config = getattr(config, "text_config", config)
    if hasattr(text_config, "to_dict"):
        tc = text_config.to_dict() if hasattr(text_config, "to_dict") else vars(text_config)
    else:
        tc = text_config

    # Handle both dict and object
    def get(key, default="N/A"):
        if isinstance(tc, dict):
            return tc.get(key, default)
        return getattr(tc, key, default)

    print_section("1. TEXT MODEL CONFIG (from text_config)")

    fields = {
        "Model Geometry": [
            "num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "global_head_dim", "num_global_key_value_heads",
        ],
        "MoE": [
            "enable_moe_block", "num_experts", "top_k_experts", "moe_intermediate_size",
        ],
        "Attention": [
            "sliding_window", "attention_bias", "attention_k_eq_v",
            "attn_logit_softcapping", "final_logit_softcapping",
        ],
        "KV Sharing": [
            "num_kv_shared_layers", "use_double_wide_mlp",
        ],
        "Special": [
            "hidden_size_per_layer_input", "vocab_size_per_layer_input",
            "hidden_activation", "rms_norm_eps",
        ],
    }

    for group, keys in fields.items():
        print(f"  [{group}]")
        for k in keys:
            print(f"    {k:40s} = {get(k)}")
        print()

    # Layer types
    layer_types = get("layer_types", [])
    if layer_types:
        print(f"  [Layer Pattern] (S=sliding/256d, F=full/512d)")
        print(f"    Total: {len(layer_types)} layers")
        line = "    "
        for i, lt in enumerate(layer_types):
            line += "F" if "full" in lt else "S"
            if (i + 1) % 6 == 0:
                line += " | "
        print(line)
        print()

    # RoPE
    rope_params = get("rope_parameters", {})
    if rope_params:
        print(f"  [RoPE per layer type]")
        for lt, params in rope_params.items():
            print(f"    {lt}: {json.dumps(params, default=str)}")
        print()

    return tc


def inspect_model_structure(model):
    """Print full model module tree with parameter shapes."""
    print_section("2. MODEL MODULE TREE (with param shapes)")

    total_params = 0
    layer_params = {}

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        # Track per-layer params
        parts = name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                layer_num = int(parts[idx + 1])
                layer_params[layer_num] = layer_params.get(layer_num, 0) + numel

    print(f"  Total parameters: {format_size(total_params)} ({total_params:,})")
    print()

    # Print module tree (first 2 layers detailed, rest summarized)
    printed_layers = set()
    for name, module in model.named_modules():
        depth = name.count(".")
        if depth > 4:
            continue  # Skip too deep

        parts = name.split(".")
        # Only print detailed structure for layers 0 (sliding) and 5 (full)
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                layer_num = int(parts[idx + 1])
                if layer_num not in (0, 5) and depth > 2:
                    if layer_num not in printed_layers:
                        printed_layers.add(layer_num)
                        print(f"  {'  ' * 2}layers.{layer_num}: (same structure as above)")
                    continue

        indent = "  " * (depth + 1)
        class_name = module.__class__.__name__
        # Count direct params
        direct_params = sum(p.numel() for p in module.parameters(recurse=False))
        param_info = f" [{format_size(direct_params)}]" if direct_params > 0 else ""
        print(f"  {indent}{name}: {class_name}{param_info}")

    print()
    print(f"  [Per-layer parameter count]")
    for layer_num in sorted(layer_params.keys()):
        print(f"    Layer {layer_num:2d}: {format_size(layer_params[layer_num])}")


def inspect_single_layer(model, layer_idx):
    """Deep inspect one decoder layer."""
    print_section(f"3. DECODER LAYER {layer_idx} — DETAILED STRUCTURE")

    # Navigate to the layer
    layers = None
    for name, mod in model.named_modules():
        if name.endswith(f"layers.{layer_idx}"):
            layers = mod
            break

    if layers is None:
        print(f"  Layer {layer_idx} not found!")
        return

    print(f"  Class: {layers.__class__.__name__}")
    print(f"  Is sliding: {getattr(layers, 'is_sliding', 'N/A')}")
    print(f"  Enable MoE: {getattr(layers, 'enable_moe_block', 'N/A')}")
    print()

    print(f"  [Sublayers in forward() execution order]")
    print()
    print(f"  ┌──────────────────────────────────────────────────────────────────┐")
    print(f"  │  input_ids → hidden_states                                       │")
    print(f"  ├──────────────────────────────────────────────────────────────────┤")
    print(f"  │                                                                  │")
    print(f"  │  1. input_layernorm (RMSNorm)                                    │")
    print(f"  │     │                                                            │")
    print(f"  │     ▼                                                            │")
    print(f"  │  2. self_attn (Gemma4Attention)                                  │")

    # Attention details
    attn = getattr(layers, "self_attn", None)
    if attn:
        for pname, p in attn.named_parameters():
            if "qkv" in pname or "o_proj" in pname:
                print(f"  │     ├─ {pname}: {list(p.shape)}")
        print(f"  │     ├─ is_sliding: {getattr(attn, 'is_sliding', '?')}")
        print(f"  │     ├─ head_dim: {getattr(attn, 'head_dim', '?')}")
        print(f"  │     ├─ num_heads: {getattr(attn, 'num_heads', '?')}")
        print(f"  │     ├─ num_kv_heads: {getattr(attn, 'num_kv_heads', '?')}")
        sliding_window = getattr(attn, 'sliding_window', None)
        if sliding_window:
            print(f"  │     └─ sliding_window: {sliding_window}")

    print(f"  │     │                                                            │")
    print(f"  │     ▼                                                            │")
    print(f"  │  3. post_attention_layernorm (RMSNorm)                           │")
    print(f"  │     │                                                            │")
    print(f"  │     ▼  (+residual)                                               │")
    print(f"  │                                                                  │")
    print(f"  │  4. pre_feedforward_layernorm (RMSNorm)                          │")
    print(f"  │     │                                                            │")
    print(f"  │     ▼                                                            │")
    print(f"  │  5. mlp (Gemma4MLP)                                              │")

    # MLP details
    mlp = getattr(layers, "mlp", None)
    if mlp:
        for pname, p in mlp.named_parameters():
            print(f"  │     ├─ {pname}: {list(p.shape)}")

    # MoE block
    if getattr(layers, "enable_moe_block", False):
        print(f"  │     │                                                            │")
        print(f"  │     ▼                                                            │")
        print(f"  │  6. post_feedforward_layernorm_1 (RMSNorm) → MLP output          │")
        print(f"  │                                                                  │")
        print(f"  │  7. PARALLEL MoE PATH (from pre-MLP residual):                   │")
        print(f"  │     ├─ router (Gemma4Router):                                    │")

        router = getattr(layers, "router", None)
        if router:
            for pname, p in router.named_parameters():
                print(f"  │     │   ├─ {pname}: {list(p.shape)}")

        print(f"  │     ├─ pre_feedforward_layernorm_2 (RMSNorm)                     │")
        print(f"  │     ├─ moe (Gemma4MoE / FusedMoE):                              │")

        moe = getattr(layers, "moe", None)
        if moe:
            for pname, p in moe.named_parameters():
                if p.numel() > 1000:  # Only show large params
                    print(f"  │     │   ├─ {pname}: {list(p.shape)}")

        print(f"  │     └─ post_feedforward_layernorm_2 (RMSNorm)                    │")
        print(f"  │                                                                  │")
        print(f"  │  8. COMBINE: norm(MLP) + norm(MoE)                               │")

    print(f"  │     │                                                            │")
    print(f"  │     ▼                                                            │")
    print(f"  │  9. post_feedforward_layernorm (RMSNorm)                         │")
    print(f"  │     │                                                            │")
    print(f"  │     ▼  (+residual)                                               │")
    print(f"  │                                                                  │")

    # PLE
    if hasattr(layers, "per_layer_input_gate"):
        print(f"  │  10. Per-Layer Embedding Gate:                                   │")
        print(f"  │      gate = gelu(per_layer_input_gate(h))                        │")
        print(f"  │      h = h + norm(per_layer_projection(gate * per_layer_input))  │")
        print(f"  │                                                                  │")

    # Layer scalar
    layer_scalar = getattr(layers, "layer_scalar", None)
    if layer_scalar is not None:
        print(f"  │  11. layer_scalar: {layer_scalar.item():.6f}                            │")
        print(f"  │      hidden_states = hidden_states * {layer_scalar.item():.6f}          │")

    print(f"  │                                                                  │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")


def inspect_mtp(model_path, assistant_path):
    """Inspect MTP assistant model structure."""
    print_section("4. MTP (MULTI-TOKEN PREDICTION) STRUCTURE")

    if not assistant_path:
        print("  (No assistant model path provided, printing expected structure)")
        print()
        print("""
  MTP Forward Flow:
  ═════════════════

  Target Model (main forward):
  ┌─────────────────────────────────────────────────────────┐
  │  Layer 0  → Layer 1 → ... → Layer 29 → LM Head         │
  │                                                         │
  │  At each layer, hidden_states are cached as             │
  │  "auxiliary hidden states" for MTP KV sharing           │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             │ backbone_hidden_states (from last layer)
                             │ + input token embedding (for next token)
                             ▼
  MTP Draft Model:
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  Step 1: Embed predicted token                          │
  │    input_embed = embed_tokens(predicted_token_id)       │
  │    input_embed = input_embed * sqrt(backbone_hidden)    │
  │                                                         │
  │  Step 2: Concatenate                                    │
  │    combined = cat([input_embed, backbone_hidden],dim=-1)│
  │    shape: [T, 2 * backbone_hidden_size]                 │
  │                                                         │
  │  Step 3: Pre-projection                                 │
  │    hidden = pre_projection(combined)                    │
  │    shape: [T, 2*backbone_hidden] → [T, draft_hidden]    │
  │                                                         │
  │  Step 4: Draft decoder layers (Q-only attention + MLP)  │
  │    ┌───────────────────────────────────────────────┐    │
  │    │  MTPDecoderLayer:                             │    │
  │    │    - input_layernorm                          │    │
  │    │    - self_attn (Q-only, KV from target cache) │    │
  │    │    - post_attention_layernorm                 │    │
  │    │    - pre_feedforward_layernorm                │    │
  │    │    - mlp (same arch as target MLP)            │    │
  │    │    - post_feedforward_layernorm               │    │
  │    │    - layer_scalar                             │    │
  │    │    NOTE: NO MoE in MTP layers                 │    │
  │    └───────────────────────────────────────────────┘    │
  │    (repeated for each MTP layer)                        │
  │                                                         │
  │  Step 5: Final norm                                     │
  │    draft_hidden = norm(hidden)                          │
  │                                                         │
  │  Step 6: Post-projection                                │
  │    backbone_hidden = post_projection(draft_hidden)      │
  │    shape: [T, draft_hidden] → [T, backbone_hidden]      │
  │                                                         │
  │  Step 7: LM Head                                        │
  │    logits = lm_head(draft_hidden)                       │
  │    → next predicted token                               │
  │                                                         │
  │  Step 8: Repeat steps 1-7 for k speculative tokens      │
  │    (feeding backbone_hidden back as input)              │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  KV Sharing between Target ↔ MTP Draft:
  ────────────────────────────────────────
  Target Layer (self_attn)     MTP Layer (Q-only attn)
  ┌─────────────────────┐     ┌──────────────────────────┐
  │  Q, K, V computed   │     │  Q computed              │
  │  K, V → KV cache    │────►│  K, V read from target   │
  │                     │     │  cache (no own KV)       │
  └─────────────────────┘     └──────────────────────────┘

  Draft layers map to target layers by attention type:
    - MTP sliding attn layer → last target sliding attn layer's KV
    - MTP full attn layer → last target full attn layer's KV

  Optional: Centroids Masking (for faster sampling)
  ──────────────────────────────────────────────────
  Instead of computing logits over full 262144 vocab:
    1. Compute hidden → centroid scores (2048 centroids)
    2. Select top-K centroids
    3. Only compute logits for tokens in selected centroids
    → Much faster greedy/top-p sampling for speculative tokens
""")
        return

    # If assistant path provided, load and inspect
    try:
        from transformers import AutoConfig
        ast_config = AutoConfig.from_pretrained(assistant_path, trust_remote_code=True)
        print(f"  Assistant model: {assistant_path}")
        print(f"  Config type: {ast_config.model_type}")
        for k in ["num_hidden_layers", "hidden_size", "num_attention_heads",
                   "num_key_value_heads", "head_dim", "intermediate_size"]:
            print(f"    {k:40s} = {getattr(ast_config, k, 'N/A')}")
    except Exception as e:
        print(f"  Failed to load assistant config: {e}")


def print_full_inference_flow(tc):
    """Print the complete inference flow diagram."""
    print_section("5. COMPLETE INFERENCE FLOW (single forward pass)")

    # Extract config values
    def get(key, default=0):
        if isinstance(tc, dict):
            return tc.get(key, default)
        return getattr(tc, key, default)

    num_layers = get("num_hidden_layers", 30)
    hidden = get("hidden_size", 2816)
    inter = get("intermediate_size", 2112)
    n_experts = get("num_experts", 128)
    top_k = get("top_k_experts", 8)
    moe_inter = get("moe_intermediate_size", 704)
    head_dim = get("head_dim", 256)
    global_head_dim = get("global_head_dim", 512)
    n_heads = get("num_attention_heads", 16)
    n_kv_heads = get("num_key_value_heads", 8)
    n_global_kv = get("num_global_key_value_heads", 2)
    sw = get("sliding_window", 1024)
    vocab = get("vocab_size", 262144)

    print(f"""
  ┌────────────────────────────────────────────────────────────────────────────┐
  │                    GEMMA4-26B-A4B INFERENCE FLOW                            │
  │                    (Single Token Generation Step)                           │
  └────────────────────────────────────────────────────────────────────────────┘

  INPUT: token_ids [batch, seq_len]
         positions [batch, seq_len]
    │
    ▼
  ╔══════════════════════════════════════════════════════════════════════════════╗
  ║  EMBEDDING                                                                 ║
  ║  embed_tokens: [{vocab}] → [{hidden}]                            ║
  ║  hidden_states = embed(token_ids) * sqrt({hidden})                      ║
  ╚══════════════════════════════════════════════════════════════════════════════╝
    │
    ▼
  ╔══════════════════════════════════════════════════════════════════════════════╗
  ║  DECODER LAYERS × {num_layers}                                                    ║
  ║                                                                            ║
  ║  Pattern: [S S S S S F] × 5  (S=sliding, F=full attention)                ║
  ║                                                                            ║
  ║ ┌────────────────────────────────────────────────────────────────────────┐ ║
  ║ │ SLIDING ATTENTION LAYER (layers 0-4, 6-10, 12-16, 18-22, 24-28)       │ ║
  ║ │ Total: 25 layers                                                       │ ║
  ║ │                                                                        │ ║
  ║ │   ┌─────────────────────────────────────────────────────────────┐      │ ║
  ║ │   │ Attention:                                                   │      │ ║
  ║ │   │   Q: [{hidden}] → [{n_heads} heads × {head_dim}d] = [{n_heads * head_dim}]         │      │ ║
  ║ │   │   K: [{hidden}] → [{n_kv_heads} heads × {head_dim}d] = [{n_kv_heads * head_dim}]          │      │ ║
  ║ │   │   V: [{hidden}] → [{n_kv_heads} heads × {head_dim}d] = [{n_kv_heads * head_dim}]          │      │ ║
  ║ │   │   Q_norm, K_norm (learned), V_norm (no weight)              │      │ ║
  ║ │   │   RoPE: theta=10000, full rotation                          │      │ ║
  ║ │   │   Window: {sw} tokens (local context only)              │      │ ║
  ║ │   │   GQA ratio: {n_heads}Q / {n_kv_heads}KV = {n_heads // n_kv_heads}:1                             │      │ ║
  ║ │   │   KV cache per layer: {n_kv_heads} × {head_dim} × seq_len              │      │ ║
  ║ │   │   Backend: TRITON_ATTN (forced)                             │      │ ║
  ║ │   └─────────────────────────────────────────────────────────────┘      │ ║
  ║ └────────────────────────────────────────────────────────────────────────┘ ║
  ║                                                                            ║
  ║ ┌────────────────────────────────────────────────────────────────────────┐ ║
  ║ │ FULL ATTENTION LAYER (layers 5, 11, 17, 23, 29)                        │ ║
  ║ │ Total: 5 layers                                                        │ ║
  ║ │                                                                        │ ║
  ║ │   ┌─────────────────────────────────────────────────────────────┐      │ ║
  ║ │   │ Attention:                                                   │      │ ║
  ║ │   │   Q: [{hidden}] → [{n_heads} heads × {global_head_dim}d] = [{n_heads * global_head_dim}]        │      │ ║
  ║ │   │   K: [{hidden}] → [{n_global_kv} heads × {global_head_dim}d] = [{n_global_kv * global_head_dim}]          │      │ ║
  ║ │   │   V: SHARED WITH K (attention_k_eq_v=True)                  │      │ ║
  ║ │   │   Q_norm, K_norm (learned), V_norm (no weight)              │      │ ║
  ║ │   │   RoPE: theta=1000000, partial_rotary_factor=0.25           │      │ ║
  ║ │   │         (only 25% of 512d = 128d gets RoPE, rest zero-pad)  │      │ ║
  ║ │   │   Window: FULL (attends to all previous tokens)             │      │ ║
  ║ │   │   GQA ratio: {n_heads}Q / {n_global_kv}KV = {n_heads // n_global_kv}:1                             │      │ ║
  ║ │   │   KV cache per layer: {n_global_kv} × {global_head_dim} × seq_len (K only, V=K) │      │ ║
  ║ │   │   Backend: TRITON_ATTN (forced, 512 > FA limit)             │      │ ║
  ║ │   └─────────────────────────────────────────────────────────────┘      │ ║
  ║ └────────────────────────────────────────────────────────────────────────┘ ║
  ║                                                                            ║
  ║ ┌────────────────────────────────────────────────────────────────────────┐ ║
  ║ │ FEEDFORWARD (every layer, after attention)                             │ ║
  ║ │                                                                        │ ║
  ║ │   ┌─ MLP (always runs) ──────────────────────────────────────┐         │ ║
  ║ │   │  gate_proj: [{hidden}] → [{inter}]                  │         │ ║
  ║ │   │  up_proj:   [{hidden}] → [{inter}]                  │         │ ║
  ║ │   │  down_proj: [{inter}] → [{hidden}]                  │         │ ║
  ║ │   │  activation: gelu_pytorch_tanh                           │         │ ║
  ║ │   │  output: gate(x) * up(x) → down → [{hidden}]          │         │ ║
  ║ │   └──────────────────────────────────────────────────────────┘         │ ║
  ║ │                              +                                         │ ║
  ║ │   ┌─ MoE (parallel path) ────────────────────────────────────┐         │ ║
  ║ │   │  Router:                                                  │         │ ║
  ║ │   │    1. RMSNorm(residual) (no learned weight)              │         │ ║
  ║ │   │    2. × root_scale ({hidden}^-0.5 = {hidden**-0.5:.6f})           │         │ ║
  ║ │   │    3. × per_dim_scale (learned [{hidden}])              │         │ ║
  ║ │   │    4. gate_linear → [{n_experts}] logits                     │         │ ║
  ║ │   │    5. softmax → top-{top_k} → renormalize                    │         │ ║
  ║ │   │                                                          │         │ ║
  ║ │   │  Experts (FusedMoE):                                     │         │ ║
  ║ │   │    {n_experts} experts × (gate+up+down)                          │         │ ║
  ║ │   │    gate: [{hidden}] → [{moe_inter}]                        │         │ ║
  ║ │   │    up:   [{hidden}] → [{moe_inter}]                        │         │ ║
  ║ │   │    down: [{moe_inter}] → [{hidden}]                        │         │ ║
  ║ │   │    × per_expert_scale (learned, folded into routing)     │         │ ║
  ║ │   │                                                          │         │ ║
  ║ │   │  Active per token: {top_k}/{n_experts} experts                       │         │ ║
  ║ │   │  Active params: {top_k}×3×{hidden}×{moe_inter} = {top_k*3*hidden*moe_inter/1e6:.1f}M          │         │ ║
  ║ │   └──────────────────────────────────────────────────────────┘         │ ║
  ║ │                                                                        │ ║
  ║ │   Final: norm(MLP_out) + norm(MoE_out) → post_norm → +residual        │ ║
  ║ │   × layer_scalar (per-layer learned)                                   │ ║
  ║ └────────────────────────────────────────────────────────────────────────┘ ║
  ║                                                                            ║
  ╚══════════════════════════════════════════════════════════════════════════════╝
    │
    ▼
  ╔══════════════════════════════════════════════════════════════════════════════╗
  ║  OUTPUT                                                                    ║
  ║  final_norm: RMSNorm                                                       ║
  ║  lm_head: [{hidden}] → [{vocab}] (tied to embedding)        ║
  ║  logit_softcap: 30.0 (tanh-based capping)                                 ║
  ╚══════════════════════════════════════════════════════════════════════════════╝
    │
    ▼
  logits [{vocab}] → sample → next token


  ═══════════════════════════════════════════════════════════════════════════════
  KV CACHE MEMORY BUDGET (per token, per layer):
  ═══════════════════════════════════════════════════════════════════════════════

  Sliding layers (25): {n_kv_heads} heads × {head_dim}d × 2(K+V) = {n_kv_heads * head_dim * 2} elements/token
    BUT: only stores last {sw} tokens (window)
    Max KV per sliding layer: {n_kv_heads}×{head_dim}×2×{sw} = {n_kv_heads*head_dim*2*sw/1024:.0f}K elements

  Full layers (5):     {n_global_kv} heads × {global_head_dim}d × 1(K only, V=K) = {n_global_kv * global_head_dim} elements/token
    Stores ALL tokens (no window limit)
    KV per full layer per token: {n_global_kv}×{global_head_dim}×1 = {n_global_kv*global_head_dim} elements

  Total KV per token (full context): 25×{n_kv_heads*head_dim*2} + 5×{n_global_kv*global_head_dim} = {25*n_kv_heads*head_dim*2 + 5*n_global_kv*global_head_dim} elements
  In FP8: {(25*n_kv_heads*head_dim*2 + 5*n_global_kv*global_head_dim) / 1024:.1f} KB/token
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--assistant", default=None)
    ap.add_argument("--config-only", action="store_true",
                    help="Only inspect config (no GPU needed)")
    args = ap.parse_args()

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # Get text_config (Gemma4 is multimodal, text params are nested)
    text_config = getattr(config, "text_config", config)
    if hasattr(text_config, "to_dict"):
        tc = text_config.to_dict()
    else:
        tc = vars(text_config) if hasattr(text_config, "__dict__") else text_config

    # Section 1: Config
    inspect_config(config)

    # Section 5: Full inference flow diagram
    print_full_inference_flow(tc)

    # Section 4: MTP structure
    inspect_mtp(args.model, args.assistant)

    if args.config_only:
        print("\n  [--config-only mode, skipping model loading]")
        return 0

    # Load actual model for detailed inspection
    print_section("LOADING MODEL (requires GPU)...")
    try:
        from vllm import LLM
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            max_model_len=4096,  # Small for inspection only
            gpu_memory_utilization=0.50,
            enforce_eager=True,
        )
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model

        # Section 2: Module tree
        inspect_model_structure(model)

        # Section 3: Single layer detail (one sliding, one full)
        inspect_single_layer(model, 0)   # First sliding layer
        inspect_single_layer(model, 5)   # First full attention layer

    except Exception as e:
        print(f"  Model loading failed: {e}")
        print(f"  (Use --config-only to skip model loading)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
