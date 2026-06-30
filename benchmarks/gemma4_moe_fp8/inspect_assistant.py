#!/usr/bin/env python3
"""Inspect Gemma4 MTP assistant (draft) model structure.

Usage:
    python inspect_assistant.py
    python inspect_assistant.py --model google/gemma-4-26B-A4B-it-assistant
    python inspect_assistant.py --model /path/to/local/assistant
"""
import argparse
import sys
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it-assistant")
    args = ap.parse_args()

    from transformers import AutoConfig

    print("=" * 80)
    print(f"  Gemma4 MTP Assistant Model Inspector")
    print(f"  Model: {args.model}")
    print("=" * 80)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # Print all config fields
    print(f"\n[CONFIG] model_type = {getattr(config, 'model_type', 'N/A')}")
    print(f"[CONFIG] architectures = {getattr(config, 'architectures', 'N/A')}")
    print()

    config_dict = config.to_dict()
    print("[ALL CONFIG FIELDS]")
    for k, v in sorted(config_dict.items()):
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        elif isinstance(v, list) and len(v) > 20:
            print(f"  {k}: [{v[0]}, {v[1]}, ... ({len(v)} items)]")
        else:
            print(f"  {k}: {v}")

    # Compare with target
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON: Target vs Draft")
    print(f"{'=' * 80}")

    compare_fields = [
        "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "vocab_size", "num_experts", "top_k_experts", "moe_intermediate_size",
    ]

    print(f"\n  {'Field':40s} {'Draft':>15s}  {'Target (26B)':>15s}")
    print(f"  {'-'*40} {'-'*15}  {'-'*15}")

    # Target known values
    target_vals = {
        "num_hidden_layers": 30,
        "hidden_size": 2816,
        "intermediate_size": 2112,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "vocab_size": 262144,
        "num_experts": 128,
        "top_k_experts": 8,
        "moe_intermediate_size": 704,
    }

    for f in compare_fields:
        draft_val = config_dict.get(f, "N/A")
        # Check nested configs
        if draft_val == "N/A":
            for nested_key in ["text_config", "backbone_config"]:
                nested = config_dict.get(nested_key, {})
                if isinstance(nested, dict) and f in nested:
                    draft_val = nested[f]
                    break
        target_val = target_vals.get(f, "N/A")
        marker = "  ←" if str(draft_val) != str(target_val) and draft_val != "N/A" else ""
        print(f"  {f:40s} {str(draft_val):>15s}  {str(target_val):>15s}{marker}")

    # Count total params estimate
    print(f"\n{'=' * 80}")
    print(f"  PARAMETER ESTIMATE")
    print(f"{'=' * 80}")

    n_layers = config_dict.get("num_hidden_layers", None)
    hidden = config_dict.get("hidden_size", None)
    inter = config_dict.get("intermediate_size", None)
    n_heads = config_dict.get("num_attention_heads", None)
    n_kv = config_dict.get("num_key_value_heads", None)
    hd = config_dict.get("head_dim", None)

    # Check nested
    for nested_key in ["text_config", "backbone_config"]:
        nested = config_dict.get(nested_key, {})
        if isinstance(nested, dict):
            n_layers = n_layers or nested.get("num_hidden_layers")
            hidden = hidden or nested.get("hidden_size")
            inter = inter or nested.get("intermediate_size")
            n_heads = n_heads or nested.get("num_attention_heads")
            n_kv = n_kv or nested.get("num_key_value_heads")
            hd = hd or nested.get("head_dim")

    if all(v is not None for v in [n_layers, hidden, inter, n_heads, hd]):
        # Q only attention (no K, V proj in draft)
        q_proj = hidden * n_heads * hd if hd else 0
        o_proj = n_heads * hd * hidden if hd else 0
        attn_per_layer = q_proj + o_proj  # Q + O only (K,V from target)

        # MLP
        mlp_per_layer = 3 * hidden * inter if inter else 0

        # Norms etc (~small)
        norms = hidden * 6  # rough estimate

        per_layer = attn_per_layer + mlp_per_layer + norms
        total_layers = per_layer * n_layers

        # Embedding + pre/post projection
        backbone_hidden = 2816  # target's hidden size
        pre_proj = 2 * backbone_hidden * hidden  # concat(embed, backbone) → draft
        post_proj = hidden * backbone_hidden  # draft → backbone
        vocab = 262144
        embed = vocab * backbone_hidden  # shared with target usually

        total = total_layers + pre_proj + post_proj
        total_with_embed = total + embed

        print(f"\n  Draft model layers: {n_layers}")
        print(f"  Draft hidden_size: {hidden}")
        print(f"  Draft intermediate_size: {inter}")
        print(f"  Draft head_dim: {hd}")
        print()
        print(f"  Per layer:")
        print(f"    Q-only attention (Q+O): {attn_per_layer/1e6:.1f}M params")
        print(f"    MLP (gate+up+down):     {mlp_per_layer/1e6:.1f}M params")
        print(f"    Total per layer:        {per_layer/1e6:.1f}M params")
        print()
        print(f"  Projections:")
        print(f"    pre_projection:         {pre_proj/1e6:.1f}M params")
        print(f"    post_projection:        {post_proj/1e6:.1f}M params")
        print()
        print(f"  Total (excl embedding):   {total/1e6:.1f}M params")
        print(f"  Total (incl embedding):   {total_with_embed/1e6:.1f}M params")
        print()
        print(f"  vs Target 26B: {total/26e9*100:.2f}% of target params (excl embed)")
    else:
        print(f"  Could not estimate params (missing config fields)")

    # Try to load and list modules
    print(f"\n{'=' * 80}")
    print(f"  MODULE STRUCTURE (loading on CPU)")
    print(f"{'=' * 80}")
    try:
        from transformers import AutoModel
        import torch
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Actual total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print()

        # Print module tree (depth ≤ 3)
        print("  [Module tree]")
        for name, module in model.named_modules():
            depth = name.count(".")
            if depth <= 3:
                direct_params = sum(p.numel() for p in module.parameters(recurse=False))
                param_info = f" [{direct_params/1e6:.2f}M]" if direct_params > 100000 else f" [{direct_params}]" if direct_params > 0 else ""
                indent = "  " * (depth + 1)
                print(f"  {indent}{name}: {module.__class__.__name__}{param_info}")

        # Print all parameter shapes
        print(f"\n  [All parameter shapes]")
        for name, p in model.named_parameters():
            print(f"    {name}: {list(p.shape)} ({p.numel()/1e6:.2f}M)")

    except Exception as e:
        print(f"  Failed to load model: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"  DONE")
    print(f"{'=' * 80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
