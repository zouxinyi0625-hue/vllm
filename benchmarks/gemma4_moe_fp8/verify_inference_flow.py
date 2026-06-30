#!/usr/bin/env python3
"""Gemma4 推理全流程验证 — 用 hook 捕获真实 tensor.

加载实际模型，用 "Hello, world" 跑一次 forward，
hook 每一层打印真实的 shape、数值、routing 决策。

Usage:
    python verify_inference_flow.py --model /path/to/text_only
    python verify_inference_flow.py --model google/gemma-4-26B-A4B-it
"""
from __future__ import annotations
import argparse
import sys
import torch
import json
from collections import OrderedDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model path")
    ap.add_argument("--prompt", default="Hello, world", help="Test prompt")
    ap.add_argument("--max-tokens", type=int, default=1, help="Tokens to generate")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("=" * 80)
    print(f"  Loading model: {args.model}")
    print(f"  Prompt: '{args.prompt}'")
    print("=" * 80)

    # Load tokenizer to show tokenization
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_ids = tok.encode(args.prompt, add_special_tokens=False)
    tokens = [tok.decode([tid]) for tid in token_ids]
    print(f"\n[TOKENIZATION]")
    print(f"  Input: '{args.prompt}'")
    print(f"  Token IDs: {token_ids}")
    print(f"  Tokens: {tokens}")
    print(f"  Num tokens: {len(token_ids)}")

    # Load model
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        max_num_seqs=1,
    )

    # Access internal model
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        # vLLM v1 path
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # =========================================================================
    # Register hooks to capture intermediate tensors
    # =========================================================================
    captured = OrderedDict()

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                captured[name] = {
                    "shape": list(out.shape),
                    "dtype": str(out.dtype),
                    "mean": out.float().mean().item(),
                    "std": out.float().std().item(),
                    "min": out.float().min().item(),
                    "max": out.float().max().item(),
                }
            elif out is not None:
                captured[name] = {"type": str(type(out))}
        return hook_fn

    hooks = []

    # Hook embedding
    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and "per_layer" not in name:
            hooks.append(module.register_forward_hook(make_hook("embedding")))
            break

    # Hook each layer's key sublayers
    for name, module in model.named_modules():
        # Attention output
        if ".self_attn.attn" in name and name.endswith(".attn"):
            parts = name.split(".")
            if "layers" in parts:
                idx = int(parts[parts.index("layers") + 1])
                hooks.append(module.register_forward_hook(make_hook(f"layer{idx}.attention")))

        # MLP output
        if name.endswith(".mlp.down_proj"):
            parts = name.split(".")
            if "layers" in parts:
                idx = int(parts[parts.index("layers") + 1])
                hooks.append(module.register_forward_hook(make_hook(f"layer{idx}.mlp")))

        # Router (capture routing logits)
        if name.endswith(".router"):
            parts = name.split(".")
            if "layers" in parts:
                idx = int(parts[parts.index("layers") + 1])

                def make_router_hook(layer_idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, torch.Tensor):
                            logits = output.detach().float()
                            probs = torch.softmax(logits, dim=-1)
                            # Per token: top-8 experts
                            top_vals, top_ids = torch.topk(probs, k=min(8, probs.shape[-1]), dim=-1)
                            captured[f"layer{layer_idx}.router"] = {
                                "logits_shape": list(logits.shape),
                                "num_experts": logits.shape[-1],
                                "last_token_top8_experts": top_ids[-1].tolist(),
                                "last_token_top8_probs": [f"{v:.4f}" for v in top_vals[-1].tolist()],
                                "expert_load_std": probs.mean(dim=0).std().item(),
                            }
                    return hook_fn

                hooks.append(module.register_forward_hook(make_router_hook(idx)))

        # MoE output
        if name.endswith(".moe"):
            parts = name.split(".")
            if "layers" in parts:
                idx = int(parts[parts.index("layers") + 1])
                hooks.append(module.register_forward_hook(make_hook(f"layer{idx}.moe")))

    # Final norm
    for name, module in model.named_modules():
        if name.endswith(".norm") and "layer" not in name and "feed" not in name and "attention" not in name:
            if "model.norm" in name or name == "model.model.norm":
                hooks.append(module.register_forward_hook(make_hook("final_norm")))
                break

    print(f"\n[HOOKS] Registered {len(hooks)} hooks")

    # =========================================================================
    # Run inference
    # =========================================================================
    print(f"\n[INFERENCE] Running forward pass...")
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    outputs = llm.generate([args.prompt], sampling)

    generated_text = outputs[0].outputs[0].text
    generated_ids = outputs[0].outputs[0].token_ids
    print(f"\n[OUTPUT]")
    print(f"  Generated text: '{generated_text}'")
    print(f"  Generated token IDs: {list(generated_ids)}")
    print(f"  Generated tokens: {[tok.decode([tid]) for tid in generated_ids]}")

    # =========================================================================
    # Print captured data
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"  CAPTURED INTERMEDIATE TENSORS")
    print(f"{'=' * 80}\n")

    for name, info in captured.items():
        print(f"  [{name}]")
        for k, v in info.items():
            if isinstance(v, float):
                print(f"    {k:20s} = {v:.6f}")
            else:
                print(f"    {k:20s} = {v}")
        print()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: DATA FLOW VERIFICATION")
    print(f"{'=' * 80}\n")

    # Check which layers have attention, mlp, router, moe
    attn_layers = sorted([int(k.split(".")[0].replace("layer", ""))
                          for k in captured if ".attention" in k])
    mlp_layers = sorted([int(k.split(".")[0].replace("layer", ""))
                         for k in captured if ".mlp" in k])
    router_layers = sorted([int(k.split(".")[0].replace("layer", ""))
                            for k in captured if ".router" in k])
    moe_layers = sorted([int(k.split(".")[0].replace("layer", ""))
                         for k in captured if ".moe" in k])

    print(f"  Layers with attention hook fired: {attn_layers}")
    print(f"  Layers with MLP hook fired:       {mlp_layers}")
    print(f"  Layers with Router hook fired:    {router_layers}")
    print(f"  Layers with MoE hook fired:       {moe_layers}")
    print()

    # Verify layer types
    if router_layers:
        sliding_with_moe = [l for l in router_layers if l not in [5, 11, 17, 23, 29]]
        full_with_moe = [l for l in router_layers if l in [5, 11, 17, 23, 29]]
        print(f"  Sliding layers with MoE: {sliding_with_moe}")
        print(f"  Full attention layers with MoE: {full_with_moe}")
        print(f"  → MoE runs in ALL layers (both sliding and full): {'YES' if len(router_layers) == 30 else 'NO'}")
    print()

    # Show routing diversity
    print(f"  [ROUTING DIVERSITY — do different layers pick different experts?]")
    for layer_idx in router_layers[:6]:  # First 6 layers
        key = f"layer{layer_idx}.router"
        if key in captured:
            info = captured[key]
            experts = info.get("last_token_top8_experts", [])
            probs = info.get("last_token_top8_probs", [])
            print(f"    Layer {layer_idx:2d}: experts={experts}  probs={probs}")
    print()

    # Cleanup hooks
    for h in hooks:
        h.remove()

    print("  DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
