#!/usr/bin/env python3
"""Deep dive into Gemma4 Attention & MoE mechanics.

Explains with concrete examples:
  1. Sliding Window Attention — how the 1024-token window works
  2. Full Attention — global context with K=V, partial RoPE
  3. MoE Router — what decides which experts activate
  4. Token-level routing trace — see actual expert assignments

Usage:
    # Explanation only (no GPU)
    python inspect_attention_moe.py --explain

    # Live trace with actual model (needs GPU + model)
    python inspect_attention_moe.py --model /path/to/text_only --trace --num-tokens 8
"""
from __future__ import annotations
import argparse
import sys
import math


def explain_sliding_window():
    """Explain how sliding window attention works in Gemma4."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  SLIDING WINDOW ATTENTION (25 layers: 0-4, 6-10, 12-16, 18-22, 24-28)        ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  Config: window_size = 1024 tokens
          head_dim = 256
          Q heads = 16, KV heads = 8 (GQA 2:1)
          RoPE: theta=10000, full rotation (all 256 dims)

  ═══════════════════════════════════════════════════════════════════════════════
  HOW IT WORKS:
  ═══════════════════════════════════════════════════════════════════════════════

  Each token can ONLY attend to the most recent 1024 tokens (including itself).
  Tokens outside this window are invisible — their KV entries are evicted.

  Example: generating token at position 2500

    Position: 0    500   1000  1476  1500  2000  2476  2500
              |     |     |     |     |     |     |     |
              ├─────────────────┤ INVISIBLE (evicted from KV cache)
                                ├─────────────────────────┤ VISIBLE (window)
                                                          ↑
                                                     current token

    Attention mask (for token at pos=2500):
      pos < 1476:  MASKED (−∞)    ← outside window
      pos ≥ 1476:  ATTEND (causal) ← within window, causal mask still applies

  ═══════════════════════════════════════════════════════════════════════════════
  COMPUTATION FLOW (for one sliding attention layer):
  ═══════════════════════════════════════════════════════════════════════════════

  Input: hidden_states [T, 2816]  (T = num tokens in batch)

  Step 1: QKV Projection
  ┌────────────────────────────────────────────────────────────────────────┐
  │  QKV = qkv_proj(hidden_states)   shape: [T, 4096+2048+2048] = [T,8192]│
  │                                                                        │
  │  Split into:                                                           │
  │    Q: [T, 16, 256]   (16 query heads, 256 dim each)                   │
  │    K: [T,  8, 256]   (8 KV heads, 256 dim each)                       │
  │    V: [T,  8, 256]   (8 KV heads, 256 dim each)                       │
  └────────────────────────────────────────────────────────────────────────┘

  Step 2: Per-Head Normalization
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Q = Q_norm(Q)     RMSNorm with LEARNED weight [256] per head          │
  │  K = K_norm(K)     RMSNorm with LEARNED weight [256] per head          │
  │  V = V_norm(V)     RMSNorm WITHOUT learned weight (pure norm)          │
  │                                                                        │
  │  Why V_norm has no weight? → V just normalizes magnitude, no direction │
  │  adjustment needed. K_norm/Q_norm learn to adjust direction for RoPE.  │
  └────────────────────────────────────────────────────────────────────────┘

  Step 3: Rotary Position Embedding (RoPE)
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Q, K = apply_rotary_emb(Q, K, positions)                              │
  │                                                                        │
  │  For sliding attention:                                                │
  │    rope_theta = 10000                                                  │
  │    ALL 256 dimensions get RoPE rotation                                │
  │    (partial_rotary_factor = 1.0 implicitly)                            │
  │                                                                        │
  │  Frequency formula:                                                    │
  │    freq_i = 1 / (theta^(2i/head_dim))  for i = 0..127                 │
  │           = 1 / (10000^(2i/256))                                       │
  │                                                                        │
  │  This encodes POSITION information so the model knows token ordering   │
  │  within the 1024-token window.                                         │
  └────────────────────────────────────────────────────────────────────────┘

  Step 4: KV Cache Update
  ┌────────────────────────────────────────────────────────────────────────┐
  │  KV Cache stores the last 1024 K and V vectors for this layer.         │
  │                                                                        │
  │  On prefill (first forward):                                           │
  │    cache[layer].K = K[:1024]   (store first 1024 tokens)               │
  │    cache[layer].V = V[:1024]                                           │
  │                                                                        │
  │  On decode (token-by-token):                                           │
  │    cache[layer].K[pos % 1024] = K_new   (circular buffer overwrite)    │
  │    cache[layer].V[pos % 1024] = V_new                                  │
  │                                                                        │
  │  Memory per layer: 8 heads × 256 dim × 1024 window × 2(K+V)           │
  │                  = 4,194,304 elements = 4MB (FP8) / 8MB (BF16)         │
  │                                                                        │
  │  Total for 25 sliding layers: 100MB (FP8) / 200MB (BF16)              │
  │  This is FIXED regardless of sequence length!                          │
  └────────────────────────────────────────────────────────────────────────┘

  Step 5: Attention Computation (TRITON_ATTN backend)
  ┌────────────────────────────────────────────────────────────────────────┐
  │  For each query head q (0..15):                                        │
  │    kv_head = q // 2        (GQA: 2 query heads share 1 KV head)        │
  │                                                                        │
  │    scores = Q[q] @ K[kv_head].T / sqrt(256)                            │
  │                                                                        │
  │    Masking:                                                            │
  │      scores[pos_k > pos_q] = -inf        (causal: no future)           │
  │      scores[pos_k < pos_q - 1023] = -inf (window: no distant past)    │
  │                                                                        │
  │    attn_weights = softmax(scores)                                      │
  │    output[q] = attn_weights @ V[kv_head]                               │
  │                                                                        │
  │  Why TRITON_ATTN forced?                                               │
  │    FlashAttention kernel has head_dim ≤ 256 limit.                     │
  │    Sliding layers (256d) COULD use FlashAttn alone, but the model      │
  │    also has full attention layers with 512d heads. vLLM forces a        │
  │    single backend for the whole model → TRITON for compatibility.      │
  └────────────────────────────────────────────────────────────────────────┘

  Step 6: Output Projection
  ┌────────────────────────────────────────────────────────────────────────┐
  │  attn_output = concat(output[0..15])    shape: [T, 16×256] = [T, 4096]│
  │  output = o_proj(attn_output)           shape: [T, 4096] → [T, 2816]  │
  └────────────────────────────────────────────────────────────────────────┘

  ═══════════════════════════════════════════════════════════════════════════════
  WHY SLIDING WINDOW?
  ═══════════════════════════════════════════════════════════════════════════════

  1. Memory: O(1) KV cache per layer instead of O(seq_len)
     → Can handle 262144 token context without OOM

  2. Speed: Attention is O(window²) per token, not O(seq_len²)
     → 1024² = 1M ops vs 262144² = 68B ops (68000x savings!)

  3. Accuracy: Local context is usually sufficient for most tokens
     → Full attention layers (every 6th) handle long-range dependencies

  The design is: "cheap local attention × 25 + expensive global attention × 5"
""")


def explain_full_attention():
    """Explain how full attention works in Gemma4."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  FULL ATTENTION (5 layers: 5, 11, 17, 23, 29)                                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  Config: NO window limit (attends to ALL previous tokens)
          head_dim = 512 (2x larger than sliding!)
          Q heads = 16, KV heads = 2 (GQA 8:1, very aggressive)
          K = V (attention_k_eq_v=True, same tensor!)
          RoPE: theta=1000000, partial_rotary_factor=0.25

  ═══════════════════════════════════════════════════════════════════════════════
  KEY DIFFERENCES FROM SLIDING:
  ═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────┬──────────────────────┬──────────────────────┐
  │ Property                │ Sliding (25 layers)  │ Full (5 layers)      │
  ├─────────────────────────┼──────────────────────┼──────────────────────┤
  │ Head dimension          │ 256                  │ 512                  │
  │ Q heads                 │ 16                   │ 16                   │
  │ KV heads                │ 8                    │ 2                    │
  │ GQA ratio               │ 2:1                  │ 8:1                  │
  │ K=V sharing             │ No (separate K, V)   │ Yes (V reuses K)     │
  │ Context window          │ 1024 tokens          │ ALL tokens           │
  │ RoPE theta              │ 10,000               │ 1,000,000            │
  │ RoPE coverage           │ 100% of dims         │ 25% of dims (128/512)│
  │ KV per token per layer  │ 8×256×2 = 4096 elem  │ 2×512×1 = 1024 elem  │
  └─────────────────────────┴──────────────────────┴──────────────────────┘

  ═══════════════════════════════════════════════════════════════════════════════
  K = V SHARING (attention_k_eq_v):
  ═══════════════════════════════════════════════════════════════════════════════

  In full attention layers, the Value tensor IS the Key tensor:

    Normal attention:   output = softmax(Q @ K.T) @ V
    K=V attention:      output = softmax(Q @ K.T) @ K   ← V replaced by K!

  Why this works:
    - K already encodes "what information is at this position"
    - V normally encodes "what to retrieve" — but K can serve both roles
    - Saves 50% of KV cache memory for these layers
    - Saves weight parameters (no v_proj weights needed)

  In the checkpoint:
    - k_proj weights exist: [2816] → [2, 512] = [1024]
    - v_proj weights: ABSENT (loaded as copy of k_proj at init)

  KV cache stores ONLY K:
    cache[layer].K = K    shape: [2, 512, seq_len]
    cache[layer].V = K    (same pointer / same data)

  ═══════════════════════════════════════════════════════════════════════════════
  PARTIAL ROTARY (25% only):
  ═══════════════════════════════════════════════════════════════════════════════

  For 512-dim heads with partial_rotary_factor=0.25:
    - First 128 dims (25%): get RoPE rotation (position-encoded)
    - Last 384 dims (75%): NO rotation (position-independent features)

  ┌──────────────────────────────────────────────────────────────────┐
  │  Q/K vector [512 dims]:                                          │
  │                                                                  │
  │  [  RoPE-rotated (128d)  |    Zero-padded / Identity (384d)   ]  │
  │  ├──── position info ────┤├──── content features ─────────────┤  │
  │                                                                  │
  │  cos/sin applied here     cos=1, sin=0 here (no rotation)       │
  └──────────────────────────────────────────────────────────────────┘

  Why?
    - 128 dims is enough to encode position for 262144-length context
    - Remaining 384 dims are PURE content features (position-agnostic)
    - This separates "where" (position) from "what" (content)
    - theta=1,000,000 (vs 10,000 for sliding) → much longer wavelengths
      for encoding far-apart positions distinctly

  ═══════════════════════════════════════════════════════════════════════════════
  GQA 8:1 — Very Aggressive Grouping:
  ═══════════════════════════════════════════════════════════════════════════════

  16 query heads share only 2 KV heads:
    Q heads  0, 1, 2, 3, 4, 5, 6, 7 → all use KV head 0
    Q heads  8, 9,10,11,12,13,14,15 → all use KV head 1

  This means:
    - Very compact KV cache (only 2 × 512 = 1024 elements per token)
    - Each KV head serves 8 query heads (must be very informative)
    - Compensated by large head_dim (512) — more capacity per head

  ═══════════════════════════════════════════════════════════════════════════════
  WHY THIS DESIGN?
  ═══════════════════════════════════════════════════════════════════════════════

  Full attention layers are EXPENSIVE (attend to all tokens), so they are:
    1. Rare: only 5/30 layers
    2. Memory-efficient: K=V + 8:1 GQA → tiny KV cache per layer
    3. High-capacity: 512d heads hold more information per head
    4. Long-range: theta=1M + full context → captures global dependencies

  The model's strategy:
    - Local patterns (syntax, adjacent tokens) → sliding attention (cheap, frequent)
    - Global patterns (topic, long-range ref) → full attention (expensive, rare)
""")


def explain_moe():
    """Explain MoE routing in Gemma4."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  MIXTURE OF EXPERTS (MoE) — ROUTING & ACTIVATION                              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  Config: 128 experts, top-8 active per token
          Expert FFN: [2816] → [704] → [2816] (gate + up + down)
          Runs in PARALLEL with MLP (not after!)

  ═══════════════════════════════════════════════════════════════════════════════
  WHAT DECIDES WHICH EXPERTS ACTIVATE?
  ═══════════════════════════════════════════════════════════════════════════════

  The ROUTER decides. It's a learned linear classifier that looks at the
  hidden state and assigns each token to its best 8 experts out of 128.

  ┌────────────────────────────────────────────────────────────────────────────┐
  │  Router Input: residual hidden_states [T, 2816]                            │
  │  (NOTE: this is the PRE-MLP residual, not the MLP output!)                 │
  │                                                                            │
  │  Step 1: Normalize                                                         │
  │    x = RMSNorm(hidden_states)      (no learned weight, pure normalization) │
  │                                                                            │
  │  Step 2: Scale                                                             │
  │    x = x * (2816^-0.5)            (root-size scaling = 0.01884)            │
  │    x = x * per_dim_scale           (learned [2816] vector)                 │
  │                                                                            │
  │  Step 3: Route                                                             │
  │    logits = gate_linear(x)          shape: [T, 2816] → [T, 128]           │
  │                                                                            │
  │  Step 4: Expert Selection                                                  │
  │    probs = softmax(logits)          shape: [T, 128] (probability dist)     │
  │    top8_indices = top-k(probs, k=8) shape: [T, 8] (expert IDs)            │
  │    top8_weights = probs[top8_indices]                                      │
  │    top8_weights = top8_weights / sum(top8_weights)  (renormalize)          │
  │                                                                            │
  │  Step 5: Apply per-expert scale                                            │
  │    for each selected expert e:                                             │
  │      top8_weights[e] *= per_expert_scale[e]  (learned scalar per expert)   │
  │                                                                            │
  └────────────────────────────────────────────────────────────────────────────┘

  ═══════════════════════════════════════════════════════════════════════════════
  WHAT DETERMINES EXPERT ASSIGNMENT?
  ═══════════════════════════════════════════════════════════════════════════════

  1. TOKEN CONTENT (hidden_states): The primary signal.
     Different tokens produce different hidden_states, which route to
     different experts. Similar tokens tend to route similarly.

  2. LEARNED gate_linear WEIGHTS: A [2816 × 128] matrix.
     Each column represents one expert's "specialty". The dot product
     between a token's hidden_state and an expert's column determines
     the affinity. High dot product → expert is relevant for this token.

  3. LEARNED per_dim_scale: Amplifies/attenuates specific hidden dims
     before routing. This lets the router focus on specific features.

  4. LEARNED per_expert_scale: After selection, scales each expert's
     contribution. Some experts have higher default influence.

  ═══════════════════════════════════════════════════════════════════════════════
  CONCRETE EXAMPLE:
  ═══════════════════════════════════════════════════════════════════════════════

  Token "Python" at layer 12:
    hidden_state = [0.3, -0.1, 0.7, ...]  (2816-dim vector)
    ↓ normalize + scale
    x = [0.005, -0.002, 0.012, ...]
    ↓ gate_linear
    logits = [0.1, -0.5, 0.3, ..., 2.1, ..., -0.2]  (128 expert scores)
    ↓ softmax
    probs = [0.01, 0.005, 0.012, ..., 0.15, ..., 0.007]
    ↓ top-8
    selected = [Expert#47: 0.15, Expert#3: 0.12, Expert#99: 0.11, ...]
    ↓ renormalize
    weights = [Expert#47: 0.22, Expert#3: 0.18, Expert#99: 0.16, ...]

  Each selected expert processes the token independently:
    expert_output_47 = Expert47.down(gelu(Expert47.gate(x)) * Expert47.up(x))
    expert_output_3  = Expert3.down(gelu(Expert3.gate(x)) * Expert3.up(x))
    ...

  Final MoE output = Σ (weight_e × expert_output_e × per_expert_scale_e)

  ═══════════════════════════════════════════════════════════════════════════════
  MOE + MLP PARALLEL EXECUTION:
  ═══════════════════════════════════════════════════════════════════════════════

  IMPORTANT: MoE and MLP run in PARALLEL, not sequentially!

                        residual (from attention output)
                              │
                    ┌─────────┼─────────┐
                    │                    │
                    ▼                    ▼
            ┌──────────────┐    ┌──────────────────────────┐
            │     MLP      │    │     Router → Experts     │
            │  (all tokens)│    │  (token-specific routing) │
            └──────┬───────┘    └────────────┬─────────────┘
                   │                         │
                   ▼                         ▼
            post_norm_1(mlp)         post_norm_2(moe)
                   │                         │
                   └─────────┬───────────────┘
                             │
                             ▼
                      mlp_out + moe_out
                             │
                             ▼
                      post_feedforward_norm
                             │
                             ▼
                        + residual
                             │
                             ▼
                    × layer_scalar

  Why parallel (not serial)?
    - MLP provides a DENSE baseline transformation (every token same weights)
    - MoE provides SPARSE specialized processing (different experts per token)
    - Combined: guaranteed baseline quality + specialized enhancement
    - If MoE were serial after MLP, it would lose the original residual signal

  ═══════════════════════════════════════════════════════════════════════════════
  EXPERT SPECIALIZATION — WHAT DO EXPERTS LEARN?
  ═══════════════════════════════════════════════════════════════════════════════

  Research shows MoE experts tend to specialize by:
    - Language/script (Expert#12 → Chinese, Expert#47 → code)
    - Token type (Expert#3 → punctuation, Expert#89 → numbers)
    - Semantic domain (Expert#55 → medical, Expert#72 → legal)
    - Syntactic role (Expert#8 → verbs, Expert#34 → adjectives)

  The 128-expert × top-8 design means:
    - Each token gets a unique COMBINATION of 8 specialists
    - C(128, 8) = ~4 × 10^11 possible combinations
    - Effectively infinite specialization capacity

  ═══════════════════════════════════════════════════════════════════════════════
  FusedMoE KERNEL — HOW vLLM EXECUTES THIS EFFICIENTLY:
  ═══════════════════════════════════════════════════════════════════════════════

  Naive implementation: loop over 128 experts, process assigned tokens each
  Problem: terrible GPU utilization (most experts process few tokens)

  FusedMoE (vLLM optimized):
    1. Sort all tokens by assigned expert (permute)
    2. Batch-process same-expert tokens together (GEMM fusion)
    3. Scatter results back to original token order (unpermute)

  On A100 with FP8:
    - Uses Marlin FP8 MoE kernel (FlashInfer FP8 needs sm_90+)
    - Routing uses custom Triton kernel (vectorized softmax + topk)
    - Expert GEMM batched across tokens for maximum GPU occupancy

  ═══════════════════════════════════════════════════════════════════════════════
  PARAMETER BUDGET:
  ═══════════════════════════════════════════════════════════════════════════════

  Per layer:
    MLP:   3 × 2816 × 2112 = 17.8M params (dense, always active)
    MoE:   128 × 3 × 2816 × 704 = 762M params (sparse, 8/128 active)
    Router: 2816 × 128 + 2816 + 128 = 363K params (always active, tiny)

  Total model MoE params: 30 layers × 762M = 22.9B (majority of 26B total!)
  Active MoE per token: 30 × 8 × 3 × 2816 × 704 = 1.43B params

  This is why it's called "26B-A4B":
    - 26B total parameters
    - ~4B active parameters per token (MLP + active experts + attention)
""")


def trace_routing(model_path, num_tokens):
    """Load model and trace actual expert routing for sample tokens."""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  LIVE ROUTING TRACE (model: {model_path})
╚══════════════════════════════════════════════════════════════════════════════════╝
""")

    try:
        import torch
        from transformers import AutoTokenizer, AutoConfig
        from vllm import LLM, SamplingParams

        print("  Loading model for routing trace...")
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            max_num_seqs=1,
        )

        # Get underlying model
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model

        # Find a router and hook into it
        hooks = []
        router_outputs = {}

        def make_router_hook(layer_idx):
            def hook_fn(module, input, output):
                router_outputs[layer_idx] = output.detach().cpu()
            return hook_fn

        # Register hooks on router gate_linear
        for name, module in model.named_modules():
            if "router" in name and "gate" in name and hasattr(module, "weight"):
                parts = name.split(".")
                if "layers" in parts:
                    idx = int(parts[parts.index("layers") + 1])
                    # Hook on the router module (parent of gate)
                    parent_name = ".".join(parts[:-1])
                    for n, m in model.named_modules():
                        if n == parent_name:
                            hooks.append(m.register_forward_hook(make_router_hook(idx)))
                            break

        if not hooks:
            print("  Could not find router modules to hook. Trying alternative...")
            # Try hooking moe modules instead
            for name, module in model.named_modules():
                if name.endswith(".router") and hasattr(module, "gate"):
                    parts = name.split(".")
                    if "layers" in parts:
                        idx = int(parts[parts.index("layers") + 1])
                        hooks.append(module.register_forward_hook(make_router_hook(idx)))

        print(f"  Registered {len(hooks)} router hooks")

        # Generate a few tokens to observe routing
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt = "The capital of France is"
        sampling = SamplingParams(temperature=0, max_tokens=num_tokens)

        print(f"  Prompt: '{prompt}'")
        print(f"  Generating {num_tokens} tokens...\n")
        outputs = llm.generate([prompt], sampling)

        generated = outputs[0].outputs[0].text
        print(f"  Generated: '{generated}'")
        print()

        # Print routing results
        if router_outputs:
            print(f"  ┌────────────────────────────────────────────────────────────┐")
            print(f"  │  Expert Routing (last forward pass, layer → top-8 experts) │")
            print(f"  ├────────────────────────────────────────────────────────────┤")
            for layer_idx in sorted(router_outputs.keys())[:5]:  # First 5 layers
                logits = router_outputs[layer_idx]
                if logits.dim() == 2:
                    # Take last token's routing
                    token_logits = logits[-1]
                    probs = torch.softmax(token_logits, dim=-1)
                    top_vals, top_ids = torch.topk(probs, k=8)
                    experts_str = ", ".join([f"E{i.item()}({v.item():.3f})" for i, v in zip(top_ids, top_vals)])
                    print(f"  │  Layer {layer_idx:2d}: {experts_str}")
            print(f"  └────────────────────────────────────────────────────────────┘")
        else:
            print("  (No routing data captured — hooks may not have triggered)")

        # Cleanup
        for h in hooks:
            h.remove()

    except Exception as e:
        print(f"  Trace failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Model path for live trace")
    ap.add_argument("--explain", action="store_true", help="Print explanations only")
    ap.add_argument("--trace", action="store_true", help="Run live routing trace")
    ap.add_argument("--num-tokens", type=int, default=8, help="Tokens to generate for trace")
    args = ap.parse_args()

    if not args.explain and not args.trace:
        args.explain = True  # Default to explain mode

    if args.explain:
        explain_sliding_window()
        explain_full_attention()
        explain_moe()

    if args.trace:
        if not args.model:
            print("ERROR: --model required for --trace mode")
            return 1
        trace_routing(args.model, args.num_tokens)

    return 0


if __name__ == "__main__":
    sys.exit(main())
