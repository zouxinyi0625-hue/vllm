#!/usr/bin/env python3
"""Gemma4 推理全流程 — 用一个具体 prompt 走一遍.

用 "Hello, world" 这个 3-token prompt 作为例子，一步一步展示
从输入到输出的完整数据流转，包括每一步的 tensor shape 变化。

Usage:
    python walkthrough_example.py
"""


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  GEMMA4-26B-A4B 推理全流程 — 具体例子                                         ║
║                                                                              ║
║  Prompt: "Hello, world"                                                      ║
║  目标: 生成下一个 token                                                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1: TOKENIZATION (在 CPU 上)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  输入文本: "Hello, world"

  Tokenizer 把文本切分成 token IDs:
    "Hello"  →  token_id = 17534
    ","      →  token_id = 235269
    " world" →  token_id = 3134

  结果:
    token_ids = [17534, 235269, 3134]     shape: [3]
    positions = [0, 1, 2]                  shape: [3]

  （实际上 chat template 会加上 <bos>、role tags 等，这里简化）


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2: EMBEDDING (GPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Embedding 表: 262144 个词 × 2816 维  (每个词一个向量)

  查表:
    embed[17534]  → [0.12, -0.03, 0.45, ..., 0.08]   ← "Hello" 的向量 (2816维)
    embed[235269] → [-0.05, 0.22, 0.11, ..., -0.31]  ← "," 的向量
    embed[3134]   → [0.33, 0.07, -0.19, ..., 0.14]   ← " world" 的向量

  缩放: hidden_states = embeddings × √2816 = embeddings × 53.05

  结果:
    hidden_states: shape [3, 2816]
    ┌─────────────────────────────────────────────────────┐
    │  token 0 ("Hello"):  [6.37, -1.59, 23.87, ..., 4.24]  │
    │  token 1 (","):      [-2.65, 11.67, 5.83, ..., -16.44] │
    │  token 2 (" world"): [17.51, 3.71, -10.08, ..., 7.43]  │
    └─────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3: 过 30 层 DECODER LAYER (这里详细展示第 0 层和第 5 层)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


╭─────────────────────────────────────────────────────────────────────────────╮
│  LAYER 0 (Sliding Attention)                                                │
│  输入: hidden_states [3, 2816]                                              │
╰─────────────────────────────────────────────────────────────────────────────╯

  ┌─ Step 3.1: Input LayerNorm ─────────────────────────────────────────────┐
  │                                                                         │
  │  residual = hidden_states                    (保存，后面要加回来)       │
  │  hidden_states = RMSNorm(hidden_states)      shape: [3, 2816] (不变)   │
  │                                                                         │
  │  RMSNorm 做什么？                                                       │
  │    对每个 token 的 2816 维向量做归一化:                                  │
  │    x_norm = x / sqrt(mean(x²) + eps) × weight                         │
  │    让每个 token 的向量长度标准化，方便后续计算                            │
  └─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─ Step 3.2: Sliding Attention ───────────────────────────────────────────┐
  │                                                                         │
  │  输入: hidden_states [3, 2816]                                          │
  │                                                                         │
  │  (a) QKV 投影 — 一次矩阵乘法产生 Q, K, V:                               │
  │                                                                         │
  │      qkv = hidden_states @ W_qkv                                       │
  │      W_qkv shape: [2816, 8192]  (Q=4096 + K=2048 + V=2048)            │
  │      qkv shape: [3, 8192]                                              │
  │                                                                         │
  │      拆分:                                                              │
  │        Q: [3, 16, 256]  → 3个token，每个有16个头，每头256维             │
  │        K: [3,  8, 256]  → 3个token，每个有8个KV头，每头256维            │
  │        V: [3,  8, 256]  → 同K                                          │
  │                                                                         │
  │  (b) Q/K/V 归一化:                                                      │
  │      Q = RMSNorm_Q(Q)    (每个头独立归一化，带学习权重)                  │
  │      K = RMSNorm_K(K)    (同上)                                         │
  │      V = RMSNorm_V(V)    (纯归一化，不带学习权重)                        │
  │                                                                         │
  │  (c) 加入位置信息 (RoPE):                                               │
  │                                                                         │
  │      token 0 ("Hello") 在 position=0:                                   │
  │        Q[0] 和 K[0] 被旋转 0 度（不动）                                 │
  │                                                                         │
  │      token 1 (",") 在 position=1:                                       │
  │        Q[1] 和 K[1] 被旋转一个小角度                                    │
  │        角度 = position / theta^(2i/256)                                 │
  │              = 1 / 10000^(2i/256) 弧度                                  │
  │        第0-1维: 旋转 0.0001 弧度 (高频，变化快)                          │
  │        第254-255维: 旋转 1.0 弧度 (低频，变化慢)                         │
  │                                                                         │
  │      token 2 (" world") 在 position=2:                                  │
  │        旋转角度是 token 1 的 2 倍                                        │
  │                                                                         │
  │      为什么旋转？这样 Q·K 的点积会自然包含两个 token 的                   │
  │      相对距离信息：Q[pos=2] · K[pos=0] 的值取决于 |2-0|=2               │
  │                                                                         │
  │  (d) KV Cache 存储:                                                     │
  │      这是第一次 (prefill)，直接存入 cache:                               │
  │        cache.K[layer=0] = K   shape: [3, 8, 256] (3个位置的K)           │
  │        cache.V[layer=0] = V   shape: [3, 8, 256]                        │
  │                                                                         │
  │      注意: sliding window=1024，这里只有3个token，远没到上限              │
  │      如果已有 >1024 token，最老的会被覆盖掉                              │
  │                                                                         │
  │  (e) 注意力计算 (3个token互相看):                                        │
  │                                                                         │
  │      以 Q head 0 为例 (使用 KV head 0，因为 GQA 2:1):                   │
  │                                                                         │
  │      scores = Q[head=0] @ K[kv_head=0].T / sqrt(256)                   │
  │             = [3, 256] @ [256, 3] / 16                                 │
  │             = [3, 3] 的注意力分数矩阵:                                  │
  │                                                                         │
  │         看→   "Hello"   ","    "world"                                  │
  │        "Hello"  [ 0.8    -inf    -inf  ]  ← 只能看自己(causal)          │
  │        ","      [ 0.3     0.7    -inf  ]  ← 能看 Hello 和自己           │
  │        "world"  [ 0.1     0.2     0.9  ]  ← 能看所有之前的              │
  │                                                                         │
  │      (-inf 是 causal mask: 不能看未来)                                   │
  │                                                                         │
  │      attention_weights = softmax(scores):                               │
  │        "Hello":  [1.0,   0.0,   0.0 ]                                  │
  │        ",":      [0.4,   0.6,   0.0 ]                                  │
  │        "world":  [0.15,  0.25,  0.60]                                  │
  │                                                                         │
  │      output = attention_weights @ V[kv_head=0]                          │
  │             = [3, 3] @ [3, 256] = [3, 256]                             │
  │                                                                         │
  │      "world" 的输出 = 0.15×V("Hello") + 0.25×V(",") + 0.60×V("world") │
  │      → 融合了之前所有 token 的信息，但最关注自己                          │
  │                                                                         │
  │      对所有 16 个 Q head 重复，得到 [3, 16, 256] = [3, 4096]            │
  │                                                                         │
  │  (f) 输出投影:                                                          │
  │      output = concat(all_heads) @ W_o                                   │
  │      W_o shape: [4096, 2816]                                            │
  │      output shape: [3, 2816]                                            │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─ Step 3.3: Post-Attention Norm + Residual ──────────────────────────────┐
  │                                                                         │
  │  hidden_states = RMSNorm(attention_output)     [3, 2816]                │
  │  hidden_states = hidden_states + residual      (加回原始输入)            │
  │                                                                         │
  │  为什么加 residual？ → "跳跃连接"，保证梯度流动，防止信息丢失            │
  └─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
  ┌─ Step 3.4: MLP + MoE (并行!) ──────────────────────────────────────────┐
  │                                                                         │
  │  residual = hidden_states      (再次保存)                               │
  │                                                                         │
  │  ═══ MLP 路径 (所有 token 走相同权重) ═══                                │
  │                                                                         │
  │  mlp_input = RMSNorm(hidden_states)    [3, 2816]                        │
  │                                                                         │
  │  gate = mlp_input @ W_gate     [3, 2816] × [2816, 2112] = [3, 2112]    │
  │  up   = mlp_input @ W_up       [3, 2816] × [2816, 2112] = [3, 2112]    │
  │  mlp_out = (gelu(gate) * up) @ W_down                                  │
  │          = [3, 2112] × [2112, 2816] = [3, 2816]                        │
  │                                                                         │
  │  ═══ MoE 路径 (每个 token 走不同 expert) ═══                             │
  │                                                                         │
  │  moe_input = residual          (注意! 用的是 attention 输出，不是 MLP 输出)│
  │                                                                         │
  │  ── Router 决策过程 ──                                                  │
  │                                                                         │
  │  以 token 2 ("world") 为例:                                             │
  │                                                                         │
  │    (i)  归一化: x = RMSNorm(moe_input[2])                 [2816]        │
  │    (ii) 缩放:  x = x × 0.01884 × per_dim_scale           [2816]        │
  │    (iii) 打分: logits = x @ W_gate_router                [128]          │
  │                                                                         │
  │    每个 expert 得到一个分数，代表"我有多适合处理这个 token":              │
  │                                                                         │
  │    logits = [                                                           │
  │      Expert 0: -0.3,    Expert 1: 0.1,    Expert 2: 1.8,  ← 高分!     │
  │      Expert 3: -0.5,    Expert 4: 0.7,    Expert 5: 2.1,  ← 最高!     │
  │      Expert 6: 0.2,     Expert 7: 1.5,    Expert 8: -0.1,             │
  │      ...                                                                │
  │      Expert 47: 1.9,    ← 高分                                         │
  │      ...                                                                │
  │      Expert 127: -0.8                                                   │
  │    ]                                                                    │
  │                                                                         │
  │    (iv) softmax → 概率分布:                                              │
  │      probs = softmax(logits)                                            │
  │      Expert 5: 0.18,  Expert 47: 0.15,  Expert 2: 0.13, ...            │
  │                                                                         │
  │    (v) 选 top-8:                                                        │
  │      选中: [5, 47, 2, 7, 99, 33, 61, 88]                               │
  │      权重: [0.18, 0.15, 0.13, 0.12, 0.11, 0.11, 0.10, 0.10]           │
  │      重归一化 (使和=1)                                                   │
  │                                                                         │
  │  ── Expert 计算 ──                                                      │
  │                                                                         │
  │  每个选中的 expert 独立处理 "world" 这个 token:                          │
  │                                                                         │
  │    Expert 5 处理 "world":                                               │
  │      gate5 = gelu(moe_normed_input @ Expert5.W_gate)  [2816]→[704]     │
  │      up5   = moe_normed_input @ Expert5.W_up          [2816]→[704]     │
  │      out5  = (gate5 * up5) @ Expert5.W_down           [704]→[2816]     │
  │                                                                         │
  │    Expert 47 处理 "world":                                              │
  │      (同样的计算，但用 Expert47 自己的权重)                              │
  │      out47 = ...                                       [2816]           │
  │                                                                         │
  │    ... (8个 expert 并行计算)                                            │
  │                                                                         │
  │  ── 加权合并 ──                                                         │
  │                                                                         │
  │    moe_output["world"] = 0.18 × out5 × scale5                          │
  │                        + 0.15 × out47 × scale47                         │
  │                        + 0.13 × out2 × scale2                           │
  │                        + ... (共8项)                                    │
  │                                                                         │
  │  对 "Hello" 和 "," 也做同样的路由，但它们可能选到不同的 expert!           │
  │                                                                         │
  │    "Hello" 选中: [12, 3, 67, 5, 22, 41, 8, 95]    ← 和 "world" 不同! │
  │    ","     选中: [0, 5, 12, 77, 43, 91, 2, 16]    ← 也不同!           │
  │                                                                         │
  │  ═══ 合并 MLP + MoE ═══                                                │
  │                                                                         │
  │  combined = RMSNorm(mlp_out) + RMSNorm(moe_out)    [3, 2816]           │
  │  combined = RMSNorm(combined)                                           │
  │  hidden_states = combined + residual                                    │
  │  hidden_states = hidden_states × layer_scalar  (比如 0.98)             │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  Layer 0 完成! hidden_states [3, 2816] 传给 Layer 1...


╭─────────────────────────────────────────────────────────────────────────────╮
│  LAYER 1, 2, 3, 4 (都是 Sliding Attention)                                  │
│  和 Layer 0 完全相同的结构，只是权重不同                                      │
│  每层的 attention 都只看最近 1024 个 token (对我们3个token来说就是全部)       │
╰─────────────────────────────────────────────────────────────────────────────╯


╭─────────────────────────────────────────────────────────────────────────────╮
│  LAYER 5 (Full Attention — 第一个全局注意力层)                                │
│  输入: hidden_states [3, 2816] (经过了5层sliding处理)                         │
╰─────────────────────────────────────────────────────────────────────────────╯

  和 Layer 0 的区别（只在 Attention 部分）:

  ┌─ Full Attention 的不同之处 ─────────────────────────────────────────────┐
  │                                                                         │
  │  (a) QKV 投影:                                                          │
  │      Q: [3, 16, 512]   ← 每头 512 维 (不是256!)                        │
  │      K: [3,  2, 512]   ← 只有 2 个 KV 头 (不是8!)                      │
  │      V: 不存在!         ← K = V，V 直接复用 K 的值                       │
  │                                                                         │
  │  (b) RoPE (位置编码):                                                   │
  │      512 维中只有前 128 维做旋转 (partial_rotary_factor=0.25)            │
  │      后 384 维保持不变 (cos=1, sin=0)                                    │
  │                                                                         │
  │      为什么？                                                            │
  │        前128维: 编码位置 → "这个token在哪里"                             │
  │        后384维: 纯内容特征 → "这个token是什么"                            │
  │        分开处理让模型同时记住 WHERE 和 WHAT                              │
  │                                                                         │
  │      theta = 1,000,000 (比 sliding 的 10,000 大100倍!)                  │
  │      → 旋转更慢 → 能区分更远的位置 (262144 token context)                │
  │                                                                         │
  │  (c) 注意力计算:                                                        │
  │      NO window mask! 每个token可以看到所有之前的token                    │
  │                                                                         │
  │      GQA 8:1: 16个Q头共享2个KV头                                        │
  │        Q head 0~7  全部和 KV head 0 计算                                │
  │        Q head 8~15 全部和 KV head 1 计算                                │
  │                                                                         │
  │      K=V: attention_weights @ V 实际上是 attention_weights @ K           │
  │        → 你"关注"到的信息 = 你的查询和键的匹配度 × 键本身                 │
  │                                                                         │
  │  (d) KV Cache:                                                          │
  │      只存 K (V=K 不需要额外存储)                                         │
  │      2 heads × 512 dim × seq_len                                        │
  │      没有窗口限制，随 seq_len 增长!                                      │
  │                                                                         │
  │  其他 (MLP, MoE, residual) 和 sliding layer 完全一样                    │
  └─────────────────────────────────────────────────────────────────────────┘


╭─────────────────────────────────────────────────────────────────────────────╮
│  Layer 6~29: 继续交替...                                                    │
│  [S S S S S F] × 重复                                                       │
│  每过5层sliding就有1层full attention "刷新全局信息"                           │
╰─────────────────────────────────────────────────────────────────────────────╯


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 4: 输出 (生成下一个 token)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  经过30层后: hidden_states [3, 2816]

  但我们只需要最后一个token的输出 (因为要预测 "world" 之后的下一个词):

  last_hidden = hidden_states[2]        shape: [2816] (只取 "world" 的)

  Step 4.1: Final LayerNorm
    last_hidden = RMSNorm(last_hidden)   [2816]

  Step 4.2: LM Head (预测下一个 token)
    logits = last_hidden @ W_lm_head     [2816] × [2816, 262144] = [262144]

    得到 262144 个分数，每个分数对应词表中一个词:
      logits[0]     ("") = -2.3
      logits[1]     ("the") = 0.5
      ...
      logits[235269] (",") = -0.8
      logits[235248] ("!") = 3.7       ← 高分！模型觉得下一个词可能是 "!"
      ...

  Step 4.3: Logit Softcapping
    logits = 30.0 × tanh(logits / 30.0)
    → 把极端值限制在 [-30, 30] 范围内，防止过度自信

  Step 4.4: Sampling
    probs = softmax(logits / temperature)
    next_token = sample(probs)          比如选到了 "!" (token_id=235248)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 5: DECODE (逐 token 生成，之后每步都走这个流程)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  现在要生成 "!" 之后的 token:

  输入: token_ids = [235248]  (只有新token "!")
        positions = [3]        (位置3)

  和 Prefill 的区别:
    - 只有 1 个 token (不是3个)
    - Attention 从 KV Cache 读之前的 K,V (不用重算)
    - 每层只算 1 个 Q，和 cache 里的 4 个 K (3旧+1新) 做 attention

  每层计算:
    Q = proj(embed("!"))                [1, 16, 256] (只有1个token的Q)
    K_new, V_new = proj(embed("!"))     [1, 8, 256]
    cache.append(K_new, V_new)          cache 现在有 4 个位置

    scores = Q @ cache.K.T              [1,16,256] × [4,8,256].T → [16, 1, 4]
    (token "!" 对 "Hello", ",", "world", "!" 各有一个注意力分数)

    weights = softmax(causal_mask(scores))
    output = weights @ cache.V          [16, 1, 256]

  → 比 Prefill 快很多! (只算1个token，但能"看到"之前所有的)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 6: MTP 投机解码 (如果开启 --spec-tokens 5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  目标: 不是一次生成 1 个 token，而是"猜" 5 个 token，然后验证

  ┌─ 正常 decode (无 MTP): ─────────────────────────────────────────────────┐
  │  Step 1: target 模型生成 token A     (跑完整30层)                        │
  │  Step 2: target 模型生成 token B     (跑完整30层)                        │
  │  Step 3: target 模型生成 token C     (跑完整30层)                        │
  │  总计: 3 × full forward pass                                            │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ MTP decode (k=5): ────────────────────────────────────────────────────┐
  │  Step 1: target 模型生成 token A     (跑完整30层)                        │
  │          同时保存 hidden_states 给 MTP draft 模型                        │
  │                                                                         │
  │  Step 2: MTP draft 猜接下来 5 个 token:                                 │
  │                                                                         │
  │    MTP轮次1:                                                            │
  │      input = [embed(A), hidden_states_from_target]  concat在一起        │
  │      → pre_projection → draft_layers → lm_head                          │
  │      → 猜出 token B'                                                    │
  │                                                                         │
  │    MTP轮次2:                                                            │
  │      input = [embed(B'), backbone_hidden_from轮次1]                     │
  │      → pre_projection → draft_layers → lm_head                          │
  │      → 猜出 token C'                                                    │
  │                                                                         │
  │    MTP轮次3~5: 继续猜 D', E', F'                                       │
  │                                                                         │
  │    Draft 模型很轻量 (没有MoE! 只有MLP)                                   │
  │    每轮只需要: pre_proj + 几层(Q-only attention + MLP) + lm_head         │
  │                                                                         │
  │  Step 3: target 模型一次性验证 [B', C', D', E', F']:                     │
  │      把 5 个猜测的 token 一起送入 target 模型                            │
  │      (就像 prefill 5 个 token 一样，可以并行计算!)                       │
  │                                                                         │
  │      验证: target 在每个位置的预测 == draft 的猜测?                       │
  │        位置1: target 也认为是 B' → ✓ 接受                               │
  │        位置2: target 也认为是 C' → ✓ 接受                               │
  │        位置3: target 说应该是 D (不是 D') → ✗ 拒绝!                     │
  │        → 接受 B', C'，生成正确的 D，丢弃 E', F'                         │
  │                                                                         │
  │  结果: 用 1次target完整forward + 5次轻量draft + 1次target验证            │
  │        生成了 3 个 token (A, B', C')                                     │
  │        而不是常规的 3次target完整forward                                  │
  │                                                                         │
  │  为什么快?                                                               │
  │    - Draft 模型没有 MoE (30层×128expert 的计算全省了!)                    │
  │    - Draft 只做 Q-only attention (KV 复用 target 的 cache)               │
  │    - 验证可以并行 (5个token一起过target，像prefill)                      │
  │    - 如果猜对率高 (accept rate ~60-70%)，等效吞吐翻倍                     │
  └─────────────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总结: 信息如何流动
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  "Hello, world" → 预测下一个 token:

  Token嵌入 [3,2816]
    → Layer 0-4 (sliding, 看1024窗口): 学习局部关系 ("Hello"和","的语法关系)
    → Layer 5 (full, 看所有): 建立全局理解 ("Hello, world"是完整问候)
    → Layer 6-10 (sliding): 在局部细化
    → Layer 11 (full): 再次全局整合
    → ... (重复)
    → Layer 29 (full): 最终全局理解
    → LM Head: 基于最终理解预测 → "!"

  每层里:
    Attention 决定 "关注哪些之前的 token"
    MLP 做 "通用变换" (所有token一样的权重)
    MoE 做 "专业变换" (不同token走不同expert，个性化处理)
""")


if __name__ == "__main__":
    main()
