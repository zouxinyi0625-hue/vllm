#!/usr/bin/env python3
"""Check if model outputs contain <think> tokens (reasoning mode leak).

Usage:
    python check_thinking.py --model Qwen/Qwen3.6-35B-A3B-FP8
    python check_thinking.py --model /path/to/local/model
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "datasets" / "sc1_delta_v2.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--prompts", type=int, default=5)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load a few prompts
    prompts_raw = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts_raw.append(d["prompt"])
            if len(prompts_raw) >= args.prompts:
                break

    # Render with enable_thinking=False
    prompts = []
    for p in prompts_raw:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        prompts.append(text)

    print(f"=== Rendered prompt[0] (last 200 chars) ===")
    print(prompts[0][-200:])
    print()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=24576,
        max_num_seqs=8,
        gpu_memory_utilization=0.95,
        seed=0,
    )

    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512,
        seed=1,
    )

    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    think_found = False
    for i, o in enumerate(outputs):
        for comp in o.outputs:
            text = comp.text
            has_think = "<think>" in text or "</think>" in text
            if has_think:
                think_found = True
            print(f"--- Output {i} (len={len(comp.token_ids)} tokens, has_think={has_think}) ---")
            print(text[:500])
            print()

    print("=" * 60)
    if think_found:
        print("WARNING: <think> tokens detected in output!")
        print("Reasoning is NOT fully disabled.")
    else:
        print("OK: No <think> tokens found. Reasoning appears disabled.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
