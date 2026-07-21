#!/usr/bin/env python3
"""Convert DSpark MAI Profile eval datasets → benchmark sc1-style {"prompt": ...} jsonl.

WHY
---
We want to measure the Google MTP assistant's acceptance rate on the SAME MAI
Profile eval prompts that DSpark trained/evaluated on, broken down PER LAYER.

The vLLM benchmark (`vllm bench serve --dataset-name custom`) only reads the
`prompt` field of each jsonl line and mixes all prompts into one aggregate run —
it cannot group by layer. So to get a PER-LAYER acceptance rate you must run the
benchmark once per layer, on a per-layer prompt file.

This script therefore emits, for each source eval file:
  - one per-layer file   sc1_maiprofile_<layer>.jsonl   (run the bench on each)
  - one combined file     sc1_maiprofile_all.jsonl        (mixed, aggregate only)

SOURCE FORMAT (DSpark, from prepare_maiprofile_splits.py:156-165)
-----------------------------------------------------------------
Each line of eval_datasets/maiprofile_<layer>.jsonl:
  {
    "id": "layer3_seasonality:<hash>",
    "source_layer": "layer3_seasonality",
    "user_id": ..., "prompt_hash": ...,
    "messages": [{"role":"system",...},{"role":"user",...}],
    "turns": ["<system+user folded into one prompt string>"]
  }
`turns[0]` is already the full folded prompt the DSpark evaluator feeds the
model, so it is the correct thing to benchmark. We fall back to folding
`messages` ourselves if `turns` is absent.

TARGET FORMAT (benchmark sc1_delta_v2.jsonl)
--------------------------------------------
Each line: {"prompt": "<text>"}   (one field; the vLLM custom reader ignores
the rest, but we keep source_layer/id too for traceability — harmless.)

USAGE (run on the server where the Azure ML mount is visible)
------------------------------------------------------------
  # default: read all 5 short layers from the mount, write next to this script
  python convert_maiprofile_eval_to_sc1.py

  # explicit input dir (the eval_datasets folder) + output dir
  python convert_maiprofile_eval_to_sc1.py \
      --input-dir "$AZURE_ML_INPUT_msndni/shares/users/zxy/maiprofile/prepared_prompts/20260615/short_layers/eval_datasets" \
      --output-dir ./maiprofile_bench_prompts

  # only some layers
  python convert_maiprofile_eval_to_sc1.py --layers layer3_seasonality,layer1_intent

Then benchmark the Google MTP model per layer, e.g.:
  for L in layer1_actual layer1_intent layer2_temporal layer3_seasonality layer4_commercial_preference; do
    # point the config's dataset at the per-layer file (env override or edit),
    # then run the existing online bench and record accept rate for that layer.
    GEMMA4_MODEL_PATH=/tmp/models/gemma4_12b/model \
    GEMMA4_ASSISTANT_MODEL_PATH=/tmp/models/gemma4_12b/assistant \
      bash run_online_align.sh configs/12b_e011_mtp.json \
        --dataset-path maiprofile_bench_prompts/sc1_maiprofile_${L}.jsonl \
        --max-concurrency none --num-prompts 200
  done
(NB: run_online_align.sh reads dataset_path from the JSON config; if it has no
 --dataset-path flag, either edit the config's dataset_path or copy the config.
 See the note printed at the end of this script.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The 5 "short" layers DSpark uses (prepare_maiprofile_splits.py:16-22).
DEFAULT_LAYERS = [
    "layer1_actual",
    "layer1_intent",
    "layer2_temporal",
    "layer3_seasonality",
    "layer4_commercial_preference",
]


def default_input_dir() -> Path | None:
    mount = os.environ.get("AZURE_ML_INPUT_msndni")
    if not mount:
        return None
    return (
        Path(mount)
        / "shares/users/zxy/maiprofile/prepared_prompts/20260615/short_layers/eval_datasets"
    )


def fold_messages(messages: list[dict]) -> str:
    """Fold system+user messages into one prompt string.

    Mirrors prepare_maiprofile_splits.py:prompt_text_for_eval (system first,
    then user, blank-line separated). Only used when `turns` is missing.
    """
    system_parts, user_parts = [], []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    parts = []
    if system_parts:
        parts.append("\n\n".join(system_parts).strip())
    if user_parts:
        parts.append("\n\n".join(user_parts).strip())
    return "\n\n".join(p for p in parts if p)


def prompt_from_record(rec: dict) -> str:
    """Extract the benchmark prompt text from one DSpark eval record."""
    turns = rec.get("turns")
    if isinstance(turns, list) and turns and isinstance(turns[0], str) and turns[0].strip():
        return turns[0]
    # Fallback: fold messages ourselves.
    return fold_messages(rec.get("messages") or rec.get("conversations") or [])


def convert_layer(src: Path, layer: str, out_path: Path) -> tuple[int, int]:
    """Convert one layer file. Returns (written, skipped)."""
    written = skipped = 0
    with src.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {src.name}:{line_no} bad JSON ({e}); skipping", file=sys.stderr)
                skipped += 1
                continue
            prompt = prompt_from_record(rec)
            if not prompt.strip():
                skipped += 1
                continue
            out = {
                "prompt": prompt,
                # Traceability (vLLM custom reader ignores extra fields):
                "source_layer": rec.get("source_layer", layer),
                "id": rec.get("id"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input-dir",
        default=None,
        help="DSpark eval_datasets dir containing maiprofile_<layer>.jsonl. "
        "Defaults to $AZURE_ML_INPUT_msndni/.../short_layers/eval_datasets.",
    )
    ap.add_argument(
        "--output-dir",
        default="./maiprofile_bench_prompts",
        help="Where to write sc1_maiprofile_<layer>.jsonl (+ _all). Default ./maiprofile_bench_prompts",
    )
    ap.add_argument(
        "--layers",
        default="short",
        help="Comma-separated layer names, or 'short' for the 5 default layers.",
    )
    ap.add_argument(
        "--prefix",
        default="maiprofile_",
        help="Filename prefix of source eval files: <prefix><layer>.jsonl. Default 'maiprofile_'.",
    )
    args = ap.parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
    else:
        d = default_input_dir()
        if d is None:
            ap.error(
                "AZURE_ML_INPUT_msndni not set and --input-dir not given. "
                "Pass --input-dir pointing at the eval_datasets folder."
            )
        input_dir = d.resolve()

    if not input_dir.is_dir():
        ap.error(f"input dir does not exist: {input_dir}")

    layers = (
        list(DEFAULT_LAYERS)
        if args.layers.strip().lower() in {"short", "short_layers"}
        else [x.strip() for x in args.layers.split(",") if x.strip()]
    )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "sc1_maiprofile_all.jsonl"

    print(f"Input : {input_dir}")
    print(f"Output: {out_dir}")
    print(f"Layers: {layers}\n")

    total_written = total_skipped = 0
    per_layer_summary = []
    with combined_path.open("w", encoding="utf-8") as fcomb:
        for layer in layers:
            src = input_dir / f"{args.prefix}{layer}.jsonl"
            if not src.exists() or src.stat().st_size == 0:
                print(f"[{layer}] MISSING/empty: {src} — skipping", file=sys.stderr)
                per_layer_summary.append((layer, 0, 0, "MISSING"))
                continue
            per_layer_path = out_dir / f"sc1_{args.prefix}{layer}.jsonl"
            w, s = convert_layer(src, layer, per_layer_path)
            total_written += w
            total_skipped += s
            per_layer_summary.append((layer, w, s, str(per_layer_path)))
            # Append same records to combined file.
            with per_layer_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    fcomb.write(line)
            print(f"[{layer}] wrote {w}, skipped {s} -> {per_layer_path.name}")

    print("\n=== Summary ===")
    for layer, w, s, path in per_layer_summary:
        print(f"  {layer:32s} written={w:4d} skipped={s:3d}")
    print(f"  {'TOTAL':32s} written={total_written:4d} skipped={total_skipped:3d}")
    print(f"\nCombined (aggregate only): {combined_path}")
    print("\nNext: benchmark the Google MTP model per layer for per-layer accept rate.")
    print("Each sc1_maiprofile_<layer>.jsonl is a drop-in replacement for sc1_delta_v2.jsonl.")


if __name__ == "__main__":
    main()
