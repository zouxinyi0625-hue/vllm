#!/usr/bin/env python3
"""Compare speculative-decoding benchmark runs against the MTP baseline.

Parses the text output produced by ``bench_online_align.sh`` (which tees the
``vllm bench serve`` stdout to ``online_results/<name>_online_<ts>.txt``) and
prints a side-by-side table of the key acceleration metrics:

    - Output token throughput (tok/s)   <- primary success metric
    - Total token throughput (tok/s)
    - Acceptance length                 <- user explicitly wants this
    - Acceptance rate (%)
    - Per-position acceptance (%)

It also computes the relative delta of each candidate vs. the baseline so you
can immediately see whether a draft model reaches the +10-20% throughput goal.

Usage:
    # Compare the newest EAGLE-3 run against the newest MTP run:
    python compare_spec_results.py \
        --baseline online_results/26b_e011_mtp_online_*.txt \
        --candidate online_results/26b_eagle3_online_*.txt

    # Or let it auto-pick the newest file per config name:
    python compare_spec_results.py --auto online_results/

This script is pure stdlib and read-only. It does NOT run any benchmark; it
only summarizes result files produced on the server.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# --- metric line patterns from vllm bench serve stdout -----------------------
# See vllm/benchmarks/serve.py print_and_save_result_json for exact labels.
PATTERNS = {
    "output_throughput": re.compile(
        r"Output token throughput \(tok/s\):\s*([\d.]+)"
    ),
    "total_throughput": re.compile(
        r"Total [Tt]oken throughput \(tok/s\):\s*([\d.]+)"
    ),
    "acceptance_rate": re.compile(r"Acceptance rate \(%\):\s*([\d.]+)"),
    "acceptance_length": re.compile(r"Acceptance length:\s*([\d.]+)"),
    "num_drafts": re.compile(r"Drafts:\s*(\d+)"),
    "draft_tokens": re.compile(r"Draft tokens:\s*(\d+)"),
    "accepted_tokens": re.compile(r"Accepted tokens:\s*(\d+)"),
    "req_throughput": re.compile(r"Request throughput \(req/s\):\s*([\d.]+)"),
}
# Per-position lines look like "  Position 0: 93.61" (indented list after header)
PER_POS = re.compile(r"[Pp]osition\s*(\d+):\s*([\d.]+)")


def parse_result_file(path: str) -> dict:
    """Extract metrics from one bench result file (online .txt or offline .json)."""
    if path.endswith(".json"):
        return parse_offline_json(path)
    return parse_online_txt(path)


def parse_offline_json(path: str) -> dict:
    """Parse bench_offline_align.py output (JSON with mean_output_tps + spec_metrics)."""
    import json
    with open(path, encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    out: dict = {"_path": path, "_name": os.path.basename(path), "_mode": "offline"}
    out["output_throughput"] = data.get("mean_output_tps")
    out["total_throughput"] = data.get("mean_total_tps")
    rows = data.get("rows") or [{}]
    out["req_throughput"] = rows[0].get("requests_per_second")
    sm = data.get("spec_metrics") or {}
    out["acceptance_rate"] = sm.get("acceptance_rate")
    out["acceptance_length"] = sm.get("acceptance_length")
    out["num_drafts"] = sm.get("num_drafts")
    out["draft_tokens"] = sm.get("num_draft_tokens")
    out["accepted_tokens"] = sm.get("num_accepted_tokens")
    # per_position_acceptance is {pos: rate}; normalize keys to int.
    pp = sm.get("per_position_acceptance") or {}
    out["per_position"] = {int(k): float(v) for k, v in pp.items()}
    return out


def parse_online_txt(path: str) -> dict:
    """Parse bench_online_align.sh output (vllm bench serve stdout text)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    out: dict = {"_path": path, "_name": os.path.basename(path), "_mode": "online"}
    for key, pat in PATTERNS.items():
        m = pat.search(text)
        out[key] = float(m.group(1)) if m else None
    # Per-position acceptance: only capture lines that appear AFTER the
    # "Per-position acceptance" header to avoid false matches.
    per_pos: dict[int, float] = {}
    hdr = text.find("Per-position acceptance")
    scope = text[hdr:] if hdr >= 0 else ""
    for m in PER_POS.finditer(scope):
        per_pos[int(m.group(1))] = float(m.group(2))
    out["per_position"] = per_pos
    return out


def newest(paths: list[str]) -> str | None:
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def fmt(v, width=12, prec=2):
    if v is None:
        return f"{'-':<{width}}"
    if isinstance(v, float):
        return f"{v:<{width}.{prec}f}"
    return f"{str(v):<{width}}"


def delta_pct(cand, base):
    if cand is None or base in (None, 0):
        return None
    return (cand - base) / base * 100.0


def short_name(name: str) -> str:
    """Derive a compact config label from a result filename."""
    base = name.split("_online_")[0]
    base = re.sub(r"_\d{8}_\d{6}\.json$", "", base)
    base = re.sub(r"\.(json|txt)$", "", base)
    return base


def print_comparison(base: dict, cands: list[dict]) -> None:
    rows = [
        ("Output tok/s  (PRIMARY)", "output_throughput", 2),
        ("Total tok/s", "total_throughput", 2),
        ("Acceptance length", "acceptance_length", 2),
        ("Acceptance rate (%)", "acceptance_rate", 2),
        ("Request throughput", "req_throughput", 2),
    ]
    col = 18
    header = f"{'Metric':<26}{'BASELINE':<{col}}"
    for c in cands:
        short = short_name(c["_name"])
        header += f"{short:<{col}}{'Δ%':<8}"
    print("=" * len(header))
    print(f"Baseline: {base['_name']}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, key, prec in rows:
        line = f"{label:<26}{fmt(base.get(key), col, prec)}"
        for c in cands:
            d = delta_pct(c.get(key), base.get(key))
            dstr = f"{d:+.1f}" if d is not None else "-"
            line += f"{fmt(c.get(key), col, prec)}{dstr:<8}"
        print(line)

    # Per-position acceptance block
    print("-" * len(header))
    print("Per-position acceptance (%):")
    max_pos = 0
    for d in [base] + cands:
        if d.get("per_position"):
            max_pos = max(max_pos, max(d["per_position"].keys()))
    for pos in range(max_pos + 1):
        line = f"  pos {pos:<22}{fmt(base['per_position'].get(pos), col)}"
        for c in cands:
            line += f"{fmt(c['per_position'].get(pos), col)}{'':<8}"
        print(line)

    # Goal check on primary metric
    print("=" * len(header))
    base_tps = base.get("output_throughput")
    if base_tps:
        print(f"Goal: +10~20% output tok/s over baseline "
              f"({base_tps:.1f}) = {base_tps*1.10:.1f} ~ {base_tps*1.20:.1f}")
        for c in cands:
            d = delta_pct(c.get("output_throughput"), base_tps)
            if d is None:
                continue
            verdict = "MET" if d >= 10 else ("PARTIAL" if d > 0 else "REGRESSION")
            short = short_name(c["_name"])
            print(f"  {short:<28} {c.get('output_throughput'):.1f} tok/s "
                  f"({d:+.1f}%)  -> {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", help="baseline result .txt (glob ok)")
    ap.add_argument("--candidate", action="append", default=[],
                    help="candidate result .txt (glob ok, repeatable)")
    ap.add_argument("--auto", metavar="DIR",
                    help="auto-pick newest file per config name in DIR")
    args = ap.parse_args()

    if args.auto:
        # Online results: online_results/<name>_online_<ts>.txt
        # Offline results: offline_results/<name>_<ts>.json
        files = glob.glob(os.path.join(args.auto, "*_online_*.txt"))
        files += glob.glob(os.path.join(args.auto, "*.json"))
        by_name: dict[str, list[str]] = {}
        for f in files:
            base = os.path.basename(f)
            if "_online_" in base:
                name = base.split("_online_")[0]
            else:
                # offline: strip trailing _<date>_<time>.json
                name = re.sub(r"_\d{8}_\d{6}\.json$", "", base)
                name = re.sub(r"\.json$", "", name)
            by_name.setdefault(name, []).append(f)
        picked = {n: newest(fs) for n, fs in by_name.items()}
        # baseline = the MTP config if present, else first
        base_name = next((n for n in picked if "mtp" in n and "no_mtp" not in n),
                         next(iter(picked), None))
        if base_name is None:
            print("No result files found in", args.auto, file=sys.stderr)
            return 1
        base = parse_result_file(picked.pop(base_name))
        cands = [parse_result_file(p) for p in picked.values() if p]
    else:
        if not args.baseline or not args.candidate:
            ap.error("provide --baseline and --candidate, or --auto DIR")
        base_path = newest(glob.glob(args.baseline))
        if not base_path:
            print("No baseline file matched:", args.baseline, file=sys.stderr)
            return 1
        base = parse_result_file(base_path)
        cands = []
        for c in args.candidate:
            p = newest(glob.glob(c))
            if p:
                cands.append(parse_result_file(p))
        if not cands:
            print("No candidate files matched", file=sys.stderr)
            return 1

    print_comparison(base, cands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
