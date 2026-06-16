#!/usr/bin/env python3
"""Analyze autobench results: ranking, parameter importance, report generation.

Usage:
    python analyze.py                         # analyze local results.tsv
    python analyze.py --tsv /path/to/results.tsv
    python analyze.py --json-dir run_results/ # detailed analysis from JSON files
    python analyze.py --reproduce 20260609_031753  # print reproduce command for a run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from config_space import parse_summary, reproduce_command

SCRIPT_DIR = Path(__file__).parent
DEFAULT_TSV = SCRIPT_DIR / "results_all.tsv"
DEFAULT_JSON_DIR = SCRIPT_DIR / "run_results"


def load_tsv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            row = dict(zip(header, vals))
            try:
                row["output_tps"] = float(row["output_tps"])
                row["total_tps"] = float(row["total_tps"])
                row["stdev"] = float(row.get("stdev", "0"))
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def load_json_results(json_dir: Path) -> list[dict]:
    results = []
    for f in sorted(json_dir.glob("*.json")):
        with f.open() as fh:
            results.append(json.load(fh))
    return results


def rank_experiments(rows: list[dict]) -> list[dict]:
    valid = [r for r in rows if r.get("status") == "ok" and r["output_tps"] > 0]
    return sorted(valid, key=lambda r: r["output_tps"], reverse=True)


def parameter_importance(results: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """Group results by each parameter value, compute mean output_tps per group."""
    if not results:
        return {}

    valid = [r for r in results if r.get("status") == "ok" and r.get("mean_output_tps", 0) > 0]
    if not valid:
        return {}

    param_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in valid:
        cfg = r.get("config", {})
        for key, val in cfg.items():
            param_groups[key][str(val)].append(r["mean_output_tps"])

    importance: dict[str, list[tuple[str, float]]] = {}
    for param, val_map in param_groups.items():
        if len(val_map) < 2:
            continue
        means = [(val, sum(tps_list) / len(tps_list)) for val, tps_list in val_map.items()]
        means.sort(key=lambda x: x[1], reverse=True)
        importance[param] = means

    return importance


def generate_report(rows: list[dict], json_results: list[dict]) -> str:
    lines = ["# Autobench Results Report\n"]

    ranked = rank_experiments(rows)
    if not ranked:
        lines.append("No valid results yet.\n")
        return "\n".join(lines)

    best = ranked[0]
    lines.append(f"## Best Result: {best['output_tps']:.1f} output tok/s\n")
    lines.append(f"- Run ID: `{best.get('run_id', 'unknown')}`")
    lines.append(f"- Config: {best.get('config_summary', 'N/A')}")
    lines.append(f"- Stdev: ±{best.get('stdev', 0):.1f}")
    lines.append("")

    # Top 10 table
    lines.append("## Top 10 Experiments\n")
    lines.append("| # | output_tps | ±stdev | config |")
    lines.append("|---|-----------|--------|--------|")
    for i, r in enumerate(ranked[:10], 1):
        lines.append(
            f"| {i} | {r['output_tps']:.1f} | ±{r.get('stdev', 0):.1f} | "
            f"{r.get('config_summary', '')} |"
        )
    lines.append("")

    # Crashes
    crashes = [r for r in rows if r.get("status") == "crash"]
    if crashes:
        lines.append(f"## Crashes: {len(crashes)}\n")
        for c in crashes[-5:]:
            lines.append(f"- `{c.get('run_id', '?')}`: {c.get('config_summary', 'unknown config')}")
        lines.append("")

    # Parameter importance
    if json_results:
        importance = parameter_importance(json_results)
        if importance:
            lines.append("## Parameter Importance\n")
            sorted_params = sorted(
                importance.items(),
                key=lambda kv: max(t[1] for t in kv[1]) - min(t[1] for t in kv[1]),
                reverse=True,
            )
            for param, means in sorted_params[:8]:
                spread = max(t[1] for t in means) - min(t[1] for t in means)
                lines.append(f"### `{param}` (spread: {spread:.1f} tok/s)")
                for val, mean_tps in means:
                    lines.append(f"  - {val}: {mean_tps:.1f} tok/s")
                lines.append("")

    # Summary stats
    valid_tps = [r["output_tps"] for r in ranked]
    lines.append("## Summary Stats\n")
    lines.append(f"- Total experiments: {len(rows)}")
    lines.append(f"- Valid runs: {len(ranked)}")
    lines.append(f"- Crashes: {len(crashes)}")
    lines.append(f"- Best: {max(valid_tps):.1f} tok/s")
    lines.append(f"- Worst valid: {min(valid_tps):.1f} tok/s")
    lines.append(f"- Mean: {sum(valid_tps)/len(valid_tps):.1f} tok/s")
    lines.append(f"- vs baseline (2020): {((max(valid_tps) - 2020) / 2020 * 100):+.1f}%")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Analyze autobench results")
    ap.add_argument("--tsv", type=Path, default=DEFAULT_TSV,
                    help="Path to results.tsv")
    ap.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR,
                    help="Path to JSON results directory")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write report to file (default: stdout)")
    ap.add_argument("--reproduce", type=str, default=None,
                    help="Print reproduce command for a run_id (or 'best')")
    args = ap.parse_args()

    if not args.tsv.exists():
        print(f"No results file at {args.tsv}", file=sys.stderr)
        return 1

    rows = load_tsv(args.tsv)

    if args.reproduce:
        target = args.reproduce
        if target == "best":
            ranked = rank_experiments(rows)
            if not ranked:
                print("No valid results.", file=sys.stderr)
                return 1
            row = ranked[0]
        else:
            matches = [r for r in rows if r.get("run_id") == target]
            if not matches:
                print(f"run_id '{target}' not found in results.tsv", file=sys.stderr)
                return 1
            row = matches[0]

        summary = row.get("config_summary", "")
        cfg = parse_summary(summary)
        print(f"# Reproduce: {row.get('run_id')} ({row.get('output_tps', 0):.1f} tok/s)")
        print(f"# Config: {summary}")
        print()
        print(reproduce_command(cfg))
        return 0

    json_results = load_json_results(args.json_dir) if args.json_dir.exists() else []

    report = generate_report(rows, json_results)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
