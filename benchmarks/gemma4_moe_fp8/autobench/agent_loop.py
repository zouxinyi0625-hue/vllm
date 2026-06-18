#!/usr/bin/env python3
"""Local agent loop for autobench — orchestrates AML GPU experiments.

This script runs on your local machine (where Claude Code lives).
It reads results, decides what configs to try next, submits AML jobs
(one job can run multiple experiments), and tracks progress.

Usage:
    python agent_loop.py --status                              # show results
    python agent_loop.py --suggest --jobs 4                    # preview next configs
    python agent_loop.py --round --jobs 4 --branch zxy_dev    # submit one job with 4 exps
    python agent_loop.py --submit configs/a.json configs/b.json  # submit specific configs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config_space import (
    BASELINE_CONFIG,
    config_summary,
    config_to_serve_cmd,
    generate_next_configs,
    validate_config,
)

SCRIPT_DIR = Path(__file__).parent


def read_results(tsv_path: Path) -> list[dict]:
    """Read results.tsv into list of dicts."""
    if not tsv_path.exists():
        return []
    rows = []
    with tsv_path.open(encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            if len(vals) != len(header):
                continue
            row = dict(zip(header, vals))
            try:
                row["output_tps"] = float(row["output_tps"])
                row["total_tps"] = float(row["total_tps"])
                row["stdev"] = float(row.get("stdev", "0"))
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def load_json_results(results_dir: Path) -> list[dict]:
    """Load detailed JSON results for config extraction."""
    results = []
    if not results_dir.exists():
        return results
    for f in sorted(results_dir.glob("*.json")):
        try:
            with f.open() as fh:
                results.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def print_status(history: list[dict]):
    """Print current experiment status."""
    if not history:
        print("No results yet.")
        return

    valid = [h for h in history if h.get("status") == "ok" and h.get("output_tps", 0) > 0]
    crashes = [h for h in history if h.get("status") == "crash"]

    print(f"\n{'='*60}")
    print(f"  Total experiments: {len(history)}")
    print(f"  Valid runs: {len(valid)}")
    print(f"  Crashes: {len(crashes)}")

    if valid:
        best = max(valid, key=lambda h: h["output_tps"])
        print(f"  Best: {best['output_tps']:.1f} tok/s ({best.get('config_summary', '')})")
        print(f"  vs baseline (2020): {((best['output_tps'] - 2020) / 2020 * 100):+.1f}%")

    print(f"{'='*60}")

    if valid:
        print("\nTop 5:")
        ranked = sorted(valid, key=lambda h: h["output_tps"], reverse=True)
        for i, r in enumerate(ranked[:5], 1):
            print(f"  {i}. {r['output_tps']:.1f} +/-{r.get('stdev', 0):.1f} | {r.get('config_summary', '')}")

    if crashes:
        print(f"\nRecent crashes ({len(crashes)}):")
        for c in crashes[-3:]:
            print(f"  - {c.get('config_summary', 'unknown')}")


def main():
    ap = argparse.ArgumentParser(description="Autobench local agent loop")
    ap.add_argument("--status", action="store_true", help="Show current results")
    ap.add_argument("--best-serve", action="store_true",
                    help="Print vllm serve command for the best config")
    ap.add_argument("--suggest", action="store_true",
                    help="Generate next configs without submitting")
    ap.add_argument("--round", action="store_true",
                    help="Generate configs and submit as one job")
    ap.add_argument("--submit", nargs="+", metavar="CONFIG_JSON",
                    help="Submit specific config JSON file(s) as one job")
    ap.add_argument("--jobs", type=int, default=4,
                    help="Number of parallel AML jobs (for --round)")
    ap.add_argument("--exps-per-job", type=int, default=1,
                    help="Experiments per job (default: 1 = one config per job)")
    ap.add_argument("--prompts", type=int, default=1000,
                    help="Prompts per experiment (default: 1000, matches sc1)")
    ap.add_argument("--reps", type=int, default=2, help="Reps per experiment")
    ap.add_argument("--branch", default="zxy_dev_autobench",
                    help="Git branch for the job to clone (default: zxy_dev_autobench)")
    ap.add_argument("--results-tsv", type=Path, default=None,
                    help="Path to results.tsv (local or mounted)")
    ap.add_argument("--debug", action="store_true",
                    help="Keep jobs alive (sleep infinity) after experiments for SSH")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be submitted without actually submitting")
    args = ap.parse_args()

    results_tsv = args.results_tsv or SCRIPT_DIR / "results_all.tsv"

    if args.status:
        history = read_results(results_tsv)
        print_status(history)
        return 0

    if args.best_serve:
        json_results = load_json_results(SCRIPT_DIR / "run_results")
        if not json_results:
            print("No JSON results found. Run experiments first.")
            return 1
        valid = [r for r in json_results if r.get("status") == "ok" and r.get("mean_output_tps", 0) > 0]
        if not valid:
            print("No valid results.")
            return 1
        best = max(valid, key=lambda r: r["mean_output_tps"])
        print(f"# Best result: {best['mean_output_tps']} output tok/s")
        print(f"# Config: {best.get('config_summary', '')}")
        print(f"# Run ID: {best.get('run_id', 'unknown')}\n")
        print(config_to_serve_cmd(best["config"]))
        return 0

    if args.suggest:
        history = read_results(results_tsv)
        json_history = load_json_results(SCRIPT_DIR / "run_results")
        configs = generate_next_configs(json_history if json_history else history, n=args.jobs)
        print(f"Suggested {len(configs)} configs:")
        for i, cfg in enumerate(configs, 1):
            print(f"\n  {i}. {config_summary(cfg)}")
            print(f"     {json.dumps(cfg)}")
        return 0

    if args.submit:
        from submit_job import submit_experiment
        configs = []
        for cfg_path in args.submit:
            with open(cfg_path) as f:
                cfg = json.load(f)
            valid, err = validate_config(cfg)
            if not valid:
                print(f"ERROR in {cfg_path}: {err}", file=sys.stderr)
                return 1
            configs.append(cfg)

        if args.dry_run:
            from submit_job import build_job_command
            print("=== DRY RUN ===\n")
            print(build_job_command(configs, prompts=args.prompts, reps=args.reps,
                                    branch=args.branch, debug=args.debug))
            return 0

        job_name = submit_experiment(
            configs, prompts=args.prompts, reps=args.reps, branch=args.branch,
            debug=args.debug,
        )
        print(f"\nJob: {job_name}")
        if args.debug:
            print("Job will sleep infinity after experiments. Cancel when done.")
        return 0

    if args.round:
        from submit_job import submit_experiment, build_job_command
        history = read_results(results_tsv)
        print_status(history)

        json_history = load_json_results(SCRIPT_DIR / "run_results")
        total_exps = args.jobs * args.exps_per_job
        configs = generate_next_configs(
            json_history if json_history else history, n=total_exps
        )

        if not configs:
            print("No new configs to try — parameter space may be exhausted.")
            return 0

        # Distribute configs across parallel jobs
        job_configs = []
        for i in range(0, len(configs), args.exps_per_job):
            job_configs.append(configs[i:i + args.exps_per_job])

        print(f"\nWill submit {len(job_configs)} parallel job(s), "
              f"{len(configs)} experiment(s) total:")
        for j, jc in enumerate(job_configs, 1):
            for cfg in jc:
                print(f"  Job {j}: {config_summary(cfg)}")

        if args.dry_run:
            print("\n=== DRY RUN ===\n")
            for j, jc in enumerate(job_configs, 1):
                print(f"--- Job {j}/{len(job_configs)} ---")
                print(build_job_command(jc, prompts=args.prompts, reps=args.reps,
                                        branch=args.branch, debug=args.debug))
                print()
            return 0

        job_names = []
        for jc in job_configs:
            name = submit_experiment(
                jc, prompts=args.prompts, reps=args.reps, branch=args.branch,
                debug=args.debug,
            )
            job_names.append(name)

        print(f"\n{len(job_names)} job(s) submitted:")
        for name in job_names:
            print(f"  - {name}")
        if args.debug:
            print("\nAll jobs will sleep infinity after experiments. Cancel when done.")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
