#!/usr/bin/env python3
"""Submit AML Singularity batch jobs for autobench experiments.

Each job:
  - Clones code from GitHub (new branch per job for traceability)
  - Copies data from mount to local disk
  - Runs one or more experiments sequentially
  - Copies results back to mount (date+config naming, no overwrites)
  - Exits when done (use --debug to keep alive for SSH inspection)

Usage:
    # Single config
    python submit_job.py --config configs/baseline_e011.json --dry-run

    # Multiple configs in one job (saves machine startup time)
    python submit_job.py --config configs/a.json configs/b.json configs/c.json

    # Keep machine alive for debugging
    python submit_job.py --config configs/a.json --debug
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities import JobResourceConfiguration, SshJobService
from azure.identity import DefaultAzureCredential

# --- Workspace ---
SUBSCRIPTION_ID = "b6dc87f3-c479-49c8-8cb5-7896da3ff895"
RESOURCE_GROUP = "AMLStudio"
WORKSPACE = "NewsFeedL2_AML"

# --- Virtual Cluster ---
VC_ARM_ID = (
    f"/subscriptions/{SUBSCRIPTION_ID}"
    "/resourceGroups/rg-cs-ranking-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
)

# --- Managed Identity ---
UAI_RESOURCE_ID = (
    f"/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}"
    "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"
)

SSH_PUB_KEY = (
    "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCpOz0QGUOBnEqMn+DwzbltVytWcFB/"
    "J10EpA0Rf5UXMtScYFKKYAi50qyhhdT5nj0LharII8p42w5MGPMLepqey6oFkVjDWrkT"
    "mzYe2nfkZpT9+GjGIEnbSvSL5CidsSWwDTzsgb5eLu0bExWHRwXscTLIfYQBNurdinw+"
    "z6k96DS1W4YTclJveoKFMJTT0ZpNd8FnGlQeJuO++xR1zVxK938rGEHO1bY3Aph3Pdg"
    "sTYliJvYNqihM/p+az8UK+zRNwRdbE175UZALbuD77mVuF8hG19ggLxi3HeyO9RE8t9V"
    "hNn6nyZDtMQtRxpgqx83tYSXqUatMwoHXiONQ1gMVbKhW6kNb7vvwCAOmUU/In4psgM"
    "1RiEv/VNVSV/9CusYDsCOvGPT0mOliaRMebA2KyHPmjkKdQNW8FTUM9No1cFsigMtsj"
    "84PwjcZYbPGFCTbifutUjav7p0PN+9AyLCOEyikX9SVGq06Qo4/oW5/aRMOQRwRala9S"
    "4pQjvj3kduQp8jNITMW+yn5AI6lgE457rbSmMpE6YxhVgVQjF1Mb6szxrQoMntqTW5O1"
    "ypr691vcnk9yph9fv9BVc+b+wdFbe8qHoYCDtDSBPYMq7GdwYqoDkpWdi5VBYRiMOGPN"
    "H5PB6S6xLBLD3Ybm+tHW/vvT6d/qjd3wYvg2dAuGwc/Mw== xinyizou@microsoft.com"
)

ENVIRONMENT = "azureml:vllm_gemma4:4"
INSTANCE_TYPE = "Singularity.ND12am_A100_v4"
DATASTORE_PATH = "azureml://datastores/adls_msn_dni_09_rankfun/paths/"

GIT_REPO = "https://github.com/zouxinyi0625-hue/vllm-msn.git"
GIT_BRANCH = "zxy_dev_autobench"

HF_MODEL_REPO = "Xinyi0625/Gemma-4-26B-A4B-it-deploy"
LOCAL_MODEL_DIR = "/tmp/models/gemma4"

MOUNT_RESULTS_PATH = "shares/users/zxy/autobench/results"
MOUNT_DATA_PATH = "shares/users/zxy/autobench/data"


def get_ml_client() -> MLClient:
    return MLClient(
        DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE,
    )


def make_resource_config() -> JobResourceConfiguration:
    return JobResourceConfiguration(
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        properties={
            "singularity": {
                "slaTier": "Premium",
                "priority": "High",
                "enableAzmlInt": False,
                "locations": ["ukwest"],
            }
        },
    )


def build_job_command(
    configs: list[dict],
    prompts: int = 1000,
    reps: int = 2,
    branch: str = GIT_BRANCH,
    job_id: str = "",
    debug: bool = False,
) -> str:
    """Build bash command that runs multiple experiments.

    Each experiment is isolated: failure in one does not affect the next.
    Results are tagged with job_id to prevent conflicts from parallel jobs.
    If debug=True, sleeps infinity at end for SSH inspection.
    """
    job_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not job_id:
        job_id = job_ts

    # Build per-experiment run blocks
    experiment_blocks = []
    for i, config in enumerate(configs, 1):
        config_json = json.dumps(config, indent=2)
        from config_space import config_summary
        summary = config_summary(config)
        # Unique result name: jobid_expN_summary-slug (no conflicts between parallel jobs)
        slug = summary.replace(" | ", "_").replace(" ", "")[:50]
        result_name = f"{job_id}_exp{i}_{slug}"

        env_exports = []
        for key in ["VLLM_HUMMING_MOE_GEMM_TYPE", "VLLM_HUMMING_USE_F16_ACCUM",
                    "VLLM_TEST_FORCE_FP8_MARLIN"]:
            if key in config:
                env_exports.append(f'export {key}="{config[key]}"')
            else:
                env_exports.append(f'unset {key} 2>/dev/null || true')
        env_block = "\n".join(env_exports)

        block = f"""
echo ""
echo "{'='*60}"
echo "  [{job_id}] Experiment {i}/{len(configs)}: {summary}"
echo "{'='*60}"
echo "  Start time: $(date)"
echo "  Prompts: {prompts}, Reps: {reps}"

# Set/unset MoE env vars for this experiment
{env_block}

cat > /tmp/config_{i}.json << 'CFGEOF'
{config_json}
CFGEOF

# Run experiment in subshell — failure does NOT stop subsequent experiments
(
    python run_experiment.py --config /tmp/config_{i}.json --prompts {prompts} --reps {reps} \\
        2>&1 | tee "/tmp/run_{i}.log"
) || echo "[WARN] EXPERIMENT {i} FAILED (exit $?), continuing..."

echo "  Experiment {i} finished at $(date)"

# Copy results with unique names (never overwrite, tagged with job_id)
if [ -d run_results ]; then
    for f in run_results/*.json; do
        if [ -f "$f" ]; then
            cp "$f" "$RESULTS_DIR/{result_name}.json" 2>/dev/null || true
            echo "  Result saved: {result_name}.json"
        fi
    done
    rm -rf run_results
else
    echo "  [WARN] No run_results/ directory — experiment likely crashed"
fi
"""
        experiment_blocks.append(block)

    experiments_str = "\n".join(experiment_blocks)

    debug_mode = "yes" if debug else "no"
    cmd = f"""#!/bin/bash
# Autobench Job — {job_id} — {len(configs)} experiment(s)
# Branch: {branch}
# Prompts: {prompts}, Reps: {reps} (matches sc1 ablation settings)
# Debug: {debug_mode}

set +e  # Do NOT exit on error — we want all experiments to run

echo "=== Autobench Job Start ==="
echo "Job ID: {job_id}"
echo "Timestamp: $(date)"
echo "Experiments: {len(configs)}"
echo "Branch: {branch}"
echo "Prompts: {prompts}, Reps: {reps}"
echo ""

# 1. Clone code on the experiment branch
echo "[Step 1/6] Cloning code from {branch}..."
git clone --branch {branch} --depth 1 {GIT_REPO} /tmp/vllm-msn
cd /tmp/vllm-msn/benchmarks/gemma4_moe_fp8/autobench
echo "  Code cloned at $(date)"

# 2. Copy data from mount to local disk (mount I/O is slow)
echo "[Step 2/6] Copying dataset to local disk..."
MOUNT=$AZURE_ML_INPUT_msndni
mkdir -p datasets
cp "$MOUNT/{MOUNT_DATA_PATH}/sc1_delta_v2.jsonl" datasets/
echo "  Data copied: $(wc -l < datasets/sc1_delta_v2.jsonl) lines"

# 3. Download model from HuggingFace (token injected via AML env vars)
echo "[Step 3/6] Downloading model from HuggingFace..."
echo "  Repo: {HF_MODEL_REPO}"
echo "  Target: {LOCAL_MODEL_DIR}"
echo "  Start: $(date)"
hf download {HF_MODEL_REPO} --local-dir {LOCAL_MODEL_DIR} --token "$HF_TOKEN"
echo "  Model download complete at $(date)"
echo "  Contents:"
ls -lh {LOCAL_MODEL_DIR}/
du -sh {LOCAL_MODEL_DIR}/

# 4. Fixed env vars for A100
echo "[Step 4/6] Setting environment variables..."
export VLLM_USE_FLASHINFER_MOE_FP8=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_MOE_USE_DEEP_GEMM=0
export GEMMA4_TEXT_ONLY_MODEL_PATH="{LOCAL_MODEL_DIR}/text_only"
export GEMMA4_ASSISTANT_MODEL_PATH="{LOCAL_MODEL_DIR}/assistant"
echo "  GEMMA4_TEXT_ONLY_MODEL_PATH=$GEMMA4_TEXT_ONLY_MODEL_PATH"
echo "  GEMMA4_ASSISTANT_MODEL_PATH=$GEMMA4_ASSISTANT_MODEL_PATH"
echo "  VLLM_USE_FLASHINFER_MOE_FP8=$VLLM_USE_FLASHINFER_MOE_FP8"
echo "  VLLM_MOE_USE_DEEP_GEMM=$VLLM_MOE_USE_DEEP_GEMM"

# 5. Prepare results directory on mount (job-specific subdir prevents conflicts)
echo "[Step 5/6] Preparing results directory..."
RESULTS_DIR="$MOUNT/{MOUNT_RESULTS_PATH}/{job_id}"
mkdir -p "$RESULTS_DIR"
echo "  Results will be saved to: $RESULTS_DIR/"

# 6. Run experiments sequentially (each isolated — failure won't stop others)
echo "[Step 6/6] Running {len(configs)} experiment(s)..."
echo ""
{experiments_str}

# 7. Copy aggregated results.tsv to mount
echo ""
echo "=== Saving aggregated results ==="
if [ -f results.tsv ]; then
    cp results.tsv "$RESULTS_DIR/results.tsv"
    echo "  Per-job results.tsv saved"
    # Append to master results file (use flock to prevent parallel write corruption)
    MASTER_TSV="$MOUNT/{MOUNT_RESULTS_PATH}/results_all.tsv"
    if command -v flock &>/dev/null; then
        flock "$MASTER_TSV.lock" bash -c '
            if [ -f "'"$MASTER_TSV"'" ]; then
                tail -n +2 results.tsv >> "'"$MASTER_TSV"'"
            else
                cp results.tsv "'"$MASTER_TSV"'"
            fi
        '
    else
        if [ -f "$MASTER_TSV" ]; then
            tail -n +2 results.tsv >> "$MASTER_TSV"
        else
            cp results.tsv "$MASTER_TSV"
        fi
    fi
    echo "  Master results_all.tsv updated"
else
    echo "  [WARN] No results.tsv generated"
fi

echo ""
echo "=== All experiments done ==="
echo "Job ID: {job_id}"
echo "Results: $RESULTS_DIR/"
echo "Finished at: $(date)"
"""

    if debug:
        cmd += """
echo ""
echo "Machine staying alive (sleep infinity). SSH in to inspect or re-run."
echo "Cancel the job when done."

sleep infinity
"""
    return cmd


def submit_experiment(
    configs: list[dict],
    prompts: int = 1000,
    reps: int = 2,
    branch: str = GIT_BRANCH,
    display_name: str | None = None,
    ml_client: MLClient | None = None,
    hf_token: str | None = None,
    debug: bool = False,
) -> str:
    """Submit one job running multiple experiments. Returns job name (ID).

    Each job gets a unique job_id embedded in result filenames to prevent
    conflicts when multiple jobs run in parallel.
    """
    if ml_client is None:
        ml_client = get_ml_client()

    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not set. Set it via environment variable or pass hf_token=."
        )

    from config_space import config_summary
    job_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = [config_summary(c) for c in configs]

    if display_name is None:
        if len(configs) == 1:
            display_name = f"autobench-{job_ts}-{summaries[0][:30]}"
        else:
            display_name = f"autobench-{job_ts}-{len(configs)}exps"

    job_cmd = build_job_command(configs, prompts=prompts, reps=reps,
                               branch=branch, job_id=job_ts, debug=debug)

    job = command(
        command=job_cmd,
        display_name=display_name,
        environment=ENVIRONMENT,
        compute=VC_ARM_ID,
        resources=make_resource_config(),
        inputs={
            "msndni": Input(
                type="uri_folder",
                path=DATASTORE_PATH,
                mode=InputOutputModes.RW_MOUNT,
            ),
        },
        environment_variables={
            "_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID,
            "HF_TOKEN": hf_token,
        },
        services={
            "ssh": SshJobService(ssh_public_keys=SSH_PUB_KEY, nodes="all"),
        },
    )

    created = ml_client.jobs.create_or_update(job)
    print(f"  Job submitted: {created.name}")
    print(f"  Display name: {display_name}")
    print(f"  Branch: {branch}")
    print(f"  Experiments: {len(configs)}")
    for i, s in enumerate(summaries, 1):
        print(f"    {i}. {s}")
    print(f"  Studio: {created.studio_url}")
    return created.name


def poll_jobs(
    job_names: list[str],
    ml_client: MLClient | None = None,
    poll_interval: int = 60,
    timeout: int = 7200,
) -> dict[str, str]:
    """Poll until all jobs reach 'Running' or terminal state.

    Note: jobs end with sleep infinity, so they won't reach 'Completed'
    on their own. We poll until Running (experiments done, sleeping) or Failed.
    """
    if ml_client is None:
        ml_client = get_ml_client()

    # With sleep infinity, 'Running' means experiments are done (or running).
    # User manually cancels when satisfied.
    done_states = {"Completed", "Failed", "Canceled"}
    results: dict[str, str] = {}
    pending = set(job_names)
    start = time.time()

    while pending and (time.time() - start) < timeout:
        for name in list(pending):
            job = ml_client.jobs.get(name)
            status = job.status
            if status in done_states:
                results[name] = status
                pending.remove(name)
                print(f"  Job {name}: {status}")
        if pending:
            elapsed = int(time.time() - start)
            print(f"  Waiting... {len(pending)} jobs pending ({elapsed}s elapsed)")
            time.sleep(poll_interval)

    for name in pending:
        results[name] = "Timeout"
    return results


def main():
    ap = argparse.ArgumentParser(description="Submit AML autobench experiment job")
    ap.add_argument("--config", nargs="+", required=True,
                    help="Path(s) to config JSON file(s). Multiple = run in same job.")
    ap.add_argument("--prompts", type=int, default=200, help="Number of prompts")
    ap.add_argument("--reps", type=int, default=2, help="Repetitions per experiment")
    ap.add_argument("--branch", default=GIT_BRANCH,
                    help=f"Git branch to clone (default: {GIT_BRANCH})")
    ap.add_argument("--name", default=None, help="Custom job display name")
    ap.add_argument("--debug", action="store_true",
                    help="Keep machine alive (sleep infinity) after experiments for SSH debugging")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print job command without submitting")
    args = ap.parse_args()

    configs = []
    for cfg_path in args.config:
        with open(cfg_path) as f:
            configs.append(json.load(f))

    if args.dry_run:
        print("=== DRY RUN — Job Command ===\n")
        print(build_job_command(configs, prompts=args.prompts, reps=args.reps,
                                branch=args.branch, debug=args.debug))
        print(f"\n=== Would submit {len(configs)} experiment(s) on branch '{args.branch}' ===")
        return 0

    job_name = submit_experiment(
        configs,
        prompts=args.prompts,
        reps=args.reps,
        branch=args.branch,
        display_name=args.name,
        debug=args.debug,
    )
    print(f"\nJob submitted: {job_name}")
    if args.debug:
        print("Job will sleep infinity after experiments. SSH in or cancel when done.")
    else:
        print("Job will exit automatically after experiments complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
