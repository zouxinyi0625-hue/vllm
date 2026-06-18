"""
Create a Singularity job with mounted storage and sleep infinity.
Holds the machine for interactive use.

Usage:
    python submit_sleep_job.py
"""
import os

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, command
from azure.ai.ml.entities import JobResourceConfiguration, SshJobService, JupyterLabJobService
from azure.ai.ml.constants import InputOutputModes

# Workspace (set via environment variables)
SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP"]
WORKSPACE = os.environ["AZURE_WORKSPACE"]

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE,
)

# Virtual Cluster
VC_ARM_ID = (
    f"/subscriptions/{SUBSCRIPTION_ID}"
    "/resourceGroups/rg-cs-ranking-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
)

# Managed Identity (required by new Singularity policy)
UAI_RESOURCE_ID = (
    f"/subscriptions/{SUBSCRIPTION_ID}"
    f"/resourceGroups/{RESOURCE_GROUP}"
    "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"
)

# SSH public key: read from SSH_PUB_KEY env var
SSH_PUB_KEY = os.environ["SSH_PUB_KEY"]

# Resource configuration
res_cfg = JobResourceConfiguration(
    instance_count=1,
    instance_type="Singularity.ND12am_A100_v4",
    properties={
        "singularity": {
            "slaTier": "Premium",
            "priority": "High",
            "enableAzmlInt": False,
            "locations": ["ukwest"],
        }
    },
)

if __name__ == "__main__":
    job = command(
        command="sleep infinity",
        environment="azureml:vllm_gemma4:3",
        compute=VC_ARM_ID,
        resources=res_cfg,
        inputs={
            "msndni": Input(
                type="uri_folder",
                path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                mode=InputOutputModes.RW_MOUNT,
            ),
        },
        environment_variables={
            "_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID,
        },
        services={
            "ssh": SshJobService(
                ssh_public_keys=SSH_PUB_KEY,
                nodes="all",
            ),
            "jupyter": JupyterLabJobService(),
        },
    )

    created = ml_client.jobs.create_or_update(job)
    print("=" * 60)
    print("Job submitted successfully!")
    print("=" * 60)
    print(f"Run ID: {created.name}")
    print(f"Studio URL: {created.studio_url}")
    print(f"Command: sleep infinity")
    print(f"Mount: adls_msn_dni_09_rankfun (RW)")
