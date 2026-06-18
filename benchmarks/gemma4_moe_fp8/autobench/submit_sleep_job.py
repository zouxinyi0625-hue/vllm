"""
Create a Singularity job with mounted storage and sleep infinity.
Holds the machine for interactive use.

Usage:
    python submit_sleep_job.py
"""
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, command
from azure.ai.ml.entities import JobResourceConfiguration, SshJobService, JupyterLabJobService
from azure.ai.ml.constants import InputOutputModes

# Workspace
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
    resource_group_name="AMLStudio",
    workspace_name="NewsFeedL2_AML",
)

# Virtual Cluster
VC_ARM_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/rg-cs-ranking-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
)

# Managed Identity (required by new Singularity policy)
UAI_RESOURCE_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/AMLStudio"
    "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"
)

# SSH public key for interactive access (paste your key here)
SSH_PUB_KEY = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCpOz0QGUOBnEqMn+DwzbltVytWcFB/J10EpA0Rf5UXMtScYFKKYAi50qyhhdT5nj0LharII8p42w5MGPMLepqey6oFkVjDWrkTmzYe2nfkZpT9+GjGIEnbSvSL5CidsSWwDTzsgb5eLu0bExWHRwXscTLIfYQBNurdinw+z6k96DS1W4YTclJveoKFMJTT0ZpNd8FnGlQeJuO++xR1zVxK938rGEHO1bY3Aph3PdgsTYliJvYNqihM/p+az8UK+zRNwRdbE175UZALbuD77mVuF8hG19ggLxi3HeyO9RE8t9VhNn6nyZDtMQtRxpgqx83tYSXqUatMwoHXiONQ1gMVbKhW6kNb7vvwCAOmUU/In4psgM1RiEv/VNVSV/9CusYDsCOvGPT0mOliaRMebA2KyHPmjkKdQNW8FTUM9No1cFsigMtsj84PwjcZYbPGFCTbifutUjav7p0PN+9AyLCOEyikX9SVGq06Qo4/oW5/aRMOQRwRala9S4pQjvj3kduQp8jNITMW+yn5AI6lgE457rbSmMpE6YxhVgVQjF1Mb6szxrQoMntqTW5O1ypr691vcnk9yph9fv9BVc+b+wdFbe8qHoYCDtDSBPYMq7GdwYqoDkpWdi5VBYRiMOGPNH5PB6S6xLBLD3Ybm+tHW/vvT6d/qjd3wYvg2dAuGwc/Mw== xinyizou@microsoft.com"

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
        environment="azureml:vllm_gemma4:5",
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
