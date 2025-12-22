# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
from dataclasses import dataclass, field

import nemo_run as run
from nemo.collections import llm


@dataclass
class SlurmConfig:
    """Configuration for SlurmExecutor."""

    account: str = ""  # Your Slurm account
    partition_cpu: str = ""  # Slurm CPU partition to use
    partition_gpu: str = ""  # Slurm GPU partition to use
    time: str = ""  # Job time limit (HH:MM:SS)
    container_image: str = ""  # Container image for jobs
    env_vars: dict[str, str] = field(default_factory=dict)  # Environment variables to set
    container_mounts: list[str] = field(default_factory=list)  # Container mounts
    use_local_tunnel: bool = False  # Set to True if running from within the cluster
    host: str = ""  # Required for SSH tunnel: Slurm cluster hostname
    user: str = ""  # Required for SSH tunnel: Your username
    job_dir: str = ""  # Required for SSH tunnel: Directory to store runs on cluster
    identity: str | None = None  # Optional for SSH tunnel: Path to SSH key for authentication

    def __post_init__(self):
        """Validate the configuration and raise descriptive errors."""
        if not self.account:
            raise ValueError("SlurmConfig.account must be set to your actual Slurm account")
        if not self.partition_cpu:
            raise ValueError("SlurmConfig.partition_cpu must be set")
        if not self.partition_gpu:
            raise ValueError("SlurmConfig.partition_gpu must be set")
        if not self.time:
            raise ValueError("SlurmConfig.time must be set to job time limit (e.g., '02:00:00')")
        if not self.container_image:
            raise ValueError("SlurmConfig.container_image must be set to container image for jobs")
        if not self.use_local_tunnel:
            # Only validate SSH tunnel settings if not using local tunnel
            if not self.host:
                raise ValueError(
                    "SlurmConfig.host must be set to your actual cluster hostname when using SSH tunnel"
                )
            if not self.user:
                raise ValueError(
                    "SlurmConfig.user must be set to your actual username when using SSH tunnel"
                )
            if not self.job_dir:
                raise ValueError(
                    "SlurmConfig.job_dir must be set to directory for storing runs on cluster"
                )

        self.env_vars |= {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",  # Disable GPU communication/computation overlap for performance
            "TRANSFORMERS_OFFLINE": "1",  # Disable online downloads from HuggingFace
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
            "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        }


def create_slurm_executor(
    slurm_cfg: SlurmConfig, nodes: int = 1, ntasks_per_node: int = 1, num_gpus: int = 0
):
    # Configure tunnel
    if slurm_cfg.use_local_tunnel:
        # Use LocalTunnel when already on the cluster
        tunnel = run.LocalTunnel(job_dir=slurm_cfg.job_dir)
    else:
        # Use SSH tunnel when launching from local machine
        tunnel = run.SSHTunnel(
            host=slurm_cfg.host,
            user=slurm_cfg.user,
            job_dir=slurm_cfg.job_dir,
            identity=slurm_cfg.identity,  # can be None
        )

    if num_gpus > 0:
        return run.SlurmExecutor(
            account=slurm_cfg.account,
            partition=slurm_cfg.partition_gpu,
            ntasks_per_node=ntasks_per_node,
            gpus_per_node=num_gpus,
            nodes=nodes,
            tunnel=tunnel,
            container_image=slurm_cfg.container_image,
            container_mounts=slurm_cfg.container_mounts,
            time=slurm_cfg.time,
            packager=run.GitArchivePackager(),
            mem="0",
            gres=f"gpu:{num_gpus}",
        )
    else:
        return run.SlurmExecutor(
            account=slurm_cfg.account,
            partition=slurm_cfg.partition_cpu,
            nodes=nodes,
            tunnel=tunnel,
            container_image=slurm_cfg.container_image,
            container_mounts=slurm_cfg.container_mounts,
            time=slurm_cfg.time,
            packager=run.GitArchivePackager(),
            mem="0",
        )


def get_finetune_recipe(recipe_name: str):
    if not hasattr(getattr(llm, recipe_name), "finetune_recipe"):
        raise ValueError(f"Recipe {recipe_name} does not have a Fine-Tuning recipe")
    return getattr(llm, recipe_name).finetune_recipe(peft_scheme=None)


def read_chat_template(template_path: str):
    with open(template_path) as f:
        return f.read().strip()


def download_hf_dataset(dataset_name: str, output_dir: str | None = None):
    """Download a dataset from HuggingFace Hub using huggingface-cli."""
    cmd = ["huggingface-cli", "download", dataset_name, "--repo-type", "dataset"]

    if output_dir:
        cmd.extend(["--local-dir", output_dir])

    subprocess.run(cmd, check=True)
    print(f"Successfully downloaded dataset: {dataset_name}")
