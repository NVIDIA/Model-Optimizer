# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Slurm configuration and factory for the ModelOpt Launcher."""

import os
from dataclasses import dataclass

import nemo_run as run


@dataclass
class SlurmConfig:
    """Cluster-agnostic Slurm configuration.

    Users define cluster details in their YAML configs or override via CLI.
    No internal cluster defaults are embedded here.
    """

    host: str = None
    port: int = 22
    account: str = None
    partition: str = "batch"
    container: str = None
    modelopt_install_path: str = "/usr/local/lib/python3.12/dist-packages/modelopt"
    container_mounts: list[str] = None
    srun_args: list[str] = None
    array: str = None
    nodes: int = 1
    ntasks_per_node: int = 1
    gpus_per_node: int = 1
    local: bool = False


@run.cli.factory
@run.autoconvert
def slurm_factory(
    host: str = os.environ.get("SLURM_HOST", ""),
    account: str = os.environ.get("SLURM_ACCOUNT", ""),
    partition: str = "batch",
    nodes: int = 1,
    ntasks_per_node: int = 1,
    gpus_per_node: int = 1,
    container: str = "nvcr.io/nvidia/tensorrt-llm/release:1.2.0",
    modelopt_install_path: str = "/usr/local/lib/python3.12/dist-packages/modelopt",
    container_mounts: list[str] = [
        "{}:/hf-local".format(os.environ.get("SLURM_HF_LOCAL", "/hf-local")),
    ],
    srun_args: list[str] = ["--no-container-mount-home"],
    array: str = None,  # noqa: RUF013
) -> SlurmConfig:
    """Generic Slurm factory — configure via environment variables or CLI overrides."""
    return SlurmConfig(
        host=host,
        account=account,
        partition=partition,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        gpus_per_node=gpus_per_node,
        container=container,
        modelopt_install_path=modelopt_install_path,
        container_mounts=container_mounts,
        srun_args=srun_args,
        array=array,
    )
