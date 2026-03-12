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

"""ModelOpt Launcher — submit quantization, training, and evaluation jobs to Slurm clusters.

Usage:
    uv run launch.py task=@configs/quantize/Qwen3-8B.yaml --yes
    uv run launch.py pipeline=@configs/pipeline/eagle3.yaml --yes
    uv run launch.py task=@configs/quantize/Qwen3-8B.yaml hf_local=/mnt/hf-local --yes

Environment variables:
    SLURM_HOST          Slurm login node hostname (required for remote jobs)
    SLURM_ACCOUNT       Slurm account/partition billing (default: from YAML)
    SLURM_JOB_DIR       Remote directory for job artifacts
    SLURM_HF_LOCAL      Path to HuggingFace model cache on the cluster
    HF_TOKEN            HuggingFace API token
    NEMORUN_HOME        NeMo Run home directory (default: current working directory)
"""

import dataclasses
import getpass
import json
import os
import re
import warnings
from dataclasses import dataclass

import nemo_run as run
import yaml

# ---------------------------------------------------------------------------
# Slurm configuration
# ---------------------------------------------------------------------------


@dataclass
class SlurmConfig:
    """Cluster-agnostic Slurm configuration.

    Users define cluster details in their YAML configs or override via CLI.
    No internal cluster defaults are embedded here.
    """

    host: str | None = None
    port: int = 22
    account: str | None = None
    partition: str = "batch"
    container: str | None = None
    modelopt_install_path: str = "/usr/local/lib/python3.12/dist-packages/modelopt"
    container_mounts: list[str] | None = None
    srun_args: list[str] | None = None
    array: str | None = None
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
    container: str = "nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5",
    modelopt_install_path: str = "/usr/local/lib/python3.12/dist-packages/modelopt",
    container_mounts: list[str] | None = None,
    srun_args: list[str] | None = None,
    array: str | None = None,
) -> SlurmConfig:
    """Generic Slurm factory — configure via environment variables or CLI overrides."""
    if container_mounts is None:
        hf_local = os.environ.get("SLURM_HF_LOCAL", "/hf-local")
        container_mounts = ["{}:/hf-local".format(hf_local)]
    if srun_args is None:
        srun_args = ["--no-container-mount-home"]
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


# ---------------------------------------------------------------------------
# Default environment variables injected into every job
# ---------------------------------------------------------------------------

DEFAULT_SLURM_ENV = {
    "HF_HOME": "/hf-cache",
    "HF_TOKEN": os.getenv("HF_TOKEN", ""),
    "MLM_SKIP_INSTALL": "1",
    "LAUNCH_SCRIPT": "python",
}

DEFAULT_LOCAL_ENV = {
    "HF_HOME": "/hf-cache",
    "HF_TOKEN": os.getenv("HF_TOKEN", ""),
    "MLM_SKIP_INSTALL": "1",
}


# ---------------------------------------------------------------------------
# Task and pipeline dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SandboxTask:
    """A single task with a script, slurm config, args, and environment."""

    script: str = None
    slurm_config: SlurmConfig = None
    args: list[str] = None
    environment: list[dict] = None
    yaml_file: str = None


@dataclass
class SandboxTask0(SandboxTask):
    """Task slot 0 in a pipeline."""


@dataclass
class SandboxTask1(SandboxTask):
    """Task slot 1 in a pipeline."""


@dataclass
class SandboxTask2(SandboxTask):
    """Task slot 2 in a pipeline."""


@dataclass
class SandboxTask3(SandboxTask):
    """Task slot 3 in a pipeline."""


@dataclass
class SandboxTask4(SandboxTask):
    """Task slot 4 in a pipeline."""


def create_task_from_yaml(yaml_file: str) -> SandboxTask:
    """Create a SandboxTask from a YAML config file."""
    with open(yaml_file) as file:
        config_from_yaml = yaml.safe_load(file)

    script = config_from_yaml["script"]
    function_name = config_from_yaml["slurm_config"].pop("_factory_")
    slurm_config = globals()[function_name](**config_from_yaml["slurm_config"])
    args = config_from_yaml.get("args", None)
    environment = config_from_yaml.get("environment", None)

    return SandboxTask(script=script, slurm_config=slurm_config, args=args, environment=environment)


@dataclass
class GlobalVariables:
    """Shared variables for <<global_vars.X>> interpolation in pipeline YAMLs."""

    hf_model: str = None
    hf_data: str = None


@dataclass
class SandboxPipeline:
    """A multi-task pipeline with shared global variables and task dependencies."""

    global_vars: GlobalVariables = None

    task_0: SandboxTask0 = None
    task_1: SandboxTask1 = None
    task_2: SandboxTask2 = None
    task_3: SandboxTask3 = None
    task_4: SandboxTask4 = None
    tasks: list[SandboxTask] = None

    test_level: int = 0
    allow_to_fail: bool = False
    skip: bool = False
    note: str = ""
    task_configs: list[str] = None
    experiment = None

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
            for i in range(5):
                task = getattr(self, "task_{}".format(i), None)
                if task is not None:
                    self.tasks += [task]
        if self.task_configs is not None:
            self.tasks += [
                create_task_from_yaml(yaml_file=yaml_file) for yaml_file in self.task_configs
            ]

        if self.global_vars is not None:
            global_vars_dict = {
                k: v for k, v in dataclasses.asdict(self.global_vars).items() if v is not None
            }

            def _resolve(s):
                if not isinstance(s, str):
                    return s
                return re.sub(
                    r"<<global_vars\.(\w+)>>",
                    lambda m: global_vars_dict.get(m.group(1), m.group(0)),
                    s,
                )

            for task in self.tasks:
                if task.environment:
                    if isinstance(task.environment, list):
                        task.environment = [
                            {k: _resolve(v) for k, v in item.items()} for item in task.environment
                        ]
                    else:
                        task.environment = {k: _resolve(v) for k, v in task.environment.items()}
                if task.args:
                    task.args = [_resolve(a) for a in task.args]


# ---------------------------------------------------------------------------
# Code packager — sync only the necessary source trees to the cluster
# ---------------------------------------------------------------------------

# Resolve paths relative to Model-Optimizer root (parent of launcher/)
LAUNCHER_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOPT_ROOT = os.path.dirname(LAUNCHER_DIR)

# All paths relative to LAUNCHER_DIR so code/ mirrors the launcher directory.
# This produces the same layout as nmm-sandbox's slurm.py:
#   code/modules/Megatron-LM/megatron/...
#   code/modules/Model-Optimizer/modelopt/...
#   code/services/...
packager = run.PatternPackager(
    include_pattern=[
        "modules/Megatron-LM/megatron/*",
        "modules/Megatron-LM/examples/*",
        "modules/Megatron-LM/*.py",
        "modules/Model-Optimizer/modelopt/*",
        "modules/Model-Optimizer/examples/*",
        "services/*",
        "tests/*",
    ],
    relative_path=[LAUNCHER_DIR] * 7,
)


# ---------------------------------------------------------------------------
# Executor builders
# ---------------------------------------------------------------------------


def get_slurm_executor(user, identity, slurm_config, experiment_id, job_dir, task_name):
    """Build a SlurmExecutor for remote job submission."""
    container_mounts = slurm_config.container_mounts or []

    scratch_dst = "/scratchspace"
    scratch_src = job_dir + "/cicd/" + experiment_id
    modelopt_dst = slurm_config.modelopt_install_path
    modelopt_src = (
        job_dir
        + "/cicd/"
        + experiment_id
        + "/{}/code/modules/Model-Optimizer/modelopt".format(task_name)
    )
    container_mounts = [
        *container_mounts,
        scratch_src + ":" + scratch_dst,
        modelopt_src + ":" + modelopt_dst,
    ]

    tunnel = run.SSHTunnel(
        host=slurm_config.host,
        user=getpass.getuser() if user is None else user,
        port=slurm_config.port,
        job_dir=job_dir,
        identity=identity,
    )

    executor = run.SlurmExecutor(
        account=slurm_config.account,
        partition=slurm_config.partition,
        ntasks_per_node=slurm_config.ntasks_per_node,
        gpus_per_node=slurm_config.gpus_per_node,
        nodes=slurm_config.nodes,
        tunnel=tunnel,
        container_image=slurm_config.container,
        container_mounts=container_mounts,
        array=slurm_config.array,
        time="04:00:00",
        mem="0",
        retries=0,
        packager=packager,
        srun_args=slurm_config.srun_args,
    )
    return executor


def get_docker_executor(hf_local, slurm_config, experiment_id, job_dir, task_name):
    """Build a DockerExecutor for local GPU jobs."""
    if slurm_config.local:
        container_mounts = list(slurm_config.container_mounts or [])
    else:
        container_mounts = []
    container_mounts += [hf_local + ":/hf-local", job_dir + "/cicd:/cicd"]

    scratch_dst = "/scratchspace"
    scratch_src = job_dir + "/cicd/" + experiment_id + "/" + task_name
    modelopt_dst = slurm_config.modelopt_install_path
    modelopt_src = os.path.join(LAUNCHER_DIR, "modules/Model-Optimizer/modelopt")
    container_mounts += [scratch_src + ":" + scratch_dst, modelopt_src + ":" + modelopt_dst]

    executor = run.DockerExecutor(
        num_gpus=-1,
        runtime="nvidia",
        ipc_mode="host",
        container_image=slurm_config.container,
        volumes=container_mounts,
        additional_kwargs={"user": "{}:{}".format(os.getuid(), os.getgid())},
        packager=packager,
    )
    return executor


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


@run.cli.entrypoint
def launch(
    ctx: run.cli.RunContext,
    job_name: str = "01_job",
    job_dir: str = os.environ.get("SLURM_JOB_DIR", os.path.expanduser("~/experiments")),
    task: SandboxTask | None = None,
    pipeline: SandboxPipeline | None = None,
    hf_local: str | None = None,
    user: str = getpass.getuser(),
    identity: str | None = None,
) -> None:
    """Launch ModelOpt jobs on Slurm or locally with Docker.

    Args:
        job_name: Name of the job.
        job_dir: Remote directory for job artifacts.
        task: Single task config (from YAML).
        pipeline: Multi-task pipeline config (from YAML).
        hf_local: Path to local HF cache (enables local Docker execution).
        user: SSH user for Slurm tunnel.
        identity: SSH identity file for Slurm tunnel.
    """
    if "NEMORUN_HOME" not in os.environ:
        warnings.warn("NEMORUN_HOME is not set. Defaulting to current working directory.")
    run.config.set_nemorun_home(os.environ.get("NEMORUN_HOME", os.getcwd()))

    if hf_local is not None:
        job_dir = os.getcwd() + "/experiments"

    job_table = {}

    if task is not None:
        job_table[job_name] = SandboxPipeline(tasks=[task])
    elif pipeline is not None:
        job_table[job_name] = pipeline
    else:
        print("No task or pipeline provided. Use task=@<yaml> or pipeline=@<yaml>.")
        return

    for job_name, job in job_table.items():  # noqa: PLR1704
        if job.skip:
            continue

        dependency = None
        exp = run.Experiment("modelopt", log_level="INFO")
        job.experiment = exp

        with exp:
            for task_id, task in enumerate(job.tasks):  # noqa: PLR1704
                task_name = job_name + "_" + str(task_id)
                task_args = [] if task.args is None else task.args

                task_env = {}
                if task.environment is not None:
                    if isinstance(task.environment, list):
                        for item in task.environment:
                            task_env.update(item.items())
                    else:
                        task_env = task.environment
                for k, v in task_env.items():
                    task_env[k] = "" if v is None else str(v)
                if hf_local is not None:
                    executor = get_docker_executor(
                        hf_local, task.slurm_config, exp._id, job_dir, task_name
                    )
                    task_env.update(DEFAULT_LOCAL_ENV)
                else:
                    executor = get_slurm_executor(
                        user, identity, task.slurm_config, exp._id, job_dir, task_name
                    )
                    task_env.update(DEFAULT_SLURM_ENV)

                task_instance = run.Script(task.script, args=task_args, env=task_env)
                print(
                    "job {} task {} slurm_config: {}".format(job_name, task_id, task.slurm_config)
                )

                if dependency is None:
                    dependency = exp.add(
                        task_instance, tail_logs=True, name=task_name, executor=executor
                    )
                else:
                    dependency = exp.add(
                        task_instance,
                        tail_logs=True,
                        name=task_name,
                        executor=executor,
                        dependencies=[dependency],
                    )

            exp.run(detach=ctx.detach)

        # Write metadata for downstream tools
        metadata = {
            "experiment_id": exp._id,
            "job_name": job_name,
            "allow_to_fail": job.allow_to_fail,
            "note": job.note,
        }
        metadata_path = os.path.join("experiments", "modelopt", exp._id, "metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)


if __name__ == "__main__":
    run.cli.main(launch)
