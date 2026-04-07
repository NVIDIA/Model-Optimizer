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

"""Core logic for the ModelOpt Launcher.

Dataclasses, executor builders, and the job run loop used by launch.py.
"""

import dataclasses
import getpass
import json
import os
import re
from dataclasses import dataclass

import nemo_run as run
import yaml

# ---------------------------------------------------------------------------
# Default environment variables injected into every job
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENT_TITLE = "cicd"


def get_default_env(experiment_title=None):
    """Return (slurm_env, local_env) dicts for the given experiment title."""
    title = experiment_title or DEFAULT_EXPERIMENT_TITLE
    slurm_env = {
        "TRITON_CACHE_DIR": f"/{title}/triton-cache",
        "HF_HOME": f"/{title}/hf-cache",
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "MLM_SKIP_INSTALL": "1",
        "LAUNCH_SCRIPT": "python",
    }
    local_env = {
        "TRITON_CACHE_DIR": f"/{title}/triton-cache",
        "HF_HOME": f"/{title}/hf-cache",
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "MLM_SKIP_INSTALL": "1",
    }
    return slurm_env, local_env


# SlurmConfig type — set by the caller via set_slurm_config_type() before use.
# This allows both slurm.py and launch.py to use their own SlurmConfig class.
_SLURM_CONFIG_TYPE = None
_FACTORY_REGISTRY = {}


def set_slurm_config_type(cls):
    """Register the SlurmConfig dataclass type used by SandboxTask."""
    global _SLURM_CONFIG_TYPE
    _SLURM_CONFIG_TYPE = cls
    # Patch SandboxTask's type annotation so nemo-run's CLI parser can resolve factories
    SandboxTask.__dataclass_fields__["slurm_config"].type = cls
    SandboxTask.__annotations__["slurm_config"] = cls


def register_factory(name, fn):
    """Register a factory function by name for task_configs YAML resolution."""
    _FACTORY_REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# Task and pipeline dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SandboxTask:
    """A single task with a script, slurm config, args, and environment."""

    script: str = None
    slurm_config: object = None  # Patched at runtime by set_slurm_config_type()
    args: list[str] = None
    environment: list[dict[str, str]] = None
    yaml_file: str = None
    skip: bool = False


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


def create_task_from_yaml(yaml_file, factory_lookup):
    """Create a SandboxTask from a YAML config file.

    Args:
        yaml_file: Path to the YAML config.
        factory_lookup: Dict mapping factory names to callable factory functions.
    """
    with open(yaml_file) as file:
        config_from_yaml = yaml.safe_load(file)

    script = config_from_yaml["script"]
    function_name = config_from_yaml["slurm_config"].pop("_factory_")
    slurm_config = factory_lookup[function_name](**config_from_yaml["slurm_config"])
    args = config_from_yaml.get("args", None)
    environment = config_from_yaml.get("environment", None)

    return SandboxTask(script=script, slurm_config=slurm_config, args=args, environment=environment)


@dataclass
class GlobalVariables:
    """Shared variables for <<global_vars.X>> interpolation in pipeline YAMLs."""

    hf_model: str = None
    hf_data: str = None
    hf_local: str = None


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

    # Set by caller — used by create_task_from_yaml
    _factory_lookup: dict = None

    def __post_init__(self):
        """Collect tasks from slots/configs and resolve <<global_vars.X>> references."""
        if self.tasks is None:
            self.tasks = []
            for i in range(5):
                task = getattr(self, f"task_{i}", None)
                if task is not None:
                    self.tasks += [task]
        if self.task_configs is not None:
            lookup = self._factory_lookup or _FACTORY_REGISTRY
            if lookup:
                self.tasks += [
                    create_task_from_yaml(yaml_file=yf, factory_lookup=lookup)
                    for yf in self.task_configs
                ]

        if self.global_vars is not None:
            global_vars_dict = {
                k: v for k, v in dataclasses.asdict(self.global_vars).items() if v is not None
            }

            def _resolve(s):
                """Replace <<global_vars.X>> with the corresponding value."""
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
# Executor builders
# ---------------------------------------------------------------------------


def build_slurm_executor(
    user,
    identity,
    slurm_config,
    experiment_id,
    job_dir,
    task_name,
    packager,
    experiment_title="cicd",
):
    """Build a SlurmExecutor for remote job submission."""
    container_mounts = list(slurm_config.container_mounts or [])

    scratch_dst = "/scratchspace"
    scratch_src = f"{job_dir}/{experiment_title}/{experiment_id}"
    modelopt_dst = slurm_config.modelopt_install_path
    modelopt_src = (
        f"{job_dir}/{experiment_title}/{experiment_id}"
        f"/{task_name}/code/modules/Model-Optimizer/modelopt"
    )
    modelopt_recipes_dst = os.path.join(
        os.path.dirname(os.path.normpath(slurm_config.modelopt_install_path)),
        "modelopt_recipes",
    )
    modelopt_recipes_src = (
        f"{job_dir}/{experiment_title}/{experiment_id}"
        f"/{task_name}/code/modules/Model-Optimizer/modelopt_recipes"
    )
    container_mounts += [
        f"{scratch_src}:{scratch_dst}",
        f"{modelopt_src}:{modelopt_dst}",
        f"{modelopt_recipes_src}:{modelopt_recipes_dst}",
        f"{job_dir}/{experiment_title}:/{experiment_title}",
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
        time=slurm_config.time,
        mem="0",
        retries=0,
        packager=packager,
        srun_args=slurm_config.srun_args,
    )
    return executor


def build_docker_executor(
    hf_local,
    slurm_config,
    experiment_id,
    job_dir,
    task_name,
    packager,
    modelopt_src_path=None,
    experiment_title="cicd",
):
    """Build a DockerExecutor for local GPU jobs."""
    if slurm_config.local:
        container_mounts = list(slurm_config.container_mounts or [])
    else:
        container_mounts = []
    container_mounts += [f"{hf_local}:/hf-local"]

    scratch_dst = "/scratchspace"
    scratch_src = os.path.join(job_dir, experiment_title, experiment_id, task_name)
    os.makedirs(scratch_src, exist_ok=True)
    modelopt_dst = slurm_config.modelopt_install_path
    if modelopt_src_path is None:
        modelopt_src_path = os.path.join(os.getcwd(), "modules/Model-Optimizer/modelopt")
    modelopt_recipes_dst = os.path.join(
        os.path.dirname(os.path.normpath(slurm_config.modelopt_install_path)),
        "modelopt_recipes",
    )
    modelopt_recipes_src_path = os.path.join(os.path.dirname(modelopt_src_path), "modelopt_recipes")
    exp_title_src = os.path.join(job_dir, experiment_title)
    os.makedirs(exp_title_src, exist_ok=True)
    container_mounts += [
        f"{scratch_src}:{scratch_dst}",
        f"{modelopt_src_path}:{modelopt_dst}",
        f"{modelopt_recipes_src_path}:{modelopt_recipes_dst}",
        f"{exp_title_src}:/{experiment_title}",
    ]

    executor = run.DockerExecutor(
        num_gpus=-1,
        runtime="nvidia",
        ipc_mode="host",
        container_image=slurm_config.container,
        volumes=container_mounts,
        additional_kwargs={"user": f"{os.getuid()}:{os.getgid()}"},
        packager=packager,
    )
    return executor


# ---------------------------------------------------------------------------
# Version reporting
# ---------------------------------------------------------------------------


def _git_info(path):
    """Get git commit hash and branch for a directory."""
    import subprocess  # nosec B404

    try:
        commit = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        branch = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return commit, branch
    except Exception:
        return "unknown", "unknown"


def report_versions(base_dir):
    """Print git commit and branch for the launcher and all submodules."""
    print("=" * 60)
    print("Version Report")
    print("=" * 60)

    # Launcher / repo root
    commit, branch = _git_info(base_dir)
    print(f"  {'Launcher':<30} {commit:<12} ({branch})")

    # Submodules
    modules_dir = os.path.join(base_dir, "modules")
    if os.path.isdir(modules_dir):
        for name in sorted(os.listdir(modules_dir)):
            sub_path = os.path.join(modules_dir, name)
            if os.path.exists(os.path.join(sub_path, ".git")):
                commit, branch = _git_info(sub_path)
                print(f"  {name:<30} {commit:<12} ({branch})")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Shared job run loop
# ---------------------------------------------------------------------------


def run_jobs(
    job_table,
    hf_local,
    user,
    identity,
    job_dir,
    packager,
    default_slurm_env,
    default_local_env,
    experiment_title="cicd",
    detach=False,
    test_level=0,
    modelopt_src_path=None,
    base_dir=None,
):
    """Run all jobs in job_table.

    Args:
        job_table: Dict mapping job_name -> SandboxPipeline.
        hf_local: Path to local HF cache (None for remote Slurm).
        user: SSH user.
        identity: SSH identity file.
        job_dir: Base directory for job artifacts.
        packager: PatternPackager instance.
        default_slurm_env: Default env vars for Slurm jobs.
        default_local_env: Default env vars for local Docker jobs.
        experiment_title: Experiment title (e.g., "cicd" or "modelopt").
        detach: Whether to detach from the experiment.
        test_level: Only run jobs with test_level <= this value.
        modelopt_src_path: Path to modelopt source for Docker mounts.
        base_dir: Base directory for version reporting (default: cwd).
    """
    report_versions(base_dir or os.getcwd())

    for job_name, job in job_table.items():
        if job.test_level > test_level:
            job.skip = True
        if job.skip:
            continue

        dependency = None
        exp = run.Experiment(experiment_title, log_level="INFO")
        job.experiment = exp

        with exp:
            for task_id, task in enumerate(job.tasks):
                if task.skip:
                    print(f"job {job_name} task {task_id}: skipped")
                    continue
                task_name = f"{job_name}_{task_id}"
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
                    executor = build_docker_executor(
                        hf_local,
                        task.slurm_config,
                        exp._id,
                        job_dir,
                        task_name,
                        packager,
                        modelopt_src_path,
                        experiment_title,
                    )
                    task_env.update(default_local_env)
                else:
                    executor = build_slurm_executor(
                        user,
                        identity,
                        task.slurm_config,
                        exp._id,
                        job_dir,
                        task_name,
                        packager,
                        experiment_title,
                    )
                    task_env.update(default_slurm_env)

                task_instance = run.Script(task.script, args=task_args, env=task_env)
                print(f"job {job_name} task {task_id} slurm_config: {task.slurm_config}")

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

            exp.run(detach=detach)

        # Write metadata for downstream tools
        metadata = {
            "experiment_id": exp._id,
            "job_name": job_name,
            "allow_to_fail": job.allow_to_fail,
            "note": job.note,
        }
        metadata_path = os.path.join("experiments", experiment_title, exp._id, "metadata.json")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
