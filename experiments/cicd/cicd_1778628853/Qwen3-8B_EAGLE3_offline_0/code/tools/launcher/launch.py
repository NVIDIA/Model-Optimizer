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
    uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml --yes
    uv run launch.py --yaml examples/Qwen/Qwen3-8B/megatron_lm_ptq.yaml hf_local=/mnt/hf-local --yes

Environment variables:
    SLURM_HOST          Slurm login node hostname (required for remote jobs)
    SLURM_ACCOUNT       Slurm account/partition billing (default: from YAML)
    SLURM_JOB_DIR       Remote directory for job artifacts
    SLURM_HF_LOCAL      Path to HuggingFace model cache on the cluster
    HF_TOKEN            HuggingFace API token
    NEMORUN_HOME        NeMo Run home directory (default: current working directory)
"""

import getpass
import os
import subprocess  # nosec B404
import warnings

import nemo_run as run
from core import SandboxPipeline, get_default_env, register_factory, run_jobs, set_slurm_config_type
from slurm_config import SlurmConfig, slurm_factory

set_slurm_config_type(SlurmConfig)
register_factory("slurm_factory", slurm_factory)

# ---------------------------------------------------------------------------
# Launcher-specific configuration
# ---------------------------------------------------------------------------

LAUNCHER_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOPT_ROOT = os.path.dirname(os.path.dirname(LAUNCHER_DIR))

# Ensure modules/Model-Optimizer symlink exists (points to parent Model-Optimizer root)
_mo_symlink = os.path.join(LAUNCHER_DIR, "modules", "Model-Optimizer")
if not os.path.exists(_mo_symlink):
    os.makedirs(os.path.join(LAUNCHER_DIR, "modules"), exist_ok=True)
    os.symlink(os.path.relpath(MODELOPT_ROOT, os.path.join(LAUNCHER_DIR, "modules")), _mo_symlink)

EXPERIMENT_TITLE = "cicd"
DEFAULT_SLURM_ENV, DEFAULT_LOCAL_ENV = get_default_env(EXPERIMENT_TITLE)

packager = run.PatternPackager(
    include_pattern=[
        "modules/Megatron-LM/megatron/*",
        "modules/Megatron-LM/examples/*",
        "modules/Megatron-LM/*.py",
        "modules/Model-Optimizer/modelopt/*",
        "modules/Model-Optimizer/modelopt_recipes/*",
        "modules/Model-Optimizer/examples/*",
        "examples/*",
        "common/*",
    ],
    relative_path=[LAUNCHER_DIR] * 8,
)

MODELOPT_SRC_PATH = os.path.join(LAUNCHER_DIR, "modules/Model-Optimizer/modelopt")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@run.cli.entrypoint
def launch(
    job_name: str = "01_job",
    job_dir: str = os.environ.get("SLURM_JOB_DIR", os.path.expanduser("~/experiments")),
    pipeline: SandboxPipeline = None,
    hf_local: str = None,  # noqa: RUF013
    user: str = getpass.getuser(),
    identity: str = None,  # noqa: RUF013
    detach: bool = False,
    clean: bool = False,
) -> None:
    """Launch ModelOpt jobs on Slurm or locally with Docker."""
    if clean:
        examples_dir = os.path.join(_mo_symlink, "examples")
        print(f"Cleaning {examples_dir} with git clean -xdf ...")
        subprocess.run(["git", "clean", "-xdf", "."], cwd=examples_dir, check=True)  # nosec B603 B607

    if "NEMORUN_HOME" not in os.environ:
        warnings.warn("NEMORUN_HOME is not set. Defaulting to current working directory.")
    run.config.set_nemorun_home(os.environ.get("NEMORUN_HOME", os.getcwd()))

    if hf_local is not None:
        job_dir = os.path.join(os.getcwd(), "local_experiments")

    job_table = {}
    if pipeline is not None:
        job_table[job_name] = pipeline
    else:
        print("No pipeline provided. Use pipeline=@<yaml> or --yaml <yaml>.")
        return

    run_jobs(
        job_table=job_table,
        hf_local=hf_local,
        user=user,
        identity=identity,
        job_dir=job_dir,
        packager=packager,
        default_slurm_env=DEFAULT_SLURM_ENV,
        default_local_env=DEFAULT_LOCAL_ENV,
        experiment_title=EXPERIMENT_TITLE,
        detach=detach,
        modelopt_src_path=MODELOPT_SRC_PATH,
        base_dir=LAUNCHER_DIR,
    )


if __name__ == "__main__":
    run.cli.main(launch)
