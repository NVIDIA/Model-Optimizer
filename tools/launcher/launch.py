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
    SLURM_USER          Remote Slurm/SSH username (default: local login name)
    SLURM_HF_LOCAL      Path to HuggingFace model cache on the cluster
    HF_TOKEN            HuggingFace API token
    NEMORUN_HOME        NeMo Run home directory (default: current working directory)
"""

import getpass
import glob
import os
import subprocess  # nosec B404 - required for explicit git clean command; no shell is used.
import warnings

import modelopt_launcher as _pkg
import nemo_run as run
from modelopt_launcher.core import (
    SandboxPipeline,
    get_default_env,
    register_factory,
    run_jobs,
    set_slurm_config_type,
)
from modelopt_launcher.slurm_config import SlurmConfig, slurm_factory

set_slurm_config_type(SlurmConfig)
register_factory("slurm_factory", slurm_factory)

# ---------------------------------------------------------------------------
# Launcher-specific configuration
# ---------------------------------------------------------------------------

LAUNCHER_DIR = _pkg.PACKAGE_DIR  # tools/launcher/ (dev or installed)

# Detect dev checkout by probing the actual MODELOPT_ROOT, not the symlink
# path (which doesn't exist yet in a clean checkout). When running as an
# installed console script the cluster container already has modelopt
# pre-installed, so we skip packaging it from source.
MODELOPT_ROOT = os.path.dirname(os.path.dirname(LAUNCHER_DIR))
_has_modelopt_src = os.path.isdir(os.path.join(MODELOPT_ROOT, "modelopt"))

# Symlink path used by the PatternPackager to resolve modules/Model-Optimizer/*
# patterns; only valid in dev mode. Initialized to None so --clean in installed
# mode gets a clear error instead of a NameError.
_mo_symlink: str | None = None

if _has_modelopt_src:
    _mo_symlink = os.path.join(LAUNCHER_DIR, "modules", "Model-Optimizer")
    if not os.path.exists(_mo_symlink):
        os.makedirs(os.path.join(LAUNCHER_DIR, "modules"), exist_ok=True)
        os.symlink(
            os.path.relpath(MODELOPT_ROOT, os.path.join(LAUNCHER_DIR, "modules")), _mo_symlink
        )

_modelopt_src = os.path.join(LAUNCHER_DIR, "modules", "Model-Optimizer", "modelopt")

EXPERIMENT_TITLE = "cicd"
DEFAULT_SLURM_ENV, DEFAULT_LOCAL_ENV = get_default_env(EXPERIMENT_TITLE)

_include_pattern = []
_relative_path = []


def _add_package_path(path: str) -> None:
    """Add an existing package path using LAUNCHER_DIR as the tar root."""
    if os.path.exists(path):
        _include_pattern.append(path)
        _relative_path.append(LAUNCHER_DIR)


def _add_package_glob(pattern: str) -> None:
    """Expand a glob and add each matching path to the launcher package."""
    for path in sorted(glob.glob(pattern)):
        _add_package_path(path)


_add_package_path(os.path.join(LAUNCHER_DIR, "examples"))
_add_package_path(os.path.join(LAUNCHER_DIR, "common"))

if _has_modelopt_src:
    _add_package_path(os.path.join(LAUNCHER_DIR, "modules/Megatron-LM/megatron"))
    _add_package_path(os.path.join(LAUNCHER_DIR, "modules/Megatron-LM/examples"))
    _add_package_glob(os.path.join(LAUNCHER_DIR, "modules/Megatron-LM/*.py"))
    _add_package_path(os.path.join(LAUNCHER_DIR, "modules/Model-Optimizer/modelopt"))
    _add_package_path(os.path.join(LAUNCHER_DIR, "modules/Model-Optimizer/modelopt_recipes"))
    _add_package_path(os.path.join(LAUNCHER_DIR, "modules/Model-Optimizer/examples"))

packager = run.PatternPackager(
    include_pattern=_include_pattern,
    relative_path=_relative_path,
)

MODELOPT_SRC_PATH = _modelopt_src if _has_modelopt_src else None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@run.cli.entrypoint
def launch(
    job_name: str = "01_job",
    job_dir: str = os.environ.get("SLURM_JOB_DIR", os.path.expanduser("~/experiments")),
    pipeline: SandboxPipeline = None,
    hf_local: str = None,  # noqa: RUF013
    user: str | None = None,
    identity: str = None,  # noqa: RUF013
    detach: bool = False,
    clean: bool = False,
) -> None:
    """Launch ModelOpt jobs on Slurm or locally with Docker."""
    if user is None:
        user = os.environ.get("SLURM_USER", getpass.getuser())

    if clean:
        if _mo_symlink is None:
            raise ValueError("--clean requires a dev checkout; modelopt source not found.")
        examples_dir = os.path.join(_mo_symlink, "examples")
        print(f"Cleaning {examples_dir} with git clean -xdf ...")
        subprocess.run(  # nosec B603 B607 - fixed git CLI argv; no shell.
            ["git", "clean", "-xdf", "."],
            cwd=examples_dir,
            check=True,
        )

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


def main() -> None:
    """Console script entry point for the ``modelopt-launcher`` command."""
    run.cli.main(launch)


if __name__ == "__main__":
    main()
