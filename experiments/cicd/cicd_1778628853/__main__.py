"""NMM Sandbox CI/CD orchestrator — internal alternative to Model-Optimizer/tools/launcher/launch.py.

Shares core logic (dataclasses, executor builders, run loop) with the public launcher
via modules/Model-Optimizer/tools/launcher/core.py. This file adds internal cluster factories,
CI batch mode (job_yaml), test_level filtering, and internal defaults.
"""

import getpass
import os
import sys
sys.path.insert(0, '/usr/local/lib/python3.12/site-packages')
import warnings

import nemo_run as run

# Add the launcher to sys.path so we can import core.py
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "tools", "launcher"))

from core import SandboxPipeline, SandboxTask, run_jobs, set_slurm_config_type, register_factory, get_default_env  # noqa: E402
from slurm_config import (  # noqa: E402
    SlurmConfig,
    slurm_factory,
)

set_slurm_config_type(SlurmConfig)

# Register the slurm factory so task_configs YAMLs can reference it by name
register_factory("slurm_factory", slurm_factory)

# ---------------------------------------------------------------------------
# nmm-sandbox-specific configuration
# ---------------------------------------------------------------------------

EXPERIMENT_TITLE = "cicd"
DEFAULT_SLURM_ENV, DEFAULT_LOCAL_ENV = get_default_env(EXPERIMENT_TITLE)


def _ensure_launcher_nvrx_install() -> None:
    """Idempotently rewrite the Model-Optimizer launcher's service_utils.sh
    util_install_extra_dep so it (a) installs nvidia-resiliency-ext from
    HEAD (container's pinned version privatized get_write_results_queue;
    HEAD keeps the public alias), (b) falls back to SLURM_LOCALID when
    OMPI/PMIX vars aren't set so only one rank per node installs, and
    (c) uses a /tmp marker as a barrier so other ranks wait. We keep
    Model-Optimizer at upstream main, so patch the working-tree file at
    startup — the PatternPackager ships the patched version.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "modules/Model-Optimizer/tools/launcher/common/service_utils.sh",
    )
    if not os.path.exists(path):
        return
    with open(path) as f:
        content = f.read()
    if "nmm_extra_dep_installed" in content:
        return  # Already patched.
    rank_old = (
        "mpi_rank=${PMIX_RANK:-$native_mpi_rank}\\n"
        "mpi_local_rank=${PMIX_LOCAL_RANK:-$native_mpi_local_rank}"
    )
    rank_new = (
        "mpi_rank=${PMIX_RANK:-${native_mpi_rank:-${SLURM_PROCID:-0}}}\\n"
        "mpi_local_rank=${PMIX_LOCAL_RANK:-${native_mpi_local_rank:-${SLURM_LOCALID:-0}}}"
    )
    func_old = (
        "function util_install_extra_dep {\\n"
        "    if [[ \\\"$mpi_local_rank\\\" -eq 0 ]]; then\\n"
        "        pip install diskcache\\n"
        "    fi\\n"
        "}"
    )
    func_new = (
        "function util_install_extra_dep {\\n"
        "    local _marker=/tmp/.nmm_extra_dep_installed\\n"
        "    if [[ -f \\\"$_marker\\\" ]]; then\\n"
        "        return 0\\n"
        "    fi\\n"
        "    if [[ \\\"$mpi_local_rank\\\" -eq 0 ]]; then\\n"
        "        pip install diskcache\\n"
        "        local _nvrx_dir\\n"
        "        _nvrx_dir=\\\"$(mktemp -d)/nvidia-resiliency-ext\\\"\\n"
        "        git clone --depth 1 https://github.com/NVIDIA/nvidia-resiliency-ext \\\"${_nvrx_dir}\\\" \\\\\\n"
        "            && pip install \\\"${_nvrx_dir}\\\"\\n"
        "        touch \\\"$_marker\\\"\\n"
        "    else\\n"
        "        local _waited=0\\n"
        "        while [[ ! -f \\\"$_marker\\\" && $_waited -lt 600 ]]; do\\n"
        "            sleep 1\\n"
        "            _waited=$((_waited + 1))\\n"
        "        done\\n"
        "    fi\\n"
        "}"
    )
    if rank_old not in content or func_old not in content:
        return  # Upstream layout changed; don't patch blindly.
    content = content.replace(rank_old, rank_new, 1).replace(func_old, func_new, 1)
    with open(path, "w") as f:
        f.write(content)


_ensure_launcher_nvrx_install()


packager = run.PatternPackager(
    include_pattern=[
        "modelopt/*",
        "modelopt_recipes/*",
        "tests/*",
        "examples/*",
        "pyproject.toml",
        "tools/launcher/common/*",
        "tools/launcher/examples/*",
        "tools/*",
    ],
    relative_path=[
        os.getcwd(),  # modelopt/*
        os.getcwd(),  # modelopt_recipes/*
        os.getcwd(),  # tests/*
        os.getcwd(),  # examples/*
        os.getcwd(),  # pyproject.toml
        os.getcwd(),  # tools/launcher/common/*
        os.getcwd(),  # tools/launcher/examples/*
        os.getcwd(),  # tools/*
    ],
)

MODELOPT_SRC_PATH = os.path.join(os.getcwd(), "modelopt")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@run.cli.entrypoint
def cicd(
    job_name: str = "01_job",
    job_dir: str = os.environ.get(
        "SLURM_JOB_DIR",
        "/lustre/fsw/portfolios/coreai/users/{}/experiments".format(getpass.getuser()),
    ),
    task: SandboxTask = None,
    pipeline: SandboxPipeline = None,
    hf_local: str = None,
    user: str = getpass.getuser(),
    identity: str = None,
    test_level: int = 0,
    detach: bool = False,
) -> None:
    """NMM Sandbox CI/CD orchestrator.

    Args:
        job_name: Name of the job.
        job_dir: Directory of the job.
        user: User name for SSH tunnel.
        identity: Identity file for SSH tunnel.
        test_level: Test level.
    """
    if "NEMORUN_HOME" not in os.environ:
        warnings.warn(
            "NEMORUN_HOME is not set. Run 'source .sandbox_credentials.sh' to set it. "
            "Defaulting to current working directory."
        )
    run.config.set_nemorun_home(os.environ.get("NEMORUN_HOME", os.getcwd()))

    if hf_local is not None:
        job_dir = os.getcwd() + "/local_experiments"

    job_table = {}

    if task is not None:
        job_table[job_name] = SandboxPipeline(tasks=[task])
    elif pipeline is not None:
        job_table[job_name] = pipeline
    else:
        print("No task or pipeline provided. Use task=@<yaml> or pipeline=@<yaml>.")
        print("For multi-job YAML files, use: bash tools/run_job_yaml.sh <yaml> [args...]")
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
        test_level=test_level,
        modelopt_src_path=MODELOPT_SRC_PATH,
    )


if __name__ == "__main__":
    run.cli.main(cicd)
