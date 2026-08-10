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
"""Nox session definitions for testing, linting, docs, and wheel builds.

Usage:
    python -m pip install nox uv                                                    # install nox and uv (once)
    nox -l                                                                          # list all sessions
    nox -s gpu_megatron                                                             # run a GPU session (inside container)
    nox -s "unit-3.12(torch_211, tf_latest)"                                        # run a specific unit test combination
    nox -s "unit-3.12(torch_211, tf_latest)" -R                                     # force-recreate venv (e.g. after dep changes)
    COVERAGE_PROCESS_START=pyproject.toml nox -s "unit-3.12(torch_211, tf_latest)"  # with coverage
"""

import glob
import json
import os
import shutil
from pathlib import Path

import nox

nox.options.default_venv_backend = "uv" if shutil.which("uv") else "virtualenv"
nox.options.envdir = "/tmp/.nox"
nox.options.reuse_existing_virtualenvs = True

TORCH_VERSIONS = {
    "torch_28": "torchvision~=0.23.0",
    "torch_29": "torchvision~=0.24.0",
    "torch_210": "torchvision~=0.25.0",
    "torch_211": "torchvision~=0.26.0",
    "torch_212": "torchvision~=0.27.0",
}

TRANSFORMERS_VERSIONS = {
    "tf_latest": "transformers~=5.9.0",
    "tf_min": "transformers~=4.56.0",
}

PUZZLETRON_V2_CI_ENVIRONMENT_PATH = (
    Path(__file__).resolve().parent / "examples/puzzletron/ci_environment.json"
)
with PUZZLETRON_V2_CI_ENVIRONMENT_PATH.open(encoding="utf-8") as environment_file:
    PUZZLETRON_V2_CI_ENVIRONMENT = json.load(environment_file)
PUZZLETRON_V2_AUTOMODEL_SOURCE = PUZZLETRON_V2_CI_ENVIRONMENT["nemo_automodel"]
PUZZLETRON_V2_AUTOMODEL = (
    "nemo-automodel @ git+"
    f"{PUZZLETRON_V2_AUTOMODEL_SOURCE['repository']}@"
    f"{PUZZLETRON_V2_AUTOMODEL_SOURCE['commit']}"
)


def _cov_args():
    """Return --cov when COVERAGE_PROCESS_START is set (CI only)."""
    return ["--cov"] if os.environ.get("COVERAGE_PROCESS_START") else []


# ─── CPU unit tests ───────────────────────────────────────────────────────────
@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
@nox.parametrize("tf_ver", [nox.param(k, id=k) for k in TRANSFORMERS_VERSIONS])
@nox.parametrize("torch_ver", [nox.param(k, id=k) for k in TORCH_VERSIONS])
def unit(session, torch_ver, tf_ver):
    """Non-Puzzletron unit tests across the generic dependency matrix."""
    session.install(TORCH_VERSIONS[torch_ver], "-e", ".[all,dev-test]")
    tf_pin = TRANSFORMERS_VERSIONS[tf_ver]
    if tf_pin:
        session.install(tf_pin)
    # Puzzletron v2 has an exact, independently tested runtime matrix.
    session.run(
        "python",
        "-m",
        "pytest",
        "tests/unit",
        "--ignore=tests/unit/torch/puzzletron",
        *_cov_args(),
    )


@nox.session(python=PUZZLETRON_V2_CI_ENVIRONMENT["python"])
def puzzletron_v2(session):
    """Run Puzzletron v2 CPU-eligible tests in its pinned Python runtime."""
    session.install(
        f"torch=={PUZZLETRON_V2_CI_ENVIRONMENT['torch']}",
        f"torchvision=={PUZZLETRON_V2_CI_ENVIRONMENT['torchvision']}",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    )
    session.install(
        f"transformers=={PUZZLETRON_V2_CI_ENVIRONMENT['transformers']}",
        "-r",
        "examples/puzzletron/requirements.txt",
        "-e",
        ".[hf,puzzletron,dev-test]",
        PUZZLETRON_V2_AUTOMODEL,
    )
    session.run("uv", "pip", "check")
    expected_versions = {
        "python": PUZZLETRON_V2_CI_ENVIRONMENT["python"],
        "torch": PUZZLETRON_V2_CI_ENVIRONMENT["torch"],
        "torchvision": PUZZLETRON_V2_CI_ENVIRONMENT["torchvision"],
        "transformers": PUZZLETRON_V2_CI_ENVIRONMENT["transformers"],
        "nemo-automodel": PUZZLETRON_V2_AUTOMODEL_SOURCE["base_version"],
    }
    session.run(
        "python",
        "-c",
        (
            "import sys; "
            "from importlib.metadata import version; "
            "from packaging.version import Version; "
            f"expected = {expected_versions!r}; "
            "actual = {"
            "'python': f'{sys.version_info.major}.{sys.version_info.minor}', "
            "'torch': Version(version('torch')).base_version, "
            "'torchvision': Version(version('torchvision')).base_version, "
            "'transformers': Version(version('transformers')).base_version, "
            "'nemo-automodel': Version(version('nemo-automodel')).base_version}; "
            "mismatches = {name: (actual[name], expected_version) "
            "for name, expected_version in expected.items() "
            "if actual[name] != expected_version}; "
            "assert not mismatches, "
            "f'Pinned Puzzletron CI environment mismatch: {mismatches}'"
        ),
    )
    session.run(
        "python",
        "-m",
        "pytest",
        "tests/unit/torch/puzzletron",
        *_cov_args(),
    )


@nox.session(python="3.12")
@nox.parametrize("subset", ["onnx", "torch", "torch_deploy"])
def partial_unit(session, subset):
    """Unit tests with partial installs."""
    if subset == "onnx":
        session.install("torchvision~=0.26.0", ".[onnx,dev-test]")
        session.run("python", "-m", "pytest", "tests/unit/onnx")
    elif subset == "torch":
        session.install("megatron-core", ".[dev-test]")
        session.run(
            "python",
            "-m",
            "pytest",
            "tests/unit/torch",
            "--ignore=tests/unit/torch/deploy",
            "--ignore=tests/unit/torch/puzzletron",
        )
    else:  # torch_deploy
        session.install(".[onnx,dev-test]")
        session.run("python", "-m", "pytest", "tests/unit/torch/deploy")


# ─── GPU sessions (run inside containers — no new venv) ──────────────────────
# `venv_backend="none"` skips creating a new venv so the session runs directly in the container's
# existing Python environment (e.g. /opt/venv in NeMo) instead of an isolated one.
# Use `python -m pip/pytest` to ensure the container's active venv Python is used,
# not a stale PATH entry (e.g. NeMo container has pip → /usr/local/bin/pip but python → /opt/venv/bin/python).
# Container: nvcr.io/nvidia/pytorch:26.01-py3 or later
@nox.session(venv_backend="none")
def gpu(session):
    # tests/gpu/_extensions/test_onnx_extensions.py fails for newer containers
    # until https://github.com/tbenthompson/cppimport/pull/98
    session.run(
        "python",
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "git+https://github.com/Dao-AILab/fast-hadamard-transform.git",
    )
    session.run("python", "-m", "pip", "install", "-e", ".[all,dev-test]")
    session.run("python", "-m", "pip", "uninstall", "-y", "cupy-cuda12x")
    session.run("python", "-m", "pip", "install", "cupy-cuda13x")
    session.run(
        "python",
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "git+https://github.com/state-spaces/mamba.git",
        "git+https://github.com/Dao-AILab/causal-conv1d.git",
    )
    session.run("python", "-m", "pytest", "tests/gpu", *_cov_args())


# Container: nvcr.io/nvidia/nemo:26.04 or later
@nox.session(venv_backend="none")
def gpu_megatron(session):
    # nemo:26.04 has transformers 5.x but system-wide installed trtllm 1.2.0 which does not support it causing import errors
    session.run("pip", "uninstall", "-y", "tensorrt_llm")
    # Pre-installed nvidia-modelopt shadows the editable install
    session.run("pip", "uninstall", "-y", "nvidia-modelopt")
    session.run("python", "-m", "pip", "install", "-e", ".[hf,dev-test]")
    session.run("python", "-m", "pytest", "tests/gpu_megatron", *_cov_args())


# Container: nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10 or later
@nox.session(venv_backend="none")
def gpu_trtllm(session):
    session.run("python", "-m", "pip", "install", "-e", ".[hf,dev-test]")
    session.run("python", "-m", "pytest", "tests/gpu_trtllm", *_cov_args())


# Container: docker.io/vllm/vllm-openai (the published image ships vLLM + CUDA + torch).
# Pin must stay in sync with examples/vllm_serve/Dockerfile.
@nox.session(venv_backend="none")
def gpu_vllm(session):
    session.run("python3", "-m", "pip", "install", "-e", ".[hf,puzzletron,dev-test]")
    session.run("python3", "-m", "pytest", "tests/gpu_vllm", *_cov_args())


# Container: nvcr.io/nvidia/pytorch:26.01-py3 or later
@nox.session(venv_backend="none")
def regression(session):
    session.run("python", "-m", "pip", "install", "-e", ".[hf,dev-test]")
    session.run("python", "-m", "pytest", "tests/regression", *_cov_args())


# ─── Code quality ─────────────────────────────────────────────────────────────
@nox.session
def pre_commit_all(session):
    session.install("-e", ".[all,dev-lint]")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure")


@nox.session
def pre_commit_diff(session):
    if len(session.posargs) not in (0, 2):
        session.error("pre_commit_diff expects optional FROM_REF and TO_REF arguments")

    from_ref, to_ref = session.posargs or ("origin/main", "HEAD")
    session.install("-e", ".[all,dev-lint]")
    session.run(
        "pre-commit",
        "run",
        "--from-ref",
        from_ref,
        "--to-ref",
        to_ref,
        "--show-diff-on-failure",
    )


# ─── Docs ─────────────────────────────────────────────────────────────────────
def _build_docs(session, warning_baseline=None):
    session.install("-e", ".[all,dev-docs]")
    shutil.rmtree("docs/build", ignore_errors=True)
    shutil.rmtree("docs/source/reference/generated", ignore_errors=True)

    warning_args = ["--fail-on-warning"]
    warning_log = Path("/tmp/modelopt-sphinx-warnings.log")
    if warning_baseline:
        warning_log.unlink(missing_ok=True)
        warning_args = ["--warning-file", str(warning_log)]

    with session.chdir("docs"):
        session.run(
            "sphinx-build",
            "-d",
            "/tmp/doctrees",
            "source",
            "build/html",
            *warning_args,
            "--show-traceback",
            "--keep-going",
        )
        if warning_baseline:
            session.run(
                "python",
                "../tools/ci/check_sphinx_warnings.py",
                str(warning_log),
                warning_baseline,
                "--repo-root",
                "..",
            )


@nox.session
def docs(session):
    _build_docs(session)


@nox.session
def docs_puzzletron_v2(session):
    _build_docs(session, "sphinx_warnings_feature_puzzletron_v2.json")


@nox.session
def docs_debug(session):
    session.install("-e", ".[all,dev-docs]")
    shutil.rmtree("docs/build", ignore_errors=True)
    shutil.rmtree("docs/source/reference/generated", ignore_errors=True)
    with session.chdir("docs"):
        session.run("sphinx-autobuild", "source", "build/html", "--host", "0.0.0.0")


# ─── Wheel build ──────────────────────────────────────────────────────────────
@nox.session
def build_wheel(session):
    shutil.rmtree("build", ignore_errors=True)
    session.install("twine")
    session.run("pip", "wheel", "--no-deps", "--wheel-dir=dist", ".")
    wheels = glob.glob("dist/*.whl")
    session.run("twine", "check", *wheels)
    (modelopt_wheel,) = glob.glob("dist/nvidia_modelopt-*.whl")
    session.install(modelopt_wheel, "-f", "dist")
    with session.chdir("dist"):
        session.run("python", "-c", "import modelopt; print(modelopt.__version__)")
