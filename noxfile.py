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
import os
import shutil
import tarfile
import tempfile

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
    "torch_213": "torchvision~=0.28.0",
    "torch_214": "torchvision~=0.29.0",
}

# Extra install pins applied per transformers matrix entry (installed after the base
# ``.[all,dev-test]`` install to constrain that env).
TRANSFORMERS_VERSIONS = {
    "tf_latest": ("transformers~=5.18.0",),
    "tf_min": ("transformers~=5.5.0",),
}


def _cov_args():
    """Return --cov when COVERAGE_PROCESS_START is set (CI only)."""
    return ["--cov"] if os.environ.get("COVERAGE_PROCESS_START") else []


# ─── CPU unit tests ───────────────────────────────────────────────────────────
_CPU_ONLY_ENV = {"CUDA_VISIBLE_DEVICES": ""}


@nox.session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
@nox.parametrize("tf_ver", [nox.param(k, id=k) for k in TRANSFORMERS_VERSIONS])
@nox.parametrize("torch_ver", [nox.param(k, id=k) for k in TORCH_VERSIONS])
def unit(session, torch_ver, tf_ver):
    """Unit tests — parametrized over torch and transformers versions."""
    session.install(TORCH_VERSIONS[torch_ver], "-e", ".[all,dev-test]")
    tf_pins = TRANSFORMERS_VERSIONS[tf_ver]
    if tf_pins:
        session.install(*tf_pins)
    run_manual = "--run-manual" in session.posargs
    pytest_args = [arg for arg in session.posargs if arg != "--run-manual"]
    session.run(
        "python", "-m", "pytest", "tests/unit", *_cov_args(), *pytest_args, env=_CPU_ONLY_ENV
    )
    if run_manual:
        # A separate process: some manual tests (e.g. multi-threaded from_pretrained) leak global
        # state that breaks later tests. No --cov here, so this run writes no coverage data.
        # Serial too (CI's PYTEST_ADDOPTS asks for xdist): a worker would run them in some order.
        session.run(
            "python",
            # Caller args first: the last ``-m`` wins, and it must be ours.
            *("-m", "pytest", "tests/unit", *pytest_args, "--run-manual", "-m", "manual"),
            env={**_CPU_ONLY_ENV, "COVERAGE_PROCESS_START": "", "PYTEST_ADDOPTS": ""},
        )


@nox.session(python="3.12")
@nox.parametrize("subset", ["onnx", "torch", "torch_deploy"])
def partial_unit(session, subset):
    """Unit tests with partial installs."""
    if subset == "onnx":
        session.install("torchvision~=0.26.0", ".[onnx,dev-test]")
        session.run("python", "-m", "pytest", "tests/unit/onnx", env=_CPU_ONLY_ENV)
    elif subset == "torch":
        session.install("megatron-core", ".[dev-test]")
        session.run(
            "python",
            "-m",
            "pytest",
            "tests/unit/torch",
            "--ignore=tests/unit/torch/deploy",
            "--ignore=tests/unit/torch/puzzletron",
            env=_CPU_ONLY_ENV,
        )
    else:  # torch_deploy
        session.install(".[onnx,dev-test]")
        session.run("python", "-m", "pytest", "tests/unit/torch/deploy", env=_CPU_ONLY_ENV)


# ─── GPU sessions (run inside containers — no new venv) ──────────────────────
# `venv_backend="none"` skips creating a new venv so the session runs directly in the container's
# existing Python environment (e.g. /opt/venv in NeMo) instead of an isolated one.
# Use `python -m pip/pytest` to ensure the container's active venv Python is used,
# not a stale PATH entry (e.g. NeMo container has pip → /usr/local/bin/pip but python → /opt/venv/bin/python).
# CUDA packages compiled against the container's torch, which takes minutes each. Pinned so the CI
# cache of the built wheels (keyed on this file and the container image, see gpu_tests.yml) stays
# valid: bump a pin deliberately, which also invalidates that cache.
_SOURCE_BUILT_WHEELS = {
    "fast_hadamard_transform": "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@e7706faf8d1c3b9f241e36860640ad1dac644ede",
    "mamba_ssm": "mamba_ssm==2.3.2.post1",
    "causal_conv1d": "causal-conv1d==1.7.0",
}


# Their setup.py files hardcode -std=c++17, but the torch headers of recent containers need C++20.
_BUILD_AS_CXX20 = ("mamba_ssm", "causal_conv1d")
# Without these, setup.py may download a matching prebuilt wheel instead of compiling the patched source.
_FORCE_BUILD = {
    "mamba_ssm": {"MAMBA_FORCE_BUILD": "TRUE"},
    "causal_conv1d": {"CAUSAL_CONV1D_FORCE_BUILD": "TRUE"},
}


def _wheelhouse():
    return os.environ.get("MODELOPT_WHEELHOUSE", os.path.expanduser("~/.cache/wheelhouse"))


def _build_wheel(session, name, wheelhouse):
    pip_wheel = ("python", "-m", "pip", "wheel", "--no-build-isolation", "--no-deps")
    pip_wheel += ("--wheel-dir", wheelhouse)
    requirement = _SOURCE_BUILT_WHEELS[name]
    if name not in _BUILD_AS_CXX20:
        session.run(*pip_wheel, requirement)
        return
    with tempfile.TemporaryDirectory() as tmp:
        session.run(
            *("python", "-m", "pip", "download", "--no-deps", "--no-binary", ":all:"),
            *("--no-build-isolation", "--dest", tmp, requirement),
        )
        sdist = glob.glob(os.path.join(tmp, "*.tar.gz"))[0]
        with tarfile.open(sdist) as tar:
            tar.extractall(tmp, filter="data")
        source = sdist.removesuffix(".tar.gz")
        setup_py = os.path.join(source, "setup.py")
        with open(setup_py) as f:
            patched = f.read().replace("-std=c++17", "-std=c++20")
        with open(setup_py, "w") as f:
            f.write(patched)
        session.run(*pip_wheel, source, env=_FORCE_BUILD[name])


def _find_wheel(wheelhouse, name):
    # ``-[0-9]`` so ``causal_conv1d`` does not also match a ``causal_conv1d_cuda-*`` wheel.
    found = sorted(glob.glob(os.path.join(wheelhouse, f"{name}-[0-9]*.whl")))
    return found[-1] if found else None


def _install_built_from_source(session, *names):
    """Install the named ``_SOURCE_BUILT_WHEELS``, building a wheel only if the wheelhouse lacks it.

    CI caches the wheelhouse (``MODELOPT_WHEELHOUSE``) per container image, so only the first run on
    an image builds.
    """
    wheelhouse = _wheelhouse()
    wheels = []
    for name in names:
        wheel = _find_wheel(wheelhouse, name)
        if wheel is None:
            _build_wheel(session, name, wheelhouse)
            wheel = _find_wheel(wheelhouse, name)
        if wheel is None:
            raise RuntimeError(f"No wheel for {name!r} in {wheelhouse} after building it")
        wheels.append(wheel)
    session.run("python", "-m", "pip", "install", *wheels)


# Container: nvcr.io/nvidia/pytorch:26.07-py3 or later with CUDA 13
@nox.session(venv_backend="none")
def gpu(session):
    _install_built_from_source(session, "fast_hadamard_transform")
    session.run("python", "-m", "pip", "install", "-e", ".[all,dev-test,dev-fla]")
    _install_built_from_source(session, "mamba_ssm", "causal_conv1d")
    # Every wheel is built: CI saves the wheelhouse only then, so a run that stopped halfway cannot
    # freeze a partial wheelhouse under the cache key.
    with open(os.path.join(_wheelhouse(), ".complete"), "w"):
        pass
    session.run("python", "-m", "pytest", "tests/gpu", *_cov_args())


# Container: nvcr.io/nvidia/nemo:26.08 or later
@nox.session(venv_backend="none")
def gpu_megatron(session):
    # NeMo containers have transformers 5.x but a system-wide installed trtllm which does not support it causing import errors
    session.run("pip", "uninstall", "-y", "tensorrt_llm")
    # Pre-installed nvidia-modelopt shadows the editable install
    session.run("pip", "uninstall", "-y", "nvidia-modelopt")
    session.run("python", "-m", "pip", "install", "-e", ".[hf,dev-test,dev-fla]")
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
    session.install("-e", ".[all,dev-lint]")
    session.run("pre-commit", "run", "--from-ref", "origin/main", "--to-ref", "HEAD")


# ─── Docs ─────────────────────────────────────────────────────────────────────
@nox.session
def docs(session):
    session.install("-e", ".[all,dev-docs]")
    shutil.rmtree("docs/build", ignore_errors=True)
    shutil.rmtree("docs/source/reference/generated", ignore_errors=True)
    with session.chdir("docs"):
        session.run(
            "sphinx-build",
            "-d",
            "/tmp/doctrees",
            "source",
            "build/html",
            "--fail-on-warning",
            "--show-traceback",
            "--keep-going",
        )


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
