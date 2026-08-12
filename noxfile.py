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
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
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
        "lmms-eval": PUZZLETRON_V2_CI_ENVIRONMENT["lmms_eval"],
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
            "'lmms-eval': Version(version('lmms-eval')).base_version, "
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


@dataclass(frozen=True)
class _ChangedPythonFile:
    base_path: str | None
    head_path: str


@dataclass(frozen=True)
class _MypyDiagnostic:
    path: str
    line: int
    column: int | None
    message: str
    code: str | None
    raw: str

    @property
    def fingerprint(self):
        return self.path, self.message, self.code


_MYPY_LOCATION = re.compile(
    r"^(?P<path>.*?):(?P<line>\d+)(?::(?P<column>\d+))?: error: (?P<message>.*)$"
)
_MYPY_CODE = re.compile(r"^(?P<message>.*)  \[(?P<code>[^]]+)\]$")
_MYPY_DIFF_DEPENDENCIES = ("types-PyYAML==6.0.12.20260724",)


def _normalize_diagnostic_path(path):
    normalized = path.replace("\\", "/")
    return normalized.removeprefix("./")


def _parse_changed_python_files(output):
    changes = []
    for line in output.splitlines():
        fields = line.split("\t")
        status = fields[0][:1]
        if status == "R" and len(fields) == 3:
            changes.append(_ChangedPythonFile(base_path=fields[1], head_path=fields[2]))
        elif status == "C" and len(fields) == 3:
            changes.append(_ChangedPythonFile(base_path=None, head_path=fields[2]))
        elif status == "A" and len(fields) == 2:
            changes.append(_ChangedPythonFile(base_path=None, head_path=fields[1]))
        elif status == "M" and len(fields) == 2:
            changes.append(_ChangedPythonFile(base_path=fields[1], head_path=fields[1]))
        else:
            raise ValueError(f"Unsupported git name-status line: {line!r}")
    return tuple(changes)


def _parse_mypy_diagnostics(output):
    diagnostics = []
    for line in output.splitlines():
        location_match = _MYPY_LOCATION.match(line)
        if location_match is None:
            continue

        message = location_match.group("message")
        code_match = _MYPY_CODE.match(message)
        code = None
        if code_match is not None:
            message = code_match.group("message")
            code = code_match.group("code")
        column = location_match.group("column")
        diagnostics.append(
            _MypyDiagnostic(
                path=_normalize_diagnostic_path(location_match.group("path")),
                line=int(location_match.group("line")),
                column=int(column) if column is not None else None,
                message=message,
                code=code,
                raw=line,
            )
        )
    return tuple(diagnostics)


def _new_mypy_diagnostics(base_output, head_output, base_to_head_paths: dict[str, str]):
    available_base_diagnostics = Counter(
        (
            _normalize_diagnostic_path(base_to_head_paths.get(diagnostic.path, diagnostic.path)),
            diagnostic.message,
            diagnostic.code,
        )
        for diagnostic in _parse_mypy_diagnostics(base_output)
    )

    new_diagnostics = []
    for diagnostic in _parse_mypy_diagnostics(head_output):
        if available_base_diagnostics[diagnostic.fingerprint]:
            available_base_diagnostics[diagnostic.fingerprint] -= 1
        else:
            new_diagnostics.append(diagnostic)
    return tuple(new_diagnostics)


def _resolve_git_commit(session, ref):
    return session.run(
        "git",
        "rev-parse",
        "--verify",
        f"{ref}^{{commit}}",
        external=True,
        silent=True,
    ).strip()


def _run_mypy(session, checkout, paths, cache_dir):
    if not paths:
        return ""
    with session.chdir(checkout):
        return session.run(
            "mypy",
            "--no-install-types",
            "--interactive",
            "--no-error-summary",
            "--no-color-output",
            "--no-pretty",
            "--show-column-numbers",
            "--show-error-codes",
            "--follow-imports=skip",
            "--ignore-missing-imports",
            "--cache-dir",
            str(cache_dir),
            "--",
            *paths,
            success_codes=[0, 1],
            silent=True,
        )


def _run_changed_file_mypy(session, from_ref, to_ref):
    changed_file_output = session.run(
        "git",
        "diff",
        "--name-status",
        "--find-renames",
        "--diff-filter=ACMR",
        f"{from_ref}...{to_ref}",
        "--",
        "*.py",
        external=True,
        silent=True,
    )
    changes = _parse_changed_python_files(changed_file_output)
    if not changes:
        session.log("mypy diff ratchet: no changed Python files")
        return

    repository = session.run(
        "git", "rev-parse", "--show-toplevel", external=True, silent=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="modelopt-mypy-diff-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        checkout = temporary_path / "checkout"
        checkout_index = checkout / ".git" / "index"
        session.run(
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            repository,
            str(checkout),
            external=True,
        )

        session.run(
            "git",
            "-C",
            str(checkout),
            "checkout",
            "--quiet",
            "--detach",
            from_ref,
            external=True,
            env={"GIT_INDEX_FILE": str(checkout_index)},
        )
        base_paths = [change.base_path for change in changes if change.base_path is not None]
        missing_base_paths = [path for path in base_paths if not (checkout / path).is_file()]
        if missing_base_paths:
            session.error(f"mypy diff base paths are missing: {', '.join(missing_base_paths)}")
        base_output = _run_mypy(session, checkout, base_paths, temporary_path / "base-mypy-cache")

        session.run(
            "git",
            "-C",
            str(checkout),
            "checkout",
            "--quiet",
            "--detach",
            to_ref,
            external=True,
            env={"GIT_INDEX_FILE": str(checkout_index)},
        )
        head_paths = [change.head_path for change in changes]
        missing_head_paths = [path for path in head_paths if not (checkout / path).is_file()]
        if missing_head_paths:
            session.error(f"mypy diff head paths are missing: {', '.join(missing_head_paths)}")
        head_output = _run_mypy(session, checkout, head_paths, temporary_path / "head-mypy-cache")

    session.log(f"mypy base diagnostics ({from_ref}):\n{base_output.rstrip() or '(none)'}")
    session.log(f"mypy head diagnostics ({to_ref}):\n{head_output.rstrip() or '(none)'}")

    base_to_head_paths = {
        change.base_path: change.head_path for change in changes if change.base_path is not None
    }
    new_diagnostics = _new_mypy_diagnostics(base_output, head_output, base_to_head_paths)
    head_diagnostics = _parse_mypy_diagnostics(head_output)
    if new_diagnostics:
        formatted_diagnostics = "\n".join(diagnostic.raw for diagnostic in new_diagnostics)
        session.log(f"New mypy diagnostics:\n{formatted_diagnostics}")
        session.error(
            f"mypy diff ratchet found {len(new_diagnostics)} new diagnostic(s) "
            f"among {len(head_diagnostics)} at the head"
        )
    session.log(
        f"mypy diff ratchet passed with {len(head_diagnostics)} head diagnostic(s) "
        "and no new diagnostics"
    )


@nox.session
def pre_commit_diff(session):
    if len(session.posargs) not in (0, 2):
        session.error("pre_commit_diff expects optional FROM_REF and TO_REF arguments")

    from_ref, to_ref = session.posargs or ("origin/main", "HEAD")
    from_ref = _resolve_git_commit(session, from_ref)
    to_ref = _resolve_git_commit(session, to_ref)
    session.install("-e", ".[all,dev-lint]", *_MYPY_DIFF_DEPENDENCIES)
    skip_hooks = {hook for hook in os.environ.get("SKIP", "").split(",") if hook}
    skip_hooks.add("mypy")
    session.run(
        "pre-commit",
        "run",
        "--from-ref",
        from_ref,
        "--to-ref",
        to_ref,
        "--show-diff-on-failure",
        env={"SKIP": ",".join(sorted(skip_hooks))},
    )
    _run_changed_file_mypy(session, from_ref, to_ref)


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
