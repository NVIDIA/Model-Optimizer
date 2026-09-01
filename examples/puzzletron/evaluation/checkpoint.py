# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared contracts for local checkpoint evaluation commands."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

__all__ = [
    "DEFAULT_PREFLIGHT_TIMEOUT_SECONDS",
    "HUGGINGFACE_CREDENTIAL_NAMES",
    "LMMS_EVAL_QWEN35_NATIVE_REVISION",
    "LMMS_EVAL_REVISION",
    "credential_free_environment",
    "lmms_eval_disabled_judge_environment",
    "positive_float",
    "positive_int",
    "run_lmms_eval_checkpoint",
    "verify_lmms_eval_revision",
    "without_huggingface_credentials",
    "write_generated",
]

REPOSITORY_ROOT = Path(__file__).absolute().parents[3]
LMMS_EVAL_REVISION = "15c32bfec165df13c269ddd3cda03b2ed9137825"
LMMS_EVAL_QWEN35_NATIVE_REVISION = "88b23e2bfa16a1edbc16e9e238ed82130b3a4f56"
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 15 * 60.0
HUGGINGFACE_CREDENTIAL_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def lmms_eval_disabled_judge_environment() -> dict[str, str]:
    """Return a loopback-only import shim for pinned tasks that eagerly create a judge."""
    return {
        "API_TYPE": "openai",
        "MODEL_VERSION": "modelopt-disabled-lmms-eval-judge",
        "OPENAI_API_KEY": "modelopt-disabled-lmms-eval-judge",
        "OPENAI_API_URL": "http://127.0.0.1:9",
    }


def _load_source_module(name: str, relative_path: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPOSITORY_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"ModelOpt source module is unavailable: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner() -> Callable[..., dict[str, object]]:
    """Load the narrow evaluator without importing unrelated ModelOpt plugins."""
    packages = (
        ("modelopt", "modelopt"),
        ("modelopt.torch", "modelopt/torch"),
        ("modelopt.torch.puzzletron", "modelopt/torch/puzzletron"),
        (
            "modelopt.torch.puzzletron.orchestration",
            "modelopt/torch/puzzletron/orchestration",
        ),
        (
            "modelopt.torch.puzzletron.evaluation",
            "modelopt/torch/puzzletron/evaluation",
        ),
    )
    source_modules = (
        "modelopt.torch.puzzletron.orchestration.mesh",
        "modelopt.torch.puzzletron.evaluation.lmms",
    )
    module_names = (*(package for package, _ in packages), *source_modules)
    missing = object()
    original_modules = {name: sys.modules.get(name, missing) for name in module_names}
    try:
        for package, relative_path in packages:
            if package not in sys.modules:
                module = ModuleType(package)
                module.__path__ = [str(REPOSITORY_ROOT / relative_path)]
                sys.modules[package] = module
        with redirect_stdout(sys.stderr):
            _load_source_module(
                source_modules[0],
                "modelopt/torch/puzzletron/orchestration/mesh.py",
            )
            module = _load_source_module(
                source_modules[1],
                "modelopt/torch/puzzletron/evaluation/lmms.py",
            )
        return cast("Callable[..., dict[str, object]]", module.run_lmms_eval_checkpoint)
    finally:
        for name, original in original_modules.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def run_lmms_eval_checkpoint(*args, **kwargs) -> dict[str, object]:
    """Load the ModelOpt runner only when an evaluation is actually executed."""
    return _load_runner()(*args, **kwargs)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def verify_lmms_eval_revision(expected_revision: str = LMMS_EVAL_REVISION) -> str:
    """Return the installed VCS revision after matching the requested evaluator pin."""
    try:
        direct_url = importlib.metadata.distribution("lmms-eval").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        direct_url = None
    try:
        provenance = json.loads(direct_url) if direct_url is not None else None
    except json.JSONDecodeError as error:
        raise RuntimeError("installed lmms-eval revision provenance is unavailable") from error
    if provenance is None:
        revision = _imported_lmms_eval_revision()
    elif isinstance(provenance, dict):
        vcs_info = provenance.get("vcs_info")
        revision = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
        if revision is None:
            revision = _editable_lmms_eval_revision(provenance)
    else:
        revision = None
    if revision != expected_revision:
        raise RuntimeError(
            "installed lmms-eval revision differs from the pinned profile: "
            f"expected {expected_revision}, found {revision or 'unknown'}"
        )
    return revision


def _editable_lmms_eval_revision(provenance: dict[str, object]) -> str | None:
    """Return the revision of a clean editable Git install, when present."""
    directory_info = provenance.get("dir_info")
    if not isinstance(directory_info, dict) or directory_info.get("editable") is not True:
        return None
    url = provenance.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        return None
    checkout = Path(url2pathname(unquote(parsed.path))).resolve()
    return _clean_checkout_revision(checkout)


def _imported_lmms_eval_revision() -> str | None:
    """Verify a source checkout imported directly through ``PYTHONPATH``."""
    spec = importlib.util.find_spec("lmms_eval")
    locations = tuple(spec.submodule_search_locations or ()) if spec is not None else ()
    if len(locations) != 1:
        return None
    return _clean_checkout_revision(Path(locations[0]).resolve().parent)


def _clean_checkout_revision(checkout: Path) -> str | None:
    """Return the revision of one clean Git checkout, if it can be verified."""
    try:
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if status:
        raise RuntimeError("installed lmms-eval checkout contains local changes")
    return revision or None


def credential_free_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Copy an environment without inherited Hugging Face credentials."""
    sanitized = dict(environment)
    for name in HUGGINGFACE_CREDENTIAL_NAMES:
        sanitized.pop(name, None)
    return sanitized


@contextmanager
def without_huggingface_credentials() -> Iterator[None]:
    """Keep inherited Hub credentials out of strict offline evaluator subprocesses."""
    inherited = {
        name: os.environ[name] for name in HUGGINGFACE_CREDENTIAL_NAMES if name in os.environ
    }
    for name in HUGGINGFACE_CREDENTIAL_NAMES:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name in HUGGINGFACE_CREDENTIAL_NAMES:
            os.environ.pop(name, None)
        os.environ.update(inherited)


def write_generated(path: Path, content: str) -> None:
    """Write deterministic generated content while rejecting path collisions."""
    if path.exists() and path.read_text() != content:
        raise FileExistsError(f"generated task config collision: {path}")
    path.write_text(content)
