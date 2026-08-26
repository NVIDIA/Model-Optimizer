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
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

__all__ = [
    "DEFAULT_PREFLIGHT_TIMEOUT_SECONDS",
    "HUGGINGFACE_CREDENTIAL_NAMES",
    "LMMS_EVAL_REVISION",
    "credential_free_environment",
    "load_runner",
    "positive_float",
    "positive_int",
    "run_lmms_eval_checkpoint",
    "verify_lmms_eval_revision",
    "without_huggingface_credentials",
    "write_generated",
]

REPOSITORY_ROOT = Path(__file__).absolute().parents[3]
LMMS_EVAL_REVISION = "88b23e2bfa16a1edbc16e9e238ed82130b3a4f56"
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 15 * 60.0
HUGGINGFACE_CREDENTIAL_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


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


def load_runner() -> Callable[..., dict[str, object]]:
    """Load the narrow evaluator without importing unrelated ModelOpt plugins."""
    for package, relative_path in (
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
    ):
        if package not in sys.modules:
            module = ModuleType(package)
            module.__path__ = [str(REPOSITORY_ROOT / relative_path)]
            sys.modules[package] = module
    with redirect_stdout(sys.stderr):
        _load_source_module(
            "modelopt.torch.puzzletron.orchestration.mesh",
            "modelopt/torch/puzzletron/orchestration/mesh.py",
        )
        module = _load_source_module(
            "modelopt.torch.puzzletron.evaluation.lmms",
            "modelopt/torch/puzzletron/evaluation/lmms.py",
        )
    return cast("Callable[..., dict[str, object]]", module.run_lmms_eval_checkpoint)


def run_lmms_eval_checkpoint(*args, **kwargs) -> dict[str, object]:
    """Load the ModelOpt runner only when an evaluation is actually executed."""
    return load_runner()(*args, **kwargs)


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


def verify_lmms_eval_revision() -> str:
    """Return the installed VCS revision after matching the shared evaluator pin."""
    try:
        direct_url = importlib.metadata.distribution("lmms-eval").read_text("direct_url.json")
        provenance = json.loads(direct_url) if direct_url is not None else None
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("installed lmms-eval revision provenance is unavailable") from error
    if not isinstance(provenance, dict):
        raise RuntimeError("installed lmms-eval revision provenance is unavailable")
    vcs_info = provenance.get("vcs_info")
    revision = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if revision != LMMS_EVAL_REVISION:
        raise RuntimeError(
            "installed lmms-eval revision differs from the pinned profile: "
            f"expected {LMMS_EVAL_REVISION}, found {revision or 'unknown'}"
        )
    return revision


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
