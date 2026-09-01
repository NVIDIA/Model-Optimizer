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

"""Versioned, executable VLM evaluation profile contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import profile

__all__ = [
    "PROFILE_NAMES",
    "ProfileContract",
    "load_profile",
]

_PROFILE_SCHEMA = "modelopt.vlm-evaluation-profile/v1"
_PROFILE_ROOT = Path(__file__).with_name("profiles")
PROFILE_NAMES = ("short-v1", "full-v1")
_PROFILE_TASKS = {
    "short-v1": ("realworldqa", "mmmu_val", "mvbench"),
    "full-v1": tuple(task for task in profile.VLM_BENCHMARK_TASKS if task != "mmvu_val"),
}
_PROFILE_SELECTIONS = {"short-v1": "exact-rows", "full-v1": "all"}


@dataclass(frozen=True)
class ProfileContract:
    """One validated profile manifest and its stable content identity."""

    name: str
    manifest: dict[str, object]
    fingerprint: str

    @property
    def source_tasks(self) -> tuple[str, ...]:
        """Return benchmark tasks in their declared evaluation order."""
        tasks = cast("dict[str, object]", self.manifest["tasks"])
        return tuple(tasks)

    @property
    def exact_rows(self) -> dict[str, object] | None:
        """Return the legacy exact-row selector payload when the profile uses one."""
        if self.manifest["selection"] != "exact-rows":
            return None
        tasks = cast("dict[str, dict[str, object]]", self.manifest["tasks"])
        return {
            "schema": "modelopt.vlm-benchmark-quick/v1",
            "lmms_eval_revision": self.manifest["lmms_eval_revision"],
            "tasks": {
                task: {
                    "dataset_revision": entry["dataset_revision"],
                    "rows": entry["rows"],
                }
                for task, entry in tasks.items()
            },
        }


def load_profile(name: str) -> ProfileContract:
    """Load a named profile after validating every executable pin."""
    if name not in PROFILE_NAMES:
        raise ValueError(f"unsupported VLM evaluation profile: {name}")
    path = _PROFILE_ROOT / f"{name}.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"VLM evaluation profile is unreadable: {path}") from error
    if not isinstance(manifest, dict):
        raise RuntimeError(f"VLM evaluation profile must contain an object: {path}")
    _validate_manifest(name, manifest)
    canonical = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    return ProfileContract(
        name=name,
        manifest=manifest,
        fingerprint=hashlib.sha256(canonical).hexdigest(),
    )


def _validate_manifest(name: str, manifest: dict[str, object]) -> None:
    if manifest.get("schema") != _PROFILE_SCHEMA:
        raise RuntimeError(f"{name} profile schema must be {_PROFILE_SCHEMA}")
    if manifest.get("name") != name:
        raise RuntimeError(f"{name} profile name does not match its filename")
    if manifest.get("lmms_eval_revision") != checkpoint.LMMS_EVAL_REVISION:
        raise RuntimeError(f"{name} profile lmms_eval_revision differs from the runtime pin")
    if manifest.get("model_family") != {
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
    }:
        raise RuntimeError(f"{name} profile model family is unsupported")
    if manifest.get("backend") != {"name": "vllm", "reasoning_parser": "qwen3"}:
        raise RuntimeError(f"{name} profile backend differs from the runtime policy")
    if manifest.get("preprocessing") != {
        "fps": 2,
        "max_frames": 32,
        "video_reader": "decord",
    }:
        raise RuntimeError(f"{name} profile preprocessing differs from the runtime policy")
    if manifest.get("generation") != {"do_sample": False, "temperature": 0}:
        raise RuntimeError(f"{name} profile generation differs from the runtime policy")
    if manifest.get("seed") != 42 or manifest.get("repetitions") != 1:
        raise RuntimeError(f"{name} profile execution identity differs from the runtime policy")
    selection = manifest.get("selection")
    if selection != _PROFILE_SELECTIONS[name]:
        raise RuntimeError(f"{name} profile selection differs from its versioned policy")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise RuntimeError(f"{name} profile tasks must contain an object")
    if tuple(tasks) != _PROFILE_TASKS[name]:
        raise RuntimeError(f"{name} profile tasks differ from its versioned policy")
    for task, entry in tasks.items():
        _validate_task(name, task, entry, selection=cast("str", selection))


def _validate_task(name: str, task: object, entry: object, *, selection: str) -> None:
    if not isinstance(task, str) or task not in profile.VLM_BENCHMARK_DATASETS:
        raise RuntimeError(f"{name} profile contains an unsupported task: {task}")
    if not isinstance(entry, dict):
        raise RuntimeError(f"{name} profile task must contain an object: {task}")
    dataset = profile.VLM_BENCHMARK_DATASETS[task]
    expected = {
        "dataset_repository": dataset.repository,
        "dataset_revision": dataset.revision,
        "max_new_tokens": dataset.max_new_tokens,
        "scoring_task_config": dataset.task_config,
    }
    observed = {key: entry.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"{name} profile task pins differ from the runtime catalog: {task}")
    rows = entry.get("rows")
    if selection == "all" and rows is not None:
        raise RuntimeError(f"{name} full-data task must not contain exact rows: {task}")
    if selection == "exact-rows" and (not isinstance(rows, list) or not rows):
        raise RuntimeError(f"{name} exact-row task must contain rows: {task}")
