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

"""Suite selection and pinned sampling policy for VLM evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import TypedDict

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import profile

__all__ = [
    "ADAPTER_SMOKE_TASKS",
    "ALL_TASKS",
    "DEFAULT_FULL_TIMEOUT_SECONDS",
    "DEFAULT_SMOKE_TIMEOUT_SECONDS",
    "DEPRECATED_SUITE_ALIASES",
    "EVALUATION_PROFILE",
    "MMVU_SMOKE_ROWS",
    "MVBENCH_LEAF_TASKS",
    "QUICK_SELECTED_ROWS",
    "QUICK_TASKS",
    "SHORT_TASKS",
    "SINGLE_TASK_SMOKE_SUITES",
    "SMOKE_SUITES",
    "TASK_PREFIX100_REPEAT2_SUITE",
    "VIDEO_MMMU_LEAF_TASKS",
    "ExecutionPolicy",
    "FramePolicy",
    "GenerationPolicy",
    "canonical_suite",
    "execution_policy",
    "generation_kwargs",
    "load_quick_manifest",
    "manifest_sha256",
    "offline_dataset_snapshot",
    "source_tasks",
    "task_name",
]

EVALUATION_PROFILE = "qwen35-vlm-benchmarks"
TASK_PREFIX100_REPEAT2_SUITE = "realworldqa-mmmu-prefix100-repeat2"
DEPRECATED_SUITE_ALIASES = {"e2e-full-eval": TASK_PREFIX100_REPEAT2_SUITE}
ALL_TASKS = profile.VLM_BENCHMARK_TASKS
SHORT_TASKS = ("realworldqa", "mmmu_val")
QUICK_TASKS = ("realworldqa", "mmmu_val", "mvbench")
ADAPTER_SMOKE_TASKS = ("videomme", "perceptiontest_val_mc")
SINGLE_TASK_SMOKE_SUITES = {
    "realworldqa-smoke": "realworldqa",
    "video-mmmu-smoke": "video_mmmu",
    "mmvu-smoke": "mmvu_val",
    "longvideobench-smoke": "longvideobench_val_v",
    "mlvu-smoke": "mlvu_dev",
}
SMOKE_SUITES = frozenset(("short", "adapter-smoke", *SINGLE_TASK_SMOKE_SUITES))

_QUICK_COUNTS = {"realworldqa": 64, "mmmu_val": 120, "mvbench": 160}
QUICK_SELECTED_ROWS = sum(_QUICK_COUNTS.values())
MMVU_SMOKE_ROWS = (
    (1, "validation_1", "videos/Chemistry/0.mp4"),
    (3, "validation_3", "videos/Thermodynamics/0.mp4"),
    (5, "validation_5", "videos/Materials_Science/0.mp4"),
    (8, "validation_8", "videos/Electromagnetism/1.mp4"),
    (12, "validation_12", "videos/Mechanical_Engineering/0.mp4"),
    (13, "validation_13", "videos/Mechanical_Engineering/0.mp4"),
    (15, "validation_15", "videos/Electrical_Engineering/1.mp4"),
    (17, "validation_17", "videos/Mechanical_Engineering/1.mp4"),
)
MVBENCH_LEAF_TASKS = (
    "action_sequence",
    "moving_count",
    "action_prediction",
    "episodic_reasoning",
    "action_antonym",
    "action_count",
    "scene_transition",
    "object_shuffle",
    "object_existence",
    "fine_grained_pose",
    "unexpected_action",
    "moving_direction",
    "state_change",
    "object_interaction",
    "character_order",
    "action_localization",
    "counterfactual_inference",
    "fine_grained_action",
    "moving_attribute",
    "egocentric_navigation",
)
VIDEO_MMMU_LEAF_TASKS = ("adaptation", "comprehension", "perception")
DEFAULT_SMOKE_TIMEOUT_SECONDS = 3_000.0
DEFAULT_FULL_TIMEOUT_SECONDS = 24 * 60 * 60.0


class FramePolicy(TypedDict):
    """Frame sampling fields shared by provenance and execution."""

    reader: str
    fps: int
    max_frames: int


class GenerationPolicy(TypedDict):
    """Generation fields shared by provenance and execution."""

    temperature: int
    do_sample: bool


class ExecutionPolicy(TypedDict):
    """Resolved execution fields for one suite invocation."""

    frame: FramePolicy
    generation: GenerationPolicy
    limit: int | None
    repetitions: int
    timeout_seconds: float


def source_tasks(suite: str) -> tuple[str, ...]:
    """Return the upstream task names selected by a VLM suite."""
    suite = canonical_suite(suite)
    if suite in {"short", TASK_PREFIX100_REPEAT2_SUITE}:
        return SHORT_TASKS
    if suite == "quick":
        return QUICK_TASKS
    if suite == "adapter-smoke":
        return ADAPTER_SMOKE_TASKS
    if suite in SINGLE_TASK_SMOKE_SUITES:
        return (SINGLE_TASK_SMOKE_SUITES[suite],)
    if suite == "full":
        return ALL_TASKS
    raise ValueError(f"unsupported VLM benchmark suite: {suite}")


def execution_policy(suite: str, *, timeout_seconds: float | None) -> ExecutionPolicy:
    """Resolve the provenance and runtime execution fields for one suite."""
    suite = canonical_suite(suite)
    source_tasks(suite)
    is_smoke = suite in SMOKE_SUITES
    default_timeout_seconds = (
        DEFAULT_SMOKE_TIMEOUT_SECONDS if is_smoke else DEFAULT_FULL_TIMEOUT_SECONDS
    )
    return {
        "frame": {"reader": "decord", "fps": 2, "max_frames": 32},
        "generation": {"temperature": 0, "do_sample": False},
        "limit": (
            2
            if suite == "realworldqa-smoke"
            else 8
            if is_smoke
            else 100
            if suite == TASK_PREFIX100_REPEAT2_SUITE
            else None
        ),
        "repetitions": 2 if suite in {"short", TASK_PREFIX100_REPEAT2_SUITE} else 1,
        "timeout_seconds": (
            timeout_seconds if timeout_seconds is not None else default_timeout_seconds
        ),
    }


def canonical_suite(suite: str) -> str:
    """Resolve a deprecated suite alias to its explicit canonical identity."""

    return DEPRECATED_SUITE_ALIASES.get(suite, suite)


def task_name(task: str, *, leaf: str | None = None) -> str:
    """Return the generated task name for one upstream task or leaf."""
    suffix = f"_{leaf}" if leaf is not None else ""
    return f"modelopt_vlm_benchmark_{task}{suffix}"


def load_quick_manifest(path: Path | None) -> dict[str, object]:
    """Load and validate the exact-row quick-suite manifest."""
    if path is None:
        raise ValueError("the quick suite requires --quick-manifest")
    try:
        manifest = json.loads(path.expanduser().absolute().read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"quick manifest is unreadable: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("quick manifest must contain an object")
    if manifest.get("schema") != "modelopt.vlm-benchmark-quick/v1":
        raise ValueError("quick manifest schema must be modelopt.vlm-benchmark-quick/v1")
    if manifest.get("lmms_eval_revision") != checkpoint.LMMS_EVAL_REVISION:
        raise ValueError("quick manifest lmms_eval_revision differs from the pinned profile")
    manifest_tasks = manifest.get("tasks")
    if not isinstance(manifest_tasks, dict) or set(manifest_tasks) != set(QUICK_TASKS):
        raise ValueError("quick manifest must contain exactly realworldqa, mmmu_val, and mvbench")
    for task, expected_count in _QUICK_COUNTS.items():
        _validate_manifest_task(task, manifest_tasks[task], expected_count)
    return manifest


def _validate_manifest_task(task: str, entry: object, expected_count: int) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"quick manifest task entry must be an object: {task}")
    if entry.get("dataset_revision") != profile.VLM_BENCHMARK_DATASETS[task].revision:
        raise ValueError(f"quick manifest dataset revision differs for {task}")
    rows = entry.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"quick manifest {task} must select exactly {expected_count} rows")

    seen: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"quick manifest {task} rows must be objects")
        index = row.get("source_row_index")
        source_id = row.get("source_sample_id")
        leaf = row.get("leaf_task")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"quick manifest {task} has an invalid source_row_index")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"quick manifest {task} has an invalid source_sample_id")
        _validate_source_identity(task, index=index, source_id=source_id, leaf=leaf)
        identity = (str(leaf or task), index)
        if identity in seen:
            raise ValueError(f"quick manifest {task} contains a duplicate source row")
        seen.add(identity)

    if task == "mvbench":
        leaf_counts = Counter(str(row["leaf_task"]) for row in rows)
        expected = {f"mvbench_{leaf}": 8 for leaf in MVBENCH_LEAF_TASKS}
        if leaf_counts != expected:
            raise ValueError("quick manifest MVBench must select exactly 8 rows per leaf task")


def _validate_source_identity(task: str, *, index: int, source_id: str, leaf: object) -> None:
    if task == "realworldqa" and source_id != f"test:{index}":
        raise ValueError("quick manifest RealWorldQA source identity mismatch")
    if task == "mvbench":
        expected_leaves = {f"mvbench_{name}" for name in MVBENCH_LEAF_TASKS}
        if leaf not in expected_leaves:
            raise ValueError("quick manifest MVBench row has an invalid leaf_task")
        config = str(leaf).removeprefix("mvbench_")
        if source_id != f"{config}:{index}":
            raise ValueError("quick manifest MVBench source identity mismatch")
    elif leaf is not None:
        raise ValueError(f"quick manifest {task} rows must not set leaf_task")


def manifest_sha256(manifest: dict[str, object]) -> str:
    """Return the digest of a canonical quick-suite manifest."""
    canonical = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def offline_dataset_snapshot(hf_home: Path, task: str, revision: str) -> Path:
    """Resolve an exact local Hub dataset snapshot without dereferencing mount aliases."""
    configured_hub_cache = os.environ.get("HF_HUB_CACHE")
    hub_cache = (
        Path(configured_hub_cache).expanduser().absolute()
        if configured_hub_cache
        else (hf_home / "hub").absolute()
    )
    repository = profile.VLM_BENCHMARK_DATASETS[task].repository
    repository_cache = f"datasets--{repository.replace('/', '--')}"
    snapshot = hub_cache / repository_cache / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError(f"pinned offline dataset snapshot is missing for {task}: {snapshot}")
    return snapshot


def generation_kwargs(task: str) -> dict[str, object]:
    """Return deterministic generation settings for one benchmark task."""
    return {
        "max_new_tokens": profile.VLM_BENCHMARK_DATASETS[task].max_new_tokens,
        "temperature": 0,
        "do_sample": False,
    }
