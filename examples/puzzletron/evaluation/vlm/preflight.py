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

"""Validate VLM suite inputs and build the shared evaluator settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import model, profile, suites, tasks

__all__ = ["PreparedSuite", "prepare", "settings"]


@dataclass(frozen=True)
class PreparedSuite:
    """Validated local inputs and policy for one VLM suite execution."""

    suite: str
    source_tasks: tuple[str, ...]
    dataset_snapshots: dict[str, Path]
    quick_manifest: dict[str, object] | None
    hf_home: Path
    judge_env: dict[str, str]
    report: dict[str, object]


def prepare(args: argparse.Namespace) -> PreparedSuite:
    """Validate model, task, dataset, media, and judge inputs for one suite."""
    suite = args.suite or "short"
    model.verify_checkpoint(args.checkpoint, profile="VLM benchmark")

    source_tasks = suites.source_tasks(suite)
    revisions = suites.dataset_revisions(source_tasks)
    quick_manifest = suites.load_quick_manifest(args.quick_manifest) if suite == "quick" else None
    if suite != "quick" and args.quick_manifest is not None:
        raise ValueError("--quick-manifest is valid only for the quick suite")

    lmms_eval_revision = checkpoint.verify_lmms_eval_revision()
    for task in source_tasks:
        tasks.task_config(profile.VLM_BENCHMARK_DATASETS[task].task_config)

    hf_home = _hf_home(args.hf_home)
    _verify_media_roots(hf_home, source_tasks)
    dataset_snapshots = {
        task: suites.offline_dataset_snapshot(hf_home, task, revisions[task])
        for task in source_tasks
    }
    judge_env, judge_policy = _judge_policy(args, suite)
    report = _report(
        args,
        suite=suite,
        source_tasks=source_tasks,
        revisions=revisions,
        dataset_snapshots=dataset_snapshots,
        quick_manifest=quick_manifest,
        hf_home=hf_home,
        judge_policy=judge_policy,
        lmms_eval_revision=lmms_eval_revision,
    )
    return PreparedSuite(
        suite=suite,
        source_tasks=source_tasks,
        dataset_snapshots=dataset_snapshots,
        quick_manifest=quick_manifest,
        hf_home=hf_home,
        judge_env=judge_env,
        report=report,
    )


def _hf_home(configured: Path | None) -> Path:
    value = configured or os.environ.get("HF_HOME")
    if value is None:
        raise ValueError("VLM benchmark requires an explicit --hf-home or HF_HOME")
    hf_home = Path(value).expanduser().absolute()
    if not hf_home.is_dir():
        raise ValueError(f"Hugging Face cache root is not a directory: {hf_home}")
    return hf_home


def _verify_media_roots(hf_home: Path, source_tasks: tuple[str, ...]) -> None:
    media_roots = {
        task: hf_home / media_dir
        for task in source_tasks
        if (media_dir := profile.VLM_BENCHMARK_DATASETS[task].media_dir) is not None
    }
    missing = [
        str(root) for root in media_roots.values() if not root.is_dir() or not any(root.iterdir())
    ]
    if missing:
        raise ValueError(f"VLM benchmark media roots are missing or empty: {missing}")


def _judge_policy(
    args: argparse.Namespace, suite: str
) -> tuple[dict[str, str], dict[str, object] | None]:
    options_set = any(
        (
            args.allow_judge_calls,
            args.mmvu_judge_api_type is not None,
            args.mmvu_judge_model is not None,
        )
    )
    if suite != "full":
        if options_set:
            raise ValueError("MMVU judge options are valid only for the full suite")
        return {}, None
    if not args.allow_judge_calls:
        raise ValueError("the full suite requires explicit --allow-judge-calls for MMVU")
    if args.mmvu_judge_api_type is None or not args.mmvu_judge_model:
        raise ValueError("the full suite requires --mmvu-judge-api-type and --mmvu-judge-model")

    credential_names = (
        ("OPENAI_API_KEY",)
        if args.mmvu_judge_api_type == "openai"
        else ("AZURE_API_KEY", "AZURE_ENDPOINT")
    )
    missing = [name for name in credential_names if not os.environ.get(name)]
    if missing:
        raise ValueError(f"MMVU judge credentials are missing: {missing}")
    judge_env = {
        "API_TYPE": args.mmvu_judge_api_type,
        "MODEL_VERSION": args.mmvu_judge_model,
    }
    policy = {
        "api_type": args.mmvu_judge_api_type,
        "model": args.mmvu_judge_model,
        "cache": "none",
        "retries": 5,
        "failure": "fail_closed_for_unjudged_open_ended_answers",
        "credential_variables": list(credential_names),
    }
    return judge_env, policy


def _report(
    args: argparse.Namespace,
    *,
    suite: str,
    source_tasks: tuple[str, ...],
    revisions: dict[str, str],
    dataset_snapshots: dict[str, Path],
    quick_manifest: dict[str, object] | None,
    hf_home: Path,
    judge_policy: dict[str, object] | None,
    lmms_eval_revision: str,
) -> dict[str, object]:
    return {
        "profile": suites.EVALUATION_PROFILE,
        "suite": suite,
        "checkpoint": str(args.checkpoint),
        "expected_source_model_revision": model.SOURCE_MODEL_REVISION,
        "lmms_eval_revision": lmms_eval_revision,
        "source_tasks": list(source_tasks),
        "dataset_revisions": revisions,
        "dataset_snapshots": {task: str(snapshot) for task, snapshot in dataset_snapshots.items()},
        "hf_home": str(hf_home),
        "frame_policy": {"reader": "torchcodec", "fps": 2, "max_frames": 32},
        "generation_policy": {
            "enable_thinking": False,
            "temperature": 0,
            "do_sample": False,
        },
        "quick_selected_rows": 344 if suite == "quick" else None,
        "judge_free_mmvu_rows": (
            [row[0] for row in suites.MMVU_SMOKE_ROWS] if suite == "mmvu-smoke" else None
        ),
        "quick_manifest_sha256": (
            suites.manifest_sha256(quick_manifest) if quick_manifest is not None else None
        ),
        "short_repetitions": 2 if suite == "short" else None,
        "judge_policy": judge_policy,
        "network_policy": "offline_cached_data_only",
    }


def settings(
    args: argparse.Namespace,
    *,
    tasks_root: Path,
    configured_tasks: tuple[str, ...],
    prepared: PreparedSuite,
) -> dict[str, object]:
    """Build shared runner settings for a validated VLM suite."""
    timeout_seconds = args.timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = (
            suites.DEFAULT_SMOKE_TIMEOUT_SECONDS
            if prepared.suite in suites.SMOKE_SUITES
            else suites.DEFAULT_FULL_TIMEOUT_SECONDS
        )
    return {
        "model": "qwen3_5",
        "checkpoint_arg": "pretrained",
        "tasks": ",".join(configured_tasks),
        "limit": 8 if prepared.suite in suites.SMOKE_SUITES else None,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timeout_seconds": timeout_seconds,
        "model_args": {"enable_thinking": False, "fps": 2, "max_frames": 32},
        "gen_kwargs": {"temperature": 0, "do_sample": False},
        "env": {
            "FORCE_QWENVL_VIDEO_READER": "torchcodec",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(prepared.hf_home),
            "HF_HUB_OFFLINE": "1",
            **prepared.judge_env,
        },
        "extra_args": ["--include_path", str(tasks_root)],
    }
