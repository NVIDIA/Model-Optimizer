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

import importlib.util
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, TemplateError

if TYPE_CHECKING:
    import argparse

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import contracts, model, profile, suites, tasks

__all__ = ["PreparedSuite", "prepare", "settings"]

_NO_THINK_TEMPLATE_PREFIX = "{%- set enable_thinking = false %}\n"
_NO_THINK_GENERATION_PREFIX = "<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class PreparedSuite:
    """Validated local inputs and policy for one VLM suite execution."""

    suite: str
    source_tasks: tuple[str, ...]
    profile_task_leaves: tuple[str, ...] | None
    dataset_snapshots: dict[str, Path]
    quick_manifest: dict[str, object] | None
    hf_home: Path
    judge_env: dict[str, str]
    execution_policy: suites.ExecutionPolicy
    profile_contract: contracts.ProfileContract | None
    report: dict[str, object]


def prepare(args: argparse.Namespace) -> PreparedSuite:
    """Validate model, task, dataset, media, and judge inputs for one suite."""
    profile_name = getattr(args, "profile", None)
    profile_contract = contracts.load_profile(profile_name) if profile_name is not None else None
    requested_suite = profile_contract.name if profile_contract is not None else args.suite or "short"
    suite = suites.canonical_suite(requested_suite)
    if suite != requested_suite:
        warnings.warn(
            f"VLM suite {requested_suite!r} is deprecated; use {suite!r}",
            FutureWarning,
            stacklevel=2,
        )
    if profile_contract is not None and args.seed != profile_contract.manifest["seed"]:
        raise ValueError("--seed cannot override a versioned evaluation profile")
    if profile_contract is not None and args.batch_size != profile_contract.manifest["batch_size"]:
        raise ValueError("--batch-size cannot override a versioned evaluation profile")
    profile_task = getattr(args, "profile_task", None)
    profile_task_shard = getattr(args, "profile_task_shard", None)
    if profile_task is not None:
        if profile_contract is None:
            raise ValueError("--profile-task requires a versioned evaluation profile")
        if profile_contract.name != "full-v1":
            raise ValueError("--profile-task is supported only for full-v1")
    if profile_task_shard is not None and profile_task is None:
        raise ValueError("--profile-task-shard requires --profile-task")
    model.verify_checkpoint(args.checkpoint, profile="VLM benchmark")

    source_tasks = suites.source_tasks(suite)
    if profile_task is not None:
        if profile_task not in source_tasks:
            raise ValueError(f"--profile-task is not part of {suite}: {profile_task}")
        source_tasks = (profile_task,)
    profile_task_leaves = _profile_task_leaves(profile_task, profile_task_shard)
    execution_policy = suites.execution_policy(suite, timeout_seconds=args.timeout_seconds)
    revisions = {task: profile.VLM_BENCHMARK_DATASETS[task].revision for task in source_tasks}
    if profile_contract is not None and profile_contract.exact_rows is not None:
        quick_manifest = suites.validate_quick_manifest(profile_contract.exact_rows)
    elif suite == "quick":
        quick_manifest = suites.load_quick_manifest(args.quick_manifest)
    else:
        quick_manifest = None
    if suite not in {"quick"} and profile_contract is None and args.quick_manifest is not None:
        raise ValueError("--quick-manifest is valid only for the quick suite")
    if profile_contract is not None and args.quick_manifest is not None:
        raise ValueError("--quick-manifest cannot override a versioned evaluation profile")

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
        execution_policy=execution_policy,
        profile_contract=profile_contract,
        profile_task_leaves=profile_task_leaves,
    )
    return PreparedSuite(
        suite=suite,
        source_tasks=source_tasks,
        profile_task_leaves=profile_task_leaves,
        dataset_snapshots=dataset_snapshots,
        quick_manifest=quick_manifest,
        hf_home=hf_home,
        judge_env=judge_env,
        execution_policy=execution_policy,
        profile_contract=profile_contract,
        report=report,
    )


def _profile_task_leaves(
    profile_task: str | None, shard: tuple[int, int] | None
) -> tuple[str, ...] | None:
    """Resolve a grouped profile task's deterministic leaf partition."""
    if shard is None:
        return None
    if profile_task is None:
        raise ValueError("--profile-task-shard requires --profile-task")
    leaves = {
        "mvbench": suites.MVBENCH_LEAF_TASKS,
        "video_mmmu": suites.VIDEO_MMMU_LEAF_TASKS,
    }.get(profile_task)
    if leaves is None:
        raise ValueError("--profile-task-shard supports only mvbench and video_mmmu")
    index, count = shard
    selected = leaves[index::count]
    if not selected:
        raise ValueError("--profile-task-shard selects no task leaves")
    return selected


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
    execution_policy: suites.ExecutionPolicy,
    profile_contract: contracts.ProfileContract | None,
    profile_task_leaves: tuple[str, ...] | None,
) -> dict[str, object]:
    profile_task_shard = getattr(args, "profile_task_shard", None)
    return {
        "schema": "modelopt.vlm-evaluation-preflight/v1",
        "profile": suites.EVALUATION_PROFILE,
        "profile_name": profile_contract.name if profile_contract is not None else None,
        "profile_schema": (
            profile_contract.manifest["schema"] if profile_contract is not None else None
        ),
        "profile_fingerprint": (
            profile_contract.fingerprint if profile_contract is not None else None
        ),
        "suite": suite,
        "checkpoint": str(args.checkpoint),
        "lmms_eval_revision": lmms_eval_revision,
        "model_backend": "vllm",
        "backend_limitations": [
            "generic vLLM video messages do not preserve native Qwen 3.5 timestamps",
        ],
        "source_tasks": list(source_tasks),
        "profile_task": getattr(args, "profile_task", None),
        "profile_task_shard": (
            {
                "index": profile_task_shard[0],
                "count": profile_task_shard[1],
                "leaves": list(profile_task_leaves or ()),
            }
            if profile_task_shard is not None
            else None
        ),
        "dataset_revisions": revisions,
        "dataset_snapshots": {task: str(snapshot) for task, snapshot in dataset_snapshots.items()},
        "hf_home": str(hf_home),
        "frame_policy": execution_policy["frame"],
        "generation_policy": execution_policy["generation"],
        "batch_size": args.batch_size,
        "sample_limit": execution_policy["limit"],
        "timeout_seconds": execution_policy["timeout_seconds"],
        "quick_selected_rows": (
            suites.QUICK_SELECTED_ROWS if suite in {"quick", "short-v1"} else None
        ),
        "judge_free_mmvu_rows": (
            [row[0] for row in suites.MMVU_SMOKE_ROWS] if suite == "mmvu-smoke" else None
        ),
        "quick_manifest_sha256": (
            suites.manifest_sha256(quick_manifest) if quick_manifest is not None else None
        ),
        "short_repetitions": execution_policy["repetitions"] if suite == "short" else None,
        "repetitions": execution_policy["repetitions"],
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
    _verify_video_reader(prepared.source_tasks)
    execution_policy = prepared.execution_policy
    frame_policy = execution_policy["frame"]
    generation_policy = execution_policy["generation"]
    chat_template = _no_think_chat_template(args.checkpoint, tasks_root)
    return {
        "model": "vllm",
        "checkpoint_arg": "model",
        "tasks": ",".join(configured_tasks),
        "limit": execution_policy["limit"],
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timeout_seconds": execution_policy["timeout_seconds"],
        "reasoning_parser": "qwen3",
        "log_samples": prepared.suite in {"quick", "short", suites.TASK_PREFIX100_REPEAT2_SUITE},
        "model_args": {
            "chat_template": str(chat_template),
            "fps": frame_policy["fps"],
            "max_frame_num": frame_policy["max_frames"],
        },
        "gen_kwargs": {
            "temperature": generation_policy["temperature"],
            "do_sample": generation_policy["do_sample"],
        },
        "env": {
            "FORCE_QWENVL_VIDEO_READER": frame_policy["reader"],
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(prepared.hf_home),
            "HF_HUB_OFFLINE": "1",
            **({} if prepared.judge_env else checkpoint.lmms_eval_disabled_judge_environment()),
            **prepared.judge_env,
        },
        "extra_args": ["--include_path", str(tasks_root)],
    }


def _no_think_chat_template(checkpoint_path: Path, tasks_root: Path) -> Path:
    """Materialize the checkpoint template with Qwen thinking disabled."""
    source = checkpoint_path / "chat_template.jinja"
    try:
        content = source.read_text()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Qwen 3.5 chat template is unreadable: {source}") from error
    rendered = _render_no_think_template(_NO_THINK_TEMPLATE_PREFIX + content, source=source)
    if not rendered.endswith(_NO_THINK_GENERATION_PREFIX):
        raise ValueError(f"Qwen 3.5 chat template cannot disable thinking: {source}")
    target = tasks_root / "modelopt_qwen35_no_think.jinja"
    target.write_text(_NO_THINK_TEMPLATE_PREFIX + content)
    return target


def _render_no_think_template(content: str, *, source: Path) -> str:
    try:
        return (
            Environment()
            .from_string(content)
            .render(
                add_generation_prompt=True,
                messages=[{"content": "test", "role": "user"}],
            )
        )
    except TemplateError as error:
        raise ValueError(f"Qwen 3.5 chat template is invalid: {source}") from error


def _verify_video_reader(source_tasks: tuple[str, ...]) -> None:
    """Fail before evaluation when a selected video task has no decord reader."""
    video_selected = any(
        profile.VLM_BENCHMARK_DATASETS[task].media_dir is not None for task in source_tasks
    )
    if video_selected and importlib.util.find_spec("decord") is None:
        raise RuntimeError(
            "video evaluation requires an installed decord-compatible reader; install one "
            "supported by this Python and platform, or use the supported Puzzletron environment"
        )
