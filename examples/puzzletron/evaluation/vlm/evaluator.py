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

"""Reusable VLM checkpoint evaluation interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import preflight, tasks

__all__ = ["evaluate"]


def evaluate(
    args: argparse.Namespace,
    *,
    settings_overrides: Mapping[str, object] | None = None,
    preflight_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Prepare and run one pinned VLM profile invocation."""

    prepared = preflight.prepare(args)
    task_root, configured_tasks = tasks.prepare(
        args.output_dir,
        suite=prepared.suite,
        dataset_snapshots=prepared.dataset_snapshots,
        quick_manifest=prepared.quick_manifest,
    )
    offline_task_preflight = tasks.verify_offline(
        task_root,
        configured_tasks,
        hf_home=prepared.hf_home,
        timeout_seconds=checkpoint.DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
    )
    report = dict(prepared.report)
    report.update(
        {
            "configured_tasks": list(configured_tasks),
            "offline_task_preflight": offline_task_preflight,
            "task_config_root": str(task_root),
        }
    )
    if preflight_callback is not None:
        preflight_callback(report)
    if args.preflight_only:
        return {"preflight": report, "runs": []}
    settings = preflight.settings(
        args,
        tasks_root=task_root,
        configured_tasks=configured_tasks,
        prepared=prepared,
    )
    settings.update(settings_overrides or {})
    repetitions = prepared.execution_policy["repetitions"]
    runs = []
    with checkpoint.without_huggingface_credentials():
        for repetition in range(1, repetitions + 1):
            output_root = args.output_dir
            if repetitions > 1:
                output_root = output_root / f"short-repetition-{repetition}"
            runs.append(
                checkpoint.run_lmms_eval_checkpoint(
                    args.checkpoint,
                    output_root=output_root,
                    settings=settings,
                )
            )
    return {"preflight": report, "runs": runs}
