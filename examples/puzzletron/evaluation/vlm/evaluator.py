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

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import preflight, tasks

__all__ = ["evaluate"]

_COMPLETED_RUN_SCHEMA = "modelopt.vlm-evaluation-completed-run/v1"
_COMPLETED_RUN_FILENAME = "completed_run.json"


def _completion_identity(
    checkpoint_path: Path,
    settings: Mapping[str, object],
    *,
    repetition: int,
) -> dict[str, object]:
    """Describe the immutable inputs for one resumable evaluation repetition."""
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "repetition": repetition,
        "settings": dict(settings),
    }


def _load_completed_run(
    output_root: Path,
    *,
    identity: Mapping[str, object],
) -> dict[str, object] | None:
    """Load a completed repetition only when its inputs and artifacts still match."""
    completion_path = output_root / _COMPLETED_RUN_FILENAME
    if not completion_path.exists():
        return None
    try:
        payload = json.loads(completion_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid completed VLM evaluation record: {completion_path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != _COMPLETED_RUN_SCHEMA:
        raise RuntimeError(f"invalid completed VLM evaluation record: {completion_path}")
    if payload.get("identity") != dict(identity):
        raise RuntimeError(f"completed VLM evaluation inputs do not match: {completion_path}")
    result = payload.get("result")
    if not isinstance(result, Mapping) or not isinstance(result.get("metrics"), Mapping):
        raise RuntimeError(f"invalid completed VLM evaluation result: {completion_path}")
    result_path = result.get("result_path")
    if not isinstance(result_path, str) or not Path(result_path).is_file():
        raise RuntimeError(f"completed VLM evaluation artifact is missing: {result_path}")
    return dict(result)


def _write_completed_run(
    output_root: Path,
    *,
    identity: Mapping[str, object],
    result: Mapping[str, object],
) -> None:
    """Atomically mark one repetition complete after its result artifact exists."""
    completion_path = output_root / _COMPLETED_RUN_FILENAME
    output_root.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            {
                "identity": dict(identity),
                "result": dict(result),
                "schema": _COMPLETED_RUN_SCHEMA,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_root,
            prefix=f".{_COMPLETED_RUN_FILENAME}.",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, completion_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
            identity = _completion_identity(
                args.checkpoint,
                settings,
                repetition=repetition,
            )
            run_result = _load_completed_run(output_root, identity=identity)
            if run_result is None:
                run_result = checkpoint.run_lmms_eval_checkpoint(
                    args.checkpoint,
                    output_root=output_root,
                    settings=settings,
                )
                _write_completed_run(output_root, identity=identity, result=run_result)
            runs.append(run_result)
    return {"preflight": report, "runs": runs}
