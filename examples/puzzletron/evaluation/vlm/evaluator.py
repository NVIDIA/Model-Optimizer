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
from hashlib import sha256
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
    checkpoint_identity: Mapping[str, object],
    settings: Mapping[str, object],
    *,
    profile_identity: Mapping[str, object],
    repetition: int,
) -> dict[str, object]:
    """Describe the immutable inputs for one resumable evaluation repetition."""
    return {
        "checkpoint": dict(checkpoint_identity),
        "profile": dict(profile_identity),
        "repetition": repetition,
        "settings": dict(settings),
    }


def _file_identity(path: Path, *, root: Path) -> dict[str, object]:
    """Return a content identity for one checkpoint or evaluation artifact."""
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(root)),
        "sha256": digest.hexdigest(),
        "size": path.stat().st_size,
    }


def _checkpoint_identity(checkpoint_path: Path) -> dict[str, object]:
    """Fingerprint every local file that defines the evaluated checkpoint."""
    from modelopt.torch.puzzletron.distributed_eval.config import checkpoint_identity

    root = checkpoint_path.resolve()
    files = [_file_identity(path, root=root) for path in sorted(root.rglob("*")) if path.is_file()]
    if not files:
        raise RuntimeError(f"VLM evaluation checkpoint contains no files: {root}")
    return {**checkpoint_identity(root), "content_files": files}


def _artifact_inventory(output_root: Path) -> list[dict[str, object]]:
    """Fingerprint evaluator outputs needed to reuse a completed repetition."""
    return [
        _file_identity(path, root=output_root)
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and path.name != _COMPLETED_RUN_FILENAME
        and not path.name.startswith(f".{_COMPLETED_RUN_FILENAME}.")
    ]


def _artifacts_match(output_root: Path, artifacts: object) -> bool:
    """Return whether every recorded evaluator output remains unchanged."""
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(f"invalid completed VLM evaluation artifacts: {output_root}")
    for item in artifacts:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise RuntimeError(f"invalid completed VLM evaluation artifacts: {output_root}")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"invalid completed VLM evaluation artifact path: {relative}")
        path = output_root / relative
        if not path.is_file():
            return False
        if _file_identity(path, root=output_root) != dict(item):
            return False
    return True


def _validated_run_result(result: object, *, label: str) -> dict[str, object]:
    """Validate a fresh evaluator result before publishing its completion record."""
    if not isinstance(result, Mapping) or not isinstance(result.get("metrics"), Mapping):
        raise RuntimeError(f"invalid {label} result")
    result_path = result.get("result_path")
    if not isinstance(result_path, str) or not Path(result_path).is_file():
        raise RuntimeError(f"{label} result artifact is missing: {result_path}")
    return dict(result)


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
        return None
    result = payload.get("result")
    if not isinstance(result, Mapping) or not isinstance(result.get("metrics"), Mapping):
        raise RuntimeError(f"invalid completed VLM evaluation result: {completion_path}")
    result_path = result.get("result_path")
    if not isinstance(result_path, str) or not Path(result_path).is_file():
        return None
    if not _artifacts_match(output_root, payload.get("artifacts")):
        return None
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
    result = _validated_run_result(result, label="VLM evaluation")
    content = (
        json.dumps(
            {
                "artifacts": _artifact_inventory(output_root),
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
        source_tasks=prepared.source_tasks,
        profile_task_leaves=prepared.profile_task_leaves,
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
    settings = preflight.settings(
        args,
        tasks_root=task_root,
        configured_tasks=configured_tasks,
        prepared=prepared,
    )
    settings.update(settings_overrides or {})
    if preflight_callback is not None:
        preflight_callback(report)
    if args.preflight_only:
        return {
            "schema": "modelopt.vlm-evaluation-result/v1",
            "preflight": report,
            "runs": [],
        }
    repetitions = prepared.execution_policy["repetitions"]
    checkpoint_identity = _checkpoint_identity(args.checkpoint)
    profile_identity = {
        "dataset_revisions": report["dataset_revisions"],
        "lmms_eval_revision": report["lmms_eval_revision"],
        "quick_manifest_sha256": report.get("quick_manifest_sha256"),
        "profile_task": report.get("profile_task"),
        "profile_task_shard": report.get("profile_task_shard"),
        "source_tasks": report["source_tasks"],
        "suite": prepared.suite,
        "profile_fingerprint": report.get("profile_fingerprint"),
        "profile_name": report.get("profile_name"),
        "profile_schema": report.get("profile_schema"),
    }
    runs = []
    with checkpoint.without_huggingface_credentials():
        for repetition in range(1, repetitions + 1):
            output_root = args.output_dir
            if repetitions > 1:
                output_root = output_root / f"{prepared.suite}-repetition-{repetition}"
            identity = _completion_identity(
                checkpoint_identity,
                settings,
                profile_identity=profile_identity,
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
    return {
        "schema": "modelopt.vlm-evaluation-result/v1",
        "preflight": report,
        "runs": runs,
    }
