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

"""Identity contracts for post-MIP evaluation evidence."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..identity import canonicalize

__all__ = [
    "checkpoint_fingerprint",
    "collect_kd_exposure",
    "downstream_evaluation_identity",
    "evaluation_contract",
    "kd_exposure_metrics",
]

if TYPE_CHECKING:
    from .records import CandidateRevision

_PROFILE_FIELDS = (
    "profile",
    "suite",
    "lmms_eval_revision",
    "source_tasks",
    "dataset_revisions",
    "frame_policy",
    "generation_policy",
    "sample_limit",
    "quick_manifest_sha256",
    "repetitions",
)
_EXPOSURE_FIELDS = (
    "cumulative_steps",
    "global_batch_size",
    "cumulative_examples",
    "max_sample_length",
    "effective_tokens",
    "effective_tokens_source",
    "token_upper_bound",
)


def collect_kd_exposure(
    output: Path,
    configured: Mapping[str, Any],
    *,
    max_steps: int,
    elapsed_gpu_hours: float,
    resumed_completed_milestone: bool,
) -> dict[str, Any]:
    """Combine configured exposure with durable token and GPU-hour evidence."""

    exposure = copy.deepcopy(dict(configured))
    training_log = output / "checkpoints" / "training.jsonl"
    records = (
        [json.loads(line) for line in training_log.read_text().splitlines() if line.strip()]
        if training_log.is_file()
        else []
    )
    effective_tokens = sum(int(record.get("num_label_tokens", 0)) for record in records)
    if effective_tokens <= 0:
        raise RuntimeError("global KD produced no non-padding token accounting")

    exposure_root = output / "exposure"
    milestone_path = exposure_root / f"step_{max_steps:06d}.json"
    prior_milestone = json.loads(milestone_path.read_text()) if milestone_path.is_file() else {}
    prior_gpu_hours = sum(
        float(json.loads(path.read_text()).get("actual_incremental_gpu_hours", 0.0))
        for path in exposure_root.glob("step_*.json")
        if path != milestone_path and int(path.stem.removeprefix("step_")) < max_steps
    )
    incremental_gpu_hours = elapsed_gpu_hours
    cumulative_gpu_hours = prior_gpu_hours + elapsed_gpu_hours
    if resumed_completed_milestone and prior_milestone:
        incremental_gpu_hours = float(
            prior_milestone.get("actual_incremental_gpu_hours", elapsed_gpu_hours)
        )
        cumulative_gpu_hours = float(
            prior_milestone.get(
                "actual_cumulative_gpu_hours", prior_gpu_hours + incremental_gpu_hours
            )
        )
    exposure.update(
        effective_tokens=effective_tokens,
        effective_tokens_source="training.jsonl:num_label_tokens",
        token_upper_bound=(
            int(exposure["cumulative_examples"]) * int(exposure["max_sample_length"])
        ),
        actual_incremental_gpu_hours=incremental_gpu_hours,
        actual_cumulative_gpu_hours=cumulative_gpu_hours,
    )
    return exposure


def kd_exposure_metrics(exposure: Mapping[str, Any]) -> dict[str, float]:
    """Publish the comparable numeric exposure fields as node metrics."""

    return {
        f"exposure.{field}": float(exposure[field])
        for field in (
            "global_batch_size",
            "cumulative_examples",
            "effective_tokens",
            "estimated_cumulative_gpu_hours",
            "actual_incremental_gpu_hours",
            "actual_cumulative_gpu_hours",
        )
    }


def checkpoint_fingerprint(checkpoint: str | Path) -> str:
    """Return the checkpoint fingerprint used by evaluation identities."""

    from ..distributed_eval.config import checkpoint_identity

    return str(checkpoint_identity(checkpoint)["fingerprint"])


def _selected_json(path: Any, fields: tuple[str, ...]) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(str(path)).read_text())
    return {field: payload.get(field) for field in fields}


def downstream_evaluation_identity(
    *,
    source: CandidateRevision,
    reference_checkpoint: str | Path,
    profile: Any,
    evaluator_revision: Any,
    settings: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference_checkpoint_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Bind one comparison to its checkpoints, training, data, and evaluator."""

    evaluator_settings = {
        key: copy.deepcopy(value)
        for key, value in settings.items()
        if key not in {"row_manifest", "timeout_seconds"}
    }
    return canonicalize(
        {
            "candidate_checkpoint_fingerprint": checkpoint_fingerprint(
                source.artifact["checkpoint"]
            ),
            "reference_checkpoint_fingerprint": reference_checkpoint_fingerprint
            or checkpoint_fingerprint(reference_checkpoint),
            "architecture_id": source.architecture_id,
            "kd": {
                "producer_node": source.producer_node,
                "exposure": _selected_json(source.artifact.get("exposure_path"), _EXPOSURE_FIELDS),
            },
            "evaluator": {
                "profile": profile,
                "revision": evaluator_revision,
                "settings": evaluator_settings,
                "resolved_profile": _selected_json(candidate.get("profile_path"), _PROFILE_FIELDS),
            },
        }
    )


def evaluation_contract(
    identity: Any,
    *,
    label: str,
    expected_profile: str,
    expected_manifest_sha256: str,
    expected_reference_fingerprint: str,
) -> dict[str, Any]:
    """Validate and project the fields that must match across evaluations."""

    if not isinstance(identity, Mapping):
        raise RuntimeError(f"{label} evaluation identity must be a mapping")
    evaluator = identity.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise RuntimeError(f"{label} evaluation identity is missing evaluator")
    if not evaluator.get("revision"):
        raise RuntimeError(f"{label} evaluation identity is missing evaluator revision")
    evaluator_settings = evaluator.get("settings")
    resolved_profile = evaluator.get("resolved_profile")
    if not isinstance(evaluator_settings, Mapping) or not isinstance(resolved_profile, Mapping):
        raise RuntimeError(
            f"{label} evaluation identity is missing evaluator settings or resolved profile"
        )
    missing = set(_PROFILE_FIELDS) - resolved_profile.keys()
    if missing:
        raise RuntimeError(
            f"{label} evaluation identity resolved profile is missing {sorted(missing)}"
        )
    if evaluator.get("profile") != expected_profile:
        raise RuntimeError(f"{label} evaluation profile differs from the result contract")
    if evaluator_settings.get("row_manifest_sha256") != expected_manifest_sha256:
        raise RuntimeError(f"{label} evaluator row manifest differs from the result contract")
    if resolved_profile.get("quick_manifest_sha256") != expected_manifest_sha256:
        raise RuntimeError(f"{label} resolved row manifest differs from the result contract")
    if identity.get("reference_checkpoint_fingerprint") != expected_reference_fingerprint:
        raise RuntimeError(f"{label} reference checkpoint differs from the result contract")
    return canonicalize(
        {
            "reference_checkpoint_fingerprint": identity["reference_checkpoint_fingerprint"],
            "evaluator": {
                "profile": evaluator["profile"],
                "revision": evaluator["revision"],
                "settings": evaluator_settings,
                "resolved_profile": resolved_profile,
            },
        }
    )
