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

"""One portable structured result for live and completed Puzzletron runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from .recipe_config import bundle_for_run_root
from .state import CampaignStateStore

if TYPE_CHECKING:
    from .dashboard import StageView
    from .result_render import RenderedView
    from .schema import CampaignPlan

__all__ = [
    "RESULT_SCHEMA",
    "ResultValidationError",
    "canonical_json_bytes",
    "compare_metrics",
    "export_run_result",
    "finalized_result_path",
    "inspect_run",
    "publish_controller_result",
    "refresh_run_report",
    "result_path",
    "result_sha256",
    "validate_result",
]

RESULT_SCHEMA = "modelopt.puzzletron.run-result/v1"
_RENDERER_REVISION = "modelopt.puzzletron.html-summary/v1"
_TERMINAL_STATES = {"cancelled", "completed", "failed"}
_RUN_STATES = _TERMINAL_STATES | {"planned", "running", "submitted", "unknown", "waiting_for_input"}
_STAGE_STATES = {
    "blocked",
    "cancelled",
    "completed",
    "failed",
    "pending",
    "ready",
    "running",
    "skipped",
    "submitted",
    "unknown",
}
_TOTAL_KINDS = {"configured", "discovered", "estimated", "exact", "unavailable"}
_TOKEN_METRICS = {
    "quality.token_accuracy": ("ratio", "target_token_weighted_mean"),
    "serving.observed_input_sequence_length": ("tokens", None),
    "serving.observed_output_sequence_length": ("tokens", None),
    "serving.output_token_throughput": ("tokens_per_second", None),
    "serving.output_token_throughput_per_user": ("tokens_per_second_per_user", None),
    "serving.requested_input_tokens": ("tokens", None),
    "serving.requested_output_tokens": ("tokens", None),
    "training.effective_tokens": ("tokens", "sum"),
}
_WORKLOAD_FIELDS = ("workload_id", "task", "row_manifest_id", "prompt_template_id", "decoding_id")
_EVALUATION_PROGRESS = re.compile(r"^evaluation\s+(?P<task>\S+)\s+")


class ResultValidationError(ValueError):
    """Raised when a structured run result violates its public contract."""


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return deterministic JSON bytes suitable for atomic storage and hashing."""

    try:
        return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    except (TypeError, ValueError) as exc:
        raise ResultValidationError("result must contain only finite JSON values") from exc


def result_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_timestamp(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultValidationError(f"{description} must be a mapping")
    return value


def _sequence(value: object, description: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ResultValidationError(f"{description} must be a sequence")
    return value


def _string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResultValidationError(f"{description} must be a non-empty string")
    return value


def _number(value: object, description: str, *, nullable: bool = False) -> int | float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ResultValidationError(f"{description} must be a non-negative number")
    return value


def _timestamp(value: object, description: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    value = _string(value, description)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResultValidationError(f"{description} must be ISO 8601") from exc
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise ResultValidationError(f"{description} must be in UTC")
    return value


def _enum(value: object, allowed: set[str], description: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ResultValidationError(f"{description} must be one of: {', '.join(sorted(allowed))}")
    return value


def _relative_path(value: object, description: str) -> str:
    value = _string(value, description)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or "\\" in value:
        raise ResultValidationError(f"{description} must be a normalized relative POSIX path")
    return value


def _validate_progress(progress: Mapping[str, Any], description: str) -> None:
    _enum(progress.get("status"), _STAGE_STATES, f"{description} status")
    measures = _sequence(progress.get("measures"), f"{description} measures")
    if not measures:
        raise ResultValidationError(f"{description} measures must not be empty")
    primary = []
    for index, raw in enumerate(measures):
        measure = _mapping(raw, f"{description} measure {index}")
        _string(measure.get("name"), f"{description} measure name")
        completed = _number(measure.get("completed"), f"{description} completed")
        total = _number(measure.get("total"), f"{description} total", nullable=True)
        total_kind = _enum(measure.get("total_kind"), _TOTAL_KINDS, f"{description} total kind")
        _string(measure.get("unit"), f"{description} unit")
        if (total_kind == "unavailable") != (total is None):
            raise ResultValidationError("progress total and total_kind disagree")
        if total is not None and completed is not None and completed > total:
            raise ResultValidationError("progress completed cannot exceed total")
        if not isinstance(measure.get("primary"), bool):
            raise ResultValidationError("progress primary flags must be boolean")
        if measure["primary"]:
            primary.append(total_kind)
    if len(primary) != 1:
        raise ResultValidationError("progress must have exactly one primary measure")
    eta = _mapping(progress.get("eta"), f"{description} ETA")
    qualified = eta.get("qualified")
    if not isinstance(qualified, bool):
        raise ResultValidationError("progress ETA qualified must be boolean")
    seconds = _number(eta.get("seconds"), "progress ETA seconds", nullable=True)
    reason = eta.get("unavailable_reason")
    if qualified:
        if seconds is None or primary[0] not in {"configured", "discovered", "exact"}:
            raise ResultValidationError("qualified ETA requires seconds and a stable total")
        if reason is not None:
            raise ResultValidationError("qualified ETA cannot have an unavailable reason")
    elif seconds is not None or not isinstance(reason, str) or not reason:
        raise ResultValidationError("unqualified ETA requires a reason and null seconds")
    _timestamp(progress.get("updated_at"), f"{description} updated_at")


def _validate_metric(metric: Mapping[str, Any], index: int) -> None:
    prefix = f"metric {index}"
    _string(metric.get("metric_id"), f"{prefix} id")
    name = _string(metric.get("name"), f"{prefix} name")
    unit = _string(metric.get("unit"), f"{prefix} unit")
    aggregation = _string(metric.get("aggregation"), f"{prefix} aggregation")
    _string(metric.get("subject_id"), f"{prefix} subject")
    _string(metric.get("checkpoint_id"), f"{prefix} checkpoint")
    _string(metric.get("producer_execution_id"), f"{prefix} producer")
    _enum(
        metric.get("direction"),
        {"higher_is_better", "lower_is_better", "neutral"},
        f"{prefix} direction",
    )
    _mapping(metric.get("workload"), f"{prefix} workload")
    dimensions = _mapping(metric.get("dimensions", {}), f"{prefix} dimensions")
    state = _enum(
        metric.get("value_state"),
        {"invalid", "missing", "not_applicable", "present"},
        f"{prefix} state",
    )
    value, reason = metric.get("value"), metric.get("missing_reason")
    numeric_collection = isinstance(value, (Mapping, Sequence)) and not isinstance(value, str)
    if state == "present":
        if value is None or isinstance(value, bool) or numeric_collection or reason is not None:
            raise ResultValidationError(f"{prefix} present value must be a scalar without a reason")
    elif value is not None or not isinstance(reason, str) or not reason:
        raise ResultValidationError(f"{prefix} missing value requires a reason")
    lm_loss_contract = (aggregation, dimensions.get("denominator"))
    if name == "quality.lm_loss" and (
        unit != "nats_per_target_token"
        or metric.get("direction") != "lower_is_better"
        or lm_loss_contract
        not in {
            ("target_token_weighted_mean", "unmasked_target_tokens"),
            ("mean_of_sample_token_means", "unmasked_target_tokens_per_sample"),
        }
    ):
        raise ResultValidationError("quality.lm_loss uses an unsupported measurement contract")
    if name == "quality.token_accuracy" and (
        metric.get("direction") != "higher_is_better"
        or dimensions.get("denominator") != "unmasked_target_tokens"
    ):
        raise ResultValidationError("quality.token_accuracy uses an unsupported denominator")
    if name in _TOKEN_METRICS:
        expected_unit, expected_aggregation = _TOKEN_METRICS[name]
        if (
            unit != expected_unit
            or expected_aggregation is not None
            and aggregation != expected_aggregation
        ):
            raise ResultValidationError(f"{name} uses an unsupported unit or aggregation")
    if name == "training.effective_tokens":
        for field in ("tokenizer_id", "data_id", "exposure_kind", "value_source"):
            _string(dimensions.get(field), f"training.effective_tokens {field}")


def validate_result(result: Mapping[str, Any]) -> None:
    """Validate the single-file public result contract."""

    if result.get("schema") != RESULT_SCHEMA:
        raise ResultValidationError("unsupported result schema")
    run = _mapping(result.get("run"), "run")
    identity = _mapping(run.get("identity"), "run identity")
    run_id = _string(identity.get("run_id"), "run id")
    status = _mapping(run.get("status"), "run status")
    _enum(status.get("execution"), _RUN_STATES, "run execution status")
    _enum(
        status.get("attachment"),
        {"attached", "detached", "not_running", "unknown"},
        "run attachment status",
    )
    _enum(
        status.get("evidence"),
        {"complete", "none", "partial", "preliminary", "teacher_only"},
        "run evidence status",
    )
    _enum(
        status.get("support"),
        {"supported", "superseded", "unreviewed", "unsupported"},
        "run support status",
    )
    timing = _mapping(run.get("timing"), "run timing")
    for field in ("created_at", "updated_at"):
        _timestamp(timing.get(field), f"run timing {field}")
    for field in ("started_at", "ended_at", "finalized_at", "last_executor_observation_at"):
        _timestamp(timing.get(field), f"run timing {field}", nullable=True)
    _number(
        timing.get("elapsed_since_first_submission_seconds"),
        "run elapsed since first submission",
        nullable=True,
    )
    freshness = _mapping(run.get("freshness"), "run freshness")
    _timestamp(freshness.get("as_of"), "run freshness as_of", nullable=True)
    _number(freshness.get("stale_after_seconds"), "run stale_after_seconds")
    _enum(freshness.get("state"), {"fresh", "stale", "unknown"}, "run freshness state")
    bundle = _mapping(
        _mapping(result.get("provenance"), "provenance").get("resolved_bundle"), "resolved bundle"
    )
    if bundle.get("bundle_id") != run_id:
        raise ResultValidationError("result run id does not match its resolved bundle")
    _string(bundle.get("producer_revision"), "resolved bundle producer revision")
    for field in ("manifest", "provenance"):
        _relative_path(bundle.get(field), f"resolved bundle {field}")

    stage_ids: set[str] = set()
    stages = _sequence(result.get("stages"), "stages")
    for index, raw in enumerate(stages):
        stage = _mapping(raw, f"stage {index}")
        stage_id = _string(stage.get("stage_id"), f"stage {index} id")
        if stage_id in stage_ids:
            raise ResultValidationError(f"duplicate stage id: {stage_id}")
        stage_ids.add(stage_id)
        _enum(stage.get("state"), _STAGE_STATES, f"stage {stage_id} state")
        _sequence(stage.get("parent_stage_ids"), f"stage {stage_id} parents")
        _sequence(
            stage.get("external_prerequisite_stage_ids", ()),
            f"stage {stage_id} external prerequisites",
        )
        _sequence(stage.get("attempts"), f"stage {stage_id} attempts")
        _number(stage.get("elapsed_seconds"), f"stage {stage_id} elapsed", nullable=True)
        _validate_progress(
            _mapping(stage.get("progress"), f"stage {stage_id} progress"),
            f"stage {stage_id} progress",
        )
    for stage in stages:
        if set(stage["parent_stage_ids"]) - stage_ids:
            raise ResultValidationError(f"stage {stage['stage_id']} has unknown parents")

    subject_ids: set[str] = set()
    for index, raw in enumerate(_sequence(result.get("subjects"), "subjects")):
        subject = _mapping(raw, f"subject {index}")
        subject_id = _string(subject.get("subject_id"), f"subject {index} id")
        if subject_id in subject_ids:
            raise ResultValidationError(f"duplicate subject id: {subject_id}")
        subject_ids.add(subject_id)
        _string(subject.get("role"), f"subject {index} role")
        _mapping(subject.get("checkpoint"), f"subject {index} checkpoint")
        _mapping(subject.get("architecture"), f"subject {index} architecture")
    metric_ids: set[str] = set()
    for index, raw in enumerate(_sequence(result.get("metrics"), "metrics")):
        metric = _mapping(raw, f"metric {index}")
        _validate_metric(metric, index)
        metric_id = str(metric["metric_id"])
        if metric_id in metric_ids:
            raise ResultValidationError(f"duplicate metric id: {metric_id}")
        if metric["subject_id"] not in subject_ids:
            raise ResultValidationError(f"metric {metric_id} references an unknown subject")
        metric_ids.add(metric_id)
    artifact_ids: set[str] = set()
    for index, raw in enumerate(_sequence(result.get("artifacts"), "artifacts")):
        artifact = _mapping(raw, f"artifact {index}")
        artifact_id = _string(artifact.get("artifact_id"), f"artifact {index} id")
        if artifact_id in artifact_ids:
            raise ResultValidationError(f"duplicate artifact id: {artifact_id}")
        artifact_ids.add(artifact_id)
        _string(artifact.get("role"), f"artifact {index} role")
        _relative_path(artifact.get("path"), f"artifact {index} path")
    limitations = _sequence(result.get("limitations"), "limitations")
    if any(not isinstance(item, str) or not item for item in limitations):
        raise ResultValidationError("limitations must contain non-empty strings")
    canonical_json_bytes(result)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def result_path(run_root: str | Path) -> Path:
    """Return the canonical structured-result path for a run root."""

    return Path(run_root) / "results" / "result.json"


def finalized_result_path(plan: CampaignPlan) -> Path | None:
    """Return the validated final result for the plan's active resolved bundle."""

    path = result_path(plan.puzzle_dir)
    try:
        result = json.loads(path.read_text())
        validate_result(result)
        expected_run_id, _, _ = _active_bundle(plan.puzzle_dir, fallback_run_id=plan.contract_hash)
        run = _mapping(result.get("run"), "run")
        identity = _mapping(run.get("identity"), "run identity")
        status = _mapping(run.get("status"), "run status")
        timing = _mapping(run.get("timing"), "run timing")
    except (OSError, TypeError, ValueError):
        return None
    if (
        identity.get("run_id") != expected_run_id
        or identity.get("workflow_id") != plan.contract_hash
        or status.get("execution") != "completed"
        or status.get("attachment") != "not_running"
        or status.get("evidence") != "complete"
        or timing.get("finalized_at") is None
    ):
        return None
    return path


def _active_bundle(run_root: Path, *, fallback_run_id: str) -> tuple[str, str, dict[str, str]]:
    if not (run_root / "orchestration" / "current_bundle.json").is_file():
        return (
            fallback_run_id,
            "modelopt.puzzletron.orchestrator/v1",
            {
                "manifest": "orchestration/compiled_plan.json",
                "provenance": "orchestration/compiled_plan.json",
            },
        )
    try:
        bundle_root = bundle_for_run_root(run_root)
        manifest = json.loads((bundle_root / "manifest.json").read_text())
        bundle_id = _string(manifest.get("bundle_id"), "active resolved bundle id")
        controller = _mapping(
            manifest.get("code", {}).get("controller", {}), "controller provenance"
        )
        relative_root = bundle_root.relative_to(run_root).as_posix()
        return (
            bundle_id,
            str(controller.get("revision") or "unknown"),
            {
                "manifest": f"{relative_root}/manifest.json",
                "provenance": f"{relative_root}/provenance.json",
            },
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ResultValidationError("invalid active resolved bundle") from exc


def _attempts(state: CampaignStateStore, stage_id: str, root: Path) -> list[dict[str, Any]]:
    return [
        {
            "attempt_id": str(attempt["attempt_id"]),
            "number": number,
            "status": str(attempt.get("status") or "unknown"),
            "submitted_at": _utc_timestamp(attempt.get("submitted_at")),
            "ended_at": _utc_timestamp(attempt.get("completed_at")),
            "log_paths": [
                relative
                for path in attempt.get("log_paths") or ()
                if (relative := _run_relative(root, path)) is not None
            ],
        }
        for number, attempt in enumerate(state.list_attempts(stage_id), start=1)
    ]


def _progress(view: StageView, observed_at: str) -> dict[str, Any]:
    if view.current is not None and view.total is not None:
        completed, total, kind, name, unit = (
            view.current,
            view.total,
            "discovered",
            "stage_native_work",
            "items",
        )
    else:
        completed, total, kind, name, unit = (
            (1 if view.status == "completed" else 0),
            1,
            "exact",
            "stage_completion",
            "stage",
        )
    qualified = view.eta_seconds is not None
    scope: dict[str, Any] = {"stage_id": view.stage_id, "configured_task_count": view.tasks}
    if match := _EVALUATION_PROGRESS.match(view.progress):
        scope["task"] = match.group("task")
    return {
        "status": "pending" if view.status == "waiting" else view.status,
        "scope": scope,
        "measures": [
            {
                "name": name,
                "completed": completed,
                "total": total,
                "unit": unit,
                "total_kind": kind,
                "primary": True,
                "dimensions": {"detail": view.progress},
            }
        ],
        "eta": {
            "seconds": view.eta_seconds if qualified else None,
            "qualified": qualified,
            "method": "observed_controller_rate",
            "unavailable_reason": None if qualified else "insufficient_observations",
        },
        "updated_at": observed_at,
    }


def _run_relative(root: Path, value: Any) -> str | None:
    """Return a portable run-relative path for an existing producer artifact."""

    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    return relative if path.exists() else None


def _metric_semantics(raw_name: str) -> tuple[str, str, str, str, dict[str, Any]]:
    """Qualify existing producer metrics without overstating their aggregation."""

    name = raw_name.removeprefix("candidate.").removeprefix("reference.")
    dimensions: dict[str, Any] = {"producer_metric": name}
    if name == "lm_loss":
        return (
            "quality.lm_loss",
            "nats_per_target_token",
            "lower_is_better",
            "mean_of_sample_token_means",
            {**dimensions, "denominator": "unmasked_target_tokens_per_sample"},
        )
    if name == "token_accuracy":
        return (
            "producer.token_accuracy",
            "ratio",
            "higher_is_better",
            "producer_defined",
            dimensions,
        )
    if name.startswith("token_accuracy") or "accuracy" in name or "exact_match" in name:
        return f"quality.{name}", "ratio", "higher_is_better", "producer_defined", dimensions
    if name in {"input_sequence_length", "output_sequence_length"}:
        return (
            f"serving.observed_{name}",
            "tokens",
            "neutral",
            "producer_defined",
            dimensions,
        )
    if "throughput" in name:
        unit = "tokens_per_second" if "token" in name else "requests_per_second"
        return f"serving.{name}", unit, "higher_is_better", "producer_defined", dimensions
    if any(term in name for term in ("latency", "ttft", "tpot")):
        return f"serving.{name}", "milliseconds", "lower_is_better", "producer_defined", dimensions
    if "loss" in name or "div" in name:
        return (
            f"quality.{name}",
            "producer_defined",
            "lower_is_better",
            "producer_defined",
            dimensions,
        )
    return f"producer.{name}", "producer_defined", "neutral", "producer_defined", dimensions


def _project_post_mip_evidence(
    root: Path,
) -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]
    | None
):
    """Project the existing candidate ledger into the portable result.

    The immutable ledger remains the detailed producer evidence.  This function
    only makes its current subjects, measurements, and artifact paths discoverable.
    """

    ledger_root = root / "artifacts" / "post_mip"
    registry_path = ledger_root / "candidate_registry.json"
    if not registry_path.is_file():
        return None
    registry = json.loads(registry_path.read_text())
    architectures = dict(registry.get("architectures") or {})
    revisions = dict(registry.get("revisions") or {})
    subjects: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    limitations: set[str] = set()
    evidence_sources = [registry_path.relative_to(root).as_posix()]
    aiperf_paths: set[Path] = set()

    def add_artifact(role: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for name, item in sorted(value.items()):
                add_artifact(f"{role}.{name}", item)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                add_artifact(role, item)
            return
        relative = _run_relative(root, value)
        if relative is None:
            if isinstance(value, str) and value:
                limitations.add("Some producer artifact paths are outside the portable run root.")
            return
        artifact_id = f"artifact:{hashlib.sha256(relative.encode()).hexdigest()[:16]}"
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "role": role,
            "path": relative,
            "availability": "available",
            "validation": "producer_recorded",
        }

    add_artifact("candidate_registry", str(registry_path))
    for current_path in sorted((ledger_root / "nodes").glob("*/current.json")):
        current = json.loads(current_path.read_text())
        execution_id = str(current.get("execution_identity") or "")
        if not execution_id:
            continue
        node_id = current_path.parent.name
        execution_root = current_path.parent / "executions" / execution_id
        observations_path = execution_root / "observations.json"
        if not observations_path.is_file():
            continue
        observations = json.loads(observations_path.read_text())
        if not isinstance(observations, list):
            raise ResultValidationError(
                f"post-MIP observations must be a list: {observations_path}"
            )
        source_path = observations_path.relative_to(root).as_posix()
        evidence_sources.append(source_path)
        add_artifact("node_observations", str(observations_path))
        add_artifact("candidate_set", str(execution_root / "candidate_set.json"))
        aiperf_paths.update(execution_root.rglob("puzzletron_aiperf_result.json"))
        for raw in observations:
            if not isinstance(raw, Mapping):
                continue
            revision_id = str(
                raw.get("output_revision_id")
                or raw.get("source_revision_id")
                or raw.get("input_revision_id")
                or ""
            )
            revision = dict(revisions.get(revision_id) or {})
            architecture_id = str(
                raw.get("architecture_id") or revision.get("architecture_id") or "unknown"
            )
            architecture = dict(architectures.get(architecture_id) or {})
            candidate_id = f"subject:candidate:{revision_id or architecture_id}"
            checkpoint = dict(revision.get("artifact") or {})
            checkpoint_id = f"checkpoint:{revision_id or architecture_id}"
            candidate_checkpoint = {"checkpoint_id": checkpoint_id}
            if relative := _run_relative(root, checkpoint.get("checkpoint")):
                candidate_checkpoint["path"] = relative
            subjects[candidate_id] = {
                "subject_id": candidate_id,
                "role": "candidate",
                "checkpoint": candidate_checkpoint,
                "architecture": {
                    "architecture_id": architecture_id,
                    "block_configs": deepcopy(list(architecture.get("block_configs") or ())),
                    "origins": deepcopy(list(architecture.get("origins") or ())),
                },
            }
            row_metrics = dict(raw.get("metrics") or {})
            has_prefixed_candidate = any(name.startswith("candidate.") for name in row_metrics)
            for raw_name, raw_value in sorted(row_metrics.items()):
                if raw_name.startswith("delta.") or (
                    has_prefixed_candidate and not raw_name.startswith(("candidate.", "reference."))
                ):
                    continue
                role = "teacher" if raw_name.startswith("reference.") else "candidate"
                subject_id = candidate_id
                metric_checkpoint_id = checkpoint_id
                if role == "teacher":
                    subject_id = f"subject:teacher:{execution_id}"
                    metric_checkpoint_id = f"checkpoint:teacher:{execution_id}"
                    subjects.setdefault(
                        subject_id,
                        {
                            "subject_id": subject_id,
                            "role": "teacher",
                            "checkpoint": {"checkpoint_id": metric_checkpoint_id},
                            "architecture": {"architecture_id": "teacher_unreported"},
                        },
                    )
                    limitations.add(
                        "Teacher architecture details are unavailable when the producer records only comparison metrics."
                    )
                name, unit, direction, aggregation, dimensions = _metric_semantics(raw_name)
                metric_id = (
                    "metric:"
                    + hashlib.sha256(
                        f"{execution_id}\0{revision_id}\0{role}\0{name}".encode()
                    ).hexdigest()[:20]
                )
                present = (
                    isinstance(raw_value, (int, float))
                    and not isinstance(raw_value, bool)
                    and math.isfinite(float(raw_value))
                )
                metrics[metric_id] = {
                    "metric_id": metric_id,
                    "name": name,
                    "value": float(raw_value) if present else None,
                    "value_state": "present" if present else "invalid",
                    "missing_reason": None if present else "producer_value_is_not_finite_numeric",
                    "unit": unit,
                    "direction": direction,
                    "subject_id": subject_id,
                    "checkpoint_id": metric_checkpoint_id,
                    "producer_execution_id": execution_id,
                    "workload": {"workload_id": execution_id, "task": node_id},
                    "aggregation": aggregation,
                    "dimensions": dimensions,
                }
            for role, value in sorted(dict(raw.get("artifacts") or {}).items()):
                add_artifact(str(role).removesuffix("_path"), value)

    subjects_by_architecture = {
        subject["architecture"]["architecture_id"]: subject for subject in subjects.values()
    }
    for aiperf_path in sorted(aiperf_paths):
        payload = json.loads(aiperf_path.read_text())
        if not isinstance(payload, Mapping) or payload.get("engine") != "aiperf":
            continue
        add_artifact("aiperf_result", str(aiperf_path))
        for role, value in sorted(dict(payload.get("raw_artifacts") or {}).items()):
            add_artifact(f"aiperf.{role}", value)
        architecture_id = str(payload.get("architecture_id") or "unknown")
        subject = subjects_by_architecture.get(architecture_id)
        if subject is None:
            continue
        workload_id = str(payload.get("workload_id") or "unknown")
        execution_id = f"aiperf:{payload.get('cache_identity') or workload_id}"
        for raw_name, raw_value in sorted(dict(payload.get("metrics") or {}).items()):
            name, unit, direction, aggregation, dimensions = _metric_semantics(str(raw_name))
            metric_id = (
                "metric:"
                + hashlib.sha256(
                    f"{execution_id}\0{subject['subject_id']}\0{raw_name}".encode()
                ).hexdigest()[:20]
            )
            present = (
                isinstance(raw_value, (int, float))
                and not isinstance(raw_value, bool)
                and math.isfinite(float(raw_value))
            )
            metrics[metric_id] = {
                "metric_id": metric_id,
                "name": name,
                "value": float(raw_value) if present else None,
                "value_state": "present" if present else "invalid",
                "missing_reason": None if present else "producer_value_is_not_finite_numeric",
                "unit": unit,
                "direction": direction,
                "subject_id": subject["subject_id"],
                "checkpoint_id": subject["checkpoint"]["checkpoint_id"],
                "producer_execution_id": execution_id,
                "workload": {
                    "workload_id": workload_id,
                    "task": "aiperf",
                    "requested": deepcopy(dict(payload.get("workload") or {})),
                },
                "aggregation": aggregation,
                "dimensions": {
                    **dimensions,
                    "concurrency": payload.get("concurrency"),
                    "repetition_index": payload.get("repetition"),
                    "topology_id": payload.get("topology_id"),
                    "gpu_count": payload.get("gpu_count"),
                    "measurement_contract": deepcopy(
                        dict(payload.get("measurement_contract") or {})
                    ),
                },
            }

    if metrics:
        limitations.add(
            "Producer-defined and sample-mean metrics retain their recorded aggregation and are not relabeled as token-weighted means."
        )
    return (
        sorted(subjects.values(), key=lambda item: item["subject_id"]),
        sorted(metrics.values(), key=lambda item: item["metric_id"]),
        sorted(artifacts.values(), key=lambda item: item["path"]),
        sorted(limitations),
        sorted(set(evidence_sources)),
    )


def publish_controller_result(
    plan: CampaignPlan,
    stage_views: Sequence[StageView],
    *,
    execution_status: str,
    attachment_status: str,
    evidence_status: str = "partial",
    finalized: bool = False,
    now: str | None = None,
) -> Path:
    """Atomically refresh the portable result from current controller state."""

    if len(plan.stages) != len(stage_views):
        raise ValueError("stage views must cover every compiled stage")
    observed_at = now or _utc_now()
    run_id, producer_revision, bundle_paths = _active_bundle(
        plan.puzzle_dir, fallback_run_id=plan.contract_hash
    )
    path = result_path(plan.puzzle_dir)
    existing: Mapping[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text())
        if isinstance(loaded, Mapping) and loaded.get("schema") == RESULT_SCHEMA:
            validate_result(loaded)
            if loaded["run"]["identity"]["run_id"] != run_id:
                raise ResultValidationError("stored result does not match the active bundle")
            existing = loaded
    state = CampaignStateStore(plan.puzzle_dir)
    stage_ids = {node.stage_id for node in plan.stages}
    attempts = state.list_attempts()
    starts = [
        float(row["submitted_at"])
        for row in attempts
        if isinstance(row.get("submitted_at"), (int, float))
    ]
    prior_timing = existing.get("run", {}).get("timing", {})
    ended_at = observed_at if execution_status in _TERMINAL_STATES else None
    observed_epoch = datetime.fromisoformat(observed_at.replace("Z", "+00:00")).timestamp()
    projected = _project_post_mip_evidence(plan.puzzle_dir)
    if projected is None:
        subjects = deepcopy(list(existing.get("subjects", ())))
        metrics = deepcopy(list(existing.get("metrics", ())))
        artifacts = deepcopy(list(existing.get("artifacts", ())))
        limitations = deepcopy(list(existing.get("limitations", ())))
        evidence_sources: list[str] = []
    else:
        subjects, metrics, artifacts, limitations, evidence_sources = projected
    result = {
        "schema": RESULT_SCHEMA,
        "run": {
            "identity": {
                "run_id": run_id,
                "workflow_id": plan.contract_hash,
                "model_family": str(plan.experiment_config.get("display_name") or "unknown"),
                "modality": plan.experiment_config.get("modality"),
            },
            "status": {
                "execution": execution_status,
                "attachment": attachment_status,
                "evidence": "complete" if finalized else evidence_status,
                "support": "unreviewed",
            },
            "timing": {
                "created_at": str(prior_timing.get("created_at") or observed_at),
                "started_at": _utc_timestamp(min(starts)) if starts else None,
                "updated_at": observed_at,
                "ended_at": ended_at,
                "finalized_at": observed_at if finalized else None,
                "last_executor_observation_at": observed_at,
                "elapsed_since_first_submission_seconds": (
                    max(0.0, observed_epoch - min(starts)) if starts else None
                ),
            },
            "freshness": {
                "as_of": observed_at,
                "stale_after_seconds": 120,
                "state": "fresh",
                "reason": None,
            },
        },
        "subjects": subjects,
        "stages": [
            {
                "stage_id": node.stage_id,
                "stage_type": node.stage_id.split(".")[-1],
                "phase_id": node.stage_id.split(".")[0],
                "parent_stage_ids": [parent for parent in node.parents if parent in stage_ids],
                "external_prerequisite_stage_ids": [
                    parent for parent in node.parents if parent not in stage_ids
                ],
                "required": True,
                "state": "pending" if view.status == "waiting" else view.status,
                "elapsed_seconds": view.elapsed_seconds,
                "attempts": _attempts(state, node.stage_id, plan.puzzle_dir),
                "progress": _progress(view, observed_at),
            }
            for node, view in zip(plan.stages, stage_views)
        ],
        "metrics": metrics,
        "artifacts": artifacts,
        "provenance": {
            "resolved_bundle": {
                "bundle_id": run_id,
                "producer_revision": producer_revision,
                **bundle_paths,
            },
            "controller_state": "orchestration/compiled_plan.json",
            "evidence_sources": evidence_sources,
        },
        "limitations": limitations,
    }
    validate_result(result)
    _atomic_write(path, canonical_json_bytes(result))
    return path


def compare_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare compatible metrics and preserve explicit exclusion reasons."""

    reasons = [
        f"metric_{field}_mismatch"
        for field in ("name", "unit", "direction", "aggregation", "dimensions")
        if left.get(field) != right.get(field)
    ]
    left_workload = _mapping(left.get("workload"), "left workload")
    right_workload = _mapping(right.get("workload"), "right workload")
    reasons.extend(
        f"workload_{field}_mismatch"
        for field in _WORKLOAD_FIELDS
        if left_workload.get(field) != right_workload.get(field)
    )
    if left_workload != right_workload and not any(
        item.startswith("workload_") for item in reasons
    ):
        reasons.append("workload_contract_mismatch")
    for side, metric in (("left", left), ("right", right)):
        if (
            metric.get("value_state") != "present"
            or isinstance(metric.get("value"), bool)
            or not isinstance(metric.get("value"), (int, float))
        ):
            reasons.append(f"{side}_metric_not_numeric")
    output = {
        "left_metric_id": left.get("metric_id"),
        "right_metric_id": right.get("metric_id"),
        "comparable": not reasons,
        "exclusion_reasons": reasons,
        "delta": None,
        "relative_delta": None,
    }
    if not reasons:
        left_value = float(left["value"])
        delta = float(right["value"]) - left_value
        output["delta"] = delta
        output["relative_delta"] = None if left_value == 0 else delta / abs(left_value)
    return output


def inspect_run(run_root: str | Path, *, viewed_at: str | None = None) -> dict[str, Any]:
    """Read the result and qualify stale detached observations."""

    path = result_path(run_root)
    result = json.loads(path.read_text())
    if not isinstance(result, Mapping):
        raise ResultValidationError("result root must be a mapping")
    validate_result(result)
    view = deepcopy(dict(result))
    run = view["run"]
    freshness = run["freshness"]
    if freshness.get("as_of") and run["status"]["execution"] in {"running", "submitted"}:
        current = datetime.fromisoformat((viewed_at or _utc_now()).replace("Z", "+00:00"))
        observed = datetime.fromisoformat(str(freshness["as_of"]).replace("Z", "+00:00"))
        if (current - observed).total_seconds() > freshness["stale_after_seconds"]:
            freshness["state"], freshness["reason"] = "stale", "result_observation_expired"
    view["stage_counts"] = dict(sorted(Counter(stage["state"] for stage in view["stages"]).items()))
    view["active_progress"] = [
        {"stage_id": stage["stage_id"], **stage["progress"]}
        for stage in view["stages"]
        if stage["progress"]["status"] in {"blocked", "running"}
    ]
    roles = {subject["subject_id"]: subject["role"] for subject in view["subjects"]}
    teachers = [
        metric for metric in view["metrics"] if roles.get(metric["subject_id"]) == "teacher"
    ]
    candidates = [
        metric for metric in view["metrics"] if roles.get(metric["subject_id"]) == "candidate"
    ]
    view["comparisons"] = [
        compare_metrics(teacher, candidate)
        for teacher in teachers
        for candidate in candidates
        if teacher["name"] == candidate["name"]
        and teacher["producer_execution_id"] == candidate["producer_execution_id"]
    ]
    view["result_path"] = str(path)
    view["result_digest"] = result_sha256(canonical_json_bytes(result))
    return view


def export_run_result(run_root: str | Path, *, exported_at: str | None = None) -> Path:
    """Validate and return the already-portable authoritative result."""

    del exported_at
    inspect_run(run_root)
    return result_path(run_root)


def refresh_run_report(run_root: str | Path, *, generated_at: str | None = None) -> RenderedView:
    """Regenerate optional HTML only from the authoritative result."""

    from .result_render import render_run_html

    rendered = render_run_html(
        inspect_run(run_root, viewed_at=generated_at),
        generated_at=generated_at or _utc_now(),
        renderer_revision=_RENDERER_REVISION,
    )
    output = Path(run_root) / "artifacts" / "campaign_report"
    _atomic_write(output / "campaign_report.html", rendered.content)
    _atomic_write(output / "report_manifest.json", canonical_json_bytes(rendered.manifest))
    return rendered
