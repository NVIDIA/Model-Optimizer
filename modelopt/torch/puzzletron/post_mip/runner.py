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

"""Execute and aggregate one compiled post-MIP node."""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..evaluation import DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS, run_lmms_eval_checkpoint
from ..identity import canonicalize, stable_hash
from ..security_policy import require_boolean_policy
from .base import CompiledPostMIPNode, NodeKind, compile_post_mip_flows
from .filters import apply_filter
from .identity import (
    expected_post_mip_execution_identity,
    post_mip_execution_contract,
    post_mip_execution_contract_identity,
)
from .records import ArtifactKind, CandidateLedger, CandidateRevision, CandidateSet, NodeObservation

__all__ = [
    "aggregate_post_mip_node",
    "expected_post_mip_execution_identity",
    "register_downstream_evaluation_profile",
    "run_post_mip_node_shard",
]

_DOWNSTREAM_EVALUATION_PROFILES: dict[str, Callable[..., dict[str, Any]]] = {}


def register_downstream_evaluation_profile(
    name: str,
    evaluator: Callable[..., dict[str, Any]],
) -> None:
    """Register an examples-layer evaluator without coupling ModelOpt to that example."""

    if not name:
        raise ValueError("downstream evaluation profile name must not be empty")
    existing = _DOWNSTREAM_EVALUATION_PROFILES.get(name)
    if existing is not None and existing is not evaluator:
        raise ValueError(f"downstream evaluation profile is already registered: {name}")
    _DOWNSTREAM_EVALUATION_PROFILES[name] = evaluator


def _puzzle_dir(config: Mapping[str, Any]) -> Path:
    return Path(config.get("puzzle_dir") or (config.get("experiment") or {})["dir"])


def _compiled_node(config: Mapping[str, Any], stage_id: str) -> CompiledPostMIPNode:
    matches = [node for node in compile_post_mip_flows(config) if node.stage_id == stage_id]
    if len(matches) != 1:
        raise ValueError(f"expected one compiled post-MIP node {stage_id!r}, found {len(matches)}")
    node = matches[0]
    if not node.capabilities.implemented:
        raise NotImplementedError(f"post-MIP node type {node.node_type!r} is not implemented")
    return node


def _ledger(config: Mapping[str, Any]) -> CandidateLedger:
    return CandidateLedger(_puzzle_dir(config) / "artifacts" / "post_mip")


def _input_set(
    ledger: CandidateLedger, config: Mapping[str, Any], node: CompiledPostMIPNode
) -> CandidateSet:
    if node.input_id == "source":
        flow = (config.get("post_mip") or {})["flows"][node.flow_id]
        return ledger.root_set(node.flow_id, flow["source"])
    return ledger.load_candidate_set(node.input_id)


def _atomic_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(canonicalize(payload), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _worker_group() -> tuple[int, int]:
    """Return the rank and size of the innermost candidate process group."""

    launcher = os.environ.get("PUZZLETRON_TASK_LAUNCHER")
    if launcher == "torchrun" or (launcher is None and "LOCAL_RANK" in os.environ):
        return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    return (
        int(os.environ.get("PUZZLETRON_GROUP_RANK", "0")),
        int(os.environ.get("PUZZLETRON_GROUP_SIZE", "1")),
    )


def _exception_diagnostics(error: Exception) -> dict[str, str]:
    """Keep candidate failures actionable after worker logs are aggregated."""

    message = str(error)
    return {
        "error": f"{type(error).__name__}: {message}" if message else type(error).__name__,
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }


def _needs_puzzletron_process_group(node_type: str) -> bool:
    """Return whether Puzzletron, rather than the node runtime, owns initialization."""

    return node_type == "evaluation"


def _is_loaded_subprocess_timeout(error: Exception) -> bool:
    """Recognize the timeout type loaded by the optional AIPerf adapter."""

    timeout_type = getattr(sys.modules.get("subprocess"), "TimeoutExpired", None)
    return timeout_type is not None and isinstance(error, timeout_type)


def _node_root(config: Mapping[str, Any], node: CompiledPostMIPNode) -> Path:
    return _puzzle_dir(config) / "artifacts" / "post_mip" / "nodes" / node.node_id


def _execution_root(
    config: Mapping[str, Any], node: CompiledPostMIPNode, execution_identity: str
) -> Path:
    return _node_root(config, node) / "executions" / execution_identity


def _raw_solution(source) -> dict[str, Any]:
    path = Path(str(source.artifact["solution_path"]))
    rows = json.loads(path.read_text())
    return dict(rows[int(source.artifact.get("solution_index", 0))])


def _materialize(
    config: dict[str, Any],
    node: CompiledPostMIPNode,
    ledger: CandidateLedger,
    input_revision_id: str,
    source,
    execution_identity: str,
) -> dict[str, Any]:
    if source.artifact_kind is ArtifactKind.CHECKPOINT:
        return {
            "artifact_kind": "checkpoint",
            "checkpoint": source.artifact["checkpoint"],
        }
    from ..anymodel.registry import resolve_descriptor_from_pretrained
    from ..replacement_library.library import ReplacementLibrary
    from ..replacement_library.replacement_utils import parse_layer_replacement

    architecture = ledger.architectures[source.architecture_id]
    origins = architecture.origins
    width = int(origins[0]["hidden_width"])
    puzzle_dir = _puzzle_dir(config)
    scenario = puzzle_dir / "scenarios" / f"width-{width:04d}" / "depth-00"
    sorted_teacher = scenario / "ckpts" / "sorted_teacher"
    model = config.get("model") or {}
    descriptor = resolve_descriptor_from_pretrained(
        str(sorted_teacher),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        descriptor_override=model.get("descriptor_override"),
    ).descriptor
    library = ReplacementLibrary(scenario / "replacement_library.json", descriptor)
    raw = _raw_solution(source)
    replacements = [
        parse_layer_replacement(item["layer_replacement"]) for item in raw["chosen_replacements"]
    ]
    output = (
        _execution_root(config, node, execution_identity) / "checkpoints" / source.architecture_id
    )
    model_config = library.create_model_config(replacements)
    library.materialize_checkpoint(
        replacements,
        output,
        model_config=model_config,
        solution_identity=f"{node.flow_id}-{node.node_id}-{source.architecture_id}",
    )
    return {"artifact_kind": "checkpoint", "checkpoint": str(output)}


def _average_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics = {}
    excluded = {"args", "puzzle_solution", "observability", "distributed_evaluation"}
    for key, value in payload.items():
        if key in excluded:
            continue
        value = value.get("avg") if isinstance(value, Mapping) and "avg" in value else value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            metrics[str(key)] = float(value)
    return metrics


def _scenario_checkpoint_roles(scenario: Path, expected_width: int) -> tuple[Path, Path | None]:
    manifest_path = scenario / "scenario_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"online evaluation scenario manifest is missing: {manifest_path}"
        ) from error
    if manifest.get("status") != "complete":
        raise RuntimeError(f"online evaluation scenario is not complete: {manifest_path}")
    if int(manifest.get("hidden_width", 0)) != expected_width:
        raise RuntimeError(
            "online evaluation scenario width does not match the candidate: "
            f"{manifest.get('hidden_width')} != {expected_width}"
        )

    source = Path(str(manifest["parent_checkpoint"]))
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"online evaluation source checkpoint is incomplete: {source}")
    bypass_value = manifest.get("bypass_checkpoint")
    bypass = Path(str(bypass_value)) if bypass_value is not None else None
    if bypass is not None and not (bypass / "config.json").is_file():
        raise FileNotFoundError(f"online evaluation bypass checkpoint is incomplete: {bypass}")
    return source, bypass


@dataclass(frozen=True)
class _ConfigEvaluationWork:
    input_revision_id: str
    source: Any
    raw_solution: dict[str, Any]
    hidden_width: int
    source_checkpoint: Path
    bypass_checkpoint: Path | None

    @property
    def cache_identity(self) -> tuple[int, str, str | None]:
        return (
            self.hidden_width,
            str(self.source_checkpoint),
            str(self.bypass_checkpoint) if self.bypass_checkpoint is not None else None,
        )


def _config_evaluation_work(
    config: Mapping[str, Any], input_revision_id: str, source: Any
) -> _ConfigEvaluationWork:
    raw = _raw_solution(source)
    raw_width = raw.get("hidden_width", source.artifact.get("hidden_width"))
    hidden_width = int(raw_width) if raw_width is not None else 0
    if not hidden_width:
        raise ValueError("online evaluation could not determine the candidate hidden width")
    scenario = _puzzle_dir(config) / "scenarios" / f"width-{hidden_width:04d}" / "depth-00"
    source_checkpoint, bypass_checkpoint = _scenario_checkpoint_roles(scenario, hidden_width)
    raw["hidden_width"] = hidden_width
    return _ConfigEvaluationWork(
        input_revision_id=input_revision_id,
        source=source,
        raw_solution=raw,
        hidden_width=hidden_width,
        source_checkpoint=source_checkpoint,
        bypass_checkpoint=bypass_checkpoint,
    )


def _merge_scoring_settings(scoring, settings: Mapping[str, Any]):
    """Merge node-local scoring overrides without dropping inherited nested fields."""

    from omegaconf import OmegaConf

    return OmegaConf.merge(scoring, dict(settings))


def _evaluate_config_group(
    config: dict[str, Any],
    node: CompiledPostMIPNode,
    work: list[_ConfigEvaluationWork],
    execution_identity: str,
) -> dict[str, dict[str, Any]]:
    if not work:
        return {}
    cache_identities = {item.cache_identity for item in work}
    if len(cache_identities) != 1:
        raise ValueError(
            "one online evaluation session requires one hidden-width/checkpoint identity, "
            f"found {sorted(cache_identities, key=repr)}"
        )

    from omegaconf import OmegaConf

    # This is a GPU-worker-only dependency; keep aggregate/filter imports lightweight.
    import modelopt.torch.utils.distributed as dist

    from ..pipeline_config import load_runtime_hydra_config
    from ..plugins.automodel.solution_launch import launch_score_solutions_automodel
    from ..stages.pipeline import _distributed
    from ..tools.hydra_utils import clone_hydra_config

    first = work[0]
    session_identity = stable_hash(
        {
            "cache_identity": first.cache_identity,
            "architecture_ids": [item.source.architecture_id for item in work],
        },
        prefix="online_eval_session",
    )
    raw_root = _execution_root(config, node, execution_identity) / "raw"
    session_output = raw_root / "sessions" / session_identity
    solutions_path = session_output / "solutions.json"
    candidate_outputs = {
        str(index): str(raw_root / item.source.architecture_id) for index, item in enumerate(work)
    }
    hydra_cfg = clone_hydra_config(load_runtime_hydra_config(config))
    OmegaConf.set_struct(hydra_cfg, False)
    settings = dict(node.config.get("config") or {})
    hydra_cfg.scoring = _merge_scoring_settings(hydra_cfg.scoring, settings)
    hydra_cfg.scoring.source_checkpoint_dir = str(first.source_checkpoint)
    hydra_cfg.scoring.target_teacher_dir = str(hydra_cfg.scoring.teacher_dir)
    hydra_cfg.scoring.bypass_checkpoint_dir = (
        str(first.bypass_checkpoint) if first.bypass_checkpoint is not None else None
    )
    hydra_cfg.scoring.solutions_path = str(solutions_path)
    hydra_cfg.scoring.output_dir = str(session_output)
    hydra_cfg.scoring.solutions_to_validate = list(range(len(work)))
    hydra_cfg.scoring.skip_existing_solutions = True
    hydra_cfg.scoring.score_source_baseline = True
    hydra_cfg.scoring.sort_solutions_by = None
    hydra_cfg.scoring.solution_output_dirs = candidate_outputs
    with _distributed(hydra_cfg):
        if dist.is_master():
            _atomic_json(solutions_path, [item.raw_solution for item in work])
            for item in work:
                _atomic_json(
                    raw_root / item.source.architecture_id / "solutions.json",
                    [item.raw_solution],
                )
        dist.barrier()
        launch_score_solutions_automodel(hydra_cfg)
    results = {}
    for item in work:
        result_path = raw_root / item.source.architecture_id / "solution_0.json"
        results[item.input_revision_id] = {
            "metrics": _average_metrics(json.loads(result_path.read_text())),
            "result_path": str(result_path),
        }
    return results


def _evaluate_config(
    config: dict[str, Any], node: CompiledPostMIPNode, source, execution_identity: str
) -> dict[str, Any]:
    input_revision_id = str(source.revision_id)
    return _evaluate_config_group(
        config,
        node,
        [_config_evaluation_work(config, input_revision_id, source)],
        execution_identity,
    )[input_revision_id]


def _evaluate_config_candidates(
    config: dict[str, Any],
    node: CompiledPostMIPNode,
    ledger: CandidateLedger,
    revision_ids: Sequence[str],
    execution_identity: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[int, str, str | None], list[_ConfigEvaluationWork]] = {}
    for revision_id in revision_ids:
        source = ledger.source_revision(revision_id, node.model_source)
        if source.artifact_kind is ArtifactKind.CHECKPOINT:
            continue
        item = _config_evaluation_work(config, revision_id, source)
        groups.setdefault(item.cache_identity, []).append(item)

    rows = {}
    for work in groups.values():
        results = _evaluate_config_group(config, node, work, execution_identity)
        for item in work:
            rows[item.input_revision_id] = {
                "input_revision_id": item.input_revision_id,
                "source_revision_id": item.source.revision_id,
                "architecture_id": item.source.architecture_id,
                "status": "success",
                **results[item.input_revision_id],
            }
    return rows


def _evaluate_checkpoint(
    config: dict[str, Any], node: CompiledPostMIPNode, source, execution_identity: str
) -> dict[str, Any]:
    from ..manifest import StageManifest, semantic_stage_config
    from ..stages.future import evaluation_stage

    candidate = copy.deepcopy(config)
    output = _execution_root(config, node, execution_identity) / "raw" / source.architecture_id
    settings = dict(node.config.get("config") or {})
    reference_checkpoint = settings.pop("reference_checkpoint", None)
    checkpoint = str(source.artifact["checkpoint"])
    checkpoints = [checkpoint]
    if reference_checkpoint is not None and str(reference_checkpoint) != checkpoint:
        checkpoints.append(str(reference_checkpoint))
    settings.update(
        enabled=True,
        checkpoints=checkpoints,
        output_dir=str(output),
    )
    candidate["zero_shot_evaluation"] = settings
    manifest = StageManifest(
        stage=f"{node.stage_id}.{source.architecture_id}",
        inputs={"config": candidate},
        config=candidate,
        semantic_config=semantic_stage_config(
            candidate, "zero_shot_evaluation", use_authored=False
        ),
    )
    evaluation_stage(candidate, manifest)
    rows = json.loads((output / "evaluation_summary.json").read_text())
    row = next(item for item in rows if str(item.get("checkpoint")) == checkpoint)
    metrics = dict(row.get("metrics") or {})
    result = {"metrics": metrics, "result_path": row.get("result_path")}
    reference_row = next(
        (
            item
            for item in rows
            if reference_checkpoint is not None
            and str(item.get("checkpoint")) == str(reference_checkpoint)
        ),
        None,
    )
    if (
        reference_checkpoint is not None
        and str(reference_checkpoint) != checkpoint
        and reference_row is None
    ):
        raise RuntimeError(
            "reference checkpoint is missing from the LM evaluation summary: "
            f"{reference_checkpoint}"
        )
    if reference_row is not None and str(reference_checkpoint) != checkpoint:
        reference_metrics = dict(reference_row.get("metrics") or {})
        if metrics.keys() != reference_metrics.keys():
            raise RuntimeError(
                "candidate and reference LM evaluations produced different metrics: "
                f"candidate_only={sorted(metrics.keys() - reference_metrics.keys())}, "
                f"reference_only={sorted(reference_metrics.keys() - metrics.keys())}"
            )
        result["metrics"] = {
            **metrics,
            **{f"candidate.{name}": value for name, value in metrics.items()},
            **{f"reference.{name}": value for name, value in reference_metrics.items()},
            **{f"delta.{name}": metrics[name] - reference_metrics[name] for name in metrics},
        }
        result["reference_result_path"] = reference_row.get("result_path")
    return result


def _evaluate(
    config: dict[str, Any], node: CompiledPostMIPNode, source, execution_identity: str
) -> dict[str, Any]:
    return (
        _evaluate_checkpoint(config, node, source, execution_identity)
        if source.artifact_kind is ArtifactKind.CHECKPOINT
        else _evaluate_config(config, node, source, execution_identity)
    )


def _aiperf(
    config: dict[str, Any], node: CompiledPostMIPNode, source, execution_identity: str
) -> dict[str, Any]:
    from ..benchmarks import run_aiperf_sweep

    settings = dict(node.config.get("config") or {})
    settings.pop("best_selection_mode", None)
    raw_concurrency = settings.pop("concurrency", [1])
    if isinstance(raw_concurrency, (int, str)):
        concurrencies = (int(raw_concurrency),)
    else:
        concurrencies = tuple(int(value) for value in raw_concurrency)
    request_count = settings.pop("request_count", None)
    minimum_request_count = int(settings.pop("minimum_request_count", 4))
    requests_per_concurrency = int(settings.pop("requests_per_concurrency", 2))
    request_counts = {
        concurrency: (
            int(request_count)
            if request_count is not None
            else max(minimum_request_count, requests_per_concurrency * concurrency)
        )
        for concurrency in concurrencies
    }
    topology = dict(settings.pop("topology", {}) or {})
    trust_remote_code = require_boolean_policy(
        settings.pop(
            "trust_remote_code",
            (config.get("model") or {}).get("trust_remote_code", False),
        ),
        path="post_mip.aiperf.config.trust_remote_code",
        default=False,
    )
    allow_online_tokenizer_resolution = require_boolean_policy(
        settings.pop("allow_aiperf_v011_online_tokenizer_resolution", False),
        path="post_mip.aiperf.config.allow_aiperf_v011_online_tokenizer_resolution",
        default=False,
    )
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu_ids:
        gpu_ids = ",".join(str(index) for index in range(int(topology.get("gpu_group_size", 1))))
    results = run_aiperf_sweep(
        source.artifact["checkpoint"],
        artifact_dir=_execution_root(config, node, execution_identity)
        / "raw"
        / source.architecture_id,
        concurrencies=concurrencies,
        input_tokens=int(settings.pop("input_tokens", 8192)),
        output_tokens=int(settings.pop("output_tokens", 1024)),
        gpu_ids=gpu_ids,
        topology=topology,
        request_counts=request_counts,
        solution_id=source.architecture_id,
        profile_id=node.flow_id,
        trust_remote_code=trust_remote_code,
        allow_aiperf_v011_online_tokenizer_resolution=allow_online_tokenizer_resolution,
        **settings,
    )
    metrics = {}
    for result in results:
        if len(results) == 1:
            metrics.update(result.metrics)
        image_batch_size = int(result.workload.get("image_batch_size", 0))
        namespace = f"concurrency_{result.concurrency}"
        if image_batch_size > 0:
            namespace = f"images_{image_batch_size}.{namespace}"
        metrics.update({f"{namespace}.{key}": value for key, value in result.metrics.items()})
    return {
        "metrics": metrics,
        "result_paths": [result.raw_artifacts for result in results],
    }


def _downstream_evaluation(
    config: dict[str, Any],
    node: CompiledPostMIPNode,
    source,
    execution_identity: str,
) -> dict[str, Any]:
    if source.artifact_kind is not ArtifactKind.CHECKPOINT:
        raise ValueError("downstream_evaluation requires materialized checkpoint artifacts")
    output_root = (
        _execution_root(config, node, execution_identity)
        / "raw"
        / source.architecture_id
        / "lmms_eval"
    )
    settings = copy.deepcopy(dict(node.config.get("config") or {}))
    reference_checkpoint = settings.pop("reference_checkpoint", None)
    reference_once = bool(settings.pop("reference_once", False))
    reference_cache_id = settings.pop("reference_cache_id", None)
    recorded_observation = settings.pop("recorded_observation", None)
    evaluator_revision = settings.pop("evaluator_revision", None)
    profile = settings.pop("profile", None)
    evaluator = run_lmms_eval_checkpoint
    if profile is not None:
        evaluator = _DOWNSTREAM_EVALUATION_PROFILES.get(str(profile))
        if evaluator is None:
            raise ValueError(f"unsupported downstream evaluation profile: {profile}")
    if recorded_observation is not None and reference_checkpoint is None:
        raise ValueError("recorded_observation requires reference_checkpoint")
    candidate = evaluator(
        source.artifact["checkpoint"],
        output_root=output_root,
        settings=settings,
    )
    if reference_checkpoint is None:
        return candidate

    if reference_once:
        if not reference_cache_id:
            raise ValueError("reference_once requires reference_cache_id")
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if any(character not in allowed for character in str(reference_cache_id)):
            raise ValueError("reference_cache_id must contain only letters, digits, '_' and '-'")
        cache_identity = stable_hash(
            {
                "checkpoint": str(reference_checkpoint),
                "profile": profile,
                "settings": settings,
            },
            prefix="post_mip_reference_evaluation",
        )
        cache_root = (
            _puzzle_dir(config)
            / "artifacts/post_mip/reference_evaluations"
            / str(reference_cache_id)
            / cache_identity
        )
        result_path = cache_root / "result.json"
        cache_root.mkdir(parents=True, exist_ok=True)
        with (cache_root / "evaluation.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if result_path.is_file():
                reference = json.loads(result_path.read_text())
            else:
                reference = evaluator(
                    reference_checkpoint,
                    output_root=cache_root / "raw",
                    settings=settings,
                )
                _atomic_json(result_path, reference)
    else:
        reference = evaluator(
            reference_checkpoint,
            output_root=output_root.parent / "reference",
            settings=settings,
        )
    candidate_metrics = dict(candidate["metrics"])
    reference_metrics = dict(reference["metrics"])
    if candidate_metrics.keys() != reference_metrics.keys():
        raise RuntimeError(
            "candidate and reference downstream evaluations produced different metrics: "
            f"candidate_only={sorted(candidate_metrics.keys() - reference_metrics.keys())}, "
            f"reference_only={sorted(reference_metrics.keys() - candidate_metrics.keys())}"
        )
    comparison_metrics = {
        **candidate_metrics,
        **{f"candidate.{name}": value for name, value in candidate_metrics.items()},
        **{f"reference.{name}": value for name, value in reference_metrics.items()},
        **{
            f"delta.{name}": value - reference_metrics[name]
            for name, value in candidate_metrics.items()
        },
    }
    evaluation_identity = _downstream_evaluation_identity(
        source=source,
        reference_checkpoint=reference_checkpoint,
        profile=profile,
        evaluator_revision=evaluator_revision,
        settings=settings,
        candidate=candidate,
    )
    observation_comparison, observation_metrics = _compare_recorded_observation(
        recorded_observation,
        comparison_metrics,
        evaluation_identity,
    )
    comparison_metrics.update(observation_metrics)
    comparison_path = _atomic_json(
        output_root / "comparison.json",
        {
            "candidate": {
                "checkpoint": source.artifact["checkpoint"],
                "metrics": candidate_metrics,
                "result_path": candidate["result_path"],
            },
            "reference": {
                "checkpoint": str(reference_checkpoint),
                "metrics": reference_metrics,
                "result_path": reference["result_path"],
            },
            "delta": {
                name: candidate_metrics[name] - reference_metrics[name]
                for name in candidate_metrics
            },
            "identity": evaluation_identity,
            "evidence": {
                "candidate_result_path": candidate["result_path"],
                "reference_result_path": reference["result_path"],
            },
            **(
                {"recorded_observation": observation_comparison}
                if observation_comparison is not None
                else {}
            ),
        },
    )
    return {
        **candidate,
        "metrics": comparison_metrics,
        "comparison_path": str(comparison_path),
        "reference_result_path": reference["result_path"],
    }


def _compare_recorded_observation(
    observation: Any,
    actual_metrics: Mapping[str, float],
    actual_identity: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, float]]:
    """Compare with a historical observation without creating an acceptance gate."""
    if observation is None:
        return None, {}
    if not isinstance(observation, Mapping):
        raise TypeError("recorded_observation must be a mapping")
    expected_identity = observation.get("identity")
    if not isinstance(expected_identity, Mapping) or not expected_identity:
        raise ValueError("recorded_observation.identity must be a non-empty mapping")
    if canonicalize(expected_identity) != canonicalize(actual_identity):
        return (
            {
                "status": "identity_mismatch",
                "identity": canonicalize(expected_identity),
                "actual_identity": canonicalize(actual_identity),
            },
            {},
        )
    raw_metrics = observation.get("metrics")
    if not isinstance(raw_metrics, Mapping) or not raw_metrics:
        raise ValueError("recorded_observation.metrics must be a non-empty mapping")
    recorded = {}
    for name, value in raw_metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"recorded observation metric {name!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"recorded observation metric {name!r} must be finite")
        recorded[str(name)] = numeric
    missing = sorted(set(recorded) - actual_metrics.keys())
    if missing:
        raise ValueError(f"recorded observation metrics were not produced: {missing}")
    differences = {name: actual_metrics[name] - value for name, value in recorded.items()}
    return (
        {
            **{
                key: copy.deepcopy(value)
                for key, value in observation.items()
                if key not in {"identity", "metrics"}
            },
            "status": "matched",
            "identity": canonicalize(expected_identity),
            "metrics": recorded,
            "difference_from_recorded": differences,
        },
        {f"observation_delta.{name}": value for name, value in differences.items()},
    )


def _downstream_evaluation_identity(
    *,
    source: CandidateRevision,
    reference_checkpoint: str | Path,
    profile: Any,
    evaluator_revision: Any,
    settings: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one comparison to its checkpoint, KD, data, and evaluator contract."""

    # Import lazily because the dependency-light orchestration controller must
    # not load the full runtime configuration stack on a login node.
    from ..distributed_eval.config import checkpoint_identity

    profile_identity = None
    profile_path = candidate.get("profile_path")
    if profile_path:
        report = json.loads(Path(str(profile_path)).read_text())
        profile_identity = {
            key: report.get(key)
            for key in (
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
        }
    exposure = None
    exposure_path = source.artifact.get("exposure_path")
    if exposure_path:
        raw_exposure = json.loads(Path(str(exposure_path)).read_text())
        exposure = {
            key: raw_exposure.get(key)
            for key in (
                "cumulative_steps",
                "global_batch_size",
                "cumulative_examples",
                "max_sample_length",
                "effective_tokens",
                "effective_tokens_source",
                "token_upper_bound",
            )
        }
    evaluator_settings = {
        key: copy.deepcopy(value)
        for key, value in settings.items()
        if key not in {"row_manifest", "timeout_seconds"}
    }
    return canonicalize(
        {
            "candidate_checkpoint_fingerprint": checkpoint_identity(source.artifact["checkpoint"])[
                "fingerprint"
            ],
            "reference_checkpoint_fingerprint": checkpoint_identity(reference_checkpoint)[
                "fingerprint"
            ],
            "architecture_id": source.architecture_id,
            "kd": {
                "producer_node": source.producer_node,
                "exposure": exposure,
            },
            "evaluator": {
                "profile": profile,
                "revision": evaluator_revision,
                "settings": evaluator_settings,
                "resolved_profile": profile_identity,
            },
        }
    )


def _post_mip_kd_settings(
    config: Mapping[str, Any],
    node_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve KD settings while preserving the transformer's checkpoint contract."""

    settings = copy.deepcopy(dict(config.get("global_distillation") or {}))
    settings.update(copy.deepcopy(dict(node_settings)))
    if str(settings.get("save_consolidated", False)).strip().lower() in {
        "false",
        "0",
        "none",
    }:
        settings["save_consolidated"] = True
    return settings


def _global_kd(
    config: dict[str, Any], node: CompiledPostMIPNode, source, execution_identity: str
) -> dict[str, Any]:
    from ..distillation.global_automodel import build_global_kd_config, run_global_kd
    from ..stages.future import _write_global_distillation_summary

    candidate = copy.deepcopy(config)
    settings = _post_mip_kd_settings(config, node.config.get("config") or {})
    trajectory = node.config.get("trajectory")
    if trajectory is None:
        output = (
            _execution_root(config, node, execution_identity)
            / "checkpoints"
            / source.architecture_id
        )
    else:
        resume_contract = {
            key: value
            for key, value in settings.items()
            if key not in {"max_steps", "checkpoint_every_steps"}
        }
        trajectory_identity = stable_hash(
            {
                "trajectory": trajectory,
                "source_revision": source.revision_id,
                "student_checkpoint": source.artifact["checkpoint"],
                "resume_contract": resume_contract,
            },
            prefix="post_mip_kd_trajectory",
        )
        output = (
            _puzzle_dir(config)
            / "artifacts/post_mip/kd_trajectories"
            / str(trajectory)
            / trajectory_identity
            / source.architecture_id
        )
    settings.update(student_dir=source.artifact["checkpoint"], output_dir=str(output))
    candidate["distillation"] = settings
    kd_config = build_global_kd_config(candidate)
    started = time.monotonic()
    result = run_global_kd(kd_config)
    elapsed_gpu_hours = (
        (time.monotonic() - started) * max(1, int(os.environ.get("WORLD_SIZE", "1"))) / 3600
    )
    summary_path = _write_global_distillation_summary(kd_config, result)
    summary = json.loads(summary_path.read_text())
    checkpoint = summary.get("post_kd_checkpoint")
    if not checkpoint:
        raise RuntimeError("global KD produced no consolidated checkpoint")
    metrics = dict(result.metrics)
    exposure = copy.deepcopy(dict(node.config.get("exposure") or {}))
    exposure_path = None
    if exposure:
        training_records = []
        training_log = output / "checkpoints" / "training.jsonl"
        if training_log.is_file():
            training_records = [
                json.loads(line) for line in training_log.read_text().splitlines() if line.strip()
            ]
        effective_tokens = sum(
            int(record.get("num_label_tokens", 0)) for record in training_records
        )
        if effective_tokens <= 0:
            raise RuntimeError("global KD produced no non-padding token accounting")
        exposure_root = output / "exposure"
        milestone_path = exposure_root / f"step_{kd_config.max_steps:06d}.json"
        prior_gpu_hours = 0.0
        for path in exposure_root.glob("step_*.json"):
            if path != milestone_path:
                prior_gpu_hours += float(
                    json.loads(path.read_text()).get("actual_incremental_gpu_hours", 0.0)
                )
        exposure.update(
            effective_tokens=effective_tokens,
            effective_tokens_source="training.jsonl:num_label_tokens",
            token_upper_bound=(
                int(exposure["cumulative_examples"]) * int(exposure["max_sample_length"])
            ),
            actual_incremental_gpu_hours=elapsed_gpu_hours,
            actual_cumulative_gpu_hours=prior_gpu_hours + elapsed_gpu_hours,
        )
        exposure_path = _atomic_json(milestone_path, exposure)
        metrics.update(
            {
                "exposure.global_batch_size": float(exposure["global_batch_size"]),
                "exposure.cumulative_examples": float(exposure["cumulative_examples"]),
                "exposure.effective_tokens": float(effective_tokens),
                "exposure.estimated_cumulative_gpu_hours": float(
                    exposure["estimated_cumulative_gpu_hours"]
                ),
                "exposure.actual_incremental_gpu_hours": elapsed_gpu_hours,
                "exposure.actual_cumulative_gpu_hours": float(
                    exposure["actual_cumulative_gpu_hours"]
                ),
            }
        )
    return {
        "artifact_kind": "checkpoint",
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "summary_path": str(summary_path),
        **({"exposure_path": str(exposure_path)} if exposure_path is not None else {}),
    }


def _run_candidate(
    config: dict[str, Any],
    node: CompiledPostMIPNode,
    ledger: CandidateLedger,
    input_revision_id: str,
    execution_identity: str,
) -> dict[str, Any]:
    source = ledger.source_revision(input_revision_id, node.model_source)
    if node.node_type == "materialize":
        result = _materialize(config, node, ledger, input_revision_id, source, execution_identity)
    elif node.node_type == "evaluation":
        result = _evaluate(config, node, source, execution_identity)
    elif node.node_type == "aiperf":
        result = _aiperf(config, node, source, execution_identity)
    elif node.node_type == "downstream_evaluation":
        result = _downstream_evaluation(config, node, source, execution_identity)
    elif node.node_type == "global_kd":
        result = _global_kd(config, node, source, execution_identity)
    else:
        raise ValueError(f"node type {node.node_type!r} is not a candidate executor")
    return {
        "input_revision_id": input_revision_id,
        "source_revision_id": source.revision_id,
        "architecture_id": source.architecture_id,
        "status": "success",
        **result,
    }


@contextmanager
def _distributed_shard(config: dict[str, Any], node: CompiledPostMIPNode) -> Iterator[None]:
    """Keep one process group alive across every candidate in a distributed shard."""

    if int(os.environ.get("WORLD_SIZE", "1")) <= 1 or not _needs_puzzletron_process_group(
        node.node_type
    ):
        yield
        return

    # Distributed runtime configuration is intentionally loaded only on GPU workers.
    from ..pipeline_config import load_runtime_hydra_config
    from ..stages.pipeline import _distributed

    with _distributed(load_runtime_hydra_config(config)):
        yield


def run_post_mip_node_shard(
    config: dict[str, Any], stage_id: str, *, shard_index: int = 0, shard_count: int = 1
) -> Path:
    node = _compiled_node(config, stage_id)
    ledger = _ledger(config)
    ledger.ingest_mip(_puzzle_dir(config))
    candidate_set = _input_set(ledger, config, node)
    execution_contract = post_mip_execution_contract(config, node, candidate_set, ledger)
    execution_identity = post_mip_execution_contract_identity(execution_contract)
    revision_ids = candidate_set.revision_ids[shard_index::shard_count]
    output_path = (
        _execution_root(config, node, execution_identity)
        / "shards"
        / f"shard_{shard_index:05d}.json"
    )
    group_rank, group_size = _worker_group()
    distributed = group_size > 1
    rows = []
    with _distributed_shard(config, node):
        cached_evaluation_rows = (
            _evaluate_config_candidates(
                config,
                node,
                ledger,
                revision_ids,
                execution_identity,
            )
            if node.node_type == "evaluation"
            else {}
        )
        for revision_id in revision_ids:
            try:
                row = cached_evaluation_rows.get(revision_id)
                if row is None:
                    row = _run_candidate(config, node, ledger, revision_id, execution_identity)
                row = {**row, "execution_identity": execution_identity}
            except Exception as error:
                subprocess_timeout = _is_loaded_subprocess_timeout(error)
                timed_out = node.node_type in {"aiperf", "downstream_evaluation"} and (
                    subprocess_timeout or isinstance(error, TimeoutError)
                )
                row = {
                    "input_revision_id": revision_id,
                    "source_revision_id": revision_id,
                    "architecture_id": ledger.revisions[revision_id].architecture_id,
                    "status": "timed_out" if timed_out else "failed",
                    "execution_identity": execution_identity,
                    **_exception_diagnostics(error),
                }
                if timed_out:
                    timeout_field = "benchmark_timeout"
                    if node.node_type == "downstream_evaluation":
                        timeout_field = "timeout_seconds"
                    elif not subprocess_timeout:
                        timeout_field = "readiness_timeout"
                    default_timeout = (
                        DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS
                        if node.node_type == "downstream_evaluation"
                        else 600
                        if timeout_field == "benchmark_timeout"
                        else 1200
                    )
                    row["timeout_seconds"] = float(
                        getattr(error, "timeout", None)
                        or (node.config.get("config") or {}).get(timeout_field, default_timeout)
                    )
                rows.append(row)
                if group_rank == 0:
                    _atomic_json(output_path, rows)
                if distributed or node.config.get("failure_policy") == "strict":
                    raise
            else:
                rows.append(row)
                if group_rank == 0:
                    _atomic_json(output_path, rows)
    return output_path


def _aggregate_filter(
    ledger: CandidateLedger,
    node: CompiledPostMIPNode,
    input_set: CandidateSet,
    execution_identity: str,
) -> tuple[list[NodeObservation], CandidateSet]:
    selected, excluded, scores = apply_filter(ledger, input_set.revision_ids, node.config)
    observations = [
        NodeObservation(
            node_id=node.node_id,
            input_revision_id=revision_id,
            source_revision_id=revision_id,
            output_revision_id=revision_id if revision_id in selected else None,
            status="selected" if revision_id in selected else "excluded",
            metrics=({"aggregate_rank": scores[revision_id]} if revision_id in scores else {}),
            warnings=[excluded[revision_id]] if revision_id in excluded else [],
        )
        for revision_id in input_set.revision_ids
    ]
    return observations, CandidateSet.create(
        node.flow_id,
        node.node_id,
        selected,
        producer_execution_identity=execution_identity,
    )


def _aggregate_result_manifest(
    config: Mapping[str, Any],
    ledger: CandidateLedger,
    node: CompiledPostMIPNode,
    input_set: CandidateSet,
    execution_identity: str,
) -> tuple[list[NodeObservation], CandidateSet]:
    """Freeze pre-KD lineage, learning-curve metrics, and exposure in one artifact."""

    settings = dict(node.config.get("config") or {})
    pre_kd_source = str(settings["pre_kd_source"])
    pre_kd_evaluation = str(settings["pre_kd_evaluation"])
    observations = []
    for revision_id in input_set.revision_ids:
        revision = ledger.revisions[revision_id]
        pre_kd = ledger.source_revision(revision_id, pre_kd_source)
        pre_kd_evaluation_observation = ledger._observation_for_revision(
            pre_kd_evaluation, revision_id
        )
        if pre_kd_evaluation_observation is None:
            raise RuntimeError(
                f"missing pre-KD evaluation {pre_kd_evaluation!r} for {revision.architecture_id}"
            )
        pre_kd_comparison = json.loads(
            Path(pre_kd_evaluation_observation.artifacts["comparison_path"]).read_text()
        )
        milestones = []
        for milestone in settings["milestones"]:
            kd_observation = ledger._observation_for_revision(str(milestone["kd"]), revision_id)
            evaluation_observation = ledger._observation_for_revision(
                str(milestone["evaluation"]), revision_id
            )
            if kd_observation is None or kd_observation.output_revision_id is None:
                raise RuntimeError(
                    f"missing KD milestone {milestone['kd']!r} for {revision.architecture_id}"
                )
            if evaluation_observation is None:
                raise RuntimeError(
                    "missing evaluation milestone "
                    f"{milestone['evaluation']!r} for {revision.architecture_id}"
                )
            kd_revision = ledger.revisions[kd_observation.output_revision_id]
            milestones.append(
                {
                    "steps": int(milestone["steps"]),
                    "kd_node": str(milestone["kd"]),
                    "evaluation_node": str(milestone["evaluation"]),
                    "checkpoint": kd_revision.artifact["checkpoint"],
                    "kd_metrics": kd_observation.metrics,
                    "evaluation_metrics": evaluation_observation.metrics,
                    "kd_artifacts": kd_observation.artifacts,
                    "evaluation_artifacts": evaluation_observation.artifacts,
                    "evaluation_identity": json.loads(
                        Path(evaluation_observation.artifacts["comparison_path"]).read_text()
                    )["identity"],
                }
            )
        payload = {
            "schema": "modelopt.puzzletron.vlm-kd-learning-curve/v1",
            "execution_identity": execution_identity,
            "architecture_id": revision.architecture_id,
            "architecture": canonicalize(asdict(ledger.architectures[revision.architecture_id])),
            "pre_kd": {
                "revision_id": pre_kd.revision_id,
                "checkpoint": pre_kd.artifact["checkpoint"],
                "evaluation_node": pre_kd_evaluation,
                "evaluation_metrics": pre_kd_evaluation_observation.metrics,
                "evaluation_artifacts": pre_kd_evaluation_observation.artifacts,
                "evaluation_identity": pre_kd_comparison["identity"],
            },
            "evaluation_identity": {
                "profile": settings.get("profile"),
                "row_manifest": settings.get("row_manifest"),
                "row_manifest_sha256": settings.get("row_manifest_sha256"),
                "reference_checkpoint": settings.get("reference_checkpoint"),
                "reference_cache_id": settings.get("reference_cache_id"),
            },
            "milestones": milestones,
        }
        manifest_path = (
            _execution_root(config, node, execution_identity)
            / "results"
            / f"{revision.architecture_id}.json"
        )
        canonical_payload = canonicalize(payload)
        if manifest_path.is_file() and json.loads(manifest_path.read_text()) != canonical_payload:
            raise RuntimeError(f"immutable learning-curve result changed: {manifest_path}")
        if not manifest_path.is_file():
            _atomic_json(manifest_path, payload)
        observations.append(
            NodeObservation(
                node_id=node.node_id,
                input_revision_id=revision_id,
                source_revision_id=revision_id,
                output_revision_id=revision_id,
                status="selected",
                artifacts={"result_manifest_path": str(manifest_path)},
            )
        )
    return observations, CandidateSet.create(
        node.flow_id,
        node.node_id,
        input_set.revision_ids,
        producer_execution_identity=execution_identity,
    )


def aggregate_post_mip_node(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    node = _compiled_node(config, stage_id)
    ledger = _ledger(config)
    ledger.ingest_mip(_puzzle_dir(config))
    input_set = _input_set(ledger, config, node)
    execution_contract = post_mip_execution_contract(config, node, input_set, ledger)
    execution_identity = post_mip_execution_contract_identity(execution_contract)
    timed_out_candidates = []
    if node.node_type == "filter":
        observations, output_set = _aggregate_filter(ledger, node, input_set, execution_identity)
    elif node.node_type == "result_manifest":
        observations, output_set = _aggregate_result_manifest(
            config, ledger, node, input_set, execution_identity
        )
    elif node.node_type == "manual_filter":
        decision_path = _node_root(config, node) / "manual_decision.json"
        decision = json.loads(decision_path.read_text()) if decision_path.is_file() else {}
        if decision.get("execution_identity") != execution_identity:
            review = {
                "status": "waiting_for_input",
                "node_id": node.node_id,
                "revision_ids": list(input_set.revision_ids),
                "execution_identity": execution_identity,
                "candidates": [
                    ledger.candidate_metadata(revision_id) for revision_id in input_set.revision_ids
                ],
            }
            _atomic_json(_node_root(config, node) / "manual_review.json", review)
            return review
        selected = tuple(decision.get("revision_ids") or ())
        unknown = set(selected) - set(input_set.revision_ids)
        if unknown:
            raise ValueError(f"manual decision contains unknown revisions: {sorted(unknown)}")
        observations = [
            NodeObservation(
                node_id=node.node_id,
                input_revision_id=revision_id,
                source_revision_id=revision_id,
                output_revision_id=revision_id if revision_id in selected else None,
                status="selected" if revision_id in selected else "excluded",
            )
            for revision_id in input_set.revision_ids
        ]
        output_set = CandidateSet.create(
            node.flow_id,
            node.node_id,
            selected,
            producer_execution_identity=execution_identity,
        )
    else:
        rows = []
        for path in sorted(
            (_execution_root(config, node, execution_identity) / "shards").glob("shard_*.json")
        ):
            rows.extend(
                row
                for row in json.loads(path.read_text())
                if row.get("execution_identity") == execution_identity
            )
        by_input = {row["input_revision_id"]: row for row in rows}
        missing = set(input_set.revision_ids) - set(by_input)
        if missing:
            raise RuntimeError(
                f"post-MIP node {node.node_id} is missing shard results: {sorted(missing)}"
            )
        observations = []
        output_ids = []
        for input_revision_id in input_set.revision_ids:
            row = by_input[input_revision_id]
            if row["status"] == "timed_out":
                timed_out_candidates.append(
                    {
                        "architecture_id": row["architecture_id"],
                        "input_revision_id": input_revision_id,
                        "timeout_seconds": row.get("timeout_seconds"),
                        "error": row.get("error"),
                    }
                )
            output_revision_id = input_revision_id
            if row["status"] == "success" and node.capabilities.kind is NodeKind.TRANSFORMER:
                artifact_kind = ArtifactKind(row["artifact_kind"])
                revision = ledger.add_revision(
                    architecture_id=row["architecture_id"],
                    artifact_kind=artifact_kind,
                    artifact={
                        key: value
                        for key, value in row.items()
                        if key in {"checkpoint", "exposure_path", "summary_path", "result_path"}
                    },
                    parent_revision_id=row["source_revision_id"],
                    producer_node=node.node_id,
                )
                output_revision_id = revision.revision_id
            if row["status"] == "success":
                output_ids.append(output_revision_id)
            observations.append(
                NodeObservation(
                    node_id=node.node_id,
                    input_revision_id=input_revision_id,
                    source_revision_id=row["source_revision_id"],
                    output_revision_id=(output_revision_id if row["status"] == "success" else None),
                    status=row["status"],
                    metrics=dict(row.get("metrics") or {}),
                    artifacts={
                        key: value
                        for key, value in row.items()
                        if key.endswith(("_path", "_paths"))
                    },
                    error=row.get("error"),
                )
            )
        if not output_ids and input_set.revision_ids:
            raise RuntimeError(f"post-MIP node {node.node_id} produced no successful candidates")
        output_set = CandidateSet.create(
            node.flow_id,
            node.node_id,
            output_ids,
            producer_execution_identity=execution_identity,
        )
    observations_path, candidate_set_path = ledger.publish_node(
        node.node_id, observations, output_set, execution_identity
    )
    summary = {
        "status": "success",
        "stage_id": stage_id,
        "node_id": node.node_id,
        "input_count": len(input_set.revision_ids),
        "output_count": len(output_set.revision_ids),
        "observations_path": str(observations_path),
        "candidate_set_path": str(candidate_set_path),
        "execution_identity": execution_identity,
        "execution_contract": execution_contract,
        "checkpoints": sorted(
            {
                str(ledger.revisions[revision_id].artifact["checkpoint"])
                for revision_id in output_set.revision_ids
                if "checkpoint" in ledger.revisions[revision_id].artifact
            }
        ),
        "metric_names": sorted(
            {metric for observation in observations for metric in observation.metrics}
        ),
        "status_counts": {
            status: sum(observation.status == status for observation in observations)
            for status in sorted({observation.status for observation in observations})
        },
    }
    if timed_out_candidates:
        summary["timed_out_candidates"] = timed_out_candidates
    execution_summary = _execution_root(config, node, execution_identity) / "summary.json"
    if execution_summary.is_file():
        if json.loads(execution_summary.read_text()) != canonicalize(summary):
            raise RuntimeError(f"immutable post-MIP execution summary changed: {execution_summary}")
    else:
        _atomic_json(execution_summary, summary)
    _atomic_json(_node_root(config, node) / "summary.json", summary)
    return summary
