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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute and aggregate one compiled post-MIP node."""

from __future__ import annotations

import copy
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..identity import canonicalize, stable_hash
from ..orchestration.mesh import normalize_vllm_topology
from .base import CompiledPostMIPNode, NodeKind, compile_post_mip_flows
from .filters import apply_filter
from .records import ArtifactKind, CandidateLedger, CandidateSet, NodeObservation

__all__ = [
    "aggregate_post_mip_node",
    "expected_post_mip_execution_identity",
    "run_post_mip_node_shard",
]


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


def _node_root(config: Mapping[str, Any], node: CompiledPostMIPNode) -> Path:
    return _puzzle_dir(config) / "artifacts" / "post_mip" / "nodes" / node.node_id


def _execution_root(
    config: Mapping[str, Any], node: CompiledPostMIPNode, execution_identity: str
) -> Path:
    return _node_root(config, node) / "executions" / execution_identity


def _execution_contract(
    config: Mapping[str, Any],
    node: CompiledPostMIPNode,
    candidate_set: CandidateSet,
    ledger: CandidateLedger,
) -> dict[str, Any]:
    dependency_owners = {
        reference.partition(".")[0]
        for reference in node.metric_references
        if not reference.startswith("mip.")
    }
    if node.model_source not in {"latest", "origin"}:
        dependency_owners.add(node.model_source)
    dependency_executions = {}
    for owner in sorted(dependency_owners):
        current_path = (
            _puzzle_dir(config) / "artifacts" / "post_mip" / "nodes" / owner / "current.json"
        )
        dependency_executions[owner] = json.loads(current_path.read_text())["execution_identity"]
    source_revisions = {
        revision_id: ledger.source_revision(revision_id, node.model_source).revision_id
        for revision_id in candidate_set.revision_ids
    }
    return {
        "candidate_set": candidate_set.identity,
        "node": node.config,
        "dependency_executions": dependency_executions,
        "source_revisions": source_revisions,
    }


def _execution_identity(
    config: Mapping[str, Any],
    node: CompiledPostMIPNode,
    candidate_set: CandidateSet,
    ledger: CandidateLedger,
) -> str:
    return stable_hash(
        _execution_contract(config, node, candidate_set, ledger),
        prefix="post_mip_execution",
    )


def expected_post_mip_execution_identity(config: Mapping[str, Any], stage_id: str) -> str:
    """Return the identity a completed post-MIP stage must have right now."""

    node = _compiled_node(config, stage_id)
    ledger = _ledger(config)
    active = json.loads((_puzzle_dir(config) / "mip" / "active_profiles.json").read_text())
    if active.get("status") != "success" or ledger.active_mip_execution_identity != active.get(
        "execution_identity"
    ):
        raise RuntimeError("post-MIP ledger does not reflect the active MIP execution")
    return _execution_identity(config, node, _input_set(ledger, config, node), ledger)


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
        return {"artifact_kind": "checkpoint", "checkpoint": source.artifact["checkpoint"]}
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
    settings.update(
        enabled=True,
        checkpoints=[source.artifact["checkpoint"]],
        output_dir=str(output),
    )
    candidate["zero_shot_evaluation"] = settings
    manifest = StageManifest(
        stage=f"{node.stage_id}.{source.architecture_id}",
        inputs={"config": candidate},
        config=candidate,
        semantic_config=semantic_stage_config(candidate, "zero_shot_evaluation"),
    )
    evaluation_stage(candidate, manifest)
    rows = json.loads((output / "evaluation_summary.json").read_text())
    checkpoint = str(source.artifact["checkpoint"])
    row = next(item for item in rows if str(item.get("checkpoint")) == checkpoint)
    return {"metrics": dict(row.get("metrics") or {}), "result_path": row.get("result_path")}


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
        **settings,
    )
    metrics = {}
    for result in results:
        if len(results) == 1:
            metrics.update(result.metrics)
        metrics.update(
            {
                f"concurrency_{result.concurrency}.{key}": value
                for key, value in result.metrics.items()
            }
        )
    return {
        "metrics": metrics,
        "result_paths": [result.raw_artifacts for result in results],
    }


_LMMS_EVAL_MODEL_ARG_FIELDS = frozenset(
    {
        "dtype",
        "gpu_memory_utilization",
        "max_model_len",
        "trust_remote_code",
        "tokenizer",
        "tokenizer_mode",
        "enforce_eager",
        "limit_mm_per_prompt",
        "reasoning_parser",
    }
)
_LMMS_EVAL_RESERVED_TOPOLOGY_MODEL_ARG_FIELDS = frozenset(
    {
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "enable_expert_parallel",
        "distributed_executor_backend",
        "expert_parallel_size",
        "gpu_group_size",
        "tp",
        "pp",
        "dp",
        "prefill_cp",
        "decode_cp",
        "ep",
    }
)
_LMMS_EVAL_RESERVED_EXTRA_ARG_FLAGS = frozenset(
    {
        "--batch-size",
        "--batch_size",
        "--model",
        "--model_args",
        "--model-args",
        "--output_path",
        "--output-path",
        "--tasks",
    }
)
_DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS = 3600.0
_LMMS_EVAL_PROCESS_CLEANUP_TIMEOUT_SECONDS = 10.0


def _as_cli_bool(value: bool) -> str:
    return "True" if value else "False"


def _as_lmms_eval_arg(value: Any) -> str:
    if isinstance(value, bool):
        return _as_cli_bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _join_cli_values(value: Any, *, path: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{path} must not be empty")
        return text
    if not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a string or sequence")
    values = [str(item).strip() for item in value]
    if not values or any(not item for item in values):
        raise ValueError(f"{path} must contain at least one non-empty value")
    return ",".join(values)


def _lmms_eval_model_arg_keys(value: str) -> tuple[str, ...]:
    keys: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False

    def append(segment: str) -> None:
        key, separator, _ = segment.strip().partition("=")
        if separator and key.strip():
            keys.append(key.strip())

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if quote:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            append(value[start:index])
            start = index + 1
    append(value[start:])
    return tuple(keys)


def _lmms_eval_reserved_model_arg_fields(checkpoint_arg: str) -> frozenset[str]:
    return frozenset(
        key
        for key in (
            str(checkpoint_arg).strip(),
            *_LMMS_EVAL_RESERVED_TOPOLOGY_MODEL_ARG_FIELDS,
        )
        if key
    )


def _reject_reserved_lmms_eval_model_args(
    keys: Sequence[Any], reserved_fields: frozenset[str]
) -> None:
    reserved = sorted({str(key).strip() for key in keys} & reserved_fields)
    if reserved:
        raise ValueError(
            "downstream_evaluation.config.model_args must not set reserved "
            f"lmms-eval model arguments: {', '.join(reserved)}"
        )


def _configured_lmms_eval_tasks(settings: Mapping[str, Any]) -> tuple[str, ...]:
    tasks = _join_cli_values(settings.get("tasks"), path="downstream_evaluation.config.tasks")
    values = tuple(task.strip() for task in tasks.split(","))
    if not values or any(not task for task in values):
        raise ValueError("downstream_evaluation.config.tasks must contain non-empty task names")
    return values


def _model_arg_string(values: Mapping[str, Any]) -> str:
    parts = []
    for key, value in values.items():
        if value is None:
            continue
        key_text = str(key).strip()
        if not key_text or "," in key_text or "=" in key_text:
            raise ValueError(f"invalid lmms-eval model_args key: {key!r}")
        rendered = _as_lmms_eval_arg(value)
        if "," in rendered:
            raise ValueError(
                f"lmms-eval model_args value for {key_text!r} contains a comma; "
                "provide model_args as a preformatted string instead"
            )
        parts.append(f"{key_text}={rendered}")
    if not parts:
        raise ValueError("lmms-eval model_args must contain at least the checkpoint path")
    return ",".join(parts)


def _merge_lmms_eval_model_args(settings: Mapping[str, Any], checkpoint: str) -> str:
    raw = settings.get("model_args")
    checkpoint_arg = str(settings.get("checkpoint_arg", "model"))
    if checkpoint_arg != "model":
        raise ValueError("downstream_evaluation.config.checkpoint_arg must be 'model'")
    topology = dict(settings.get("topology") or {})
    canonical_topology = normalize_vllm_topology(topology) if topology else {}
    reserved_fields = _lmms_eval_reserved_model_arg_fields(checkpoint_arg)
    derived = {
        checkpoint_arg: checkpoint,
    }
    if canonical_topology:
        derived.update(
            {
                "tensor_parallel_size": canonical_topology["tp"],
                "pipeline_parallel_size": canonical_topology["pp"],
                "data_parallel_size": canonical_topology["dp"],
                "enable_expert_parallel": canonical_topology["enable_expert_parallel"],
                "distributed_executor_backend": canonical_topology["distributed_executor_backend"],
            }
        )
    for key in sorted(_LMMS_EVAL_MODEL_ARG_FIELDS):
        if key in settings:
            derived[key] = settings[key]

    if isinstance(raw, str):
        _reject_reserved_lmms_eval_model_args(_lmms_eval_model_arg_keys(raw), reserved_fields)
        prefix = raw.strip().strip(",")
        suffix = _model_arg_string(derived)
        return ",".join(part for part in (prefix, suffix) if part)
    if raw is not None and not isinstance(raw, Mapping):
        raise TypeError("downstream_evaluation.config.model_args must be a mapping or string")
    _reject_reserved_lmms_eval_model_args(tuple((raw or {}).keys()), reserved_fields)
    merged = dict(raw or {})
    merged.update(derived)
    return _model_arg_string(merged)


def _command_prefix(settings: Mapping[str, Any]) -> list[str]:
    raw = settings.get("command_prefix")
    if raw is None:
        return [sys.executable, "-m", "lmms_eval"]
    if isinstance(raw, str):
        values = [raw]
    else:
        values = [str(item) for item in raw]
    if not values or any(not value for value in values):
        raise ValueError("downstream_evaluation.config.command_prefix must not be empty")
    return values


def _lmms_eval_extra_args(settings: Mapping[str, Any]) -> list[str]:
    raw = settings.get("extra_args")
    if raw is None:
        return []
    if isinstance(raw, str):
        values = shlex.split(raw)
    elif isinstance(raw, Sequence):
        values = [str(item) for item in raw]
    else:
        raise TypeError("downstream_evaluation.config.extra_args must be a string or sequence")
    if any(not value for value in values):
        raise ValueError("downstream_evaluation.config.extra_args must not contain empty values")
    reserved = sorted(
        {
            value.split("=", 1)[0]
            for value in values
            if value.split("=", 1)[0] in _LMMS_EVAL_RESERVED_EXTRA_ARG_FLAGS
        }
    )
    if reserved:
        raise ValueError(
            "downstream_evaluation.config.extra_args must not set reserved "
            f"lmms-eval flags: {', '.join(reserved)}"
        )
    return values


def _lmms_eval_command(
    settings: Mapping[str, Any],
    *,
    checkpoint: str,
    output_path: Path,
) -> tuple[list[str], dict[str, str], float | None]:
    """Build a deterministic lmms-eval CLI invocation for one realized checkpoint."""

    model = str(settings.get("model", "vllm"))
    if model != "vllm":
        raise ValueError("downstream_evaluation.config.model must be 'vllm'")
    tasks = ",".join(_configured_lmms_eval_tasks(settings))
    argv = [
        *_command_prefix(settings),
        "--model",
        model,
        "--model_args",
        _merge_lmms_eval_model_args(settings, checkpoint),
        "--tasks",
        tasks,
        "--batch_size",
        str(settings.get("batch_size", 1)),
        "--output_path",
        str(output_path),
    ]
    optional_fields = {
        "limit": "--limit",
        "num_fewshot": "--num_fewshot",
        "seed": "--seed",
        "verbosity": "--verbosity",
        "device": "--device",
        "use_cache": "--use_cache",
    }
    for key, flag in optional_fields.items():
        value = settings.get(key)
        if value is not None:
            argv.extend([flag, str(value)])
    if settings.get("gen_kwargs") is not None:
        argv.extend(
            [
                "--gen_kwargs",
                (
                    settings["gen_kwargs"]
                    if isinstance(settings["gen_kwargs"], str)
                    else _model_arg_string(dict(settings["gen_kwargs"]))
                ),
            ]
        )
    if bool(settings.get("log_samples", False)):
        argv.append("--log_samples")
    argv.extend(_lmms_eval_extra_args(settings))

    env = os.environ.copy()
    for key, value in dict(settings.get("env") or {}).items():
        if value is not None:
            env[str(key)] = str(value)
    if settings.get("cache_dir") is not None:
        env.setdefault("LMMS_EVAL_HOME", str(settings["cache_dir"]))
    timeout = settings.get("timeout_seconds", settings.get("timeout"))
    if timeout is None:
        timeout = _DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS
    return argv, env, float(timeout)


def _metric_key(value: Any) -> str:
    return (
        str(value).strip().replace(" ", "_").replace(",", "_").replace("/", "_").replace("\\", "_")
    )


def _numeric_metrics(task_payload: Mapping[str, Any]) -> dict[str, float]:
    metrics = {}
    for metric_name, value in task_payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            metrics[str(metric_name)] = float(value)
    return metrics


def _flatten_lmms_eval_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    results = payload.get("results")
    if not isinstance(results, Mapping):
        return {}
    metrics = {}
    for task_name, task_payload in results.items():
        if not isinstance(task_payload, Mapping):
            continue
        for metric_name, value in _numeric_metrics(task_payload).items():
            metrics[f"{_metric_key(task_name)}.{_metric_key(metric_name)}"] = value
    return metrics


def _resolved_lmms_eval_tasks(
    payload: Mapping[str, Any], configured_tasks: Sequence[str]
) -> tuple[str, ...]:
    group_subtasks = payload.get("group_subtasks")
    if not isinstance(group_subtasks, Mapping):
        group_subtasks = {}

    def expand(task: str, seen: frozenset[str]) -> tuple[str, ...]:
        raw_subtasks = group_subtasks.get(task)
        if (
            isinstance(raw_subtasks, Sequence)
            and not isinstance(raw_subtasks, str)
            and raw_subtasks
            and task not in seen
        ):
            expanded = []
            for raw_subtask in raw_subtasks:
                expanded.extend(expand(str(raw_subtask), seen | {task}))
            return tuple(dict.fromkeys(expanded))
        return (task,)

    resolved = []
    for task in configured_tasks:
        resolved.extend(expand(task, frozenset()))
    return tuple(dict.fromkeys(resolved))


def _sample_count(payload: Mapping[str, Any], task: str) -> float | None:
    samples = payload.get("n-samples", payload.get("n_samples"))
    if not isinstance(samples, Mapping):
        return None
    value = samples.get(task)
    if isinstance(value, Mapping):
        if "effective" in value:
            value = value["effective"]
        elif "original" in value:
            value = value["original"]
        else:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _validate_lmms_eval_completion(
    payload: Mapping[str, Any], configured_tasks: Sequence[str]
) -> dict[str, float]:
    results = payload.get("results")
    if not isinstance(results, Mapping):
        raise RuntimeError("lmms-eval result is missing the results mapping")

    expected_tasks = _resolved_lmms_eval_tasks(payload, configured_tasks)
    missing_results = [task for task in expected_tasks if task not in results]
    if missing_results:
        raise RuntimeError(
            f"lmms-eval result is missing configured task results: {sorted(missing_results)}"
        )

    missing_metrics = [
        task
        for task in expected_tasks
        if not isinstance(results[task], Mapping) or not _numeric_metrics(results[task])
    ]
    if missing_metrics:
        raise RuntimeError(
            "lmms-eval result has no numeric metrics for configured tasks: "
            f"{sorted(missing_metrics)}"
        )

    sample_counts = {}
    missing_samples = []
    zero_samples = []
    for task in expected_tasks:
        sample_count = _sample_count(payload, task)
        if sample_count is None:
            missing_samples.append(task)
        elif sample_count <= 0:
            zero_samples.append(task)
        else:
            sample_counts[task] = sample_count
    if missing_samples:
        raise RuntimeError(
            "lmms-eval result is missing sample counts for configured tasks: "
            f"{sorted(missing_samples)}"
        )
    if zero_samples:
        raise RuntimeError(
            "lmms-eval result has zero effective samples for configured tasks: "
            f"{sorted(zero_samples)}"
        )
    return sample_counts


def _lmms_eval_result_payload(output_path: Path) -> tuple[dict[str, Any], Path]:
    candidates = []
    for path in sorted(output_path.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(payload, Mapping) and isinstance(payload.get("results"), Mapping):
            candidates.append((path.stat().st_mtime_ns, path, dict(payload)))
    if not candidates:
        raise FileNotFoundError(f"lmms-eval wrote no JSON results below {output_path}")
    _mtime, path, payload = max(candidates, key=lambda item: item[0])
    return payload, path


def _write_lmms_eval_streams(
    output_path: Path, result: subprocess.CompletedProcess[str]
) -> dict[str, str]:
    stream_paths = {}
    for stream_name, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        if not text:
            continue
        stream_path = output_path / f"{stream_name}.txt"
        stream_path.write_text(text)
        stream_paths[f"{stream_name}_path"] = str(stream_path)
    return stream_paths


def _lmms_eval_output_tail(result: subprocess.CompletedProcess[str], *, max_lines: int = 20) -> str:
    sections = []
    for stream_name, text in (("stderr", result.stderr), ("stdout", result.stdout)):
        lines = (text or "").strip().splitlines()
        if lines:
            sections.append(f"{stream_name} tail:")
            sections.extend(lines[-max_lines:])
    return "\n".join(sections)


def _signal_lmms_eval_process_group(process: subprocess.Popen[str], signal_number: int) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal_number)
        else:
            process.send_signal(signal_number)
    except ProcessLookupError:
        pass


def _lmms_eval_process_group_exists(process: subprocess.Popen[str]) -> bool:
    if os.name != "posix":
        return process.poll() is None
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_lmms_eval_process(
    argv: list[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _signal_lmms_eval_process_group(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=_LMMS_EVAL_PROCESS_CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_lmms_eval_process_group(process, signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(
                    timeout=_LMMS_EVAL_PROCESS_CLEANUP_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired as kill_error:
                stdout, stderr = kill_error.output, kill_error.stderr
        else:
            if _lmms_eval_process_group_exists(process):
                _signal_lmms_eval_process_group(process, signal.SIGKILL)
        raise subprocess.TimeoutExpired(
            argv,
            error.timeout,
            output=stdout if stdout is not None else error.output,
            stderr=stderr if stderr is not None else error.stderr,
        ) from error
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _downstream_evaluation(
    config: dict[str, Any],
    node: CompiledPostMIPNode,
    source,
    execution_identity: str,
) -> dict[str, Any]:
    if source.artifact_kind is not ArtifactKind.CHECKPOINT:
        raise ValueError("downstream_evaluation requires materialized checkpoint artifacts")
    settings = dict(node.config.get("config") or {})
    output_root = (
        _execution_root(config, node, execution_identity)
        / "raw"
        / source.architecture_id
        / "lmms_eval"
    )
    output = output_root / f"attempt_{uuid.uuid4().hex}"
    output.mkdir(parents=True, exist_ok=True)
    argv, env, timeout = _lmms_eval_command(
        settings,
        checkpoint=str(source.artifact["checkpoint"]),
        output_path=output,
    )
    command_path = output / "command.json"
    _atomic_json(
        command_path,
        {
            "argv": argv,
            "env_overrides": sorted(str(key) for key in dict(settings.get("env") or {})),
            "timeout": timeout,
        },
    )
    # Campaign config controls the executable and arguments, but subprocess receives
    # an argv list directly; no shell parsing is involved.
    result = _run_lmms_eval_process(
        argv,
        cwd=str(output),
        env=env,
        timeout=timeout,
    )
    stream_paths = _write_lmms_eval_streams(output, result)
    if result.returncode:
        tail = _lmms_eval_output_tail(result)
        raise RuntimeError(
            f"lmms-eval failed with exit code {result.returncode}" + (f": {tail}" if tail else "")
        )
    try:
        payload, result_path = _lmms_eval_result_payload(output)
    except FileNotFoundError as error:
        tail = _lmms_eval_output_tail(result)
        raise FileNotFoundError(str(error) + (f": {tail}" if tail else "")) from error
    sample_counts = _validate_lmms_eval_completion(payload, _configured_lmms_eval_tasks(settings))
    metrics = _flatten_lmms_eval_metrics(payload)
    if not metrics:
        raise RuntimeError(f"lmms-eval result has no numeric task metrics: {result_path}")
    summary_path = output / "summary.json"
    _atomic_json(
        summary_path,
        {
            "architecture_id": source.architecture_id,
            "checkpoint": source.artifact["checkpoint"],
            "metrics": metrics,
            "result_path": str(result_path),
            "sample_counts": sample_counts,
        },
    )
    return {
        "metrics": metrics,
        "result_path": str(summary_path),
        "raw_result_path": str(result_path),
        "command_path": str(command_path),
        **stream_paths,
    }


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
    output = (
        _execution_root(config, node, execution_identity) / "checkpoints" / source.architecture_id
    )
    settings.update(student_dir=source.artifact["checkpoint"], output_dir=str(output))
    candidate["distillation"] = settings
    kd_config = build_global_kd_config(candidate)
    result = run_global_kd(kd_config)
    summary_path = _write_global_distillation_summary(kd_config, result)
    summary = json.loads(summary_path.read_text())
    checkpoint = summary.get("post_kd_checkpoint")
    if not checkpoint:
        raise RuntimeError("global KD produced no consolidated checkpoint")
    return {
        "artifact_kind": "checkpoint",
        "checkpoint": str(checkpoint),
        "metrics": dict(result.metrics),
        "summary_path": str(summary_path),
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
    execution_identity = _execution_identity(config, node, candidate_set, ledger)
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
                timed_out = node.node_type in {"aiperf", "downstream_evaluation"} and isinstance(
                    error, (subprocess.TimeoutExpired, TimeoutError)
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
                    elif not isinstance(error, subprocess.TimeoutExpired):
                        timeout_field = "readiness_timeout"
                    default_timeout = (
                        _DEFAULT_LMMS_EVAL_TIMEOUT_SECONDS
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


def aggregate_post_mip_node(config: dict[str, Any], stage_id: str) -> dict[str, Any]:
    node = _compiled_node(config, stage_id)
    ledger = _ledger(config)
    ledger.ingest_mip(_puzzle_dir(config))
    input_set = _input_set(ledger, config, node)
    execution_identity = _execution_identity(config, node, input_set, ledger)
    timed_out_candidates = []
    if node.node_type == "filter":
        observations, output_set = _aggregate_filter(ledger, node, input_set, execution_identity)
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
                        if key in {"checkpoint", "summary_path", "result_path"}
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
        "execution_contract": _execution_contract(config, node, input_set, ledger),
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
