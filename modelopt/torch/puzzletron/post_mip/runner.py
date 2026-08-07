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
    """Run an AI performance sweep for a checkpoint across configured concurrency levels.
    
    Parameters:
        config (dict[str, Any]): Workflow configuration used to determine the execution directory.
        node (CompiledPostMIPNode): Compiled post-MIP node containing benchmark settings.
        source: Candidate source containing the checkpoint and architecture identifier.
        execution_identity (str): Identifier for the current node execution.
    
    Returns:
        dict[str, Any]: Benchmark metrics and paths to the raw result artifacts.
    """
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


def _as_cli_bool(value: bool) -> str:
    """Convert a Boolean value to the CLI-compatible ``"True"`` or ``"False"`` string.
    
    Parameters:
    	value (bool): The Boolean value to convert.
    
    Returns:
    	str: ``"True"`` for true values and ``"False"`` for false values.
    """
    return "True" if value else "False"


def _as_lmms_eval_arg(value: Any) -> str:
    """
    Convert a value to the command-line argument format expected by lmms-eval.
    
    Parameters:
    	value (Any): The value to convert.
    
    Returns:
    	str: The formatted command-line argument value.
    """
    if isinstance(value, bool):
        return _as_cli_bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _join_cli_values(value: Any, *, path: str) -> str:
    """
    Convert a string or sequence of values into a comma-separated CLI value.
    
    Parameters:
        value (Any): String or sequence of values to normalize.
        path (str): Configuration path used in validation errors.
    
    Returns:
        str: The normalized comma-separated value.
    
    Raises:
        TypeError: If value is neither a string nor a sequence.
        ValueError: If value is empty or contains an empty item.
    """
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


def _model_arg_string(values: Mapping[str, Any]) -> str:
    """
    Convert model arguments to lmms-eval's comma-separated argument format.
    
    Parameters:
        values (Mapping[str, Any]): Model argument names and values.
    
    Returns:
        str: A comma-separated string of rendered key-value arguments.
    
    Raises:
        ValueError: If an argument key or value is invalid, or if no arguments are provided.
    """
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
    """
    Merge checkpoint, topology, and supported model settings into lmms-eval model arguments.
    
    Parameters:
        settings (Mapping[str, Any]): Downstream evaluation settings containing optional model arguments and configuration overrides.
        checkpoint (str): Path to the checkpoint used for evaluation.
    
    Returns:
        str: Comma-separated lmms-eval model arguments.
    """
    raw = settings.get("model_args")
    checkpoint_arg = str(settings.get("checkpoint_arg", "model"))
    topology = dict(settings.get("topology") or {})
    canonical_topology = normalize_vllm_topology(topology) if topology else {}
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
                "distributed_executor_backend": canonical_topology[
                    "distributed_executor_backend"
                ],
            }
        )
    for key in _LMMS_EVAL_MODEL_ARG_FIELDS:
        if key in settings:
            derived[key] = settings[key]

    if isinstance(raw, str):
        prefix = raw.strip().strip(",")
        suffix = _model_arg_string(derived)
        return ",".join(part for part in (prefix, suffix) if part)
    if raw is not None and not isinstance(raw, Mapping):
        raise TypeError("downstream_evaluation.config.model_args must be a mapping or string")
    merged = dict(raw or {})
    for key, value in derived.items():
        merged.setdefault(key, value)
    return _model_arg_string(merged)


def _command_prefix(settings: Mapping[str, Any]) -> list[str]:
    """
    Resolve the command prefix used to invoke lmms-eval.
    
    Parameters:
    	settings (Mapping[str, Any]): Downstream evaluation settings containing an optional command prefix.
    
    Returns:
    	list[str]: The configured command prefix, or the current Python interpreter followed by the lmms-eval module.
    
    Raises:
    	ValueError: If the configured command prefix is empty or contains an empty value.
    """
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


def _lmms_eval_command(
    settings: Mapping[str, Any],
    *,
    checkpoint: str,
    output_path: Path,
) -> tuple[list[str], dict[str, str], float | None]:
    """
    Builds an lmms-eval command, environment, and optional timeout for a realized checkpoint.
    
    Parameters:
    	settings (Mapping[str, Any]): Downstream evaluation settings.
    	checkpoint (str): Path to the realized checkpoint.
    	output_path (Path): Directory for lmms-eval output.
    
    Returns:
    	tuple[list[str], dict[str, str], float | None]: The command arguments, environment variables, and timeout in seconds.
    """

    tasks = _join_cli_values(settings.get("tasks"), path="downstream_evaluation.config.tasks")
    argv = [
        *_command_prefix(settings),
        "--model",
        str(settings.get("model", "vllm")),
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
    argv.extend(str(item) for item in settings.get("extra_args") or ())

    env = os.environ.copy()
    for key, value in dict(settings.get("env") or {}).items():
        if value is not None:
            env[str(key)] = str(value)
    if settings.get("cache_dir") is not None:
        env.setdefault("LMMS_EVAL_HOME", str(settings["cache_dir"]))
    timeout = settings.get("timeout_seconds", settings.get("timeout"))
    return argv, env, (float(timeout) if timeout is not None else None)


def _metric_key(value: Any) -> str:
    """
    Normalize a metric name component for use in metric keys.
    
    Parameters:
    	value (Any): The value to convert into a normalized metric name component.
    
    Returns:
    	str: The stripped string representation with spaces, commas, and slashes replaced by underscores.
    """
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace(",", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _flatten_lmms_eval_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """
    Flatten finite numeric task metrics from an lmms-eval result payload.
    
    Parameters:
    	payload (Mapping[str, Any]): Result payload containing task metrics under the ``results`` key.
    
    Returns:
    	dict[str, float]: Metric names mapped to finite numeric values, or an empty dictionary when no valid results are present.
    """
    results = payload.get("results")
    if not isinstance(results, Mapping):
        return {}
    metrics = {}
    for task_name, task_payload in results.items():
        if not isinstance(task_payload, Mapping):
            continue
        for metric_name, value in task_payload.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                metrics[f"{_metric_key(task_name)}.{_metric_key(metric_name)}"] = float(value)
    return metrics


def _lmms_eval_result_payload(output_path: Path) -> tuple[dict[str, Any], Path]:
    """
    Finds the newest valid lmms-eval result payload under an output directory.
    
    Parameters:
        output_path (Path): Directory containing lmms-eval output files.
    
    Returns:
        tuple[dict[str, Any], Path]: The result payload and path of the newest JSON file containing a `results` mapping.
    
    Raises:
        FileNotFoundError: If no valid result JSON file is found.
    """
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
    """
    Persist non-empty lmms-eval subprocess output streams and return their artifact paths.
    
    Parameters:
        output_path (Path): Directory where stream files are written.
        result (subprocess.CompletedProcess[str]): Completed subprocess result containing captured output.
    
    Returns:
        dict[str, str]: Mapping of stream path keys to the paths of written output files.
    """
    stream_paths = {}
    for stream_name, text in (("stdout", result.stdout), ("stderr", result.stderr)):
        if not text:
            continue
        stream_path = output_path / f"{stream_name}.txt"
        stream_path.write_text(text)
        stream_paths[f"{stream_name}_path"] = str(stream_path)
    return stream_paths


def _lmms_eval_output_tail(result: subprocess.CompletedProcess[str], *, max_lines: int = 20) -> str:
    """
    Format the most recent subprocess output lines from stderr and stdout.
    
    Parameters:
        result (subprocess.CompletedProcess[str]): Completed process containing captured output.
        max_lines (int): Maximum number of lines to include from each stream.
    
    Returns:
        str: Formatted stderr and stdout output tails.
    """
    sections = []
    for stream_name, text in (("stderr", result.stderr), ("stdout", result.stdout)):
        lines = (text or "").strip().splitlines()
        if lines:
            sections.append(f"{stream_name} tail:")
            sections.extend(lines[-max_lines:])
    return "\n".join(sections)


def _downstream_evaluation(
    config: dict[str, Any],
    node: CompiledPostMIPNode,
    source,
    execution_identity: str,
) -> dict[str, Any]:
    """
    Run downstream lmms-eval benchmarking for a materialized checkpoint.
    
    Parameters:
    	config (dict[str, Any]): Campaign configuration used to determine execution paths.
    	node (CompiledPostMIPNode): Post-MIP node containing lmms-eval settings.
    	source: Checkpoint artifact to evaluate.
    	execution_identity (str): Identity of the current node execution.
    
    Returns:
    	dict[str, Any]: Paths to the evaluation summary, raw result, command record, and captured streams, together with numeric metrics.
    
    Raises:
    	ValueError: If the source is not a checkpoint artifact.
    	RuntimeError: If lmms-eval fails or produces no numeric task metrics.
    	FileNotFoundError: If no valid lmms-eval result file is produced.
    """
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
    result = subprocess.run(
        argv,
        cwd=str(output),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    stream_paths = _write_lmms_eval_streams(output, result)
    if result.returncode:
        tail = _lmms_eval_output_tail(result)
        raise RuntimeError(
            f"lmms-eval failed with exit code {result.returncode}"
            + (f": {tail}" if tail else "")
        )
    try:
        payload, result_path = _lmms_eval_result_payload(output)
    except FileNotFoundError as error:
        tail = _lmms_eval_output_tail(result)
        raise FileNotFoundError(str(error) + (f": {tail}" if tail else "")) from error
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
    """
    Execute a candidate according to the node type and return its execution result.
    
    Parameters:
    	config (dict[str, Any]): Runtime configuration for the candidate execution.
    	node (CompiledPostMIPNode): Compiled node defining the execution type and model source.
    	ledger (CandidateLedger): Ledger containing the input candidate revision.
    	input_revision_id (str): Identifier of the candidate revision to execute.
    	execution_identity (str): Identifier for the current node execution.
    
    Returns:
    	dict[str, Any]: A successful result containing the input and source revision identifiers, architecture identifier, and executor-specific metadata.
    
    Raises:
    	ValueError: If the node type is not a supported candidate executor.
    """
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
    """
    Execute the assigned candidate revisions for a post-MIP node shard and persist the results.
    
    Parameters:
    	config (dict[str, Any]): Post-MIP configuration.
    	stage_id (str): Identifier of the compiled node to execute.
    	shard_index (int): Zero-based index of this shard.
    	shard_count (int): Total number of shards distributing the candidate revisions.
    
    Returns:
    	Path: Path to the shard result artifact.
    """
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
                    default_timeout = 3600 if node.node_type == "downstream_evaluation" else (
                        600 if timeout_field == "benchmark_timeout" else 1200
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
