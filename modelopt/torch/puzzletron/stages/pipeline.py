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

from __future__ import annotations

import dataclasses
import json
import math
import os
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from omegaconf import OmegaConf

import modelopt.torch.utils.distributed as dist

from ..identity import canonicalize, stable_hash
from ..pipeline_config import load_runtime_hydra_config
from ..rpc_eval import EvaluationCache, EvaluationRequest, EvaluationResult
from ..scoring_parent import ensure_scoring_parent
from ..subblock_stats.measurements import apply_vllm_measurement, normalize_vllm_measurements
from ..tools.hydra_utils import clone_hydra_config
from .common import complete_stage, experiment_dir, stage_manifest_path

if TYPE_CHECKING:
    from ..manifest import StageManifest

__all__ = [
    "activation_stage",
    "sort_stage",
    "bypass_overfit_stage",
    "bypass_stage",
    "build_library_stage",
    "configure_vllm_stats_widths",
    "emit_runtime_subblock_library",
    "finalize_vllm_measurements",
    "finalize_vllm_stats_report",
    "prepare_vllm_stats_workspace",
    "vllm_stats_stage",
    "scoring_stage",
    "mip_stage",
]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _timeout_minutes(hydra_cfg: Any) -> timedelta:
    raw = _get(hydra_cfg, "nccl_timeout_minutes", None)
    if raw is None:
        return timedelta(minutes=10)
    if isinstance(raw, timedelta):
        return raw
    return timedelta(minutes=int(raw))


@contextmanager
def _distributed(hydra_cfg: Any) -> Iterator[None]:
    already_initialized = dist.is_initialized()
    failed = False
    if not already_initialized:
        dist.setup(timeout=_timeout_minutes(hydra_cfg))
    try:
        yield
    except BaseException:
        # Any process-group operation during exception unwinding (including
        # destroy_process_group) can permanently hide the originating error
        # when other PP ranks are still inside stage-local FSDP collectives.
        # Let this rank escape; torchrun will terminate its peers.
        failed = True
        raise
    finally:
        if not failed and not already_initialized and dist.is_initialized():
            dist.cleanup()


def _runtime_split(config: dict[str, Any]) -> tuple[int, int]:
    runtime = dict(config.get("_runtime") or {})
    requested_nodes = int(runtime.get("num_nodes", 1))
    requested_index = int(runtime.get("node_index", 0))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))))
    torchrun_nodes = (world_size + local_world_size - 1) // local_world_size
    if torchrun_nodes > requested_nodes:
        return torchrun_nodes, int(os.environ.get("GROUP_RANK", requested_index))
    return requested_nodes, requested_index


def _puzzle_dir(config: dict[str, Any], hydra_cfg: Any) -> Path:
    return Path(_get(hydra_cfg, "puzzle_dir", None) or experiment_dir(config))


def _teacher_dir(config: dict[str, Any], hydra_cfg: Any) -> Path:
    convert_cfg = config.get("convert") or {}
    return Path(
        _get(hydra_cfg, "teacher_dir", None)
        or convert_cfg.get("teacher_dir")
        or _puzzle_dir(config, hydra_cfg) / "ckpts" / "teacher"
    )


def _activations_log_dir(config: dict[str, Any], hydra_cfg: Any) -> Path:
    pruning_cfg = _get(hydra_cfg, "pruning", {}) or {}
    return Path(
        _get(pruning_cfg, "activations_log_dir", None)
        or _get(hydra_cfg, "activations_log_dir", None)
        or _puzzle_dir(config, hydra_cfg) / "activations_log"
    )


def _hf_checkpoint_complete(path: Path, *, required_files: tuple[str, ...] = ()) -> bool:
    if not (path / "config.json").is_file():
        return False
    if any(not (path / name).is_file() for name in required_files):
        return False
    if (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file():
        return True
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            return False
        shards = set(index.get("weight_map", {}).values())
        if shards and all((path / shard).is_file() for shard in shards):
            return True
    return False


def _manifest_success(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "success"
    except json.JSONDecodeError:
        return False


def _inputs_not_newer_than(output_manifest: Path, input_paths: tuple[Path, ...]) -> bool:
    if not output_manifest.is_file():
        return False
    output_mtime = output_manifest.stat().st_mtime
    return all((not path.exists()) or path.stat().st_mtime <= output_mtime for path in input_paths)


def _sorted_teacher_complete(config: dict[str, Any], sorted_teacher_dir: Path) -> bool:
    manifest_path = stage_manifest_path(config, "sort")
    activation_manifest = stage_manifest_path(config, "activation")
    convert_manifest = stage_manifest_path(config, "convert")
    permutations_path = sorted_teacher_dir / "sorted_permutations.json"
    try:
        has_permutations = bool(json.loads(permutations_path.read_text()))
    except (OSError, json.JSONDecodeError):
        has_permutations = False
    return (
        _manifest_success(manifest_path)
        and _hf_checkpoint_complete(
            sorted_teacher_dir, required_files=("sorted_permutations.json",)
        )
        and has_permutations
        and _inputs_not_newer_than(manifest_path, (activation_manifest, convert_manifest))
    )


def _scoring_output_dir(hydra_cfg: Any) -> Path:
    from ..scoring import resolve_scoring_output_dir

    return Path(resolve_scoring_output_dir(hydra_cfg))


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Path):
        return str(value)
    return value


def _runtime_stats_are_sharded(hydra_cfg: Any) -> bool:
    stats_cfg = _get(hydra_cfg, "calc_subblock_stats", {}) or {}
    runtime_cfg = _get(stats_cfg, "runtime_stats", {}) or {}
    return str(_get(runtime_cfg, "execution", "inline")).lower() == "sharded"


def _vllm_stats_is_explicit(config: dict[str, Any]) -> bool:
    """Return whether the canonical DAG selects the standalone vLLM producer."""
    return bool((config.get("vllm_stats") or {}).get("enabled", False))


def _calculate_static_workload_stats(config: dict[str, Any], hydra_cfg: Any) -> None:
    """Append one analytical memory profile for every configured MIP workload."""
    from ..subblock_stats.calc_subblock_stats import launch_calc_subblock_stats

    workloads = dict((config.get("mip") or {}).get("workloads") or {})
    if not workloads:
        workloads = {
            "default": {
                "isl": int(hydra_cfg.calc_subblock_stats.prefill_seq_len),
                "osl": int(hydra_cfg.calc_subblock_stats.generation_seq_len),
                "batch_size": int(hydra_cfg.calc_subblock_stats.batch_sizes[0]),
            }
        }
    for raw_workload in workloads.values():
        workload = dict(raw_workload or {})
        selected = clone_hydra_config(hydra_cfg)
        stats_cfg = selected.calc_subblock_stats
        concurrency = int(workload.get("concurrency", workload.get("batch_size", 1)))
        stats_cfg.batch_sizes = [int(workload.get("batch_size", concurrency))]
        stats_cfg.prefill_seq_len = int(
            workload.get("isl", workload.get("prefill_seq_len", stats_cfg.prefill_seq_len))
        )
        stats_cfg.generation_seq_len = int(
            workload.get(
                "osl",
                workload.get("generation_seq_len", stats_cfg.generation_seq_len),
            )
        )
        if stats_cfg.get("runtime_stats") is None:
            stats_cfg.runtime_stats = {}
        stats_cfg.runtime_stats.enabled = False
        stats_cfg.merge_with_existing_stats = True
        launch_calc_subblock_stats(selected)


def _write_runtime_subblock_library(path: Path, block_configs: tuple[Any, ...]) -> None:
    """Write the legacy subblock-library input without assembling a replacement library."""
    rows = []
    for block_config in block_configs:
        row = {
            f"{subblock.kind}_config": subblock.to_dict()
            for subblock in block_config.subblock_configs
        }
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def emit_runtime_subblock_library(
    config: dict[str, Any],
    *,
    teacher_dir: Path | str,
    puzzle_dir: Path | str | None = None,
) -> Path:
    """Write the pre-build_library runtime candidate list used by vLLM stats.

    Convert is the sole producer. This is intentionally not the final
    replacement/subblock library from ``build_library``.
    """
    from ..anymodel.model_descriptor import ModelDescriptorFactory
    from ..anymodel.registry import resolve_descriptor_from_pretrained
    from ..subblock_stats.calc_runtime_stats import enumerate_runtime_block_configs

    teacher_dir = Path(teacher_dir)
    puzzle_dir = Path(puzzle_dir) if puzzle_dir is not None else experiment_dir(config)
    model_cfg = config.get("model") or {}
    resolution = resolve_descriptor_from_pretrained(
        str(teacher_dir),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        descriptor_override=model_cfg.get("descriptor_override"),
    )
    descriptor = ModelDescriptorFactory.get(resolution.name)
    block_configs = enumerate_runtime_block_configs(
        teacher_dir,
        descriptor,
        search_space=config.get("search_space") or {},
        include_noops=bool((config.get("build_library") or {}).get("include_noops", True)),
    )
    puzzle_dir.mkdir(parents=True, exist_ok=True)
    path = puzzle_dir / "subblock_library.json"
    _write_runtime_subblock_library(path, block_configs)
    return path


def configure_vllm_stats_widths(config: dict[str, Any], hydra_cfg: Any) -> tuple[int, ...]:
    """Merge embedding search widths into the vLLM measurement configuration."""
    OmegaConf.set_struct(hydra_cfg, False)
    if not hasattr(hydra_cfg, "calc_subblock_stats"):
        raise ValueError("vLLM statistics require calc_subblock_stats runtime configuration")
    configured_widths = tuple(
        int(width)
        for width in (_get(hydra_cfg.calc_subblock_stats, "model_hidden_sizes", ()) or ())
    )
    embedding_widths = tuple(
        int(width) for width in (config.get("embedding_pruning") or {}).get("widths", ())
    )
    requested_widths = tuple(dict.fromkeys((*configured_widths, *embedding_widths)))
    if requested_widths:
        hydra_cfg.calc_subblock_stats.model_hidden_sizes = list(requested_widths)
    return requested_widths


def prepare_vllm_stats_workspace(config: dict[str, Any], hydra_cfg: Any) -> dict[str, Any]:
    """Validate convert-emitted inputs and enable runtime measurement config."""
    puzzle_dir = _puzzle_dir(config, hydra_cfg)
    teacher_dir = _teacher_dir(config, hydra_cfg)
    OmegaConf.set_struct(hydra_cfg, False)
    if not hasattr(hydra_cfg, "calc_subblock_stats"):
        raise ValueError("vLLM statistics require calc_subblock_stats runtime configuration")
    runtime_cfg = _get(hydra_cfg.calc_subblock_stats, "runtime_stats", None)
    if runtime_cfg is None:
        hydra_cfg.calc_subblock_stats.runtime_stats = {}
        runtime_cfg = hydra_cfg.calc_subblock_stats.runtime_stats
    runtime_cfg.enabled = True
    hydra_cfg.teacher_dir = str(teacher_dir)
    configure_vllm_stats_widths(config, hydra_cfg)

    subblock_library_path = puzzle_dir / "subblock_library.json"
    if not (subblock_library_path.is_file() and subblock_library_path.stat().st_size > 0):
        raise FileNotFoundError(
            f"missing runtime subblock library {subblock_library_path}; "
            "convert must emit it when vllm_stats is enabled"
        )

    from ..anymodel.model_descriptor import ModelDescriptorFactory
    from ..subblock_stats.calc_runtime_stats import enumerate_runtime_block_configs

    descriptor = ModelDescriptorFactory.get(_get(hydra_cfg, "descriptor", None))
    block_configs = enumerate_runtime_block_configs(
        teacher_dir,
        descriptor,
        search_space=config.get("search_space") or {},
        include_noops=bool((config.get("build_library") or {}).get("include_noops", True)),
    )
    sparse_selection = _prepare_sparse_runtime_selection(
        config,
        runtime_cfg=runtime_cfg,
        teacher_dir=teacher_dir,
        puzzle_dir=puzzle_dir,
    )
    return {
        "puzzle_dir": puzzle_dir,
        "teacher_dir": teacher_dir,
        "subblock_library_path": subblock_library_path,
        "block_configs": block_configs,
        "runtime_cfg": runtime_cfg,
        "sparse_selection": sparse_selection,
    }


def _prepare_sparse_runtime_selection(
    config: dict[str, Any],
    *,
    runtime_cfg: Any,
    teacher_dir: Path,
    puzzle_dir: Path,
) -> dict[str, Any]:
    """Create the deterministic sparse subblock view consumed by vLLM stats."""
    sparse_cfg = _get(runtime_cfg, "sparse_sampling", None) or {}
    if not bool(_get(sparse_cfg, "enabled", False)):
        return {}

    from ..candidates import build_candidate_library, load_block_configs_from_checkpoint
    from ..sampling.sparse import SparseSamplingPolicy, sample_subblock_configs

    candidates = build_candidate_library(
        load_block_configs_from_checkpoint(teacher_dir),
        search_space=config.get("search_space") or {},
        parent_checkpoint_identity=str(teacher_dir.resolve()),
        include_self=True,
        include_noops=bool((config.get("build_library") or {}).get("include_noops", True)),
    )
    sampled = sample_subblock_configs(
        candidates,
        policy=SparseSamplingPolicy(
            max_pairwise_per_family=int(
                _get(sparse_cfg, "max_pairwise_per_family", 4)
            ),
            seed=int(_get(sparse_cfg, "seed", 42)),
        ),
    )
    sparse_manifest = sampled.to_dict()
    sparse_manifest.update(
        {
            "teacher_dir": str(teacher_dir.resolve()),
            "candidate_count": len(candidates),
        }
    )
    output = puzzle_dir / "artifacts" / "vllm_stats" / "sparse_subblock_samples.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(sparse_manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    runtime_cfg.selection_manifest = str(output)
    return {
        "path": str(output),
        "identity": sparse_manifest["identity"],
        "candidate_count": len(candidates),
        "selected_count": len(sparse_manifest["selected"]),
        "excluded_count": len(sparse_manifest["excluded"]),
    }


def finalize_vllm_stats_report(
    config: dict[str, Any], hydra_cfg: Any | None = None
) -> dict[str, Any]:
    """Validate the runtime aggregate and generate canonical vLLM report artifacts."""

    if hydra_cfg is None:
        hydra_cfg = load_runtime_hydra_config(config)
    puzzle_dir = _puzzle_dir(config, hydra_cfg)
    runtime_cfg = _get(hydra_cfg.calc_subblock_stats, "runtime_stats", {})
    stats_path = puzzle_dir / _get(
        hydra_cfg.calc_subblock_stats, "subblock_stats_filename", "subblock_stats.json"
    )
    if not stats_path.is_file() or stats_path.stat().st_size == 0:
        raise RuntimeError(f"vLLM statistics aggregate is missing or empty: {stats_path}")
    stats = json.loads(stats_path.read_text())
    measured_widths = {
        int(entry["args"]["n_embd"])
        for entry in stats
        if isinstance(entry, dict)
        and isinstance(entry.get("args"), dict)
        and entry["args"].get("runtime_stats") is True
        and entry["args"].get("n_embd") is not None
    }
    expected_widths = {
        int(width) for width in (config.get("embedding_pruning") or {}).get("widths", ())
    }
    missing_widths = expected_widths - measured_widths
    if missing_widths:
        raise RuntimeError(
            f"vLLM statistics aggregate {stats_path} is missing configured hidden widths "
            f"{sorted(missing_widths)}; measured widths are {sorted(measured_widths)}"
        )

    from ..diagnostics import generate_vllm_stats_report

    return generate_vllm_stats_report(
        puzzle_dir,
        stats_path=stats_path,
        output_dir=puzzle_dir / "artifacts" / "vllm_stats",
        granularity=str(_get(runtime_cfg, "granularity", "block")),
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def finalize_vllm_measurements(config: dict[str, Any]) -> dict[str, Any]:
    """Merge named measurement results and publish their completion index."""

    measurements = normalize_vllm_measurements(config)
    if len(measurements) == 1 and next(iter(measurements.values())).legacy:
        return finalize_vllm_stats_report(config)

    puzzle_dir = Path(config["puzzle_dir"]).resolve()
    rows: list[dict[str, Any]] = []
    index: dict[str, Any] = {"schema_version": 1, "measurements": {}}
    for measurement_id, measurement in measurements.items():
        path = puzzle_dir / measurement.relative_stats_path
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"vLLM measurement {measurement_id!r} is missing or invalid: {path}"
            ) from error
        if not isinstance(payload, list) or not payload:
            raise RuntimeError(f"vLLM measurement {measurement_id!r} is empty: {path}")
        for raw in payload:
            if not isinstance(raw, dict):
                raise RuntimeError(
                    f"vLLM measurement {measurement_id!r} contains a non-mapping row."
                )
            row = dict(raw)
            row["args"] = dict(row.get("args") or {})
            row["args"]["workload_id"] = measurement_id
            rows.append(row)
        index["measurements"][measurement_id] = {
            "identity": measurement.identity,
            "path": str(measurement.relative_stats_path),
            "rows": len(payload),
            "workload": {
                "batch_size": measurement.batch_size,
                "prefill_seq_len": measurement.prefill_seq_len,
                "generation_seq_len": measurement.generation_seq_len,
            },
        }

    stats_name = str(
        (config.get("vllm_stats") or {}).get("subblock_stats_filename", "subblock_stats.json")
    )
    _write_json_atomic(puzzle_dir / stats_name, rows)
    index_path = puzzle_dir / "artifacts" / "vllm_stats" / "measurements" / "index.json"
    _write_json_atomic(index_path, index)
    report = finalize_vllm_stats_report(config)
    return {"index": str(index_path), "measurements": index["measurements"], "report": report}


def vllm_stats_stage(config: dict[str, Any], manifest: StageManifest):
    """Collect runtime statistics from converted-teacher candidates before library assembly."""
    from ..subblock_stats.calc_subblock_stats import launch_calc_subblock_stats

    measurements = normalize_vllm_measurements(config)
    prepared = None
    for measurement in measurements.values():
        selected = apply_vllm_measurement(config, measurement)
        hydra_cfg = load_runtime_hydra_config(selected)
        prepared = prepare_vllm_stats_workspace(selected, hydra_cfg)
        stats_path = prepared["puzzle_dir"] / _get(
            hydra_cfg.calc_subblock_stats, "subblock_stats_filename", "subblock_stats.json"
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        launch_calc_subblock_stats(hydra_cfg)
    assert prepared is not None
    puzzle_dir = prepared["puzzle_dir"]
    runtime_cfg = prepared["runtime_cfg"]
    subblock_library_path = prepared["subblock_library_path"]
    block_configs = prepared["block_configs"]
    sparse_selection = prepared["sparse_selection"]
    stats_path = puzzle_dir / str(
        (config.get("vllm_stats") or {}).get("subblock_stats_filename", "subblock_stats.json")
    )
    report_summary: dict[str, Any] = {}
    if int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0"))) == 0:
        report_summary = finalize_vllm_measurements(config)
    elif not stats_path.is_file() or stats_path.stat().st_size == 0:
        raise RuntimeError(f"vLLM statistics aggregate is missing or empty: {stats_path}")
    return complete_stage(
        config,
        manifest,
        outputs={
            "subblock_stats_path": str(stats_path),
            "subblock_library_path": str(subblock_library_path),
            "runtime_candidate_count": len(block_configs),
            "runtime_granularity": _get(runtime_cfg, "granularity", "block"),
            "sparse_runtime_selection": sparse_selection,
            "report": report_summary,
        },
    )


def _extract_average_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "avg" in value:
            metrics[key] = value["avg"]
            metrics[f"one_minus_{key}"] = 1 - value["avg"]
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = value
    return metrics


def _index_scoring_results(config: dict[str, Any], hydra_cfg: Any) -> dict[str, Any]:
    output_dir = _scoring_output_dir(hydra_cfg)
    if not dist.is_master():
        return {"scoring_output_dir": str(output_dir), "indexed_scores": 0}
    if not output_dir.exists():
        raise RuntimeError(f"Replace-one-block scoring produced no output directory: {output_dir}")

    cache = EvaluationCache(_puzzle_dir(config, hydra_cfg) / "rpc_eval" / "replace_one_block")
    index_entries = []
    settings = {
        "scoring": _plain(_get(hydra_cfg, "scoring", {})),
        "teacher_dir": str(_teacher_dir(config, hydra_cfg)),
        "sorted_teacher_dir": str(_puzzle_dir(config, hydra_cfg) / "ckpts" / "sorted_teacher"),
    }
    for result_path in sorted(output_dir.glob("solution*.json")):
        raw = json.loads(result_path.read_text())
        request = EvaluationRequest(
            handler="replace_one_block",
            payload={
                "puzzle_solution": raw.get("puzzle_solution"),
                "i_solution": raw.get("i_solution"),
                "source_result_path": str(result_path),
            },
            settings=settings,
        )
        result = EvaluationResult(
            request_id=request.identity,
            metrics=_extract_average_metrics(raw),
            artifacts={"source_result_path": str(result_path)},
            metadata={
                "result_kind": "replace_one_block",
                "candidate_score_identity": stable_hash(
                    {"request": request.to_dict(), "metrics": _extract_average_metrics(raw)},
                    prefix="score",
                ),
            },
        )
        cache.put(result, request)
        index_entries.append(
            {
                "request_id": request.identity,
                "source_result_path": str(result_path),
                "metrics": result.metrics,
                "metadata": result.metadata,
            }
        )
    if not index_entries:
        raise RuntimeError(
            f"Replace-one-block scoring produced no solution_*.json files in {output_dir}"
        )
    index_path = cache.root / "score_index.json"
    index_path.write_text(
        json.dumps(
            canonicalize({"version": 1, "scores": index_entries}),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "scoring_output_dir": str(output_dir),
        "score_cache_dir": str(cache.root),
        "score_index_path": str(index_path),
        "indexed_scores": len(index_entries),
    }


def _index_mip_results(
    config: dict[str, Any],
    hydra_cfg: Any,
    solution_paths: list[str],
) -> dict[str, Any]:
    if not dist.is_master():
        return {"indexed_mip_solutions": 0}
    cache = EvaluationCache(_puzzle_dir(config, hydra_cfg) / "rpc_eval" / "mip_solutions")
    index_entries = []
    settings = {
        "mip": _plain(_get(hydra_cfg, "mip", {})),
        "realize_model": _plain(_get(hydra_cfg, "realize_model", {})),
        "skip_realize_model": bool(_get(hydra_cfg, "skip_realize_model", False)),
    }
    for solution_path in solution_paths:
        path = Path(solution_path)
        raw = json.loads(path.read_text()) if path.exists() else {"solution_path": str(path)}
        request = EvaluationRequest(
            handler="mip_solution",
            payload={"solution_path": str(path), "solution": raw},
            settings=settings,
        )
        result = EvaluationResult(
            request_id=request.identity,
            metrics={},
            artifacts={"solution_path": str(path)},
            metadata={"solution_identity": stable_hash(raw, prefix="solution")},
        )
        cache.put(result, request)
        index_entries.append(result.to_dict())
    index_path = cache.root / "mip_solution_index.json"
    index_path.write_text(
        json.dumps(
            canonicalize({"version": 1, "solutions": index_entries}),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "mip_cache_dir": str(cache.root),
        "mip_solution_index_path": str(index_path),
        "indexed_mip_solutions": len(index_entries),
    }


def activation_stage(config: dict[str, Any], manifest: StageManifest):
    hydra_cfg = load_runtime_hydra_config(config)
    num_nodes, node_index = _runtime_split(config)
    if not hasattr(hydra_cfg, "pruning"):
        raise ValueError(
            "Activation stage needs runtime pruning target config. Provide "
            "pruning.activation_passes or pruning.pruning_mixin for the descriptor."
        )
    if not hydra_cfg.pruning.get("activation_passes", None) and not hydra_cfg.pruning.get(
        "pruning_mixin", None
    ):
        raise ValueError(
            "Activation stage needs target selectors: set pruning.activation_passes "
            "for unified multi-hook scoring or pruning.pruning_mixin for a single pass."
        )
    with _distributed(hydra_cfg):
        from ..activation_scoring import launch_score_activations

        launch_score_activations(hydra_cfg, num_nodes=num_nodes, node_index=node_index)
        dist.barrier()
    return complete_stage(
        config,
        manifest,
        outputs={
            "activations_log_dir": str(_activations_log_dir(config, hydra_cfg)),
            "num_nodes": num_nodes,
            "node_index": node_index,
        },
    )


def sort_stage(config: dict[str, Any], manifest: StageManifest):
    hydra_cfg = load_runtime_hydra_config(config)
    sorted_teacher_dir = _puzzle_dir(config, hydra_cfg) / "ckpts" / "sorted_teacher"
    if _sorted_teacher_complete(config, sorted_teacher_dir):
        return complete_stage(
            config,
            manifest,
            outputs={
                "sorted_teacher_dir": str(sorted_teacher_dir),
                "parent_teacher_dir": str(_teacher_dir(config, hydra_cfg)),
                "activations_log_dir": str(_activations_log_dir(config, hydra_cfg)),
                "skipped_existing": True,
            },
        )
    with _distributed(hydra_cfg):
        from ..anymodel.model_descriptor import ModelDescriptorFactory
        from ..pruning.sorted_teacher import build_sorted_teacher

        descriptor = ModelDescriptorFactory.get(_get(hydra_cfg, "descriptor", None))
        sort_cfg = _get(hydra_cfg, "sort", {})
        deferred_axes = tuple(_get(sort_cfg, "deferred_axes", ()) or ())
        mamba_state_score_key = str(_get(sort_cfg, "mamba_state_score_key", "ssm_channel_contrib"))
        embedding_cfg = _get(hydra_cfg, "embedding_pruning", {})
        embedding_widths = tuple(_get(embedding_cfg, "widths", ()) or ())
        build_sorted_teacher(
            _teacher_dir(config, hydra_cfg),
            _activations_log_dir(config, hydra_cfg),
            sorted_teacher_dir,
            descriptor,
            deferred_axes=deferred_axes,
            mamba_state_score_key=mamba_state_score_key,
            embedding_widths=embedding_widths,
        )
        dist.barrier()
    return complete_stage(
        config,
        manifest,
        outputs={
            "sorted_teacher_dir": str(sorted_teacher_dir),
            "parent_teacher_dir": str(_teacher_dir(config, hydra_cfg)),
            "activations_log_dir": str(_activations_log_dir(config, hydra_cfg)),
        },
    )


def bypass_stage(config: dict[str, Any], manifest: StageManifest):
    hydra_cfg = load_runtime_hydra_config(config)
    bypass_cfg = config.get("bypass") or {}
    if bypass_cfg.get("enabled") is False:
        parent_artifact = _puzzle_dir(config, hydra_cfg) / "artifacts" / "scoring_parent.json"
        result = complete_stage(
            config,
            manifest,
            outputs={
                "skipped": True,
                "scoring_parent_artifact": str(parent_artifact),
            },
            status="skipped",
            message="Bypass is disabled.",
        )
        if os.environ.get("RANK") in (None, "", "0"):
            ensure_scoring_parent(config, refresh=True)
        return result
    num_nodes, node_index = _runtime_split(config)
    puzzle_dir = _puzzle_dir(config, hydra_cfg)
    with _distributed(hydra_cfg):
        from ..bypass_distillation import launch_bypass_distillation

        launch_bypass_distillation(hydra_cfg, num_nodes=num_nodes, node_index=node_index)
        dist.barrier()
    result = complete_stage(
        config,
        manifest,
        outputs={
            "puzzle_dir": str(puzzle_dir),
            "ckpts_dir": str(puzzle_dir / "ckpts"),
            "num_nodes": num_nodes,
            "node_index": node_index,
            "scoring_parent_artifact": str(puzzle_dir / "artifacts" / "scoring_parent.json"),
        },
    )
    if os.environ.get("RANK") in (None, "", "0"):
        # The parent fingerprints this completed manifest.  Never rewrite the
        # manifest afterward: embedding its own fingerprint would create a
        # self-referential identity that is stale by construction.
        ensure_scoring_parent(config, refresh=True)
    return result


def _finalize_bypass_sanity_summary(
    puzzle_dir: Path,
    modes: list[str],
    repetitions: int,
) -> Path | None:
    """Publish one completion artifact after every independent probe is complete."""

    mode_summaries: dict[str, dict[str, Any]] = {}
    findings: list[dict[str, Any]] = []
    passed = True
    for mode in modes:
        history_path = (
            puzzle_dir
            / "artifacts"
            / "bypass"
            / "overfit_probe"
            / mode
            / "local_kd_loss_history.json"
        )
        if not history_path.is_file():
            return None
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        records = [row for row in history.get("records", ()) if isinstance(row, dict)]
        steps = [int(row.get("step", -1)) for row in records]
        losses = [row.get("loss") for row in records]
        valid_losses = all(
            isinstance(loss, (int, float)) and math.isfinite(float(loss)) for loss in losses
        )
        if (
            len(records) != repetitions
            or steps != list(range(1, repetitions + 1))
            or not valid_losses
        ):
            return None
        history_summary = history.get("summary") or {}
        mode_passed = bool(history_summary.get("passed", True))
        mode_findings = list(history_summary.get("findings") or ())
        passed = passed and mode_passed
        findings.extend(mode_findings)
        mode_summaries[mode] = {
            "history_path": str(history_path),
            "record_count": len(records),
            "first_loss": float(losses[0]),
            "last_loss": float(losses[-1]),
            "finite": True,
            "max_step": steps[-1],
            "history_summary": history_summary,
            "passed": mode_passed,
            "findings": mode_findings,
        }

    summary_path = puzzle_dir / "artifacts" / "bypass_sanity" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "stage": "bypass_sanity",
                "complete": True,
                "passed": passed,
                "findings": findings,
                "verdict": "passed" if passed else "warning",
                "modes": modes,
                "repetitions": repetitions,
                "mode_summaries": mode_summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, summary_path)
    return summary_path


def _bypass_sanity_overfit_config(config: dict[str, Any]) -> dict[str, Any]:
    """Map the public sanity controls onto the legacy local-KD probe config."""

    bypass = dict(config.get("bypass") or {})
    overfit = dict(bypass.get("overfit") or {})
    sanity = dict(config.get("bypass_sanity") or {})
    if "steps" in sanity:
        overfit["repetitions"] = int(sanity["steps"])
    mode_keys = ("fixed_smallest", "diverse_nested")
    if any(key in sanity for key in mode_keys):
        modes = []
        if bool(sanity.get("fixed_smallest", False)):
            modes.append("smallest_fixed")
        if bool(sanity.get("diverse_nested", False)):
            modes.append("diverse_resampled")
        overfit["modes"] = modes
    if any(key in sanity for key in ("steps", *mode_keys)):
        overfit.setdefault("learning_rate", 3.0e-4)
        overfit.setdefault("decay_lr", False)
        overfit.setdefault("weight_decay", 0.0)
        overfit.setdefault("minimum_relative_decrease", 0.05)
    return overfit


def bypass_overfit_stage(config: dict[str, Any], manifest: StageManifest):
    """Run only the isolated same-batch nested-bypass acceptance probe."""

    hydra_cfg = load_runtime_hydra_config(config)
    puzzle_dir = _puzzle_dir(config, hydra_cfg)
    num_nodes, node_index = _runtime_split(config)
    OmegaConf.set_struct(hydra_cfg, False)
    hydra_cfg.bypass.overfit = OmegaConf.create(_bypass_sanity_overfit_config(config))
    hydra_cfg.bypass.overfit.enabled = True
    hydra_cfg.bypass.overfit.only = True
    with _distributed(hydra_cfg):
        from ..bypass_distillation import launch_bypass_distillation

        launch_bypass_distillation(hydra_cfg, num_nodes=num_nodes, node_index=node_index)
        dist.barrier()
    configured_modes = hydra_cfg.bypass.overfit.get("modes", None)
    modes = (
        [str(mode) for mode in configured_modes]
        if configured_modes is not None
        else ["smallest_fixed"]
    )
    history_paths = {
        mode: str(
            puzzle_dir
            / "artifacts"
            / "bypass"
            / "overfit_probe"
            / mode
            / "local_kd_loss_history.json"
        )
        for mode in modes
    }
    repetitions = int(hydra_cfg.bypass.overfit.repetitions)
    summary_path = None
    bypass_summary = None
    if os.environ.get("RANK") in (None, "", "0"):
        summary_path = _finalize_bypass_sanity_summary(
            puzzle_dir,
            modes,
            repetitions,
        )
        if summary_path is not None and summary_path.is_file():
            bypass_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    from ..diagnostics.sanity_verdict import SanityVerdict, complete_sanity_stage

    return complete_sanity_stage(
        config,
        manifest,
        outputs={
            "puzzle_dir": str(puzzle_dir),
            "history_paths": history_paths,
            "summary_path": str(summary_path) if summary_path is not None else None,
            "modes": modes,
            "repetitions": repetitions,
            "single_batch": True,
            "nested": bool(hydra_cfg.bypass.elastic),
            "num_nodes": num_nodes,
            "node_index": node_index,
            "passed": bool((bypass_summary or {}).get("passed", True)),
            "findings": list((bypass_summary or {}).get("findings") or ()),
        },
        verdict=SanityVerdict(
            passed=bool((bypass_summary or {}).get("passed", True)),
            findings=list((bypass_summary or {}).get("findings") or ()),
        ),
    )


def build_library_stage(config: dict[str, Any], manifest: StageManifest):
    hydra_cfg = load_runtime_hydra_config(config)
    puzzle_dir = _puzzle_dir(config, hydra_cfg)
    candidate_library_path = puzzle_dir / "candidate_library.json"
    stats_path = puzzle_dir / _get(
        _get(hydra_cfg, "calc_subblock_stats", {}) or {},
        "subblock_stats_filename",
        "subblock_stats.json",
    )
    with _distributed(hydra_cfg):
        # The scoring-parent artifact is shared campaign state.  Publish it
        # once, then let every rank validate the same atomic artifact.
        if dist.is_master():
            ensure_scoring_parent(config, refresh=True)
        dist.barrier()
        scoring_parent = ensure_scoring_parent(config)
        OmegaConf.set_struct(hydra_cfg, False)
        hydra_cfg.build_replacement_library.source_checkpoint_dir = str(scoring_parent.path)
        if dist.is_master():
            if _vllm_stats_is_explicit(config):
                if not stats_path.is_file():
                    raise RuntimeError(
                        "build-library requires the standalone vLLM aggregate at "
                        f"{stats_path}; run or import vllm_stats first"
                    )
            from ..replacement_library.build_replacement_library import (
                launch_build_replacement_library,
            )

            launch_build_replacement_library(hydra_cfg)
            _calculate_static_workload_stats(config, hydra_cfg)
            try:
                from ..candidates import build_candidate_library_from_checkpoint

                library_cfg = config.get("build_replacement_library") or {}
                configured_parent = library_cfg.get("source_checkpoint_dir")
                parent_dir = (
                    Path(configured_parent).resolve() if configured_parent else scoring_parent.path
                )
                if parent_dir.resolve() != scoring_parent.path.resolve():
                    raise RuntimeError(
                        "build-library source does not match the resolved scoring parent: "
                        f"{parent_dir} != {scoring_parent.path}"
                    )
                build_candidate_library_from_checkpoint(
                    parent_dir,
                    search_space=config.get("search_space") or {},
                    output_path=candidate_library_path,
                    puzzle_dir=puzzle_dir,
                    include_noops=bool(library_cfg.get("include_noops", True)),
                    include_bypass=bool(library_cfg.get("include_bypass", True)),
                    stats_paths=(stats_path,) if stats_path.is_file() else (),
                    metadata={"library_settings": config.get("library") or {}},
                    hidden_width=library_cfg.get("hidden_width"),
                )
            except ImportError:
                pass
        dist.barrier()
    return complete_stage(
        config,
        manifest,
        outputs={
            "replacement_library_path": str(puzzle_dir / "replacement_library.json"),
            "candidate_library_path": str(candidate_library_path),
            "subblock_stats_path": str(
                puzzle_dir
                / _get(
                    _get(hydra_cfg, "calc_subblock_stats", {}) or {},
                    "subblock_stats_filename",
                    "subblock_stats.json",
                )
            ),
            "runtime_stats_execution": (
                "sharded" if _runtime_stats_are_sharded(hydra_cfg) else "inline"
            ),
            "scoring_parent": scoring_parent.to_dict(),
        },
    )


def scoring_stage(config: dict[str, Any], manifest: StageManifest):
    hydra_cfg = load_runtime_hydra_config(config)
    scoring_parent = ensure_scoring_parent(config)
    OmegaConf.set_struct(hydra_cfg, False)
    hydra_cfg.scoring.source_checkpoint_dir = str(scoring_parent.path)
    reference = str((config.get("replacement_scoring") or {}).get("reference", "scoring_parent"))
    hydra_cfg.scoring.target_teacher_dir = str(
        _teacher_dir(config, hydra_cfg) if reference == "original_teacher" else scoring_parent.path
    )
    num_nodes, node_index = _runtime_split(config)
    with _distributed(hydra_cfg):
        from ..scoring import launch_scoring

        try:
            launch_scoring(hydra_cfg, num_nodes=num_nodes, node_index=node_index)
        except BaseException:
            # ``dist.cleanup()`` enters a global barrier.  In a PP failure, the
            # failing stage otherwise waits there while another stage waits in
            # P2P receive, hiding the original traceback until NCCL times out.
            import traceback

            traceback.print_exc()
            raise
        dist.barrier()
        indexed_outputs = _index_scoring_results(config, hydra_cfg)
        dist.barrier()
        report_summary: dict[str, Any] = {}
        if dist.is_master():
            from ..diagnostics import generate_replace_block_report

            report_config = config.get("replacement_scoring") or {}
            granularity = str(_get(hydra_cfg.scoring, "granularity", "block"))
            report_summary = generate_replace_block_report(
                _puzzle_dir(config, hydra_cfg),
                scores_dir=_scoring_output_dir(hydra_cfg),
                output_dir=_puzzle_dir(config, hydra_cfg) / "artifacts" / "replacement_scoring",
                granularity=granularity,
                default_metric=str(
                    report_config.get("default_metric", "normalized_mse_loss_hidden_states")
                ),
                default_layer_count=int(report_config.get("default_layer_count", 5)),
                anchor_count=int(report_config.get("anchor_count", 3)),
                trend_relative_tolerance=float(
                    report_config.get("trend_relative_tolerance", 0.02)
                ),
            )
        dist.barrier()
    return complete_stage(
        config,
        manifest,
        outputs={
            **indexed_outputs,
            "scoring_parent": scoring_parent.to_dict(),
            "num_nodes": num_nodes,
            "node_index": node_index,
            "report": report_summary,
        },
    )


def mip_stage(config: dict[str, Any], manifest: StageManifest):
    hydra_cfg = load_runtime_hydra_config(config)
    coverage_report = None
    if str(hydra_cfg.mip.get("score_granularity", "block")) == "subblock":
        from ..artifact_coverage import verify_real_campaign_artifacts

        coverage_report = verify_real_campaign_artifacts(
            _puzzle_dir(config, hydra_cfg),
            expected_depth_scenarios=int(hydra_cfg.mip.get("depth_scenario_count", 1)),
            bypass_enabled=bool(hydra_cfg.bypass.get("enabled", False)),
            expected_checkpoint_dir=str(hydra_cfg.scoring.source_checkpoint_dir),
            expected_data_identity={
                "eval_samples": int(hydra_cfg.scoring.eval_samples),
                "block_size": int(hydra_cfg.scoring.block_size),
            },
        )
        coverage_report.require_complete()
    solution_paths: list[str] = []
    with _distributed(hydra_cfg):
        coverage_path = (
            _puzzle_dir(config, hydra_cfg) / "artifacts" / "mip" / "artifact_coverage.json"
        )
        if coverage_report is not None and dist.is_master():
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            coverage_path.write_text(
                json.dumps(dataclasses.asdict(coverage_report), indent=2, default=str) + "\n"
            )
        has_depth_trajectory = bool(
            hasattr(hydra_cfg, "mip") and hydra_cfg.mip.get("depth_trajectory_path", None)
        )
        has_mip_sweep = (
            hasattr(hydra_cfg, "mip")
            and hasattr(hydra_cfg.mip, "sweep")
            and hydra_cfg.mip.sweep.get("enabled", False)
        )
        if has_depth_trajectory:
            from ..depth.mip_scenarios import run_depth_mip_scenarios

            solution_paths = run_depth_mip_scenarios(hydra_cfg) or []
        elif has_mip_sweep:
            from ..mip import run_mip_sweep

            run_mip_sweep(hydra_cfg)
        else:
            grid_cfg = dict((config.get("mip") or {}).get("grid_budgeting") or {})
            if grid_cfg.get("enabled"):
                from ..mip.grid_budgeting import run_grid_budgeted_mip

                solution_paths = run_grid_budgeted_mip(hydra_cfg, grid_cfg) or []
            else:
                from ..mip import launch_mip_and_realize_model

                solution_paths = launch_mip_and_realize_model(hydra_cfg) or []
        dist.barrier()
        indexed_outputs = _index_mip_results(config, hydra_cfg, solution_paths)
        dist.barrier()
    return complete_stage(
        config,
        manifest,
        outputs={
            "solution_paths": [str(path) for path in solution_paths],
            "artifact_coverage_path": (
                str(_puzzle_dir(config, hydra_cfg) / "artifacts" / "mip" / "artifact_coverage.json")
                if coverage_report is not None
                else None
            ),
            **indexed_outputs,
        },
    )
