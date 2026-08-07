# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# mypy: ignore-errors

"""Calc subblock stats to compute memory and runtime statistics for subblocks."""

import copy
import dataclasses
import hashlib
import json
import math
import multiprocessing
import os
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import partial
from itertools import product
from pathlib import Path
from typing import Iterable, Type, TypeVar

import pandas as pd
import torch
from immutabledict import immutabledict
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
from transformers import PretrainedConfig

from modelopt.torch.utils import json_dump

from ..anymodel.model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ..block_config import SUBBLOCK_CLS_DICT, BlockConfig, FFNConfig, SubblockConfig
from ..distributed_eval.storage import file_lock
from ..pruning.embedding_pruning import EmbeddingPruningSpec
from ..replacement_library.replacement_utils import parse_layer_replacement
from ..tools.checkpoint_utils import load_model_config
from ..tools.logger import mprint
from ..utils.parsing import format_global_config
from .calc_subblock_params_and_memory import (
    calc_subblock_active_params,
    calculate_additive_metrics,
    calculate_non_block_memory,
    calculate_non_block_params,
    calculate_subblock_memory,
    calculate_subblock_params,
)
from .runtime_vllm import RuntimeMeasurement

__all__ = [
    "calculate_subblock_stats",
    "launch_calc_subblock_stats",
]


def _freeze_stats_args(value):
    """Return a recursively hashable key for JSON-compatible stats arguments."""
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _freeze_stats_args(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_stats_args(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze_stats_args(item) for item in value))
    return value

# Type variable for dataclasses
T_DataClass = TypeVar("T_DataClass")

_SUBBLOCK_KINDS = ("attention", "mla", "mamba", "ffn", "moe")
_ATTENTION_LIKE_KINDS = frozenset(("attention", "mla", "mamba"))
_FFN_LIKE_KINDS = frozenset(("ffn", "moe"))
_NON_BLOCK_PARAM_CACHE: dict[tuple[str, str, int, int], int] = {}
_PARAMETER_INVENTORY_SCHEMA = 2


def _atomic_json_dump(value: object, path: Path) -> None:
    """Publish JSON without exposing partial contents to resumable readers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    json_dump(value, temporary)
    temporary.replace(path)


def _unique_hidden_sizes(model_hidden_sizes: Iterable[int], teacher_hidden_size: int) -> tuple[int, ...]:
    """Return configured widths plus the teacher exactly once, preserving order."""

    return tuple(dict.fromkeys(int(width) for width in (*model_hidden_sizes, teacher_hidden_size)))


def _virtual_width_config(
    teacher_config: PretrainedConfig,
    descriptor: Type[ModelDescriptor],
    width: int,
    teacher_hidden_size: int,
    legal_widths: Iterable[int],
) -> tuple[PretrainedConfig, EmbeddingPruningSpec | None]:
    """Build the exact child config and slicing contract without writing a checkpoint."""

    if int(width) == int(teacher_hidden_size):
        return teacher_config, None
    spec = descriptor.embedding_pruning_spec(
        teacher_config,
        widths=tuple(dict.fromkeys(int(value) for value in legal_widths)),
        alignment=1,
    )
    return spec.update_config_object(teacher_config, int(width)), spec


def _parameter_inventory_key(subblock: SubblockConfig | dict, parent_layer_index: int) -> str:
    return f"{int(parent_layer_index)}:{_subblock_identity(subblock)}"


def _checkpoint_parameter_identity(
    checkpoint_dir: Path,
    descriptor: Type[ModelDescriptor],
    width: int,
    subblock_configs: list[immutabledict[str, SubblockConfig]],
    embedding_spec: EmbeddingPruningSpec | None,
) -> str:
    """Build a cheap identity over source shapes, virtual slicing, and candidates."""

    files = []
    for path in sorted(checkpoint_dir.iterdir()):
        if not (
            path.name == "config.json"
            or path.suffix == ".py"
            or path.suffix == ".safetensors"
            or path.name.endswith(".safetensors.index.json")
        ):
            continue
        stat = path.stat()
        item = {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.name == "config.json" or path.suffix == ".py":
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(item)
    candidates = [
        {
            "subblock_config": row["subblock_config"].to_dict(),
            "parent_layer_index": int(row["parent_layer_indices"][0]),
        }
        for row in sorted(
            subblock_configs,
            key=lambda row: (
                int(row["parent_layer_indices"][0]),
                str(row["subblock_config"]),
            ),
        )
    ]
    payload = {
        "schema": _PARAMETER_INVENTORY_SCHEMA,
        "descriptor": f"{descriptor.__module__}.{descriptor.__qualname__}",
        "width": int(width),
        "source_checkpoint": str(checkpoint_dir.resolve()),
        "embedding_pruning_spec": (
            dataclasses.asdict(embedding_spec) if embedding_spec is not None else None
        ),
        "files": files,
        "candidates": candidates,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_parameter_inventory_cache(
    path: Path,
    *,
    identity: str,
    total: int,
) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("identity") != identity or int(payload.get("total", -1)) != int(total):
        return None
    if not isinstance(payload.get("rows"), list):
        return None
    return payload


def _parameter_inventory_progress(
    *,
    width: int,
    completed: int,
    total: int,
    elapsed_seconds: float,
    status: str,
) -> dict[str, float | int | str | None]:
    rate = completed / elapsed_seconds if completed and elapsed_seconds > 0 else 0.0
    remaining = max(0, total - completed)
    return {
        "width": int(width),
        "status": status,
        "completed": int(completed),
        "total": int(total),
        "fraction_complete": completed / total if total else 1.0,
        "elapsed_seconds": float(elapsed_seconds),
        "rate_per_second": rate,
        "eta_seconds": remaining / rate if rate else None,
    }


def _runtime_measurement_fields(
    measurement: RuntimeMeasurement | None,
    *,
    generation_seq_len: int,
) -> dict:
    """Flatten a typed timing measurement into the stable stats schema."""
    if measurement is None:
        return {
            "runtime_ms": None,
            "prefill_runtime_ms": None,
            "decode_runtime_ms": None,
            "decode_runtime_ms_per_token": None,
            "latency_difference_negative": None,
            "additive_metric_provenance": {},
        }
    return {
        "runtime_ms": measurement.total_ms,
        "prefill_runtime_ms": measurement.prefill_ms,
        "decode_runtime_ms": measurement.decode_ms,
        "decode_runtime_ms_per_token": measurement.decode_ms_per_token(
            generation_seq_len
        ),
        "latency_difference_negative": measurement.decode_ms < 0,
        "additive_metric_provenance": {
            "runtime_ms": "vllm_measured",
            "prefill_runtime_ms": "vllm_measured_prompt_plus_one_output",
            "decode_runtime_ms": "combined_minus_prefill",
            "decode_runtime_ms_per_token": (
                "combined_minus_prefill_per_remaining_output"
            ),
        },
    }


_REUSABLE_RUNTIME_FIELDS = (
    "runtime_ms",
    "prefill_runtime_ms",
    "decode_runtime_ms",
    "decode_runtime_ms_per_token",
    "latency_difference_negative",
)
_REUSABLE_RUNTIME_ARG_FIELDS = (
    "workload_id",
    "runtime_granularity",
    "runtime_backend",
    "num_iters",
    "num_warmup_iters",
    "max_num_seqs",
    "repeat_block_n_times",
    "vllm_args",
    "runtime_selection_identity",
)


def _runtime_reuse_key_from_args(
    args: Mapping,
    *,
    fallback_workload_id: str | None = None,
) -> tuple | None:
    """
    Builds the identity used to match reusable runtime measurements.
    
    Parameters:
    	args (Mapping): Persisted calculation arguments containing runtime and workload settings.
    	fallback_workload_id (str | None): Workload identifier to use when `args` does not provide one.
    
    Returns:
    	tuple | None: Runtime-reuse identity, or `None` when runtime statistics are unavailable or the dtype is not bfloat16.
    """

    if not args.get("runtime_stats") or args.get("weights_dtype") != str(torch.bfloat16):
        return None
    workload_id = args.get("workload_id")
    if workload_id is None:
        workload_id = fallback_workload_id
    return (
        int(args["n_embd"]),
        int(args["batch_size"]),
        int(args.get("prefill_seq_len")),
        int(args.get("generation_seq_len")),
        args.get("max_num_seqs"),
        args.get("runtime_granularity", "subblock"),
        args.get("runtime_backend"),
        args.get("num_iters"),
        args.get("num_warmup_iters"),
        args.get("repeat_block_n_times"),
        _freeze_stats_args(args.get("vllm_args")),
        workload_id,
    )


def _runtime_reuse_key(
    *,
    width: int,
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    runtime_stats_config: Mapping,
) -> tuple:
    """
    Builds an exact identity for matching reusable runtime measurements.
    
    Parameters:
        runtime_stats_config (Mapping): Runtime settings that determine measurement compatibility.
    
    Returns:
        tuple: Identity containing model dimensions, runtime settings, vLLM arguments, and workload identity.
    """

    return (
        int(width),
        int(batch_size),
        int(prefill_seq_len),
        int(generation_seq_len),
        runtime_stats_config.get("max_num_seqs"),
        runtime_stats_config.get("granularity", "subblock"),
        runtime_stats_config.get("backend"),
        runtime_stats_config.get("num_iters", 30),
        runtime_stats_config.get("num_warmup_iters", 10),
        max(2, int(runtime_stats_config.get("repeat_block_n_times", 10))),
        _freeze_stats_args([str(arg) for arg in runtime_stats_config.get("vllm_args", [])]),
        runtime_stats_config.get("workload_id"),
    )


def _reuse_runtime_stats(
    target: dict,
    source: dict,
    *,
    source_path: str,
    fallback_workload_id: str | None = None,
) -> dict:
    """
    Reuse measured runtime statistics from a compatible source entry in refreshed statistics.
    
    Parameters:
        target (dict): Statistics entry to update with reusable runtime data.
        source (dict): Statistics entry containing the measured runtime data.
        source_path (str): Path identifying the source statistics entry.
        fallback_workload_id (str | None): Workload identifier to use when the source does not provide one.
    
    Returns:
        dict: The updated target statistics entry.
    
    Raises:
        KeyError: If a target subblock has no matching runtime statistics in the source entry.
    """

    # Synthetic vLLM timings are layer-independent: collection benchmarks the
    # set of unique subblock configs, while a post-scoring replacement library
    # can repeat one config at many parent layers. Reuse by config identity.
    source_by_key = {
        _subblock_identity(row["subblock_config"]): row
        for row in source.get("subblocks", [])
    }
    for target_row in target.get("subblocks", []):
        key = _subblock_identity(target_row["subblock_config"])
        source_row = source_by_key.get(key)
        if source_row is None:
            raise KeyError(f"Reusable runtime stats are missing subblock {key}")
        for field in _REUSABLE_RUNTIME_FIELDS:
            target_row[field] = source_row.get(field)
        source_provenance = source_row.get("additive_metric_provenance") or {}
        target_provenance = target_row.setdefault("additive_metric_provenance", {})
        target_provenance.update(
            {
                field: provenance
                for field, provenance in source_provenance.items()
                if field in _REUSABLE_RUNTIME_FIELDS
            }
        )

    source_args = source.get("args", {})
    target_args = target.setdefault("args", {})
    target_args["runtime_stats"] = True
    target_args["runtime_reuse_source"] = str(source_path)
    for field in _REUSABLE_RUNTIME_ARG_FIELDS:
        value = source_args.get(field)
        if field == "workload_id" and value is None:
            value = target_args.get("workload_id") or fallback_workload_id
        target_args[field] = value

    source_non_block = source.get("non_block", {})
    target_non_block = target.setdefault("non_block", {})
    for field in _REUSABLE_RUNTIME_FIELDS:
        target_non_block[field] = source_non_block.get(field)
    for field in ("runtime_decomposition", "block_runtimes", "block_runtime_records"):
        if field in source:
            target[field] = copy.deepcopy(source[field])
    return target


def _subblock_identity(subblock: SubblockConfig | dict) -> str:
    if isinstance(subblock, SubblockConfig):
        payload = subblock.to_dict()
    elif subblock.get("kind") in SUBBLOCK_CLS_DICT:
        # Robust JSON serialization of dataclasses retains explicit ``None``
        # fields, while ``to_dict`` omits them. Normalize both representations
        # so an immutable runtime record matches the same live candidate.
        payload = SUBBLOCK_CLS_DICT[subblock["kind"]](**subblock).to_dict()
    else:
        payload = subblock
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _select_runtime_subblock_configs(
    available: list[immutabledict[str, SubblockConfig]],
    manifest: dict,
) -> list[immutabledict[str, SubblockConfig]]:
    """Resolve a layer-independent sparse runtime view over a canonical library."""

    selected_payloads = [row.get("subblock_config") for row in manifest.get("selected", [])]
    if not manifest.get("identity") or not selected_payloads:
        raise ValueError("sparse runtime manifest must have an identity and selected subblocks")
    if any(payload is None for payload in selected_payloads):
        raise ValueError("sparse runtime manifest contains a block-level selection")

    available_by_identity = {
        _subblock_identity(row["subblock_config"]): row["subblock_config"]
        for row in available
    }
    selected_by_identity: dict[str, SubblockConfig] = {}
    for payload in selected_payloads:
        identity = _subblock_identity(payload)
        if identity not in available_by_identity:
            raise ValueError(
                "sparse runtime subblock is not present in the canonical library: "
                f"{payload}"
            )
        selected_by_identity[identity] = available_by_identity[identity]
    return [
        immutabledict(
            {"subblock_config": subblock, "parent_layer_indices": (-1,)}
        )
        for subblock in sorted(selected_by_identity.values())
    ]


def _validate_sparse_runtime_settings(runtime_stats_config: Mapping) -> None:
    if runtime_stats_config.get("granularity", "subblock") != "subblock":
        raise ValueError("sparse runtime collection requires subblock granularity")
    if (
        int(runtime_stats_config.get("num_warmup_iters", -1)) != 2
        or int(runtime_stats_config.get("num_iters", -1)) != 3
    ):
        raise ValueError(
            "sparse runtime collection requires num_warmup_iters=2 and num_iters=3"
        )


def _checkpoint_non_block_params(
    teacher_dir: Path,
    descriptor: Type[ModelDescriptor],
    num_layers: int,
    width: int,
    embedding_spec: EmbeddingPruningSpec | None,
) -> int | None:
    """Count every virtually sliced tensor not owned by a decoder subblock.

    This includes ViT, projector, embeddings/head, MTP, and family-specific
    fixed tensors. Shape-only slicing uses the same descriptor contract as
    checkpoint materialization without loading weights or writing a child.
    """
    cache_key = (
        str(Path(teacher_dir).resolve()),
        descriptor.__name__,
        int(num_layers),
        int(width),
    )
    if cache_key in _NON_BLOCK_PARAM_CACHE:
        return _NON_BLOCK_PARAM_CACHE[cache_key]

    from safetensors import safe_open

    from ..pruning.sorted_teacher import iter_safetensor_weight_files

    predicates = descriptor.layer_name_predicates(int(num_layers))
    block_patterns = [
        pattern for name, pattern in predicates.items() if str(name).startswith("block_")
    ]
    total = 0
    try:
        weight_files = iter_safetensor_weight_files(teacher_dir)
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    for relative in weight_files:
        try:
            with safe_open(str(Path(teacher_dir) / relative), framework="pt") as handle:
                shapes = {
                    key: tuple(int(dim) for dim in handle.get_slice(key).get_shape())
                    for key in handle.keys()
                }
        except (FileNotFoundError, OSError, RuntimeError):
            return None
        for key, source_shape in shapes.items():
            target_shape = (
                embedding_spec.sliced_shape(key, source_shape, int(width))
                if embedding_spec is not None
                else source_shape
            )
            if any(pattern.fullmatch(key) for pattern in block_patterns):
                continue
            total += math.prod(target_shape)
    _NON_BLOCK_PARAM_CACHE[cache_key] = int(total)
    return int(total)


def _write_parameter_inventory_progress_manifest(
    cache_root: Path,
    widths: Iterable[int],
) -> None:
    progress_rows = []
    for width in widths:
        path = cache_root / "progress" / f"width-{int(width):04d}.json"
        if path.is_file():
            try:
                progress_rows.append(json.loads(path.read_text()))
                continue
            except (OSError, json.JSONDecodeError):
                pass
        progress_rows.append(
            _parameter_inventory_progress(
                width=int(width),
                completed=0,
                total=0,
                elapsed_seconds=0.0,
                status="pending",
            )
        )
    _atomic_json_dump(
        {
            "schema": _PARAMETER_INVENTORY_SCHEMA,
            "updated_at_unix": time.time(),
            "widths": progress_rows,
            "completed": sum(int(row["completed"]) for row in progress_rows),
            "total": sum(int(row["total"]) for row in progress_rows),
        },
        cache_root / "progress.json",
    )


def _calculate_parameter_inventory_for_width(
    *,
    teacher_dir: Path,
    descriptor: Type[ModelDescriptor],
    width: int,
    teacher_hidden_size: int,
    legal_widths: tuple[int, ...],
    subblock_configs: list[immutabledict[str, SubblockConfig]],
    cache_root: Path,
    progress_every: int,
) -> dict:
    """Build or reuse one width inventory under single-writer ownership."""

    with file_lock(cache_root / f".width-{int(width):04d}.lock"):
        return _calculate_parameter_inventory_for_width_unlocked(
            teacher_dir=teacher_dir,
            descriptor=descriptor,
            width=width,
            teacher_hidden_size=teacher_hidden_size,
            legal_widths=legal_widths,
            subblock_configs=subblock_configs,
            cache_root=cache_root,
            progress_every=progress_every,
        )


def _calculate_parameter_inventory_for_width_unlocked(
    *,
    teacher_dir: Path,
    descriptor: Type[ModelDescriptor],
    width: int,
    teacher_hidden_size: int,
    legal_widths: tuple[int, ...],
    subblock_configs: list[immutabledict[str, SubblockConfig]],
    cache_root: Path,
    progress_every: int,
) -> dict:
    """Count one virtual width once, publishing resumable progress."""

    trust_remote_code = descriptor.requires_trust_remote_code()
    teacher_config = load_model_config(teacher_dir, trust_remote_code=trust_remote_code)
    model_config, embedding_spec = _virtual_width_config(
        teacher_config,
        descriptor,
        width,
        teacher_hidden_size,
        legal_widths,
    )
    lm_config = descriptor.get_language_model_config(model_config)
    actual_width = int(lm_config.hidden_size)
    if actual_width != int(width):
        raise ValueError(
            f"Virtual width config has hidden_size={actual_width}, expected {width}."
        )

    unique_rows: dict[str, immutabledict[str, SubblockConfig]] = {}
    for row in sorted(
        subblock_configs,
        key=lambda item: (
            int(item["parent_layer_indices"][0]),
            str(item["subblock_config"]),
        ),
    ):
        key = _parameter_inventory_key(
            row["subblock_config"], int(row["parent_layer_indices"][0])
        )
        unique_rows.setdefault(key, row)

    total = len(unique_rows)
    identity = _checkpoint_parameter_identity(
        teacher_dir,
        descriptor,
        width,
        list(unique_rows.values()),
        embedding_spec,
    )
    cache_path = cache_root / f"width-{int(width):04d}.json"
    progress_path = cache_root / "progress" / f"width-{int(width):04d}.json"
    cached = _load_parameter_inventory_cache(cache_path, identity=identity, total=total)
    if (
        cached is not None
        and cached.get("status") == "complete"
        and len(cached["rows"]) == total
        and cached.get("non_block_params") is not None
        and cached.get("model") is not None
    ):
        progress = _parameter_inventory_progress(
            width=width,
            completed=total,
            total=total,
            elapsed_seconds=float(cached.get("elapsed_seconds", 0.0)),
            status="complete",
        )
        _atomic_json_dump(progress, progress_path)
        print(f"[parameter-stats width={width}] reused complete cache ({total}/{total})", flush=True)
        return cached

    rows = list(cached.get("rows", [])) if cached is not None else []
    completed_keys = {
        str(row["inventory_key"])
        for row in rows
        if isinstance(row, dict) and row.get("inventory_key") is not None
    }
    prior_elapsed = float(cached.get("elapsed_seconds", 0.0)) if cached is not None else 0.0
    started = time.monotonic()

    def publish(status: str) -> None:
        elapsed = prior_elapsed + time.monotonic() - started
        progress = _parameter_inventory_progress(
            width=width,
            completed=len(rows),
            total=total,
            elapsed_seconds=elapsed,
            status=status,
        )
        payload = {
            "schema": _PARAMETER_INVENTORY_SCHEMA,
            "identity": identity,
            "status": status,
            "width": int(width),
            "teacher_dir": str(teacher_dir.resolve()),
            "descriptor": f"{descriptor.__module__}.{descriptor.__qualname__}",
            "total": total,
            "elapsed_seconds": elapsed,
            "rows": rows,
        }
        _atomic_json_dump(payload, cache_path)
        _atomic_json_dump(progress, progress_path)
        eta = progress["eta_seconds"]
        eta_text = "unknown" if eta is None else f"{float(eta):.1f}s"
        print(
            f"[parameter-stats width={width}] {len(rows)}/{total} "
            f"({100 * float(progress['fraction_complete']):.1f}%) "
            f"rate={float(progress['rate_per_second']):.3f}/s ETA={eta_text}",
            flush=True,
        )

    publish("running")
    for inventory_key, indexed in unique_rows.items():
        if inventory_key in completed_keys:
            continue
        subblock_config = indexed["subblock_config"]
        parent_layer_index = int(indexed["parent_layer_indices"][0])
        layer_model_config = copy.deepcopy(model_config)
        descriptor.truncate_pattern_for_subblock(
            descriptor.get_language_model_config(layer_model_config), parent_layer_index
        )
        num_params = calculate_subblock_params(
            layer_model_config, subblock_config, descriptor
        )
        active_params = calc_subblock_active_params(
            subblock_config,
            layer_model_config,
            descriptor,
            actual_width,
            num_params=num_params,
        )
        rows.append(
            {
                "inventory_key": inventory_key,
                "subblock_config": subblock_config.to_dict(),
                "parent_layer_index": parent_layer_index,
                "num_params": int(num_params),
                "active_params": int(active_params),
            }
        )
        completed_keys.add(inventory_key)
        if len(rows) % max(1, int(progress_every)) == 0:
            publish("running")

    non_block_params = _checkpoint_non_block_params(
        teacher_dir,
        descriptor,
        int(lm_config.num_hidden_layers),
        actual_width,
        embedding_spec,
    )
    if non_block_params is None:
        non_block_params = calculate_non_block_params(actual_width, int(lm_config.vocab_size))
        non_block_source = "lm_formula_fallback"
    else:
        non_block_source = "virtual_checkpoint_tensor_inventory"
    publish("complete")
    payload = json.loads(cache_path.read_text())
    payload["non_block_params"] = int(non_block_params)
    payload["non_block_parameter_count_source"] = non_block_source
    payload["model"] = {
        "hidden_size": actual_width,
        "num_attention_heads": int(lm_config.num_attention_heads),
        "num_hidden_layers": int(lm_config.num_hidden_layers),
        "vocab_size": int(lm_config.vocab_size),
    }
    _atomic_json_dump(payload, cache_path)
    return payload


def _collect_parameter_inventories(
    *,
    calc_subblock_stats_config: Mapping,
    master_puzzle_dir: Path,
    teacher_dir: Path,
    descriptor: Type[ModelDescriptor],
    teacher_hidden_size: int,
    model_hidden_sizes: tuple[int, ...],
    subblock_configs: list[immutabledict[str, SubblockConfig]],
) -> dict[int, dict]:
    cache_root = master_puzzle_dir / "artifacts" / "subblock_stats" / "parameter_inventory"
    cache_root.mkdir(parents=True, exist_ok=True)
    progress_every = max(1, int(calc_subblock_stats_config.get("parameter_progress_every", 10)))
    requested_workers = max(1, int(calc_subblock_stats_config.get("parameter_workers", 1)))
    num_workers = min(requested_workers, len(model_hidden_sizes))
    kwargs_by_width = {
        width: {
            "teacher_dir": teacher_dir,
            "descriptor": descriptor,
            "width": width,
            "teacher_hidden_size": teacher_hidden_size,
            "legal_widths": model_hidden_sizes,
            "subblock_configs": subblock_configs,
            "cache_root": cache_root,
            "progress_every": progress_every,
        }
        for width in model_hidden_sizes
    }
    _write_parameter_inventory_progress_manifest(cache_root, model_hidden_sizes)
    if num_workers == 1:
        results = {
            width: _calculate_parameter_inventory_for_width(**kwargs_by_width[width])
            for width in model_hidden_sizes
        }
        _write_parameter_inventory_progress_manifest(cache_root, model_hidden_sizes)
        return results

    results: dict[int, dict] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=context) as executor:
        pending = {
            executor.submit(_calculate_parameter_inventory_for_width, **kwargs): width
            for width, kwargs in kwargs_by_width.items()
        }
        while pending:
            done, _ = wait(tuple(pending), timeout=2.0, return_when=FIRST_COMPLETED)
            _write_parameter_inventory_progress_manifest(cache_root, model_hidden_sizes)
            for future in done:
                width = pending.pop(future)
                results[width] = future.result()
    _write_parameter_inventory_progress_manifest(cache_root, model_hidden_sizes)
    return results


def _block_config_from_subblocks(*subblocks: SubblockConfig | None) -> BlockConfig:
    return BlockConfig(subblock_configs=tuple(subblock for subblock in subblocks if subblock))


def _runtime_measurement_noop(subblock: SubblockConfig) -> SubblockConfig:
    """Return a timing-only zero-cost baseline without adding it to the library.

    Width-search libraries are allowed to be strictly no-op-free.  Marginal
    block timing still needs one disabled mixer and one disabled FFN reference,
    so synthesize those references only inside the runtime experiment.
    """
    return SUBBLOCK_CLS_DICT[subblock.kind](
        kind=subblock.kind,
        name=subblock.name,
        no_op=True,
    )


"""
Usage:
python -m modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats PUZZLE_DIR [ --runtime_stats ]

--runtime_stats_enabled=False (the default) means that the code won't benchmark runtime,
  only memory stats will be calculated. If you want to benchmark runtime, run inside an trtllm docker.

"""


def calculate_subblock_stats(
    calc_subblock_stats_config: DictConfig,
    teacher_dir: Path,
    model_config: PretrainedConfig,
    descriptor: Type[ModelDescriptor],
    master_puzzle_dir: Path,
    subblock_configs: list[immutabledict[str, SubblockConfig]],
    batch_size: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    n_embd: int,
    n_head: int,
    vocab_size: int,
    runtime_stats_enabled: bool,
    use_cuda_graph: bool,
    weights_dtype: torch.dtype,
    activations_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    runtime_selection_identity: str | None = None,
    parameter_inventory: Mapping | None = None,
) -> dict:
    """
    Compute parameter, memory, additive-metric, and optional runtime statistics for subblock configurations.
    
    Parameters:
    	calc_subblock_stats_config (DictConfig): Runtime measurement and calculation settings.
    	teacher_dir (Path): Directory containing the teacher model or checkpoint.
    	model_config (PretrainedConfig): Model configuration used for metric calculations.
    	descriptor (Type[ModelDescriptor]): Model descriptor defining architecture-specific behavior.
    	master_puzzle_dir (Path): Puzzle directory used for runtime measurement caches.
    	subblock_configs (list[immutabledict[str, SubblockConfig]]): Subblock configurations and their parent layer indices.
    	batch_size (int): Number of sequences in the workload.
    	prefill_seq_len (int): Input sequence length used for prefill calculations.
    	generation_seq_len (int): Number of generated tokens used for decode calculations.
    	n_embd (int): Model hidden size.
    	n_head (int): Number of attention heads.
    	vocab_size (int): Model vocabulary size.
    	runtime_stats_enabled (bool): Whether to measure runtime statistics.
    	use_cuda_graph (bool): Whether to use CUDA graphs during runtime measurement.
    	weights_dtype (torch.dtype): Data type used for model weights.
    	activations_dtype (torch.dtype): Data type used for activations.
    	kv_cache_dtype (torch.dtype): Data type used for the key-value cache.
    	runtime_selection_identity (str | None): Identity of the runtime subblock selection.
    	parameter_inventory (Mapping | None): Precomputed parameter inventory to use for parameter counts.
    
    Returns:
    	dict: Statistics for the requested workload, including calculation arguments, non-block statistics, and per-subblock metrics.
    """
    runtime_granularity = "subblock"
    runtime_stats_config = (
        calc_subblock_stats_config.get("runtime_stats", {}) if runtime_stats_enabled else {}
    )
    if runtime_stats_enabled and not descriptor.runtime_benchmark_supported():
        mprint(
            f"Runtime stats requested, but {descriptor.__name__} does not implement "
            "synthetic vLLM benchmark models; continuing with params/memory stats only."
        )
        runtime_stats_enabled = False
    if runtime_stats_enabled:
        from modelopt.torch.puzzletron.subblock_stats.calc_runtime_stats import (
            calc_runtime_for_blocks,
            calc_runtime_for_subblocks,
        )

        runtime_granularity = calc_subblock_stats_config.get("runtime_stats", {}).get(
            "granularity", "subblock"
        )
        if runtime_granularity not in ("subblock", "block"):
            raise ValueError(
                f"runtime_stats.granularity must be 'subblock' or 'block', got "
                f"{runtime_granularity!r}. 'subblock' times attention/ffn separately; 'block' times "
                "the whole block (attn+ffn) together for interaction-aware costs."
            )

    gpu = None if not torch.cuda.is_available() else torch.cuda.get_device_name()
    subblock_stats = {
        "args": dict(
            gpu=gpu,
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            generation_seq_len=generation_seq_len,
            n_embd=n_embd,
            n_head=n_head,
            vocab_size=vocab_size,
            runtime_stats=runtime_stats_enabled,
            runtime_granularity=runtime_granularity if runtime_stats_enabled else None,
            use_cuda_graph=use_cuda_graph,
            weights_dtype=str(weights_dtype),
            activations_dtype=str(activations_dtype),
            kv_cache_dtype=str(kv_cache_dtype),
            runtime_backend=runtime_stats_config.get("backend") if runtime_stats_enabled else None,
            num_iters=runtime_stats_config.get("num_iters", 30) if runtime_stats_enabled else None,
            num_warmup_iters=(
                runtime_stats_config.get("num_warmup_iters", 10) if runtime_stats_enabled else None
            ),
            max_num_seqs=(
                runtime_stats_config.get("max_num_seqs") if runtime_stats_enabled else None
            ),
            workload_id=runtime_stats_config.get("workload_id") if runtime_stats_enabled else None,
            repeat_block_n_times=(
                max(2, int(runtime_stats_config.get("repeat_block_n_times", 10)))
                if runtime_stats_enabled
                else None
            ),
            vllm_args=(
                [str(arg) for arg in runtime_stats_config.get("vllm_args", [])]
                if runtime_stats_enabled
                else None
            ),
            runtime_selection_identity=(
                runtime_selection_identity if runtime_stats_enabled else None
            ),
            parameter_inventory_identity=(
                str(parameter_inventory["identity"]) if parameter_inventory is not None else None
            ),
        ),
        "non_block": dict(),
        "subblocks": list(),
    }
    # Compute runtime stats for unique subblocks only
    if runtime_stats_enabled:
        subblock_configs_nolayerindex = set(
            [subblock_config["subblock_config"] for subblock_config in subblock_configs]
        )

        lm_config = descriptor.get_language_model_config(model_config)

        _runtime_common = dict(
            runtime_stats_config=runtime_stats_config,
            vocab_size=vocab_size,
            hidden_size=n_embd,
            num_attention_heads=n_head,
            num_key_value_heads=lm_config.num_key_value_heads,
            descriptor=descriptor,
            lm_config=lm_config,
            tokenizer_path=teacher_dir,
            prefill_seq_len=prefill_seq_len,
            generation_seq_len=generation_seq_len,
            batch_size=batch_size,
            cache_dir=Path(master_puzzle_dir) / "runtime_cache",
        )
        if runtime_granularity == "block":
            # Block-level timing: build the cross-product of unique (attn, ffn) subblock configs
            # seen in the library and benchmark each (attn, ffn) pair together in one vLLM call.
            # This is more accurate than summing independent subblock timings.
            _attn_set = {
                sc["subblock_config"]
                for sc in subblock_configs
                if sc["subblock_config"].kind in _ATTENTION_LIKE_KINDS
            }
            _ffn_set = {
                sc["subblock_config"]
                for sc in subblock_configs
                if sc["subblock_config"].kind in _FFN_LIKE_KINDS
            }
            if not _attn_set or not _ffn_set:
                raise ValueError(
                    "Block-level runtime decomposition needs at least one attention-like "
                    "subblock and one FFN-like subblock in the library."
                )
            include_noop_baselines = bool(
                runtime_stats_config.get("include_noop_baselines", True)
            )
            if include_noop_baselines:
                # These baselines are measurement controls, not replacement
                # candidates. Keeping them local preserves a no-op-free search
                # library while retaining the established marginal-cost equation.
                for kind in sorted({subblock.kind for subblock in _attn_set}):
                    _attn_set.add(
                        _runtime_measurement_noop(
                            min(subblock for subblock in _attn_set if subblock.kind == kind)
                        )
                    )
                _ffn_set.add(_runtime_measurement_noop(min(_ffn_set)))
            block_config_set = {
                _block_config_from_subblocks(a, f) for a in _attn_set for f in _ffn_set
            }
            runtime_by_block_dict, non_block_runtime_ms = calc_runtime_for_blocks(
                block_config_set=block_config_set, **_runtime_common
            )
            # The legacy MIP interface consumes additive per-subblock costs, so decompose the exact
            # full-block measurements without assigning the whole block time to both subblocks.
            # When zero-cost controls are available, retain the established marginal decomposition.
            # Otherwise use the two-way additive projection:
            #   attn(A) = row_mean(A) - overall_mean / 2
            #   ffn(F)  = col_mean(F) - overall_mean / 2
            # whose sum is the least-squares additive fit to the valid Cartesian block timings. This
            # keeps no-op-free searches entirely inside deployable configurations. Exact per-(A,F)
            # measurements remain available in ``block_runtime_records`` for reports and block-aware
            # optimization.
            attn_no_ops = {
                a.kind: a for a in _attn_set if getattr(a, "no_op", False)
            }
            ffn_no_op = next((f for f in _ffn_set if getattr(f, "no_op", False)), None)
            def _block_ms(a, f):
                return runtime_by_block_dict[_block_config_from_subblocks(a, f)]

            if attn_no_ops and ffn_no_op is not None:
                decomposition_method = "noop_marginal"
                active_attn = [a for a in _attn_set if not getattr(a, "no_op", False)]
                active_ffn = [f for f in _ffn_set if not getattr(f, "no_op", False)]

                def _subblock_runtime_from_block(sc):
                    if sc.kind in _ATTENTION_LIKE_KINDS:
                        return RuntimeMeasurement.mean(
                            [
                                _block_ms(sc, f) - _block_ms(attn_no_ops[sc.kind], f)
                                for f in active_ffn
                            ]
                        )
                    if sc.kind in _FFN_LIKE_KINDS:
                        return RuntimeMeasurement.mean(
                            [_block_ms(a, sc) - _block_ms(a, ffn_no_op) for a in active_attn]
                        )
                    return RuntimeMeasurement.zero()

            else:
                decomposition_method = "two_way_additive_projection"
                overall_mean = RuntimeMeasurement.mean(
                    [_block_ms(a, f) for a in _attn_set for f in _ffn_set]
                )

                def _subblock_runtime_from_block(sc):
                    if sc.kind in _ATTENTION_LIKE_KINDS:
                        return RuntimeMeasurement.mean([_block_ms(sc, f) for f in _ffn_set]) - (
                            overall_mean / 2
                        )
                    if sc.kind in _FFN_LIKE_KINDS:
                        return RuntimeMeasurement.mean([_block_ms(a, sc) for a in _attn_set]) - (
                            overall_mean / 2
                        )
                    return RuntimeMeasurement.zero()

            runtime_by_subblock_dict = {
                sc: _subblock_runtime_from_block(sc) for sc in subblock_configs_nolayerindex
            }
            subblock_stats["runtime_decomposition"] = {
                "method": decomposition_method,
                "include_noop_baselines": include_noop_baselines,
                "attention_levels": len(_attn_set),
                "ffn_levels": len(_ffn_set),
            }
            # Expose block-level data on the stats dict for downstream consumers that want it.
            subblock_stats["block_runtimes"] = {
                str(bc): ms.total_ms for bc, ms in runtime_by_block_dict.items()
            }
            # Keep the historical string-keyed map readable, while also emitting a stable,
            # machine-readable representation for diagnostic and optimization tooling.
            subblock_stats["block_runtime_records"] = [
                {
                    "block_config": bc.to_dict(),
                    **_runtime_measurement_fields(
                        ms, generation_seq_len=generation_seq_len
                    ),
                }
                for bc, ms in sorted(runtime_by_block_dict.items(), key=lambda item: str(item[0]))
            ]
        else:
            runtime_by_subblock_dict, non_block_runtime_ms = calc_runtime_for_subblocks(
                subblock_config_set=subblock_configs_nolayerindex, **_runtime_common
            )

    sorted_subblock_config = sorted(
        subblock_configs, key=lambda subblock_config: subblock_config["subblock_config"]
    )
    parameter_rows = (
        {
            str(row["inventory_key"]): row
            for row in parameter_inventory.get("rows", [])
        }
        if parameter_inventory is not None
        else {}
    )
    it = (
        tqdm(sorted_subblock_config, desc="Measuring subblock runtimes")
        if runtime_stats_enabled
        else sorted_subblock_config
    )
    for subblock_config_indexed in it:
        subblock_config = subblock_config_indexed["subblock_config"]
        parent_layer_indices = subblock_config_indexed["parent_layer_indices"]

        inventory_row = parameter_rows.get(
            _parameter_inventory_key(subblock_config, parent_layer_indices[0])
        )
        if parameter_inventory is not None and inventory_row is None:
            raise KeyError(
                "Parameter inventory is missing subblock "
                f"{subblock_config} at parent layer {parent_layer_indices[0]}"
            )
        if inventory_row is not None:
            # The inventory already used the correctly truncated physical-width
            # config. Static assembly only needs family-wide config attributes.
            layer_model_config = model_config
        else:
            layer_model_config = copy.deepcopy(model_config)
            descriptor.truncate_pattern_for_subblock(
                descriptor.get_language_model_config(layer_model_config),
                parent_layer_indices[0],
            )
        subblock_params = (
            int(inventory_row["num_params"])
            if inventory_row is not None
            else calculate_subblock_params(layer_model_config, subblock_config, descriptor)
        )
        subblock_active_params = (
            int(inventory_row["active_params"])
            if inventory_row is not None
            else calc_subblock_active_params(
                subblock_config,
                layer_model_config,
                descriptor,
                n_embd,
                num_params=subblock_params,
            )
        )

        latency_fields = _runtime_measurement_fields(
            runtime_by_subblock_dict[subblock_config] if runtime_stats_enabled else None,
            generation_seq_len=generation_seq_len,
        )

        subblock_memory = calculate_subblock_memory(
            subblock_config,
            batch_size,
            prefill_seq_len,
            generation_seq_len,
            n_embd,
            n_head,
            weights_dtype,
            kv_cache_dtype,
            model_config=layer_model_config,
            descriptor=descriptor,
            num_params=subblock_params,
        )
        if not isinstance(subblock_memory, dict):
            subblock_memory = {"memory_mib": subblock_memory, "kv_cache_memory_mib": 0.0}

        additive_fields = calculate_additive_metrics(
            subblock_config,
            model_config=layer_model_config,
            descriptor=descriptor,
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            generation_seq_len=generation_seq_len,
            n_embd=n_embd,
            n_head=n_head,
            weights_dtype=weights_dtype,
            kv_cache_dtype=kv_cache_dtype,
            num_params=subblock_params,
            active_params=subblock_active_params,
        )
        provenance = {
            **additive_fields.pop("additive_metric_provenance"),
            **latency_fields.pop("additive_metric_provenance"),
        }
        subblock_stats["subblocks"].append(
            {
                "subblock_config": subblock_config,
                "subblock_config_class": type(subblock_config).__name__,
                "num_params": subblock_params,
                "active_params": subblock_active_params,
                "parent_layer_index": parent_layer_indices[0],
                **latency_fields,
                **subblock_memory,
                **additive_fields,
                "additive_metric_provenance": provenance,
            }
        )

    if not runtime_stats_enabled:
        non_block_runtime_ms = None
    non_block_memory = calculate_non_block_memory(n_embd, vocab_size, weights_dtype)
    lm_config = descriptor.get_language_model_config(model_config)
    if parameter_inventory is not None:
        non_block_params = int(parameter_inventory["non_block_params"])
        non_block_source = str(parameter_inventory["non_block_parameter_count_source"])
    else:
        non_block_params = _checkpoint_non_block_params(
            teacher_dir,
            descriptor,
            int(lm_config.num_hidden_layers),
        )
        if non_block_params is None:
            non_block_params = calculate_non_block_params(n_embd, vocab_size)
            non_block_source = "lm_formula_fallback"
        else:
            non_block_source = "checkpoint_tensor_inventory"

    subblock_stats["non_block"] = {
        **_runtime_measurement_fields(
            non_block_runtime_ms, generation_seq_len=generation_seq_len
        ),
        "memory_mib": non_block_memory,
        "num_params": non_block_params,
        "parameter_count_source": non_block_source,
    }
    return subblock_stats


def launch_calc_subblock_stats(cfg: DictConfig) -> None:
    """
    Launch the calc subblock stats function with Hydra configuration.
    """
    mprint(f"Calculating subblock stats for puzzle directory: {cfg.puzzle_dir}")
    mprint(f"Teacher directory: {cfg.teacher_dir}")
    mprint(
        f"Calc subblock stats config: {format_global_config(cfg.calc_subblock_stats, title='Calc subblock stats')}"
    )

    descriptor = ModelDescriptorFactory.get(cfg.descriptor)
    calculate_subblock_stats_for_puzzle_dir(
        cfg.calc_subblock_stats,
        master_puzzle_dir=cfg.puzzle_dir,
        teacher_dir=cfg.teacher_dir,
        descriptor=descriptor,
        model_hidden_sizes=cfg.calc_subblock_stats.get("model_hidden_sizes", OmegaConf.create([])),
        ffn_hidden_sizes=cfg.calc_subblock_stats.get("ffn_hidden_sizes", OmegaConf.create([])),
        batch_sizes=cfg.calc_subblock_stats.batch_sizes,
        prefill_seq_len=cfg.calc_subblock_stats.prefill_seq_len,
        generation_seq_len=cfg.calc_subblock_stats.generation_seq_len,
        runtime_stats_enabled=cfg.calc_subblock_stats.get("runtime_stats", {}).get(
            "enabled", False
        ),
        merge_with_existing_stats=cfg.calc_subblock_stats.merge_with_existing_stats,
        subblock_stats_filename=cfg.calc_subblock_stats.subblock_stats_filename,
    )


def _arg_signature(args: dict) -> tuple:
    """Identity of a computed stats entry.

    Captures the knobs that distinguish one configuration from another while
    ignoring ``gpu`` (host-dependent).
    """
    runtime_stats = bool(args.get("runtime_stats", False))
    return (
        args["batch_size"],
        args.get("prefill_seq_len"),
        args.get("generation_seq_len"),
        str(args["weights_dtype"]),
        str(args["activations_dtype"]),
        str(args["kv_cache_dtype"]),
        args["n_embd"],
        runtime_stats,
        args.get("runtime_granularity") if runtime_stats else None,
        args.get("max_num_seqs") if runtime_stats else None,
        args.get("workload_id") if runtime_stats else None,
        args.get("runtime_selection_identity"),
        args.get("parameter_inventory_identity"),
    )


def _subblock_stats_already_complete(
    existing_stats: list,
    subblock_configs: list[immutabledict[str, SubblockConfig]],
    batch_sizes: Iterable[int],
    data_types: list,
    model_hidden_sizes: Iterable[int],
    runtime_stats_enabled: bool,
    runtime_granularity: str = "subblock",
    runtime_max_num_seqs: int | None = None,
    runtime_workload_id: str | None = None,
    runtime_selection_identity: str | None = None,
    parameter_inventory_identities: Mapping[int, str] | None = None,
    prefill_seq_len: int = 2048,
    generation_seq_len: int = 2048,
) -> bool:
    """
    Determine whether existing statistics cover all requested configurations.
    
    Parameters:
        existing_stats (list): Previously calculated statistics entries.
        subblock_configs (list): Subblock configurations required for coverage.
        batch_sizes (Iterable[int]): Batch sizes to verify.
        data_types (list): Weight, activation, and KV-cache dtype combinations.
        model_hidden_sizes (Iterable[int]): Model widths to verify.
        runtime_stats_enabled (bool): Whether runtime measurements are required.
        runtime_granularity (str): Required runtime measurement granularity.
        runtime_max_num_seqs (int | None): Required maximum number of runtime sequences.
        runtime_workload_id (str | None): Required runtime workload identity.
        runtime_selection_identity (str | None): Required runtime subblock-selection identity.
        parameter_inventory_identities (Mapping[int, str] | None): Inventory identity required for each model width.
        prefill_seq_len (int): Required prefill sequence length.
        generation_seq_len (int): Required generation sequence length.
    
    Returns:
        bool: True if every requested configuration and required measurement is present, False otherwise.
    """
    by_signature = {_arg_signature(entry["args"]): entry for entry in existing_stats}
    required_subblock_keys = {
        (subblock_config["subblock_config"], subblock_config["parent_layer_indices"][0])
        for subblock_config in subblock_configs
    }

    def _entry_subblock_keys(entry: dict) -> set[tuple[SubblockConfig, int]]:
        keys = set()
        for substats in entry.get("subblocks", []):
            raw_config = substats["subblock_config"]
            kind = raw_config.get("kind")
            if kind not in SUBBLOCK_CLS_DICT:
                return set()
            keys.add(
                (SUBBLOCK_CLS_DICT[kind](**raw_config), substats.get("parent_layer_index", -1))
            )
        return keys

    for batch_size, (
        weights_dtype,
        activations_dtype,
        kv_cache_dtype,
    ), model_hidden_size in product(batch_sizes, data_types, model_hidden_sizes):
        runtime_expected = runtime_stats_enabled and weights_dtype == torch.bfloat16
        signature = (
            batch_size,
            prefill_seq_len,
            generation_seq_len,
            str(weights_dtype),
            str(activations_dtype),
            str(kv_cache_dtype),
            model_hidden_size,
            runtime_expected,
            runtime_granularity if runtime_expected else None,
            runtime_max_num_seqs if runtime_expected else None,
            runtime_workload_id if runtime_expected else None,
            (
                runtime_selection_identity
                if runtime_expected
                else None
            ),
            (
                parameter_inventory_identities.get(int(model_hidden_size))
                if parameter_inventory_identities is not None
                else None
            ),
        )
        entry = by_signature.get(signature)
        if entry is None:
            return False
        if not required_subblock_keys.issubset(_entry_subblock_keys(entry)):
            return False
        # Runtime is only measured for the bf16 configuration (see the
        # ``curr_runtime_stats_enabled`` guard below); require it to be present.
        if runtime_expected:
            if not entry["args"].get("runtime_stats", False):
                return False
            # Default to "subblock" for entries written before granularity was recorded.
            if entry["args"].get("runtime_granularity", "subblock") != runtime_granularity:
                return False
            required_runtime_fields = (
                "runtime_ms",
                "prefill_runtime_ms",
                "decode_runtime_ms",
                "decode_runtime_ms_per_token",
                "weight_memory_mib",
                "kv_cache_bytes_per_token",
                "state_cache_bytes_per_sequence",
                "prefill_flops",
                "decode_flops",
            )
            for substats in entry.get("subblocks", []):
                if any(substats.get(field) is None for field in required_runtime_fields):
                    return False
                provenance = substats.get("additive_metric_provenance") or {}
                if any(field not in provenance for field in required_runtime_fields):
                    return False
    return True


def calculate_subblock_stats_for_puzzle_dir(
    calc_subblock_stats_config: DictConfig,
    master_puzzle_dir: Path | str,
    teacher_dir: Path | str,
    descriptor: Type[ModelDescriptor],
    model_hidden_sizes: ListConfig,
    ffn_hidden_sizes: ListConfig,
    batch_sizes: Iterable[int] = (1, 8, 16, 32, 64, 128, 256),
    prefill_seq_len: int = 2048,
    generation_seq_len: int = 2048,
    runtime_stats_enabled: bool = False,  # Compute runtime statistics.
    merge_with_existing_stats: bool = False,
    subblock_stats_filename: str = "subblock_stats.json",
) -> None:
    # ==== START === Setup for attach-helper ====
    # import sys
    # import os
    # sys.path.insert(0, os.environ["ATTACH_HELPER_INSTALLATION_PATH"])
    # from attach_helper import debugging_setup
    # debugging_setup()  # You can optionally pass a name to identify the job (e.g. `debugging_setup(name="my_script")`)
    # ==== END === Setup for attach-helper ====
    """
    Compute and persist subblock statistics for all requested batch sizes, data types, and model widths.
    
    Parameters:
        calc_subblock_stats_config (DictConfig): Configuration for statistics calculation and optional runtime measurement.
        master_puzzle_dir (Path | str): Puzzle directory containing subblock configurations and output files.
        teacher_dir (Path | str): Teacher checkpoint directory used for model metadata and parameter inventories.
        descriptor (Type[ModelDescriptor]): Model descriptor defining architecture-specific behavior.
        model_hidden_sizes (ListConfig): Hidden sizes to evaluate; the teacher hidden size is always included.
        ffn_hidden_sizes (ListConfig): Additional FFN sizes to include in the subblock configurations.
        batch_sizes (Iterable[int]): Batch sizes to evaluate.
        prefill_seq_len (int): Number of prompt tokens used for runtime measurements.
        generation_seq_len (int): Number of generated tokens used for runtime measurements.
        runtime_stats_enabled (bool): Whether to compute or reuse runtime statistics.
        merge_with_existing_stats (bool): Whether to update an existing incomplete statistics file.
        subblock_stats_filename (str): Name of the JSON file used to persist statistics.
    
    Raises:
        FileNotFoundError: If a configured runtime manifest or reusable runtime statistics file cannot be found.
        ValueError: If runtime settings or reusable runtime statistics do not cover the requested configurations.
    """
    if isinstance(batch_sizes, str):
        batch_sizes = [
            int(batch_size) for batch_size in batch_sizes.strip("[]").replace(" ", "").split(",")
        ]
    else:
        batch_sizes = list(batch_sizes)

    master_puzzle_dir = Path(master_puzzle_dir)
    teacher_dir = (
        Path(teacher_dir) if teacher_dir is not None else master_puzzle_dir / "ckpts" / "teacher"
    )
    trust_remote_code = descriptor.requires_trust_remote_code()
    model_config = load_model_config(teacher_dir, trust_remote_code=trust_remote_code)
    # Get language model config for LM-specific attributes (VL models have nested config)
    lm_config = descriptor.get_language_model_config(model_config)
    subblock_configs = _load_subblock_configs(master_puzzle_dir, ffn_hidden_sizes)
    runtime_selection_identity = None
    runtime_stats_config = calc_subblock_stats_config.get("runtime_stats", {})
    selection_manifest_path = runtime_stats_config.get("selection_manifest", None)
    if selection_manifest_path:
        selection_manifest_path = Path(str(selection_manifest_path))
        if not selection_manifest_path.is_file():
            candidate = master_puzzle_dir / selection_manifest_path
            if candidate.is_file():
                selection_manifest_path = candidate
            else:
                raise FileNotFoundError(
                    f"sparse runtime selection manifest does not exist: {selection_manifest_path}"
                )
        manifest = json.loads(selection_manifest_path.read_text())
        if manifest.get("mode") != "subblock_runtime":
            raise ValueError(
                f"runtime selection manifest has mode={manifest.get('mode')!r}, "
                "expected 'subblock_runtime'"
            )
        _validate_sparse_runtime_settings(runtime_stats_config)
        subblock_configs = _select_runtime_subblock_configs(subblock_configs, manifest)
        runtime_selection_identity = str(manifest["identity"])

    data_types = [
        ("nvfp4", "nvfp4", "nvfp4"),
        (torch.int8, torch.int8, torch.int8),
        (torch.int8, torch.int8, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ]

    teacher_hidden_size = int(lm_config.hidden_size)
    model_hidden_sizes = _unique_hidden_sizes(model_hidden_sizes, teacher_hidden_size)
    runtime_reuse_path = runtime_stats_config.get("reuse_stats_path")
    runtime_reuse_by_key: dict[tuple, dict] = {}
    if runtime_stats_enabled and runtime_reuse_path:
        runtime_reuse_path = Path(str(runtime_reuse_path))
        if not runtime_reuse_path.is_file():
            candidate = master_puzzle_dir / runtime_reuse_path
            if candidate.is_file():
                runtime_reuse_path = candidate
            else:
                raise FileNotFoundError(
                    f"Reusable runtime stats file does not exist: {runtime_reuse_path}"
                )
        reusable_entries = json.loads(runtime_reuse_path.read_text())
        fallback_workload_id = runtime_stats_config.get("reuse_workload_id_if_missing")
        for entry in reusable_entries:
            key = _runtime_reuse_key_from_args(
                entry.get("args", {}),
                fallback_workload_id=fallback_workload_id,
            )
            if key is not None:
                runtime_reuse_by_key[key] = entry
        requested_runtime_keys = {
            _runtime_reuse_key(
                width=int(width),
                batch_size=int(batch_size),
                prefill_seq_len=int(prefill_seq_len),
                generation_seq_len=int(generation_seq_len),
                runtime_stats_config=runtime_stats_config,
            )
            for width in model_hidden_sizes
            for batch_size in batch_sizes
        }
        missing_runtime_keys = requested_runtime_keys - set(runtime_reuse_by_key)
        if missing_runtime_keys:
            raise ValueError(
                f"Reusable runtime stats {runtime_reuse_path} are missing requested "
                f"runtime identities {sorted(missing_runtime_keys)}"
            )
        runtime_selection_identity = "reuse-" + hashlib.sha256(
            runtime_reuse_path.read_bytes()
        ).hexdigest()
    parameter_inventories = _collect_parameter_inventories(
        calc_subblock_stats_config=calc_subblock_stats_config,
        master_puzzle_dir=master_puzzle_dir,
        teacher_dir=teacher_dir,
        descriptor=descriptor,
        teacher_hidden_size=teacher_hidden_size,
        model_hidden_sizes=model_hidden_sizes,
        subblock_configs=subblock_configs,
    )
    parameter_inventory_identities = {
        width: str(inventory["identity"])
        for width, inventory in parameter_inventories.items()
    }
    width_model_configs = {
        width: _virtual_width_config(
            model_config,
            descriptor,
            width,
            teacher_hidden_size,
            model_hidden_sizes,
        )[0]
        for width in parameter_inventories
    }

    subblock_stats_file = master_puzzle_dir / subblock_stats_filename
    subblock_stats_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: the runtime (vLLM) benchmark is by far the most expensive
    # part of this step, so make it skippable on re-runs just like teacher
    # conversion, pruning, and bypass distillation. If a previous run already
    # produced stats for every (batch_size, dtype, hidden_size) configuration we
    # would compute -- and, when runtime benchmarking is enabled, the relevant
    # entries already carry runtime measurements -- there is nothing to do.
    if subblock_stats_file.exists():
        with open(subblock_stats_file) as f:
            subblock_stats = json.load(f)

        if _subblock_stats_already_complete(
            subblock_stats,
            subblock_configs,
            batch_sizes,
            data_types,
            model_hidden_sizes,
            runtime_stats_enabled,
            runtime_granularity=calc_subblock_stats_config.get("runtime_stats", {}).get(
                "granularity", "subblock"
            ),
            runtime_max_num_seqs=calc_subblock_stats_config.get("runtime_stats", {}).get(
                "max_num_seqs"
            ),
            runtime_workload_id=calc_subblock_stats_config.get("runtime_stats", {}).get(
                "workload_id"
            ),
            runtime_selection_identity=runtime_selection_identity,
            parameter_inventory_identities=parameter_inventory_identities,
            prefill_seq_len=prefill_seq_len,
            generation_seq_len=generation_seq_len,
        ):
            mprint(
                f"Subblock stats file {subblock_stats_file} already covers all requested "
                f"configurations{' (incl. runtime measurements)' if runtime_stats_enabled else ''}; "
                "skipping recomputation. Delete the file to force a rebuild."
            )
            return

        if not merge_with_existing_stats:
            raise ValueError(
                f"Subblock stats file {subblock_stats_file} already exists, is incomplete, and "
                "`merge_with_existing_stats` was set to False."
            )
    else:
        subblock_stats = []

    # Entries written before width-specific inventory identities are stale for
    # the requested widths. Remove them so one canonical row remains per dtype
    # and width rather than leaving ambiguous duplicate costs for MIP readers.
    subblock_stats = [
        entry
        for entry in subblock_stats
        if int(entry.get("args", {}).get("n_embd", -1)) not in parameter_inventory_identities
        or entry.get("args", {}).get("parameter_inventory_identity")
        == parameter_inventory_identities[int(entry["args"]["n_embd"])]
    ]

    subblock_stats_indices = {
        _freeze_stats_args(entry["args"]): index for index, entry in enumerate(subblock_stats)
    }

    for batch_size, (
        weights_dtype,
        activations_dtype,
        kv_cache_dtype,
    ), model_hidden_size in product(batch_sizes, data_types, model_hidden_sizes):
        width_model_config = width_model_configs[int(model_hidden_size)]
        width_lm_config = descriptor.get_language_model_config(width_model_config)
        curr_runtime_stats_enabled = (
            runtime_stats_enabled if weights_dtype == torch.bfloat16 else False
        )
        reused_runtime_stats = None
        if curr_runtime_stats_enabled:
            reused_runtime_stats = runtime_reuse_by_key.get(
                _runtime_reuse_key(
                    width=int(model_hidden_size),
                    batch_size=int(batch_size),
                    prefill_seq_len=int(prefill_seq_len),
                    generation_seq_len=int(generation_seq_len),
                    runtime_stats_config=runtime_stats_config,
                )
            )

        curr_subblock_stats = calculate_subblock_stats(
            calc_subblock_stats_config,
            teacher_dir=teacher_dir,
            model_config=width_model_config,
            descriptor=descriptor,
            master_puzzle_dir=master_puzzle_dir,
            subblock_configs=subblock_configs,
            batch_size=batch_size,
            prefill_seq_len=prefill_seq_len,
            generation_seq_len=generation_seq_len,
            n_embd=model_hidden_size,
            n_head=width_lm_config.num_attention_heads,
            vocab_size=width_lm_config.vocab_size,
            runtime_stats_enabled=curr_runtime_stats_enabled and reused_runtime_stats is None,
            # The vLLM benchmark runs with --optimization-level 0 (CUDA graphs disabled) for
            # accurate per-block timing, so record that rather than a misleading True.
            use_cuda_graph=False,
            weights_dtype=weights_dtype,
            activations_dtype=activations_dtype,
            kv_cache_dtype=kv_cache_dtype,
            runtime_selection_identity=runtime_selection_identity,
            parameter_inventory=parameter_inventories[int(model_hidden_size)],
        )
        if reused_runtime_stats is not None:
            curr_subblock_stats = _reuse_runtime_stats(
                curr_subblock_stats,
                reused_runtime_stats,
                source_path=str(runtime_reuse_path),
                fallback_workload_id=runtime_stats_config.get("workload_id"),
            )
            curr_subblock_stats["args"]["runtime_selection_identity"] = (
                runtime_selection_identity
            )

        curr_args = _freeze_stats_args(curr_subblock_stats["args"])
        if curr_args in subblock_stats_indices:
            subblock_stats[subblock_stats_indices[curr_args]] = curr_subblock_stats
        else:
            subblock_stats_indices[curr_args] = len(subblock_stats)
            subblock_stats.append(curr_subblock_stats)

    shard_index = int(os.environ.get("PUZZLETRON_RUNTIME_SHARD_INDEX", "0"))
    shard_count = int(os.environ.get("PUZZLETRON_RUNTIME_SHARD_COUNT", "1"))
    if (
        os.environ.get("PUZZLETRON_RUNTIME_CACHE_WARMUP_ONLY") != "1"
        and (shard_count == 1 or shard_index == 0)
    ):
        _atomic_json_dump(subblock_stats, subblock_stats_file)

    mprint(subblock_stats_file)


def _load_subblock_configs(
    master_puzzle_dir: Path, ffn_hidden_sizes: ListConfig
) -> list[SubblockConfig]:
    try:
        subblock_configs = _load_subblock_configs_from_replacement_library(master_puzzle_dir)
    except FileNotFoundError:
        subblock_configs = _load_subblock_configs_from_subblock_library(master_puzzle_dir)

    # Extend subblock stats calculation space with ffn_hidden_sizes defined in the calc_subblock_stats section of the model config yaml file.
    extra_ffn_subblock_configs = []
    for ffn_hidden_size in ffn_hidden_sizes:
        # Use FFNConfig defaults (hidden_act will use its default value)
        ffn_config = FFNConfig(intermediate_size=ffn_hidden_size)
        extra_ffn_subblock_configs.append(
            immutabledict({"subblock_config": ffn_config, "parent_layer_indices": tuple([-1])})
        )  # -1 to indicate that this sublock has no parent layer
    subblock_configs.extend(extra_ffn_subblock_configs)

    return subblock_configs


def _load_subblock_configs_from_subblock_library(master_puzzle_dir: Path) -> list[SubblockConfig]:
    subblocks_df = pd.read_json(master_puzzle_dir / "subblock_library.json")
    configs = []
    for kind in _SUBBLOCK_KINDS:
        column = f"{kind}_config"
        if column not in subblocks_df:
            continue
        subblocks_df[column] = subblocks_df[column].apply(
            partial(_dataclass_from_dict, cls=SUBBLOCK_CLS_DICT[kind])
        )
        configs.extend(subblocks_df[column].dropna().drop_duplicates().tolist())

    # Wrap in the same dict format expected by calculate_subblock_stats() callers.
    # Use parent_layer_indices=(-1,) to indicate no specific parent layer.
    subblock_configs = [
        immutabledict({"subblock_config": cfg, "parent_layer_indices": (-1,)}) for cfg in configs
    ]
    return subblock_configs


def _load_subblock_configs_from_replacement_library(
    master_puzzle_dir: Path,
) -> list[SubblockConfig]:
    """Load unique subblocks from replacement_library.json (v1 list or v2 dict with header).

    Args:
        master_puzzle_dir: Directory with "replacement_library.json" file
    """
    raw = json.loads((master_puzzle_dir / "replacement_library.json").read_text())
    if isinstance(raw, dict) and raw.get("version") == 2:
        entries = raw.get("entries", [])
        for e in entries:
            e.setdefault("weight_paths", [])
        replacement_library = entries
    else:
        replacement_library = raw
    subblock_configs = set()
    for layer_replacement in replacement_library:
        layer_replacement = parse_layer_replacement(layer_replacement)

        for block_config in layer_replacement["child_block_configs"]:
            block_config: BlockConfig
            for subblock_ref in block_config.subblocks():
                subblock_configs.add(
                    immutabledict(
                        {
                            "subblock_config": subblock_ref.config,
                            "parent_layer_indices": tuple(
                                layer_replacement["parent_layer_indices"]
                            ),
                        }
                    )
                )

    subblock_configs = list(subblock_configs)
    return subblock_configs


T_DataClass: TypeVar = Type[dataclasses.dataclass]


def _dataclass_from_dict(
    d: dict | T_DataClass | None,
    cls: T_DataClass,
) -> T_DataClass | None:
    if isinstance(d, cls):
        return d
    if isinstance(d, dict):
        kind = d.get("kind")
        if kind is not None:
            return SUBBLOCK_CLS_DICT[kind](**d)
        return cls(**d)
    if pd.isna(d):
        return None
    raise ValueError(f"_dataclass_from_dict: unrecognized {type(d)=} {d=}")
