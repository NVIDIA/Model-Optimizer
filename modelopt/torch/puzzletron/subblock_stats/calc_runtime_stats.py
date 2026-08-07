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

"""Runtime statistics calculation for NAS subblock benchmarking via vLLM."""

import hashlib
import json
import math
import os
import queue
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Type

from omegaconf import DictConfig
from tqdm import tqdm

from ..anymodel.model_descriptor import ModelDescriptor
from ..block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    MLAConfig,
    MoEConfig,
    SubblockConfig,
    maybe_cast_block_configs,
)
from ..candidates import build_candidate_library
from ..tools.checkpoint_utils import load_model_config
from ..tools.logger import mprint
from .runtime_estimator import (
    candidate_slope,
    effective_repeat_count,
    fixed_intercept,
    homogeneous_layout,
    median_measurement,
    scaffolded_layout,
)
from .runtime_utils import RuntimeConfig, save_model
from .runtime_vllm import RuntimeMeasurement, run_vllm_latency_benchmark
from .topology import RuntimeTopology

_ATTENTION_LIKE_KINDS = frozenset(("attention", "mla", "mamba"))
_FFN_LIKE_KINDS = frozenset(("ffn", "moe"))
_RUNTIME_SHARD_RESULT_SCHEMA_VERSION = 4


def enumerate_runtime_block_configs(
    teacher_dir: Path | str,
    descriptor: Type[ModelDescriptor],
    *,
    search_space: Mapping[str, Any] | None = None,
    include_noops: bool = True,
) -> tuple[BlockConfig, ...]:
    """Return deterministic unique runtime candidates from a converted teacher checkpoint."""
    teacher_dir = Path(teacher_dir)
    model_config = load_model_config(
        teacher_dir, trust_remote_code=descriptor.requires_trust_remote_code()
    )
    language_model_config = descriptor.get_language_model_config(model_config)
    block_configs = getattr(language_model_config, "block_configs", None) or getattr(
        model_config, "block_configs", None
    )
    if block_configs is None:
        raise ValueError(f"Converted teacher {teacher_dir} does not define block_configs")
    typed_block_configs = maybe_cast_block_configs(block_configs)
    candidates = build_candidate_library(
        typed_block_configs,
        search_space=dict(search_space or {}),
        parent_checkpoint_identity=str(teacher_dir.resolve()),
        include_noops=include_noops,
    )
    return tuple(
        sorted(
            {candidate.block_config for candidate in candidates},
            key=lambda block_config: json.dumps(block_config.to_dict(), sort_keys=True),
        )
    )


def _subblocks_with_kinds(block_config: BlockConfig, kinds: frozenset[str]) -> tuple[SubblockConfig, ...]:
    return tuple(
        subblock for subblock in block_config.subblock_configs if subblock.kind in kinds
    )


def _mark_noop_kinds(block_config: BlockConfig, kinds: frozenset[str]) -> BlockConfig:
    return BlockConfig(
        subblock_configs=tuple(
            replace(subblock, no_op=True) if subblock.kind in kinds else subblock
            for subblock in block_config.subblock_configs
        )
    )


def _freeze_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_config_value(val)) for key, val in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _freeze_config_fields(fields: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((key, _freeze_config_value(value)) for key, value in fields.items()))


def _extra_vllm_args(runtime_stats_config: DictConfig) -> tuple[str, ...]:
    return tuple(str(arg) for arg in runtime_stats_config.get("vllm_args", []))


def _vllm_env(runtime_stats_config: DictConfig) -> tuple[tuple[str, str], ...]:
    """Freeze explicit vLLM subprocess overrides into a stable cache identity."""

    configured = runtime_stats_config.get("vllm_env", {}) or {}
    return tuple(sorted((str(key), str(value)) for key, value in configured.items()))


def create_benchmark_model(
    runtime_config: RuntimeConfig,
    block_layout: tuple[BlockConfig, ...],
):
    """Build a descriptor-specific model from an exact immutable block layout."""

    return runtime_config.descriptor.create_runtime_benchmark_model(
        runtime_config, list(block_layout)
    )


def _block_config_for_subblock(
    runtime_config: RuntimeConfig, subblock_config: BlockConfig | SubblockConfig | None
) -> BlockConfig | None:
    """Map a subblock to the repeated ``BlockConfig`` used to benchmark it.

    ``None`` means "no repeated block" (i.e. the base block only).
    """
    if subblock_config is None:
        return None
    if isinstance(subblock_config, BlockConfig):
        return subblock_config
    exclusive = bool(
        getattr(
            runtime_config.descriptor,
            "runtime_benchmark_sublayers_are_exclusive",
            lambda: False,
        )()
    )
    if exclusive:
        return BlockConfig(subblock_configs=(subblock_config,))
    if isinstance(subblock_config, FFNConfig | MoEConfig):
        base_block_config = runtime_config.descriptor.runtime_benchmark_base_block_config(
            runtime_config
        )
        return _mark_noop_kinds(
            base_block_config.with_subblock(
                subblock_config, replace_kinds=_FFN_LIKE_KINDS
            ),
            _ATTENTION_LIKE_KINDS,
        )
    if isinstance(subblock_config, AttentionConfig | MLAConfig | MambaConfig):
        base_block_config = runtime_config.descriptor.runtime_benchmark_base_block_config(
            runtime_config
        )
        return _mark_noop_kinds(
            base_block_config.with_subblock(
                subblock_config, replace_kinds=_ATTENTION_LIKE_KINDS
            ),
            _FFN_LIKE_KINDS,
        )
    raise Exception(f"Runtime stats: Not supported subblock type: {subblock_config}")


def _validate_marginal_runtime(
    measurement: RuntimeMeasurement,
    *,
    label: str,
    ignore_negatives: bool = False,
) -> None:
    """Reject derived timings that cannot represent additive operator work."""

    if measurement.total_ms <= 0.0:
        message = (
            f"non-positive marginal runtime for {label}: {measurement.total_ms:.6g} ms"
        )
        if not ignore_negatives or measurement.total_ms == 0.0:
            raise ValueError(message)
        warnings.warn(f"Ignoring {message}", RuntimeWarning)
    negative = {
        "prefill_ms": measurement.prefill_ms,
        "decode_ms": measurement.decode_ms,
    }
    negative = {name: value for name, value in negative.items() if value < 0.0}
    if negative:
        message = f"negative marginal phase for {label}: {negative}"
        if not ignore_negatives:
            raise ValueError(message)
        warnings.warn(f"Ignoring {message}", RuntimeWarning)


def _benchmark_spec(
    runtime_config: RuntimeConfig,
    block_layout: tuple[BlockConfig, ...],
    gpu_id: str | int | None,
    cache_dir: Path | None,
) -> RuntimeMeasurement:
    """Build one exact-layout model and measure its total latency."""

    model = create_benchmark_model(runtime_config, block_layout)
    with tempfile.TemporaryDirectory() as model_tmpdir:
        save_model(
            model,
            Path(runtime_config.tokenizer_path),
            Path(model_tmpdir),
            runtime_config.descriptor,
        )
        return run_vllm_latency_benchmark(
            Path(model_tmpdir), runtime_config, gpu_id=gpu_id, cache_dir=cache_dir
        )


def _resolve_gpu_ids(group_size: int = 1) -> list[str | None]:
    """Disjoint ordered GPU groups available to this process.

    Returns ``[None]`` when no GPUs are detected, meaning "run serially and let
    vLLM choose the device".
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() != "":
        ids = [d.strip() for d in cuda_visible.split(",") if d.strip() != ""]
        if not ids:
            return [None]
        if len(ids) % group_size:
            raise ValueError(
                f"{len(ids)} visible GPUs cannot be divided into groups of {group_size}"
            )
        return [",".join(ids[i : i + group_size]) for i in range(0, len(ids), group_size)]
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            ids = [str(i) for i in range(torch.cuda.device_count())]
            if len(ids) % group_size:
                raise ValueError(
                    f"{len(ids)} detected GPUs cannot be divided into groups of {group_size}"
                )
            return [
                ",".join(ids[i : i + group_size])
                for i in range(0, len(ids), group_size)
            ]
    except Exception:  # pragma: no cover - torch optional / no CUDA
        pass
    return [None]


def _merge_runtime_shard_results(
    ordered_items: list[tuple[tuple, Any]],
    *,
    status_dir: Path,
    shard_count: int,
    spec_identity: str,
) -> dict[tuple, RuntimeMeasurement]:
    """Merge one serialized latency result per deterministic runtime shard.

    Shards exchange only scalar benchmark results.  In particular, callers do
    not reconstruct other shards' temporary models just to hit the shared
    runtime cache after the barrier.
    """
    results_by_index: dict[int, RuntimeMeasurement] = {}
    for shard_index in range(shard_count):
        result_path = status_dir / f"shard_{shard_index:04d}.json"
        if not result_path.is_file():
            raise ValueError(f"missing runtime shard result: {result_path}")
        payload = json.loads(result_path.read_text())
        if payload.get("spec_identity") != spec_identity:
            raise ValueError(
                f"runtime shard {shard_index} has spec identity "
                f"{payload.get('spec_identity')!r}, expected {spec_identity!r}"
            )
        if payload.get("shard_index", shard_index) != shard_index:
            raise ValueError(f"runtime shard index mismatch in {result_path}")
        if payload.get("shard_count", shard_count) != shard_count:
            raise ValueError(f"runtime shard count mismatch in {result_path}")
        for raw_index, raw_measurement in payload.get("results", {}).items():
            index = int(raw_index)
            if not 0 <= index < len(ordered_items):
                raise ValueError(f"runtime result index {index} is out of range")
            if index in results_by_index:
                raise ValueError(f"duplicate runtime result index {index}")
            measurement = RuntimeMeasurement.from_dict(raw_measurement)
            if not all(
                math.isfinite(value)
                for value in (measurement.total_ms, measurement.prefill_ms)
            ):
                raise ValueError(
                    f"non-finite runtime result at index {index}: {measurement}"
                )
            results_by_index[index] = measurement

    expected_indices = set(range(len(ordered_items)))
    missing = sorted(expected_indices - results_by_index.keys())
    if missing:
        raise ValueError(f"missing runtime result indices: {missing}")
    return {
        ordered_items[index][0]: results_by_index[index]
        for index in range(len(ordered_items))
    }


def _runtime_shard_spec_identity(
    ordered_items: list[tuple[tuple, Any]],
    *,
    result_schema_version: int = _RUNTIME_SHARD_RESULT_SCHEMA_VERSION,
) -> str:
    """Identity for a distributed scalar-result barrier.

    The result schema is part of the directory identity so stale ``.done``
    markers from a total-latency-only run cannot release a newer additive
    phase-timing run before its shard JSON files exist.
    """

    payload = {
        "result_schema_version": int(result_schema_version),
        "spec_keys": [str(key) for key, _ in ordered_items],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _runtime_shard_results_complete(status_dir: Path, *, shard_count: int) -> bool:
    """Return whether every runtime shard has both payload and commit marker."""

    return all(
        (status_dir / f"shard_{index:04d}.json").is_file()
        and (status_dir / f"shard_{index:04d}.done").is_file()
        for index in range(int(shard_count))
    )


def _assigned_runtime_shard_indices(
    ordered_items: list[tuple[tuple, Any]],
    *,
    shard_count: int,
    shard_index: int,
    measurement_pairs: list[tuple[tuple, tuple]] | None = None,
) -> list[int]:
    """Assign complete paired measurements to one distributed runtime shard."""

    if measurement_pairs is None:
        return [
            index for index in range(len(ordered_items)) if index % shard_count == shard_index
        ]

    indices_by_key = {key: index for index, (key, _) in enumerate(ordered_items)}
    paired_indices: list[tuple[int, int]] = []
    assigned_indices: set[int] = set()
    for short_key, long_key in measurement_pairs:
        try:
            pair = (indices_by_key[short_key], indices_by_key[long_key])
        except KeyError as exc:
            raise ValueError("runtime measurement pair is missing a benchmark spec") from exc
        if pair[0] == pair[1] or assigned_indices.intersection(pair):
            raise ValueError("runtime measurement pairs must be disjoint")
        paired_indices.append(pair)
        assigned_indices.update(pair)

    expected_indices = set(range(len(ordered_items)))
    if assigned_indices != expected_indices:
        raise ValueError("runtime measurement pairs must cover every benchmark spec")
    return [
        index
        for pair_index, pair in enumerate(paired_indices)
        if pair_index % shard_count == shard_index
        for index in pair
    ]


def _run_benchmarks(
    specs: dict[tuple, tuple[RuntimeConfig, tuple[BlockConfig, ...]]],
    gpu_ids: list[str | None],
    cache_dir: Path | None,
    measurement_pairs: list[tuple[tuple, tuple]] | None = None,
) -> dict[tuple, RuntimeMeasurement]:
    """Benchmark each unique spec, fanning out across ``gpu_ids`` concurrently.

    Each concurrent task holds one GPU (taken from a queue) for the duration of
    its vLLM subprocess, so at most ``len(gpu_ids)`` benchmarks run at once and
    no two share a device. ``subprocess.run`` releases the GIL while the child
    runs, so threads give real parallelism here.
    """
    gpu_pool: "queue.Queue[str | None]" = queue.Queue()
    for gpu in gpu_ids:
        gpu_pool.put(gpu)

    def _work(
        item: tuple[tuple, tuple[RuntimeConfig, tuple[BlockConfig, ...]]],
    ) -> tuple[tuple, RuntimeMeasurement]:
        key, (rc, block_layout) = item
        gpu = gpu_pool.get()
        try:
            try:
                ms = _benchmark_spec(
                    rc,
                    block_layout,
                    gpu_id=gpu,
                    cache_dir=cache_dir,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"vLLM runtime benchmark failed on gpu={gpu} "
                    f"with block_layout={block_layout}"
                ) from exc
        finally:
            gpu_pool.put(gpu)
        return key, ms

    max_workers = max(1, len(gpu_ids))
    ordered_items = sorted(specs.items(), key=lambda item: str(item[0]))
    shard_count = int(os.environ.get("PUZZLETRON_RUNTIME_SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("PUZZLETRON_RUNTIME_SHARD_INDEX", "0"))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"invalid runtime shard {shard_index}/{shard_count}; expected 0 <= index < count"
        )
    assigned_indices = _assigned_runtime_shard_indices(
        ordered_items,
        shard_count=shard_count,
        shard_index=shard_index,
        measurement_pairs=measurement_pairs,
    )
    assigned = [ordered_items[index] for index in assigned_indices]

    status_dir = None
    spec_identity = None
    if shard_count > 1:
        if cache_dir is None:
            raise ValueError("multi-node runtime sharding requires a shared cache_dir")
        spec_identity = _runtime_shard_spec_identity(ordered_items)
        status_dir = Path(cache_dir) / "shards" / spec_identity
        if _runtime_shard_results_complete(status_dir, shard_count=shard_count):
            mprint(
                f"All {shard_count} runtime shard results already exist; "
                "merging without replaying vLLM benchmarks"
            )
            return _merge_runtime_shard_results(
                ordered_items,
                status_dir=status_dir,
                shard_count=shard_count,
                spec_identity=spec_identity,
            )

    def _execute(items, description):
        values: dict[tuple, RuntimeMeasurement] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for key, ms in tqdm(
                executor.map(_work, items),
                total=len(items),
                desc=description,
            ):
                values[key] = ms
        return values

    results = _execute(
        assigned,
        f"Benchmarking runtime shard {shard_index + 1}/{shard_count} "
        f"({len(assigned)}/{len(specs)} specs) on {max_workers} GPU group(s)",
    )
    if shard_count == 1:
        return results
    # Every worker publishes its scalar latencies for exactly this deterministic
    # spec set.  Once all workers finish, merge these files directly rather than
    # reconstructing every other shard's temporary model to replay cache hits.
    assert status_dir is not None and spec_identity is not None
    status_dir.mkdir(parents=True, exist_ok=True)
    result_path = status_dir / f"shard_{shard_index:04d}.json"
    result_temporary = result_path.with_suffix(f".json.{os.getpid()}.tmp")
    result_temporary.write_text(
        json.dumps(
            {
                "spec_identity": spec_identity,
                "result_schema_version": _RUNTIME_SHARD_RESULT_SCHEMA_VERSION,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "results": {
                    str(index): results[ordered_items[index][0]].to_dict()
                    for index in assigned_indices
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    result_temporary.replace(result_path)
    done_path = status_dir / f"shard_{shard_index:04d}.done"
    temporary = done_path.with_suffix(f".done.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"count": len(assigned), "pid": os.getpid()}) + "\n")
    temporary.replace(done_path)

    deadline = time.monotonic() + float(
        os.environ.get("PUZZLETRON_RUNTIME_SHARD_TIMEOUT_SECONDS", "13800")
    )
    last_report = 0.0
    while True:
        completed = len(list(status_dir.glob("shard_*.done")))
        if completed == shard_count:
            break
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"runtime shard barrier timed out: {completed}/{shard_count} complete in {status_dir}"
            )
        if now - last_report >= 30:
            mprint(
                f"Waiting for runtime shards: {completed}/{shard_count} complete "
                f"({status_dir})"
            )
            last_report = now
        time.sleep(2)

    return _merge_runtime_shard_results(
        ordered_items,
        status_dir=status_dir,
        shard_count=shard_count,
        spec_identity=spec_identity,
    )


def calc_runtime_for_blocks(
    block_config_set: set["BlockConfig"],
    runtime_stats_config: "DictConfig",
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    descriptor: "Type[ModelDescriptor]",
    lm_config: "Any",
    tokenizer_path: str,
    prefill_seq_len: int,
    generation_seq_len: int,
    batch_size: int,
    cache_dir: "Path | None" = None,
) -> "tuple[dict[BlockConfig, float], float]":
    """Benchmark each full decoder block (attention + FFN together) and return block runtimes.

    Unlike :func:`calc_runtime_for_subblocks` which times attention and FFN independently and sums
    them, this function times the full ``(attention + FFN)`` block in a single vLLM call. This is
    more accurate because it captures kernel-fusion and memory-bandwidth-reuse effects that the
    sum of independent measurements misses.

    Each unique :class:`~..block_config.BlockConfig` in ``block_config_set`` is benchmarked as a
    repeated-block model (same as the subblock path) and the per-block runtime is derived via the
    same differencing formula.

    Returns ``(runtime_by_block_dict, no_block_runtime_ms)`` analogous to
    :func:`calc_runtime_for_subblocks`.
    """
    configured_repeat_count = max(
        1, int(runtime_stats_config.get("repeat_block_n_times", 4))
    )
    topology = RuntimeTopology.from_config(runtime_stats_config.get("topology", None))
    repeat_block_n_times = effective_repeat_count(
        configured_repeat_count, topology.pipeline_parallel_size
    )

    runtime_config = RuntimeConfig(
        vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        descriptor,
        _freeze_config_fields(descriptor.runtime_benchmark_config_fields(lm_config)),
        tokenizer_path,
        repeat_block_n_times,
        prefill_seq_len,
        generation_seq_len,
        batch_size,
        runtime_stats_config.get("num_iters", 30),
        runtime_stats_config.get("num_warmup_iters", 10),
        _extra_vllm_args(runtime_stats_config),
        runtime_stats_config.get("max_num_seqs", None),
        topology,
        effective_repeat_count=repeat_block_n_times,
        vllm_env=_vllm_env(runtime_stats_config),
    )
    runtime_config.topology.validate_model_dimensions(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )
    base_block_config = descriptor.runtime_benchmark_base_block_config(runtime_config)

    specs: dict[tuple, tuple[RuntimeConfig, tuple[BlockConfig, ...]]] = {}

    def _add_spec(
        block_layout: tuple[BlockConfig, ...], scaffold_policy: str
    ) -> tuple:
        spec_runtime = replace(runtime_config, scaffold_policy=scaffold_policy)
        key = (spec_runtime, block_layout)
        specs.setdefault(key, (spec_runtime, block_layout))
        return key

    def _is_noop_block(bc) -> bool:
        # ``BlockConfig`` has no ``no_op`` field; a block is a no-op only when all present
        # subblocks are no-ops.
        return not any(not subblock.no_op for subblock in bc.subblock_configs)

    block_spec_keys: dict[BlockConfig, tuple[tuple, tuple]] = {}
    fixed_overhead_keys: list[tuple[tuple, tuple]] = []
    scaffold_required = False
    for block_config in sorted(block_config_set):
        if _is_noop_block(block_config):
            continue
        policy = getattr(
            descriptor,
            "runtime_benchmark_scaffold_policy",
            lambda _candidate: "none",
        )(block_config)
        if policy == "none":
            short_layout = homogeneous_layout(block_config, repeat_block_n_times)
            long_layout = homogeneous_layout(block_config, 2 * repeat_block_n_times)
            short_key = _add_spec(short_layout, policy)
            long_key = _add_spec(long_layout, policy)
            fixed_overhead_keys.append((short_key, long_key))
        elif policy == "attention_scaffold_per_pp_stage":
            scaffold_required = True
            short_layout = scaffolded_layout(
                block_config,
                base_block_config,
                repeat_block_n_times,
                topology.pipeline_parallel_size,
            )
            long_layout = scaffolded_layout(
                block_config,
                base_block_config,
                2 * repeat_block_n_times,
                topology.pipeline_parallel_size,
            )
            short_key = _add_spec(short_layout, policy)
            long_key = _add_spec(long_layout, policy)
        else:
            raise ValueError(f"unsupported runtime scaffold policy: {policy!r}")
        block_spec_keys[block_config] = (short_key, long_key)

    scaffold_overhead_keys = None
    if scaffold_required:
        scaffold_overhead_keys = (
            _add_spec(homogeneous_layout(base_block_config, repeat_block_n_times), "none"),
            _add_spec(
                homogeneous_layout(base_block_config, 2 * repeat_block_n_times),
                "none",
            ),
        )
    if not block_spec_keys:
        raise ValueError("runtime estimation requires at least one active block")

    gpu_ids = _resolve_gpu_ids(runtime_config.topology.gpu_group_size)
    mprint(
        f"Computing block-level runtime for {len(block_config_set)} block configs "
        f"({len(specs)} unique benchmarks) across {len(gpu_ids)} GPU(s)"
    )
    measurement_pairs = list(block_spec_keys.values())
    if scaffold_overhead_keys is not None:
        measurement_pairs.append(scaffold_overhead_keys)
    results = _run_benchmarks(specs, gpu_ids, cache_dir, measurement_pairs)

    runtime_by_block_dict: dict = {}
    for block_config in sorted(block_config_set):
        if _is_noop_block(block_config):
            runtime_by_block_dict[block_config] = RuntimeMeasurement.zero()
        else:
            short_key, long_key = block_spec_keys[block_config]
            runtime_by_block_dict[block_config] = candidate_slope(
                results[short_key], results[long_key], repeat_block_n_times
            )
            _validate_marginal_runtime(
                runtime_by_block_dict[block_config],
                label=json.dumps(block_config.to_dict(), sort_keys=True),
                ignore_negatives=bool(runtime_stats_config.get("ignore_negatives", False)),
            )

    if scaffold_overhead_keys is not None:
        fixed_overhead_keys.append(scaffold_overhead_keys)
    overhead_estimates = [
        fixed_intercept(results[short_key], results[long_key])
        for short_key, long_key in fixed_overhead_keys
    ]
    no_block_runtime_ms = median_measurement(overhead_estimates)

    return runtime_by_block_dict, no_block_runtime_ms


def calc_runtime_for_subblocks(
    subblock_config_set: set[SubblockConfig],
    runtime_stats_config: DictConfig,
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    descriptor: Type[ModelDescriptor],
    lm_config: Any,
    tokenizer_path: str,
    prefill_seq_len: int,
    generation_seq_len: int,
    batch_size: int,
    cache_dir: Path | None = None,
) -> tuple[dict[SubblockConfig, float], float]:
    """Benchmark each unique subblock and return per-subblock runtimes and no-block overhead.

    The distinct vLLM benchmarks are enumerated up front and run concurrently
    across all visible GPUs (with on-disk caching via ``cache_dir`` for resume),
    then the per-subblock runtimes are derived from the cached measurements using
    the same differencing the sequential version used.
    """
    configured_repeat_count = max(
        1, int(runtime_stats_config.get("repeat_block_n_times", 4))
    )
    topology = RuntimeTopology.from_config(runtime_stats_config.get("topology", None))
    repeat_block_n_times = effective_repeat_count(
        configured_repeat_count, topology.pipeline_parallel_size
    )

    runtime_config = RuntimeConfig(
        vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        descriptor,
        _freeze_config_fields(descriptor.runtime_benchmark_config_fields(lm_config)),
        tokenizer_path,
        repeat_block_n_times,
        prefill_seq_len,
        generation_seq_len,
        batch_size,
        runtime_stats_config.get("num_iters", 30),
        runtime_stats_config.get("num_warmup_iters", 10),
        _extra_vllm_args(runtime_stats_config),
        runtime_stats_config.get("max_num_seqs", None),
        topology,
        effective_repeat_count=repeat_block_n_times,
        vllm_env=_vllm_env(runtime_stats_config),
    )
    runtime_config.topology.validate_model_dimensions(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )
    base_block_config = descriptor.runtime_benchmark_base_block_config(runtime_config)
    specs: dict[tuple, tuple[RuntimeConfig, tuple[BlockConfig, ...]]] = {}

    def _add_spec(
        block_layout: tuple[BlockConfig, ...], scaffold_policy: str
    ) -> tuple:
        spec_runtime = replace(runtime_config, scaffold_policy=scaffold_policy)
        key = (spec_runtime, block_layout)
        specs.setdefault(key, (spec_runtime, block_layout))
        return key

    def _scaffold_policy(candidate: BlockConfig) -> str:
        policy = getattr(
            descriptor,
            "runtime_benchmark_scaffold_policy",
            lambda _candidate: "none",
        )(candidate)
        if policy not in {"none", "attention_scaffold_per_pp_stage"}:
            raise ValueError(f"unsupported runtime scaffold policy: {policy!r}")
        return policy

    subblock_spec_keys: dict[SubblockConfig, tuple[tuple, tuple]] = {}
    fixed_overhead_keys: list[tuple[tuple, tuple]] = []
    scaffold_required = False
    for subblock_config in sorted(subblock_config_set):
        if subblock_config.no_op:
            continue
        candidate = _block_config_for_subblock(runtime_config, subblock_config)
        assert candidate is not None
        policy = _scaffold_policy(candidate)
        if policy == "none":
            short_layout = homogeneous_layout(candidate, repeat_block_n_times)
            long_layout = homogeneous_layout(candidate, 2 * repeat_block_n_times)
            short_key = _add_spec(short_layout, policy)
            long_key = _add_spec(long_layout, policy)
            fixed_overhead_keys.append((short_key, long_key))
        else:
            scaffold_required = True
            short_layout = scaffolded_layout(
                candidate,
                base_block_config,
                repeat_block_n_times,
                topology.pipeline_parallel_size,
            )
            long_layout = scaffolded_layout(
                candidate,
                base_block_config,
                2 * repeat_block_n_times,
                topology.pipeline_parallel_size,
            )
            short_key = _add_spec(short_layout, policy)
            long_key = _add_spec(long_layout, policy)
        subblock_spec_keys[subblock_config] = (short_key, long_key)

    scaffold_overhead_keys = None
    if scaffold_required:
        scaffold_short = homogeneous_layout(base_block_config, repeat_block_n_times)
        scaffold_long = homogeneous_layout(base_block_config, 2 * repeat_block_n_times)
        scaffold_overhead_keys = (
            _add_spec(scaffold_short, "none"),
            _add_spec(scaffold_long, "none"),
        )
    if not subblock_spec_keys:
        raise ValueError("runtime estimation requires at least one active subblock")

    # ---- Run all benchmarks (parallel across GPUs, cached/resumable) ----
    gpu_ids = _resolve_gpu_ids(runtime_config.topology.gpu_group_size)
    mprint(
        f"Computing runtime for {len(subblock_config_set)} subblocks "
        f"({len(specs)} unique benchmarks) across {len(gpu_ids)} GPU(s)"
    )
    measurement_pairs = list(subblock_spec_keys.values())
    if scaffold_overhead_keys is not None:
        measurement_pairs.append(scaffold_overhead_keys)
    results = _run_benchmarks(specs, gpu_ids, cache_dir, measurement_pairs)

    runtime_by_subblock_dict = {}
    for subblock_config in sorted(subblock_config_set):
        if not isinstance(
            subblock_config,
            AttentionConfig | MLAConfig | MambaConfig | FFNConfig | MoEConfig,
        ):
            raise ValueError(f"Unsupported subblock type: {type(subblock_config)}")
        if subblock_config.no_op:
            total_runtime_ms = RuntimeMeasurement.zero()
        else:
            short_key, long_key = subblock_spec_keys[subblock_config]
            total_runtime_ms = candidate_slope(
                results[short_key], results[long_key], repeat_block_n_times
            )
            _validate_marginal_runtime(
                total_runtime_ms,
                label=json.dumps(subblock_config.to_dict(), sort_keys=True),
                ignore_negatives=bool(runtime_stats_config.get("ignore_negatives", False)),
            )

        runtime_by_subblock_dict[subblock_config] = total_runtime_ms

    if scaffold_overhead_keys is not None:
        fixed_overhead_keys.append(scaffold_overhead_keys)
    overhead_estimates = [
        fixed_intercept(results[short_key], results[long_key])
        for short_key, long_key in fixed_overhead_keys
    ]
    no_block_runtime_ms = median_measurement(overhead_estimates)
    relative_tolerance = float(
        runtime_stats_config.get("fixed_overhead_relative_tolerance", 0.5)
    )
    denominator = max(abs(no_block_runtime_ms.total_ms), 1.0e-12)
    relative_spread = max(
        abs(value.total_ms - no_block_runtime_ms.total_ms) / denominator
        for value in overhead_estimates
    )
    if relative_spread > relative_tolerance:
        raise ValueError(
            "fixed runtime overhead estimates exceed relative tolerance: "
            f"spread={relative_spread:.6g} tolerance={relative_tolerance:.6g}"
        )

    return runtime_by_subblock_dict, no_block_runtime_ms
