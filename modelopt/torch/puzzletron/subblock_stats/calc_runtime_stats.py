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

import os
import queue
import tempfile
import hashlib
import json
import math
import time
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
    MLAConfig,
    MambaConfig,
    MoEConfig,
    SubblockConfig,
    maybe_cast_block_configs,
)
from ..candidates import build_candidate_library
from ..tools.checkpoint_utils import load_model_config
from ..tools.logger import mprint
from .runtime_utils import RuntimeConfig, save_model
from .runtime_vllm import RuntimeMeasurement, run_vllm_latency_benchmark
from .topology import RuntimeTopology

_ATTENTION_LIKE_KINDS = frozenset(("attention", "mla", "mamba"))
_FFN_LIKE_KINDS = frozenset(("ffn", "moe"))
_RUNTIME_SHARD_RESULT_SCHEMA_VERSION = 3


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


def create_benchmark_model(
    runtime_config: RuntimeConfig,
    block_config: BlockConfig | None,
    trailing_base_block: bool = False,
):
    """Build a small descriptor-specific model with repeated subblocks."""
    block_configs = [runtime_config.descriptor.runtime_benchmark_base_block_config(runtime_config)]
    if block_config:
        block_configs.extend([block_config] * runtime_config.repeat_block_n_times)
    if trailing_base_block:
        block_configs.append(
            runtime_config.descriptor.runtime_benchmark_base_block_config(runtime_config)
        )

    return runtime_config.descriptor.create_runtime_benchmark_model(runtime_config, block_configs)


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
    if isinstance(subblock_config, FFNConfig | MoEConfig):
        base_block_config = runtime_config.descriptor.runtime_benchmark_base_block_config(
            runtime_config
        )
        return base_block_config.with_subblock(
            subblock_config, replace_kinds=_FFN_LIKE_KINDS
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


def _benchmark_spec(
    runtime_config: RuntimeConfig,
    block_config: BlockConfig | None,
    gpu_id: str | int | None,
    cache_dir: Path | None,
    trailing_base_block: bool = False,
) -> RuntimeMeasurement:
    """Build the repeated-block model for a spec and measure its total latency (ms)."""
    model = create_benchmark_model(
        runtime_config,
        block_config=block_config,
        trailing_base_block=trailing_base_block,
    )
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


def _run_benchmarks(
    specs: dict[tuple, tuple[RuntimeConfig, BlockConfig | None, bool]],
    gpu_ids: list[str | None],
    cache_dir: Path | None,
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
        item: tuple[tuple, tuple[RuntimeConfig, BlockConfig | None, bool]],
    ) -> tuple[tuple, RuntimeMeasurement]:
        key, (rc, block_config, trailing_base_block) = item
        gpu = gpu_pool.get()
        try:
            try:
                ms = _benchmark_spec(
                    rc,
                    block_config,
                    gpu_id=gpu,
                    cache_dir=cache_dir,
                    trailing_base_block=trailing_base_block,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"vLLM runtime benchmark failed on gpu={gpu} "
                    f"with block_config={block_config}"
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
    assigned_indices = [
        index for index in range(len(ordered_items)) if index % shard_count == shard_index
    ]
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
    repeat_block_n_times = max(3, int(runtime_stats_config.get("repeat_block_n_times", 10)))

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
        RuntimeTopology.from_config(runtime_stats_config.get("topology", None)),
    )
    runtime_config_fewer = replace(runtime_config, repeat_block_n_times=repeat_block_n_times - 1)
    runtime_config.topology.validate_model_dimensions(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )
    base_block_config = descriptor.runtime_benchmark_base_block_config(runtime_config)

    specs: dict[tuple, tuple] = {}

    def _add_spec(rc, block_config, trailing_base_block=False):
        key = (rc, block_config, trailing_base_block)
        specs.setdefault(key, (rc, block_config, trailing_base_block))
        return key

    # A one-layer model leaves one PP stage empty when PP=2, which the hybrid
    # cache planner cannot represent.  Measure two cache-bearing base layers
    # instead, then recover the one-layer intercept from the repeated-base
    # measurement below.  Both anchors therefore use the production topology.
    two_base_key = _add_spec(runtime_config, None, trailing_base_block=True)
    ten_block_key = _add_spec(runtime_config_fewer, base_block_config)

    def _is_noop_block(bc) -> bool:
        # ``BlockConfig`` has no ``no_op`` field; a block is a no-op only when all present
        # subblocks are no-ops.
        return not any(not subblock.no_op for subblock in bc.subblock_configs)

    block_spec_keys: dict = {}
    for block_config in block_config_set:
        if not _is_noop_block(block_config):
            needs_pp_cache_anchor = any(
                subblock.no_op and subblock.kind in _ATTENTION_LIKE_KINDS
                for subblock in block_config.subblock_configs
            )
            block_spec_keys[block_config] = _add_spec(
                runtime_config,
                block_config,
                trailing_base_block=needs_pp_cache_anchor,
            )

    gpu_ids = _resolve_gpu_ids(runtime_config.topology.gpu_group_size)
    mprint(
        f"Computing block-level runtime for {len(block_config_set)} block configs "
        f"({len(specs)} unique benchmarks) across {len(gpu_ids)} GPU(s)"
    )
    results = _run_benchmarks(specs, gpu_ids, cache_dir)

    runtime_ms_two_base_blocks = results[two_base_key]
    runtime_ms_ten_blocks = results[ten_block_key]
    runtime_ms_per_base_block = (
        runtime_ms_ten_blocks - runtime_ms_two_base_blocks
    ) / (repeat_block_n_times - 2)
    runtime_ms_one_block = runtime_ms_two_base_blocks - runtime_ms_per_base_block

    runtime_by_block_dict: dict = {}
    for block_config in block_config_set:
        if _is_noop_block(block_config):
            runtime_by_block_dict[block_config] = RuntimeMeasurement.zero()
        else:
            block_total_ms = results[block_spec_keys[block_config]]
            needs_pp_cache_anchor = any(
                subblock.no_op and subblock.kind in _ATTENTION_LIKE_KINDS
                for subblock in block_config.subblock_configs
            )
            baseline_ms = (
                runtime_ms_two_base_blocks
                if needs_pp_cache_anchor
                else runtime_ms_one_block
            )
            runtime_by_block_dict[block_config] = (
                block_total_ms - baseline_ms
            ) / repeat_block_n_times

    no_block_runtime_ms = runtime_ms_one_block - runtime_ms_per_base_block

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
    repeat_block_n_times = max(2, int(runtime_stats_config.get("repeat_block_n_times", 10)))

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
        RuntimeTopology.from_config(runtime_stats_config.get("topology", None)),
    )
    # Config with one fewer repeat, used only for the no-block overhead estimate.
    runtime_config_fewer = replace(runtime_config, repeat_block_n_times=repeat_block_n_times - 1)
    runtime_config.topology.validate_model_dimensions(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )
    base_block_config = descriptor.runtime_benchmark_base_block_config(runtime_config)
    base_attention_block = _mark_noop_kinds(base_block_config, _FFN_LIKE_KINDS)

    # ---- Enumerate the distinct (runtime_config, block_config) benchmarks ----
    # A spec is uniquely identified by (runtime_config, block_config); the same
    # spec requested twice (e.g. a baseline shared by every FFN subblock) is
    # benchmarked once.
    specs: dict[tuple, tuple[RuntimeConfig, BlockConfig | None, bool]] = {}

    def _add_spec(
        rc: RuntimeConfig,
        block_config: BlockConfig | None,
        trailing_base_block: bool = False,
    ) -> tuple:
        key = (rc, block_config, trailing_base_block)
        specs.setdefault(key, (rc, block_config, trailing_base_block))
        return key

    base_key = _add_spec(runtime_config, None)  # 1 base block (attn baseline + no-block)
    ten_block_key = _add_spec(runtime_config_fewer, base_block_config)  # base + 9 base blocks
    ffn_baseline_key = _add_spec(runtime_config, base_attention_block)  # base + 10 attn-only blocks

    subblock_spec_keys: dict[SubblockConfig, tuple] = {}
    for subblock_config in subblock_config_set:
        if not subblock_config.no_op:
            subblock_spec_keys[subblock_config] = _add_spec(
                runtime_config, _block_config_for_subblock(runtime_config, subblock_config)
            )

    # ---- Run all benchmarks (parallel across GPUs, cached/resumable) ----
    gpu_ids = _resolve_gpu_ids(runtime_config.topology.gpu_group_size)
    mprint(
        f"Computing runtime for {len(subblock_config_set)} subblocks "
        f"({len(specs)} unique benchmarks) across {len(gpu_ids)} GPU(s)"
    )
    results = _run_benchmarks(specs, gpu_ids, cache_dir)

    # ---- Derive per-subblock runtimes from the measured totals ----
    runtime_by_subblock_dict = {}
    for subblock_config in sorted(subblock_config_set):
        if isinstance(subblock_config, AttentionConfig | MLAConfig | MambaConfig):
            baseline_runtime_ms = results[base_key]
        elif isinstance(subblock_config, FFNConfig | MoEConfig):
            baseline_runtime_ms = results[ffn_baseline_key]
        else:
            raise ValueError(f"Unsupported subblock type: {type(subblock_config)}")

        if subblock_config.no_op:
            total_runtime_ms = RuntimeMeasurement.zero()
        else:
            subblock_total_runtime_ms = results[subblock_spec_keys[subblock_config]]
            total_runtime_ms = (
                subblock_total_runtime_ms - baseline_runtime_ms
            ) / repeat_block_n_times

        runtime_by_subblock_dict[subblock_config] = total_runtime_ms

    # No-block overhead (embedding + LM head): extrapolate from 1- and 10-block models.
    runtime_ms_one_block = results[base_key]
    runtime_ms_ten_blocks = results[ten_block_key]
    no_block_runtime_ms = runtime_ms_one_block - (
        runtime_ms_ten_blocks - runtime_ms_one_block
    ) / (repeat_block_n_times - 1)

    return runtime_by_subblock_dict, no_block_runtime_ms
