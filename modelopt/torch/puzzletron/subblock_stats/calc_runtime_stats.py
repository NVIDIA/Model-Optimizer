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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Type

from omegaconf import DictConfig
from tqdm import tqdm

from ..anymodel.model_descriptor import ModelDescriptor
from ..block_config import AttentionConfig, BlockConfig, FFNConfig, SubblockConfig
from ..tools.logger import mprint
from .runtime_utils import RuntimeConfig, save_model
from .runtime_vllm import run_vllm_latency_benchmark


def _freeze_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_config_value(val)) for key, val in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _freeze_config_fields(fields: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((key, _freeze_config_value(value)) for key, value in fields.items()))


def create_benchmark_model(
    runtime_config: RuntimeConfig,
    block_config: BlockConfig | None,
):
    """Build a small descriptor-specific model with repeated subblocks."""
    block_configs = [runtime_config.descriptor.runtime_benchmark_base_block_config(runtime_config)]
    if block_config:
        block_configs.extend([block_config] * runtime_config.repeat_block_n_times)

    return runtime_config.descriptor.create_runtime_benchmark_model(runtime_config, block_configs)


def _block_config_for_subblock(
    runtime_config: RuntimeConfig, subblock_config: SubblockConfig | None
) -> BlockConfig | None:
    """Map a subblock to the repeated ``BlockConfig`` used to benchmark it.

    ``None`` means "no repeated block" (i.e. the base block only).
    """
    if subblock_config is None:
        return None
    if isinstance(subblock_config, BlockConfig):
        return subblock_config
    if isinstance(subblock_config, FFNConfig):
        base_block_config = runtime_config.descriptor.runtime_benchmark_base_block_config(
            runtime_config
        )
        return BlockConfig(attention=base_block_config.attention, ffn=subblock_config)
    if isinstance(subblock_config, AttentionConfig):
        return subblock_config.to_blockconfig()
    raise Exception(f"Runtime stats: Not supported subblock type: {subblock_config}")


def _benchmark_spec(
    runtime_config: RuntimeConfig,
    block_config: BlockConfig | None,
    gpu_id: str | int | None,
    cache_dir: Path | None,
) -> float:
    """Build the repeated-block model for a spec and measure its total latency (ms)."""
    model = create_benchmark_model(runtime_config, block_config=block_config)
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


def _resolve_gpu_ids() -> list[str | None]:
    """Physical GPU ids available to this process (honoring CUDA_VISIBLE_DEVICES).

    Returns ``[None]`` when no GPUs are detected, meaning "run serially and let
    vLLM choose the device".
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() != "":
        ids = [d.strip() for d in cuda_visible.split(",") if d.strip() != ""]
        return ids or [None]
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:  # pragma: no cover - torch optional / no CUDA
        pass
    return [None]


def _run_benchmarks(
    specs: dict[tuple, tuple[RuntimeConfig, BlockConfig | None]],
    gpu_ids: list[str | None],
    cache_dir: Path | None,
) -> dict[tuple, float]:
    """Benchmark each unique spec, fanning out across ``gpu_ids`` concurrently.

    Each concurrent task holds one GPU (taken from a queue) for the duration of
    its vLLM subprocess, so at most ``len(gpu_ids)`` benchmarks run at once and
    no two share a device. ``subprocess.run`` releases the GIL while the child
    runs, so threads give real parallelism here.
    """
    gpu_pool: "queue.Queue[str | None]" = queue.Queue()
    for gpu in gpu_ids:
        gpu_pool.put(gpu)

    def _work(item: tuple[tuple, tuple[RuntimeConfig, BlockConfig | None]]) -> tuple[tuple, float]:
        key, (rc, block_config) = item
        gpu = gpu_pool.get()
        try:
            ms = _benchmark_spec(rc, block_config, gpu_id=gpu, cache_dir=cache_dir)
        finally:
            gpu_pool.put(gpu)
        return key, ms

    max_workers = max(1, len(gpu_ids))
    results: dict[tuple, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, ms in tqdm(
            executor.map(_work, list(specs.items())),
            total=len(specs),
            desc=f"Benchmarking {len(specs)} subblock models on {max_workers} GPU(s)",
        ):
            results[key] = ms
    return results


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
    repeat_block_n_times = 10

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
        runtime_stats_config.get("max_num_seqs", None),
    )
    # Config with one fewer repeat, used only for the no-block overhead estimate.
    runtime_config_fewer = replace(runtime_config, repeat_block_n_times=repeat_block_n_times - 1)
    base_block_config = descriptor.runtime_benchmark_base_block_config(runtime_config)
    base_attention_block = base_block_config.attention.to_blockconfig()

    # ---- Enumerate the distinct (runtime_config, block_config) benchmarks ----
    # A spec is uniquely identified by (runtime_config, block_config); the same
    # spec requested twice (e.g. a baseline shared by every FFN subblock) is
    # benchmarked once.
    specs: dict[tuple, tuple[RuntimeConfig, BlockConfig | None]] = {}

    def _add_spec(rc: RuntimeConfig, block_config: BlockConfig | None) -> tuple:
        key = (rc, block_config)
        specs.setdefault(key, (rc, block_config))
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
    gpu_ids = _resolve_gpu_ids()
    mprint(
        f"Computing runtime for {len(subblock_config_set)} subblocks "
        f"({len(specs)} unique benchmarks) across {len(gpu_ids)} GPU(s)"
    )
    results = _run_benchmarks(specs, gpu_ids, cache_dir)

    # ---- Derive per-subblock runtimes from the measured totals ----
    runtime_by_subblock_dict = {}
    for subblock_config in sorted(subblock_config_set):
        if isinstance(subblock_config, AttentionConfig):
            baseline_runtime_ms = results[base_key]
        elif isinstance(subblock_config, FFNConfig):
            baseline_runtime_ms = results[ffn_baseline_key]
        else:
            raise ValueError(f"Unsupported subblock type: {type(subblock_config)}")

        if subblock_config.no_op:
            total_runtime_ms = 0.0
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
