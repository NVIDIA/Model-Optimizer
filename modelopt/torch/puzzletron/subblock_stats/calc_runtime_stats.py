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

import tempfile
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any, Type

from omegaconf import DictConfig
from tqdm import tqdm

from ..anymodel.model_descriptor import ModelDescriptor
from ..block_config import AttentionConfig, BlockConfig, FFNConfig, SubblockConfig
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


def calc_model_runtime(model, runtime_config: RuntimeConfig) -> float:
    """Measure total runtime of a model via vLLM latency benchmark."""
    with tempfile.TemporaryDirectory() as model_tmpdir:
        save_model(
            model,
            Path(runtime_config.tokenizer_path),
            Path(model_tmpdir),
            runtime_config.descriptor,
        )
        model_total_runtime_ms = run_vllm_latency_benchmark(Path(model_tmpdir), runtime_config)
    return model_total_runtime_ms


@cache
def calc_subblock_runtime(
    runtime_config: RuntimeConfig,
    subblock_config: SubblockConfig | None,
) -> float:
    """Measure total runtime of a repeated subblock via vLLM latency benchmark."""
    block_config: BlockConfig | None = None

    if subblock_config is not None:
        if isinstance(subblock_config, BlockConfig):
            block_config = subblock_config
        elif isinstance(subblock_config, (AttentionConfig, FFNConfig)):
            if isinstance(subblock_config, FFNConfig):
                base_block_config = runtime_config.descriptor.runtime_benchmark_base_block_config(
                    runtime_config
                )
                block_config = BlockConfig(
                    attention=base_block_config.attention,
                    ffn=subblock_config,
                )
            else:
                block_config = subblock_config.to_blockconfig()
        else:
            raise Exception(f"Runtime stats: Not supported subblock type: {subblock_config}")

    model = create_benchmark_model(runtime_config, block_config=block_config)
    return calc_model_runtime(model, runtime_config)


@cache
def calc_base_runtime(runtime_config: RuntimeConfig, subblock_config: SubblockConfig) -> float:
    """Calculate the base runtime of a model with no subblocks."""
    base_runtime_ms = None
    if isinstance(subblock_config, AttentionConfig):
        base_runtime_ms = calc_subblock_runtime(runtime_config, None)
    elif isinstance(subblock_config, FFNConfig):
        base_block_config = runtime_config.descriptor.runtime_benchmark_base_block_config(
            runtime_config
        )
        base_runtime_ms = calc_subblock_runtime(
            runtime_config, base_block_config.attention.to_blockconfig()
        )
    else:
        raise ValueError(f"Unsupported subblock type: {type(subblock_config)}")

    return base_runtime_ms


@cache
def calc_no_block_runtime(runtime_config: RuntimeConfig) -> float:
    """Estimate the overhead runtime (embedding + LM head) with no decoder blocks."""
    runtime_cfg_ten_blocks = replace(runtime_config, repeat_block_n_times=9)

    block_config = runtime_config.descriptor.runtime_benchmark_base_block_config(runtime_config)

    runtime_ms_one_block = calc_subblock_runtime(runtime_config, None)  # only one base block
    runtime_ms_ten_blocks = calc_subblock_runtime(
        runtime_cfg_ten_blocks, block_config
    )  # one base block + 9 repeated blocks

    no_block_runtime_ms = runtime_ms_one_block - (runtime_ms_ten_blocks - runtime_ms_one_block) / 9

    return no_block_runtime_ms


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
) -> tuple[dict[SubblockConfig, float], float]:
    """Benchmark each unique subblock and return per-subblock runtimes and no-block overhead."""
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
    )

    runtime_by_subblock_dict = {}

    for subblock_config in tqdm(
        sorted(subblock_config_set),
        desc=(f"Computing runtime for {len(subblock_config_set)} subblocks\n"),
    ):
        baseline_runtime_ms = calc_base_runtime(runtime_config, subblock_config)

        if subblock_config.no_op:
            total_runtime_ms = 0.0
        else:
            subblock_total_runtime_ms = calc_subblock_runtime(runtime_config, subblock_config)
            total_runtime_ms = (
                subblock_total_runtime_ms - baseline_runtime_ms
            ) / repeat_block_n_times

        runtime_by_subblock_dict[subblock_config] = total_runtime_ms

    no_block_runtime_ms = calc_no_block_runtime(runtime_config)

    return runtime_by_subblock_dict, no_block_runtime_ms
