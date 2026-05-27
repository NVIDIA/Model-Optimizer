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

from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from ..anymodel.models.llama import LlamaModelDescriptor
from ..anymodel.puzzformer import deci_x_patcher
from ..block_config import AttentionConfig, BlockConfig, FFNConfig, SubblockConfig
from .runtime_utils import RuntimeConfig, save_model
from .runtime_vllm import run_vllm_latency_benchmark


def _make_standard_block_config(num_key_value_heads: int) -> BlockConfig:
    return BlockConfig(
        attention=AttentionConfig(no_op=False, num_key_value_heads=num_key_value_heads),
        ffn=FFNConfig(no_op=False, intermediate_size=256, moe=None),
    )


def create_benchmark_model(
    vocab_size: int,
    hidden_size: int,
    num_key_value_heads: int,
    num_attention_heads: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    block_config: BlockConfig | None,
    repeat_block_n_times: int = 10,
) -> LlamaForCausalLM:
    """Build a small Llama model with repeated subblocks for latency benchmarking."""
    block_configs = [_make_standard_block_config(num_key_value_heads)]

    if block_config:
        block_configs.extend([block_config] * repeat_block_n_times)

    model_config = LlamaConfig(
        max_position_embeddings=prefill_seq_len + generation_seq_len,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=len(block_configs),
        head_dim=None,  # Compute from hidden_size // num_attention_heads instead of using default 128
        # this is required for trt-llm convertion to know which model classes to use to the checkpoint
        auto_map={
            "AutoConfig": "transformers.models.llama.configuration_llama.LlamaConfig",
            "AutoModelForCausalLM": "transformers.models.llama.modeling_llama.LlamaForCausalLM",
        },
    )

    for idx, bc in enumerate(block_configs):
        block_configs[idx] = bc.to_dict()
    model_config.block_configs = block_configs

    with deci_x_patcher(LlamaModelDescriptor, block_configs):
        model = AutoModelForCausalLM.from_config(model_config)

    model.config.architectures = ["AnyModel"]
    model.config.base_architecture = "LlamaForCausalLM"

    return model


def calc_model_runtime(model: LlamaForCausalLM, runtime_config: RuntimeConfig) -> float:
    """Measure total runtime of a model via vLLM latency benchmark."""
    with tempfile.TemporaryDirectory() as model_tmpdir:
        save_model(model, Path(runtime_config.tokenizer_path), Path(model_tmpdir))
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
                block_config = BlockConfig(
                    attention=AttentionConfig(
                        no_op=False, num_key_value_heads=runtime_config.num_key_value_heads
                    ),
                    ffn=subblock_config,
                )
            else:
                block_config = subblock_config.to_blockconfig()
        else:
            raise Exception(f"Runtime stats: Not supported subblock type: {subblock_config}")

    model = create_benchmark_model(
        runtime_config.vocab_size,
        runtime_config.hidden_size,
        runtime_config.num_key_value_heads,
        runtime_config.num_attention_heads,
        runtime_config.prefill_seq_len,
        runtime_config.generation_seq_len,
        block_config=block_config,
        repeat_block_n_times=runtime_config.repeat_block_n_times,
    )
    return calc_model_runtime(model, runtime_config)


@cache
def calc_base_runtime(runtime_config: RuntimeConfig, subblock_config: SubblockConfig) -> float:
    """Calculate the base runtime of a model with no subblocks."""
    base_runtime_ms = None
    if isinstance(subblock_config, AttentionConfig):
        base_runtime_ms = calc_subblock_runtime(runtime_config, None)
    elif isinstance(subblock_config, FFNConfig):
        attn_block_config = AttentionConfig(
            no_op=False, num_key_value_heads=runtime_config.num_key_value_heads
        ).to_blockconfig()
        base_runtime_ms = calc_subblock_runtime(runtime_config, attn_block_config)
    else:
        raise ValueError(f"Unsupported subblock type: {type(subblock_config)}")

    return base_runtime_ms


@cache
def calc_no_block_runtime(runtime_config: RuntimeConfig) -> float:
    """Estimate the overhead runtime (embedding + LM head) with no decoder blocks."""
    runtime_cfg_ten_blocks = replace(runtime_config, repeat_block_n_times=9)

    block_config = _make_standard_block_config(runtime_config.num_key_value_heads)

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
    tokenizer_path: str,
    prefill_seq_len: int,
    generation_seq_len: int,
) -> tuple[dict[SubblockConfig, float], float]:
    """Benchmark each unique subblock and return per-subblock runtimes and no-block overhead."""
    repeat_block_n_times = 10
    runtime_config = RuntimeConfig(
        vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        tokenizer_path,
        repeat_block_n_times,
        prefill_seq_len,
        generation_seq_len,
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
