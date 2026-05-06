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
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from modelopt.torch.nas.subblock_stats.runtime_utils import RuntimeConfig, save_model
from modelopt.torch.nas.subblock_stats.runtime_vllm import run_vllm_latency_benchmark
from modelopt.torch.puzzletron.anymodel.models.llama import LlamaModelDescriptor
from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher
from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    SubblockConfig,
)


def _make_standard_block_config(hidden_size: int, num_attention_heads: int) -> BlockConfig:
    return BlockConfig(
        attention=AttentionConfig(no_op=False, num_key_value_heads=num_attention_heads),
        ffn=FFNConfig(no_op=False, intermediate_size=hidden_size, moe=None),
        parallel_blocks=None,
    )


def create_benchmark_model(
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    block_config: BlockConfig | None,
    repeat_block_n_times: int = 10,
) -> LlamaForCausalLM:
    """Build a small Llama model with repeated subblocks for latency benchmarking."""
    block_configs = [_make_standard_block_config(hidden_size, num_attention_heads)]

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
        save_model(
            model,
            Path(runtime_config.tokenizer_path),
            Path(model_tmpdir),
            num_hidden_layers=runtime_config.repeat_block_n_times + 1,
        )
        model_total_runtime_ms = run_vllm_latency_benchmark(Path(model_tmpdir), runtime_config)
    return model_total_runtime_ms


def calc_subblock_runtime(
    runtime_config: RuntimeConfig,
    subblock_config: SubblockConfig,
) -> float:
    """Measure total runtime of a repeated subblock via vLLM latency benchmark."""
    block_config: BlockConfig | None = None

    if subblock_config is not None:
        if isinstance(subblock_config, BlockConfig):
            block_config = subblock_config
        elif isinstance(subblock_config, (AttentionConfig, FFNConfig)):
            block_config = subblock_config.to_blockconfig()
        else:
            raise Exception(f"Runtime stats: Not supported subblock type: {subblock_config}")

    model = create_benchmark_model(
        runtime_config.vocab_size,
        runtime_config.hidden_size,
        runtime_config.num_attention_heads,
        runtime_config.prefill_seq_len,
        runtime_config.generation_seq_len,
        block_config=block_config,
        repeat_block_n_times=runtime_config.repeat_block_n_times,
    )
    return calc_model_runtime(model, runtime_config)


def calc_no_block_runtime(runtime_config: RuntimeConfig) -> float:
    """Estimate the overhead runtime (embedding + LM head) with no decoder blocks."""
    runtime_config1 = replace(runtime_config, repeat_block_n_times=0)
    runtime_config10 = replace(runtime_config, repeat_block_n_times=9)

    block_config = _make_standard_block_config(
        runtime_config.hidden_size, runtime_config.num_attention_heads
    )

    runtime_ms1 = calc_subblock_runtime(runtime_config1, None)
    runtime_ms10 = calc_subblock_runtime(runtime_config10, block_config)

    no_block_runtime_ms = runtime_ms1 - (runtime_ms10 - runtime_ms1) / 9

    return no_block_runtime_ms


def calc_runtime_for_subblocks(
    subblock_config_set: set[SubblockConfig],
    runtime_stats_config: DictConfig,
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    master_puzzle_dir: str,
    tokenizer_path: str,
    synth_dataset_num_requests: int,
    prefill_seq_len: int,
    generation_seq_len: int,
) -> tuple[dict[SubblockConfig, float], float]:
    """Benchmark each unique subblock and return per-subblock runtimes and no-block overhead."""
    repeat_block_n_times = 10
    runtime_config = RuntimeConfig(
        vocab_size,
        hidden_size,
        num_attention_heads,
        master_puzzle_dir,
        tokenizer_path,
        synth_dataset_num_requests,
        repeat_block_n_times,
        prefill_seq_len,
        generation_seq_len,
        runtime_stats_config.get("batch_size", 1),
        runtime_stats_config.get("num_iters", 30),
        runtime_stats_config.get("num_warmup_iters", 10),
    )

    runtime_by_subblock_dict = {}

    baseline_runtime_ms = calc_subblock_runtime(runtime_config, subblock_config=None)

    for subblock_config in tqdm(
        sorted(subblock_config_set),
        desc=(f"Computing runtime for {len(subblock_config_set)} subblocks\n"),
    ):
        if isinstance(subblock_config, AttentionConfig):
            num_key_value_heads = subblock_config.num_key_value_heads
            desc = f"AttentionConfig(num_key_value_heads={num_key_value_heads})"
        elif isinstance(subblock_config, FFNConfig):
            intermediate_size = subblock_config.intermediate_size
            desc = f"FFNConfig(intermediate_size={intermediate_size})"
        else:
            raise ValueError(f"Unsupported subblock type: {type(subblock_config)}")
        print(f"Computing runtime for subblock: {desc} {subblock_config.no_op=}")

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
