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

import copy
import tempfile
from dataclasses import replace
from functools import cache
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PretrainedConfig

from ..anymodel.model_descriptor import ModelDescriptor
from ..anymodel.models.llama import LlamaModelDescriptor
from ..anymodel.puzzformer import deci_x_patcher
from ..block_config import AttentionConfig, BlockConfig, FFNConfig, SubblockConfig
from ..tools.checkpoint_utils_hf import init_model_from_config
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


def _uses_moe(subblock_config: SubblockConfig | BlockConfig | None) -> bool:
    if subblock_config is None:
        return False
    if isinstance(subblock_config, FFNConfig):
        return subblock_config.is_moe
    if isinstance(subblock_config, BlockConfig):
        if subblock_config.ffn is not None and subblock_config.ffn.is_moe:
            return True
        if subblock_config.parallel_blocks is not None:
            return any(_uses_moe(block_config) for block_config in subblock_config.parallel_blocks)
    return False


def _block_config_to_pattern_char(block_config: BlockConfig) -> str:
    if block_config.ffn is not None and block_config.ffn.is_moe:
        return "E"
    if block_config.attention is not None and block_config.attention.is_mamba:
        return "M"
    if block_config.attention is not None and not block_config.attention.no_op:
        return "*"
    return "-"


def _hybrid_override_pattern_for_block_configs(block_configs: list[BlockConfig]) -> str:
    return "".join(_block_config_to_pattern_char(block_config) for block_config in block_configs)


def _make_moe_ffn_block_config(ffn_config: FFNConfig) -> BlockConfig:
    return BlockConfig(attention=AttentionConfig(no_op=True), ffn=ffn_config)


def _make_moe_ffn_baseline_block_config() -> BlockConfig:
    return BlockConfig(attention=AttentionConfig(no_op=True), ffn=FFNConfig(no_op=True))


def create_descriptor_benchmark_model(
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    vocab_size: int,
    hidden_size: int,
    num_key_value_heads: int,
    num_attention_heads: int,
    prefill_seq_len: int,
    generation_seq_len: int,
    block_config: BlockConfig,
    repeat_block_n_times: int,
    pattern_block_config: BlockConfig | None = None,
):
    """Build a benchmark model with the caller's descriptor, preserving real MoE layers."""
    block_configs = [block_config] * repeat_block_n_times
    pattern_block_configs = [pattern_block_config or block_config] * repeat_block_n_times

    benchmark_config = copy.deepcopy(model_config)
    lm_config = descriptor.get_language_model_config(benchmark_config)
    lm_config.num_hidden_layers = len(block_configs)
    lm_config.hidden_size = hidden_size
    lm_config.num_attention_heads = num_attention_heads
    if hasattr(lm_config, "num_key_value_heads"):
        lm_config.num_key_value_heads = num_key_value_heads
    if hasattr(lm_config, "vocab_size"):
        lm_config.vocab_size = vocab_size
    if hasattr(benchmark_config, "vocab_size"):
        benchmark_config.vocab_size = vocab_size
    if hasattr(lm_config, "max_position_embeddings"):
        lm_config.max_position_embeddings = prefill_seq_len + generation_seq_len
    if hasattr(lm_config, "hybrid_override_pattern"):
        lm_config.hybrid_override_pattern = _hybrid_override_pattern_for_block_configs(
            pattern_block_configs
        )

    benchmark_config.block_configs = block_configs
    if lm_config is not benchmark_config:
        lm_config.block_configs = block_configs

    with deci_x_patcher(descriptor, block_configs):
        model = init_model_from_config(
            benchmark_config,
            trust_remote_code=descriptor.requires_trust_remote_code(),
        )

    block_config_dicts = [block_config.to_dict() for block_config in block_configs]
    model.config.block_configs = block_config_dicts
    model_lm_config = descriptor.get_language_model_config(model.config)
    if model_lm_config is not model.config:
        model_lm_config.block_configs = block_config_dicts
    model.config.architectures = ["AnyModel"]
    model.config.base_architecture = type(model).__name__
    return model


def calc_model_runtime(
    model: LlamaForCausalLM,
    runtime_config: RuntimeConfig,
    descriptor: type[ModelDescriptor] = LlamaModelDescriptor,
) -> float:
    """Measure total runtime of a model via vLLM latency benchmark."""
    with tempfile.TemporaryDirectory() as model_tmpdir:
        save_model(model, Path(runtime_config.tokenizer_path), Path(model_tmpdir), descriptor)
        model_total_runtime_ms = run_vllm_latency_benchmark(Path(model_tmpdir), runtime_config)
    return model_total_runtime_ms


@cache
def _calc_llama_subblock_runtime(
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
def _calc_llama_base_runtime(
    runtime_config: RuntimeConfig, subblock_config: SubblockConfig
) -> float:
    """Calculate the base runtime of a model with no subblocks."""
    base_runtime_ms = None
    if isinstance(subblock_config, AttentionConfig):
        base_runtime_ms = _calc_llama_subblock_runtime(runtime_config, None)
    elif isinstance(subblock_config, FFNConfig):
        attn_block_config = AttentionConfig(
            no_op=False, num_key_value_heads=runtime_config.num_key_value_heads
        ).to_blockconfig()
        base_runtime_ms = _calc_llama_subblock_runtime(runtime_config, attn_block_config)
    else:
        raise ValueError(f"Unsupported subblock type: {type(subblock_config)}")

    return base_runtime_ms


def _calc_moe_ffn_runtime(
    runtime_config: RuntimeConfig,
    ffn_config: FFNConfig,
    model_config: PretrainedConfig,
    descriptor: type[ModelDescriptor],
    baseline: bool,
) -> float:
    target_block_config = _make_moe_ffn_block_config(ffn_config)
    block_config = _make_moe_ffn_baseline_block_config() if baseline else target_block_config
    model = create_descriptor_benchmark_model(
        model_config,
        descriptor,
        runtime_config.vocab_size,
        runtime_config.hidden_size,
        runtime_config.num_key_value_heads,
        runtime_config.num_attention_heads,
        runtime_config.prefill_seq_len,
        runtime_config.generation_seq_len,
        block_config=block_config,
        repeat_block_n_times=runtime_config.repeat_block_n_times,
        pattern_block_config=target_block_config,
    )
    return calc_model_runtime(model, runtime_config, descriptor)


def calc_subblock_runtime(
    runtime_config: RuntimeConfig,
    subblock_config: SubblockConfig | None,
    model_config: PretrainedConfig | None = None,
    descriptor: type[ModelDescriptor] | None = None,
) -> float:
    """Measure total runtime of a repeated subblock via vLLM latency benchmark."""
    if isinstance(subblock_config, FFNConfig) and subblock_config.is_moe:
        if model_config is None or descriptor is None:
            raise ValueError("MoE runtime benchmarking requires model_config and descriptor.")
        return _calc_moe_ffn_runtime(
            runtime_config, subblock_config, model_config, descriptor, baseline=False
        )
    if _uses_moe(subblock_config):
        raise ValueError(f"MoE runtime stats support FFNConfig only, got {subblock_config}.")
    return _calc_llama_subblock_runtime(runtime_config, subblock_config)


def calc_base_runtime(
    runtime_config: RuntimeConfig,
    subblock_config: SubblockConfig,
    model_config: PretrainedConfig | None = None,
    descriptor: type[ModelDescriptor] | None = None,
) -> float:
    """Calculate the base runtime of a model with no target subblocks."""
    if isinstance(subblock_config, FFNConfig) and subblock_config.is_moe:
        if model_config is None or descriptor is None:
            raise ValueError("MoE runtime benchmarking requires model_config and descriptor.")
        return _calc_moe_ffn_runtime(
            runtime_config, subblock_config, model_config, descriptor, baseline=True
        )
    return _calc_llama_base_runtime(runtime_config, subblock_config)


@cache
def calc_no_block_runtime(runtime_config: RuntimeConfig) -> float:
    """Estimate the overhead runtime (embedding + LM head) with no decoder blocks."""
    runtime_cfg_ten_blocks = replace(runtime_config, repeat_block_n_times=9)

    block_config = _make_standard_block_config(runtime_config.num_key_value_heads)

    runtime_ms_one_block = _calc_llama_subblock_runtime(runtime_config, None)  # only one base block
    runtime_ms_ten_blocks = _calc_llama_subblock_runtime(
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
    batch_size: int,
    model_config: PretrainedConfig | None = None,
    descriptor: type[ModelDescriptor] | None = None,
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
        batch_size,
        runtime_stats_config.get("num_iters", 30),
        runtime_stats_config.get("num_warmup_iters", 10),
        runtime_stats_config.get("gpu_memory_utilization", 0.5),
    )

    runtime_by_subblock_dict = {}

    for subblock_config in tqdm(
        sorted(subblock_config_set),
        desc=(f"Computing runtime for {len(subblock_config_set)} subblocks\n"),
    ):
        baseline_runtime_ms = calc_base_runtime(
            runtime_config, subblock_config, model_config, descriptor
        )

        if subblock_config.no_op:
            total_runtime_ms = 0.0
        else:
            subblock_total_runtime_ms = calc_subblock_runtime(
                runtime_config, subblock_config, model_config, descriptor
            )
            total_runtime_ms = (
                subblock_total_runtime_ms - baseline_runtime_ms
            ) / repeat_block_n_times

        runtime_by_subblock_dict[subblock_config] = total_runtime_ms

    no_block_runtime_ms = calc_no_block_runtime(runtime_config)

    return runtime_by_subblock_dict, no_block_runtime_ms
