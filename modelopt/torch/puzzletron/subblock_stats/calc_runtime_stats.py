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

"""Runtime statistics calculation for NAS subblock benchmarking via vLLM.

A subblock's runtime is estimated empirically: we build a tiny benchmark model
whose decoder is the subblock repeated ``repeat_block_n_times`` times, measure
its end-to-end vLLM latency, subtract a baseline (the same model without the
measured subblock), and divide by the repeat count. The fixed per-model
overhead (embedding + LM head) is estimated separately by extrapolating from a
1-block and a 10-block model.

Both plain Transformer (Llama) and hybrid Mamba/attention (Nemotron-H) benchmark
models are supported.
"""

import copy
import tempfile
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

from omegaconf import DictConfig
from tqdm import tqdm
from transformers import LlamaConfig, PretrainedConfig, PreTrainedModel

from ..anymodel.model_descriptor import ModelDescriptor
from ..anymodel.models.llama import LlamaModelDescriptor
from ..anymodel.puzzformer import deci_x_patcher
from ..block_config import AttentionConfig, BlockConfig, FFNConfig, MambaConfig, SubblockConfig
from ..tools.checkpoint_utils_hf import init_model_from_config
from .runtime_utils import DEFAULT_MAMBA_BLOCK_SIZE, RuntimeConfig, save_model
from .runtime_vllm import run_vllm_latency_benchmark

_SUPPORTED_HYBRID_DESCRIPTOR_NAMES = {"NemotronHModelDescriptor", "NemotronHV2ModelDescriptor"}

# Default FFN/MoE intermediate size for synthesized benchmark configs. Kept tiny
# so benchmark models are cheap to build and run.
_DEFAULT_INTERMEDIATE_SIZE = 256


@dataclass(frozen=True)
class _BenchmarkDimensions:
    """Shape parameters shared by every benchmark model we construct.

    Bundling these avoids threading the same six values positionally through the
    config builders.
    """

    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    prefill_seq_len: int
    generation_seq_len: int

    @classmethod
    def from_runtime_config(cls, runtime_config: RuntimeConfig) -> "_BenchmarkDimensions":
        return cls(
            vocab_size=runtime_config.vocab_size,
            hidden_size=runtime_config.hidden_size,
            num_attention_heads=runtime_config.num_attention_heads,
            num_key_value_heads=runtime_config.num_key_value_heads,
            prefill_seq_len=runtime_config.prefill_seq_len,
            generation_seq_len=runtime_config.generation_seq_len,
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def max_position_embeddings(self) -> int:
        return self.prefill_seq_len + self.generation_seq_len


# --------------------------------------------------------------------------- #
# Block-config helpers
# --------------------------------------------------------------------------- #


def _make_standard_block_config(num_key_value_heads: int) -> BlockConfig:
    """Build a dense attention + FFN block for non-hybrid (Llama-style) models."""
    return BlockConfig(
        attention=AttentionConfig(no_op=False, num_key_value_heads=num_key_value_heads),
        ffn=FFNConfig(no_op=False, intermediate_size=_DEFAULT_INTERMEDIATE_SIZE, moe=None),
    )


def _has_mamba_block(block_config: BlockConfig | None) -> bool:
    """Return whether ``block_config`` (or any of its parallel blocks) is a Mamba block."""
    if block_config is None:
        return False
    if block_config.attention is not None and block_config.attention.is_mamba:
        return True
    if block_config.parallel_blocks is None:
        return False
    return any(_has_mamba_block(parallel_block) for parallel_block in block_config.parallel_blocks)


def _has_mamba_subblock(subblock_config: SubblockConfig | None) -> bool:
    """Return whether ``subblock_config`` represents a Mamba subblock."""
    if isinstance(subblock_config, AttentionConfig):
        return subblock_config.is_mamba
    if isinstance(subblock_config, BlockConfig):
        return _has_mamba_block(subblock_config)
    return False


def _complete_block_config(block_config: BlockConfig) -> BlockConfig:
    """Fill in missing attention/FFN subblocks with explicit no-ops."""
    return BlockConfig(
        attention=block_config.attention or AttentionConfig(no_op=True),
        ffn=block_config.ffn or FFNConfig(no_op=True),
        parallel_blocks=block_config.parallel_blocks,
    )


def _is_no_op_subblock(subblock_config: SubblockConfig) -> bool:
    """Return whether ``subblock_config`` performs no computation (zero runtime)."""
    if isinstance(subblock_config, BlockConfig):
        block_config = _complete_block_config(subblock_config)
        return block_config.attention.no_op and block_config.ffn.no_op
    return subblock_config.no_op


def _uses_hybrid_block_pattern(
    model_config: PretrainedConfig | None, descriptor: type[ModelDescriptor]
) -> bool:
    """Return whether the benchmark model uses a hybrid (Mamba/attention) layer pattern."""
    if descriptor.__name__ in _SUPPORTED_HYBRID_DESCRIPTOR_NAMES:
        return True
    if model_config is None:
        return False
    lm_config = descriptor.get_language_model_config(model_config)
    return hasattr(lm_config, "hybrid_override_pattern")


def _make_default_block_config(
    num_key_value_heads: int,
    model_config: PretrainedConfig | None,
    descriptor: type[ModelDescriptor],
    repeated_block_config: BlockConfig | None = None,
) -> BlockConfig:
    """Build the leading "base" block that every benchmark model starts with.

    For hybrid models the base block mirrors the measured block (so the layer
    pattern stays valid); for dense models it is a standard attention + FFN block.
    """
    if _uses_hybrid_block_pattern(model_config, descriptor):
        if repeated_block_config is not None:
            return _complete_block_config(repeated_block_config)
        return AttentionConfig(
            no_op=False, num_key_value_heads=num_key_value_heads
        ).to_blockconfig()
    return _make_standard_block_config(num_key_value_heads)


def _subblock_to_benchmark_block_config(
    runtime_config: RuntimeConfig,
    subblock_config: SubblockConfig | None,
) -> BlockConfig | None:
    """Wrap a subblock into the full ``BlockConfig`` used to build the benchmark model."""
    if subblock_config is None:
        return None
    if isinstance(subblock_config, BlockConfig):
        return _complete_block_config(subblock_config)
    if isinstance(subblock_config, AttentionConfig):
        return subblock_config.to_blockconfig()
    if isinstance(subblock_config, FFNConfig):
        # On hybrid models an FFN occupies a layer on its own; on dense models it
        # must be paired with an attention subblock to form a complete block.
        if _uses_hybrid_block_pattern(
            runtime_config.benchmark_model_config, runtime_config.benchmark_model_descriptor
        ):
            return subblock_config.to_blockconfig()
        return BlockConfig(
            attention=AttentionConfig(
                no_op=False, num_key_value_heads=runtime_config.num_key_value_heads
            ),
            ffn=subblock_config,
        )
    raise TypeError(f"Runtime stats: Not supported subblock type: {subblock_config}")


def _block_config_to_hybrid_pattern_char(block_config: BlockConfig) -> str:
    """Map a single block to its Nemotron-H ``hybrid_override_pattern`` character."""
    block_config = _complete_block_config(block_config)
    if block_config.parallel_blocks is not None:
        raise ValueError("Runtime stats do not support parallel_blocks for hybrid benchmark models")

    attention = block_config.attention
    ffn = block_config.ffn
    attention_active = attention is not None and not attention.no_op
    ffn_active = ffn is not None and not ffn.no_op

    if attention_active and ffn_active:
        raise ValueError(
            "Hybrid benchmark layers support a single active subblock. Got both attention and FFN."
        )
    if attention_active:
        return "M" if attention.is_mamba else "*"
    if ffn_active:
        return "E" if ffn.is_moe else "-"
    return "-"


def _hybrid_override_pattern(block_configs: list[BlockConfig]) -> str:
    """Build the full Nemotron-H ``hybrid_override_pattern`` for a list of blocks."""
    return "".join(_block_config_to_hybrid_pattern_char(block) for block in block_configs)


def _get_first_mamba_config(block_configs: list[BlockConfig]) -> MambaConfig | None:
    """Return the first ``MambaConfig`` found among ``block_configs``, if any."""
    for block_config in block_configs:
        attention = block_config.attention
        if attention is not None and attention.mamba is not None:
            return attention.mamba
    return None


def _apply_mamba_config_overrides(
    config: PretrainedConfig, block_configs: list[BlockConfig]
) -> None:
    """Copy Mamba hyperparameters from the measured block onto a benchmark config."""
    mamba_config = _get_first_mamba_config(block_configs)
    if mamba_config is None:
        return

    config.mamba_num_heads = mamba_config.num_heads
    config.mamba_head_dim = mamba_config.head_dim
    config.ssm_state_size = mamba_config.state_dim
    config.n_groups = mamba_config.num_groups
    if not hasattr(config, "conv_kernel"):
        config.conv_kernel = 4


def _get_first_mamba_subblock_config(
    subblock_config_set: set[SubblockConfig],
) -> AttentionConfig | BlockConfig | None:
    """Return the first subblock in the set that contains a Mamba subblock, if any."""
    for subblock_config in subblock_config_set:
        if _has_mamba_subblock(subblock_config):
            return subblock_config
    return None


# --------------------------------------------------------------------------- #
# Benchmark model-config builders
# --------------------------------------------------------------------------- #


def _get_base_architecture(
    model_config: PretrainedConfig, descriptor: type[ModelDescriptor]
) -> str:
    """Determine the HF ``base_architecture`` string for an AnyModel benchmark config."""
    base_architecture = getattr(model_config, "base_architecture", None)
    if base_architecture:
        return base_architecture

    architectures = getattr(model_config, "architectures", None) or []
    if architectures and architectures[0] != "AnyModel":
        return architectures[0]

    if descriptor is LlamaModelDescriptor:
        return "LlamaForCausalLM"
    if descriptor.__name__ in _SUPPORTED_HYBRID_DESCRIPTOR_NAMES:
        return "NemotronHForCausalLM"
    return f"{model_config.__class__.__name__.removesuffix('Config')}ForCausalLM"


def _make_llama_benchmark_config(dims: _BenchmarkDimensions, num_hidden_layers: int) -> LlamaConfig:
    """Build a default Llama benchmark config from the requested dimensions."""
    return LlamaConfig(
        max_position_embeddings=dims.max_position_embeddings,
        vocab_size=dims.vocab_size,
        hidden_size=dims.hidden_size,
        num_attention_heads=dims.num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        head_dim=dims.head_dim,
        # This is required for trt-llm conversion to know which model classes to use.
        auto_map={
            "AutoConfig": "transformers.models.llama.configuration_llama.LlamaConfig",
            "AutoModelForCausalLM": "transformers.models.llama.modeling_llama.LlamaForCausalLM",
        },
    )


def _resolve_nemotron_h_mamba_hyperparams(
    block_configs: list[BlockConfig],
    template: PretrainedConfig | None,
    descriptor: type[ModelDescriptor],
) -> tuple[int, int, int, int, int]:
    """Return (mamba_num_heads, mamba_head_dim, ssm_state_size, n_groups, conv_kernel)."""
    mamba_config = _get_first_mamba_config(block_configs)
    if mamba_config is not None:
        return (
            mamba_config.num_heads,
            mamba_config.head_dim,
            mamba_config.state_dim,
            mamba_config.num_groups,
            4,
        )

    if template is not None:
        lm_config = descriptor.get_language_model_config(template)
        return (
            lm_config.mamba_num_heads,
            lm_config.mamba_head_dim,
            lm_config.ssm_state_size,
            lm_config.n_groups,
            getattr(lm_config, "conv_kernel", 4),
        )

    raise ValueError("Cannot build a Nemotron-H benchmark config without Mamba hyperparameters")


def _make_default_nemotron_h_benchmark_config(
    dims: _BenchmarkDimensions,
    block_configs: list[BlockConfig],
    descriptor: type[ModelDescriptor],
    template: PretrainedConfig | None = None,
) -> PretrainedConfig:
    """Build a fresh Nemotron-H config for a hybrid benchmark layer stack.

    Always constructs a new ``NemotronHConfig`` so we never mutate
    ``hybrid_override_pattern`` on an existing config (read-only in newer
    Transformers).
    """
    try:
        from transformers import NemotronHConfig
    except ImportError as e:
        raise ValueError(
            "Hybrid runtime stats require a benchmark model_config/descriptor from a "
            "Mamba-capable model, or a Transformers version that provides NemotronHConfig."
        ) from e

    mamba_num_heads, mamba_head_dim, ssm_state_size, n_groups, conv_kernel = (
        _resolve_nemotron_h_mamba_hyperparams(block_configs, template, descriptor)
    )
    template_lm = descriptor.get_language_model_config(template) if template is not None else None

    return NemotronHConfig(
        hidden_size=dims.hidden_size,
        intermediate_size=getattr(template_lm, "intermediate_size", _DEFAULT_INTERMEDIATE_SIZE),
        num_hidden_layers=len(block_configs),
        hybrid_override_pattern=_hybrid_override_pattern(block_configs),
        num_attention_heads=dims.num_attention_heads,
        num_key_value_heads=dims.num_key_value_heads,
        head_dim=dims.head_dim,
        mamba_num_heads=mamba_num_heads,
        mamba_head_dim=mamba_head_dim,
        ssm_state_size=ssm_state_size,
        n_groups=n_groups,
        conv_kernel=conv_kernel,
        n_routed_experts=getattr(template_lm, "n_routed_experts", 8),
        num_experts_per_tok=getattr(template_lm, "num_experts_per_tok", 1),
        moe_intermediate_size=getattr(
            template_lm, "moe_intermediate_size", _DEFAULT_INTERMEDIATE_SIZE
        ),
        n_shared_experts=getattr(template_lm, "n_shared_experts", 1),
        moe_shared_expert_intermediate_size=getattr(
            template_lm, "moe_shared_expert_intermediate_size", _DEFAULT_INTERMEDIATE_SIZE
        ),
        moe_latent_size=getattr(template_lm, "moe_latent_size", None),
        vocab_size=dims.vocab_size,
        max_position_embeddings=dims.max_position_embeddings,
    )


def _make_default_mamba_benchmark_config(
    dims: _BenchmarkDimensions, block_configs: list[BlockConfig]
) -> PretrainedConfig:
    """Build a default Nemotron-H benchmark config for a hybrid (Mamba) layer stack."""
    from ..anymodel.models.nemotron_h import NemotronHModelDescriptor

    return _make_default_nemotron_h_benchmark_config(dims, block_configs, NemotronHModelDescriptor)


def _make_benchmark_model_config(
    dims: _BenchmarkDimensions,
    block_configs: list[BlockConfig],
    benchmark_model_config: PretrainedConfig | None,
    descriptor: type[ModelDescriptor],
) -> PretrainedConfig:
    """Build (or adapt) the benchmark model config for the given block stack.

    When no template ``benchmark_model_config`` is provided a default is
    synthesized (Nemotron-H for hybrid stacks, Llama otherwise); otherwise the
    template is deep-copied and its dimensions are overwritten to match ``dims``.

    Hybrid Nemotron-H stacks always get a freshly synthesized config so we never
    mutate ``hybrid_override_pattern`` (read-only on ``NemotronHConfig``).
    """
    if _uses_hybrid_block_pattern(benchmark_model_config, descriptor):
        model_config = _make_default_nemotron_h_benchmark_config(
            dims, block_configs, descriptor, template=benchmark_model_config
        )
    elif benchmark_model_config is None:
        model_config = _make_llama_benchmark_config(dims, len(block_configs))
    else:
        model_config = copy.deepcopy(benchmark_model_config)

    lm_config = descriptor.get_language_model_config(model_config)
    lm_config.vocab_size = dims.vocab_size
    lm_config.hidden_size = dims.hidden_size
    lm_config.num_attention_heads = dims.num_attention_heads
    lm_config.num_key_value_heads = dims.num_key_value_heads
    lm_config.num_hidden_layers = len(block_configs)
    lm_config.max_position_embeddings = dims.max_position_embeddings
    if hasattr(lm_config, "head_dim"):
        lm_config.head_dim = dims.head_dim
    if not hasattr(lm_config, "intermediate_size"):
        lm_config.intermediate_size = _DEFAULT_INTERMEDIATE_SIZE

    lm_config.block_configs = block_configs
    # For models that nest the language config (e.g. multimodal), the outer
    # config also needs the layer count and block configs.
    if lm_config is not model_config:
        model_config.num_hidden_layers = len(block_configs)
        model_config.block_configs = block_configs
    return model_config


def _make_benchmark_model_key(
    model_config: PretrainedConfig | None, descriptor: type[ModelDescriptor]
) -> str:
    """Build a cache key capturing the benchmark model's architecture-defining fields.

    ``RuntimeConfig`` excludes the (unhashable) config/descriptor from its own
    hash, so this string stands in for them when caching runtime measurements.
    """
    if model_config is None:
        return descriptor.__name__
    lm_config = descriptor.get_language_model_config(model_config)
    architectures = tuple(getattr(model_config, "architectures", None) or ())
    key_parts = (
        descriptor.__name__,
        model_config.__class__.__name__,
        getattr(model_config, "model_type", None),
        architectures,
        getattr(lm_config, "hybrid_override_pattern", None),
        getattr(lm_config, "conv_kernel", None),
        getattr(lm_config, "n_shared_experts", None),
        getattr(lm_config, "moe_latent_size", None),
    )
    return repr(key_parts)


def _resolve_benchmark_descriptor_and_config(
    subblock_config_set: set[SubblockConfig],
    descriptor: type[ModelDescriptor],
    model_config: PretrainedConfig | None,
    dims: _BenchmarkDimensions,
) -> tuple[type[ModelDescriptor], PretrainedConfig | None]:
    """Pick a Mamba-capable descriptor/config when measuring Mamba subblocks.

    If the caller supplied no model config but the subblock set contains a Mamba
    subblock, fall back to a synthesized Nemotron-H benchmark model. Otherwise the
    caller's descriptor and config are used unchanged.
    """
    mamba_subblock_config = _get_first_mamba_subblock_config(subblock_config_set)
    if model_config is not None or mamba_subblock_config is None:
        return descriptor, model_config

    from ..anymodel.models.nemotron_h import NemotronHModelDescriptor

    mamba_block_config = (
        mamba_subblock_config
        if isinstance(mamba_subblock_config, BlockConfig)
        else mamba_subblock_config.to_blockconfig()
    )
    model_config = _make_default_mamba_benchmark_config(dims, [mamba_block_config])
    return NemotronHModelDescriptor, model_config


# --------------------------------------------------------------------------- #
# Benchmark model construction and runtime measurement
# --------------------------------------------------------------------------- #


def create_benchmark_model(
    dims: _BenchmarkDimensions,
    block_config: BlockConfig | None,
    repeat_block_n_times: int = 10,
    benchmark_model_config: PretrainedConfig | None = None,
    benchmark_model_descriptor: type[ModelDescriptor] = LlamaModelDescriptor,
) -> PreTrainedModel:
    """Build a small benchmark model with repeated subblocks for latency benchmarking.

    The model consists of one "base" block followed by ``block_config`` repeated
    ``repeat_block_n_times`` times.
    """
    if (
        benchmark_model_config is None
        and benchmark_model_descriptor is LlamaModelDescriptor
        and _has_mamba_block(block_config)
    ):
        from ..anymodel.models.nemotron_h import NemotronHModelDescriptor

        benchmark_model_descriptor = NemotronHModelDescriptor

    block_configs = [
        _make_default_block_config(
            dims.num_key_value_heads,
            benchmark_model_config,
            benchmark_model_descriptor,
            repeated_block_config=block_config,
        )
    ]
    if block_config is not None and repeat_block_n_times > 0:
        block_configs.extend([block_config] * repeat_block_n_times)

    model_config = _make_benchmark_model_config(
        dims, block_configs, benchmark_model_config, benchmark_model_descriptor
    )
    base_architecture = _get_base_architecture(model_config, benchmark_model_descriptor)

    with deci_x_patcher(benchmark_model_descriptor, block_configs):
        model = init_model_from_config(
            model_config,
            trust_remote_code=benchmark_model_descriptor.requires_trust_remote_code(),
        )

    serializable_block_configs = [block_config.to_dict() for block_config in block_configs]
    model.config.block_configs = serializable_block_configs
    model.config.architectures = ["AnyModel"]
    model.config.base_architecture = base_architecture
    lm_config = benchmark_model_descriptor.get_language_model_config(model.config)
    if lm_config is not model.config:
        lm_config.block_configs = serializable_block_configs

    return model


def calc_model_runtime(model: PreTrainedModel, runtime_config: RuntimeConfig) -> float:
    """Measure total runtime of a model via vLLM latency benchmark."""
    with tempfile.TemporaryDirectory() as model_tmpdir:
        save_model(
            model,
            Path(runtime_config.tokenizer_path),
            Path(model_tmpdir),
            runtime_config.benchmark_model_descriptor,
        )
        return run_vllm_latency_benchmark(Path(model_tmpdir), runtime_config)


@cache
def calc_subblock_runtime(
    runtime_config: RuntimeConfig,
    subblock_config: SubblockConfig | None,
) -> float:
    """Measure total runtime of a repeated subblock via vLLM latency benchmark."""
    block_config = _subblock_to_benchmark_block_config(runtime_config, subblock_config)
    model = create_benchmark_model(
        _BenchmarkDimensions.from_runtime_config(runtime_config),
        block_config=block_config,
        repeat_block_n_times=runtime_config.repeat_block_n_times,
        benchmark_model_config=runtime_config.benchmark_model_config,
        benchmark_model_descriptor=runtime_config.benchmark_model_descriptor,
    )
    return calc_model_runtime(model, runtime_config)


@cache
def calc_base_runtime(runtime_config: RuntimeConfig, subblock_config: SubblockConfig) -> float:
    """Calculate the base runtime of a model with no repeated measured subblocks.

    The baseline is what remains once the measured subblock is removed, so that
    subtracting it from the repeated-subblock runtime isolates the subblock cost.
    """
    if _uses_hybrid_block_pattern(
        runtime_config.benchmark_model_config, runtime_config.benchmark_model_descriptor
    ):
        base_runtime_config = replace(runtime_config, repeat_block_n_times=0)
        block_config = _subblock_to_benchmark_block_config(runtime_config, subblock_config)
        return calc_subblock_runtime(base_runtime_config, block_config)

    if isinstance(subblock_config, AttentionConfig):
        return calc_subblock_runtime(runtime_config, None)
    if isinstance(subblock_config, FFNConfig):
        attn_block_config = AttentionConfig(
            no_op=False, num_key_value_heads=runtime_config.num_key_value_heads
        ).to_blockconfig()
        return calc_subblock_runtime(runtime_config, attn_block_config)
    if isinstance(subblock_config, BlockConfig):
        return calc_subblock_runtime(runtime_config, None)
    raise ValueError(f"Unsupported subblock type: {type(subblock_config)}")


@cache
def calc_no_block_runtime(runtime_config: RuntimeConfig) -> float:
    """Estimate the overhead runtime (embedding + LM head) with no decoder blocks.

    Measures a 1-block and a 10-block model, extrapolates the per-block cost from
    their difference, and subtracts it from the 1-block runtime.
    """
    num_repeated_blocks = 9  # 10-block model = 1 base block + 9 repeated blocks
    runtime_cfg_ten_blocks = replace(runtime_config, repeat_block_n_times=num_repeated_blocks)

    block_config = _make_default_block_config(
        runtime_config.num_key_value_heads,
        runtime_config.benchmark_model_config,
        runtime_config.benchmark_model_descriptor,
    )

    runtime_ms_one_block = calc_subblock_runtime(runtime_config, None)
    runtime_ms_ten_blocks = calc_subblock_runtime(runtime_cfg_ten_blocks, block_config)

    per_block_runtime_ms = (runtime_ms_ten_blocks - runtime_ms_one_block) / num_repeated_blocks
    return runtime_ms_one_block - per_block_runtime_ms


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
    descriptor: type[ModelDescriptor] = LlamaModelDescriptor,
) -> tuple[dict[SubblockConfig, float], float]:
    """Benchmark each unique subblock and return per-subblock runtimes and no-block overhead."""
    repeat_block_n_times = 10
    dims = _BenchmarkDimensions(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        prefill_seq_len=prefill_seq_len,
        generation_seq_len=generation_seq_len,
    )
    descriptor, model_config = _resolve_benchmark_descriptor_and_config(
        subblock_config_set, descriptor, model_config, dims
    )

    uses_mamba_benchmark = _uses_hybrid_block_pattern(model_config, descriptor) or any(
        _has_mamba_subblock(subblock) for subblock in subblock_config_set
    )
    mamba_block_size = (
        runtime_stats_config.get("mamba_block_size", DEFAULT_MAMBA_BLOCK_SIZE)
        if uses_mamba_benchmark
        else None
    )

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
        benchmark_model_key=_make_benchmark_model_key(model_config, descriptor),
        benchmark_model_config=model_config,
        benchmark_model_descriptor=descriptor,
        mamba_block_size=mamba_block_size,
    )

    runtime_by_subblock_dict = {}
    for subblock_config in tqdm(
        sorted(subblock_config_set),
        desc=f"Computing runtime for {len(subblock_config_set)} subblocks\n",
    ):
        if _is_no_op_subblock(subblock_config):
            total_runtime_ms = 0.0
        else:
            baseline_runtime_ms = calc_base_runtime(runtime_config, subblock_config)
            subblock_total_runtime_ms = calc_subblock_runtime(runtime_config, subblock_config)
            total_runtime_ms = (
                subblock_total_runtime_ms - baseline_runtime_ms
            ) / repeat_block_n_times

        runtime_by_subblock_dict[subblock_config] = total_runtime_ms

    no_block_runtime_ms = calc_no_block_runtime(runtime_config)
    return runtime_by_subblock_dict, no_block_runtime_ms
