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

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import (
    LayerType,
    LNImpl,
    TransformerBlockSubmodules,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_num_layers_to_build,
    get_transformer_layer_offset,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.post_training.modelopt.layers import Linear
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.quantization.utils import (
    kitchen_quantization_recipe_config,
    load_quantization_recipe,
)
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    _initialize_affine_weight_cpu,
)
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import get_te_version, is_te_min_version, is_torch_min_version

# from megatron.core.activations import squared_relu #for megatron 0.14 version in future NeMo containers
from megatron.training.activations import squared_relu
from nemo.collections.llm.gpt.model.llama import Llama31Config70B
from packaging.version import Version as PkgVersion
from torch import Tensor
from torch.nn.parameter import Parameter

try:
    import transformer_engine as te  # pylint: disable=unused-import
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TELinear,
        TENorm,
        TERowParallelLinear,
        _get_extra_te_kwargs,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

# TODO: check sharded_state_dict_keys_map => only if TE is disabled
# TODO: parallel Blocks
# TODO: multimodal
#       https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/vlm/neva/model/base.py
#       https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/vlm/qwen2vl/model/base.py


# NOTE based on https://github.com/NVIDIA/Megatron-LM/blob/aacc3b8aa5f0d3071431a94503d6233802fbaedd/megatron/core/models/gpt/heterogeneous/heterogeneous_layer_specs.py#L144
# TODO: what is the difference between this and the referenced one?
def _get_sharded_state_dict_keys_map(
    block_config: "PuzzletronTransformerBlockConfig", use_transformer_engine: bool
):
    """Generate a mapping of sharded state dictionary keys for Puzzletron transformer blocks.

    This function is a specialized version of the original Megatron-LM
    `_get_sharded_state_dict_keys_map` function, adapted for Puzzletron's
    heterogeneous transformer architecture with Mamba support.

    Key differences from the original:
    - **Mamba Support**: Adds mapping for Mamba layers (`mixer.norm_`)
    - **Enhanced Logic**: Uses if-elif-else structure instead of multiple if statements
    - **No-op Handling**: Explicit handling of no-op attention and MLP cases
    - **Simplified**: Removes `num_query_groups` check (handled in main logic)
    - **Config Type**: Uses `PuzzletronTransformerBlockConfig` instead of `TransformerBlockConfig`

    Args:
        block_config: Puzzletron transformer block configuration
        use_transformer_engine: Whether to use Transformer Engine optimizations

    Returns:
        dict: A dictionary mapping sharded state dictionary keys
    """
    mapping = {}
    if not use_transformer_engine:
        if block_config.attention.replace_with_linear:
            mapping.update({"input_layernorm.": "self_attention.layer_norm_"})
        elif block_config.attention.is_mamba:  # Mamba, not sure about this
            mapping.update({"input_layernorm.": "mixer.norm_"})
        elif not block_config.attention.no_op:  # MHA and MLA
            mapping.update({"input_layernorm.": "self_attention.linear_qkv.layer_norm_"})
        else:  # No-op
            pass

        if block_config.mlp.ffn_hidden_size is not None:  # FFN
            mapping.update({"pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_"})
        elif block_config.mlp.replace_with_linear:  # Linear
            mapping.update({"pre_mlp_layernorm.": "mlp.layer_norm_"})
        else:  # No-op, MoE
            pass
    return mapping


# NOTE: new class
@dataclass
class PuzzletronSubblockConfig:
    """Base configuration class for Puzzletron transformer subblocks.

    This is the base class for attention and MLP configurations in Puzzletron's
    heterogeneous transformer architecture. It provides common functionality
    for subblock configurations including no-op and linear replacement options.

    Key differences from the original Megatron-LM subblock configs:
    - **Enhanced Building**: Uses `build_config_from_dict()` with main config fallback
    - **Validation**: Includes `__post_init__()` validation for mutual exclusivity
    - **Flexibility**: Supports both no-op and linear replacement modes

    Attributes:
        no_op: Whether this subblock should be a no-op operation
        replace_with_linear: Whether to replace the subblock with a single linear layer
    """

    no_op: bool = False
    replace_with_linear: bool = False

    @classmethod
    def build_config_from_dict(
        cls,
        subblock_config_dict: dict[str, Any],
        main_config: "PuzzletronHeterogeneousTransformerConfig",
    ):
        field_names = {f.name for f in fields(cls)}
        subblock_config_dict = {k: v for k, v in subblock_config_dict.items() if k in field_names}
        # getting default values from the main config (if not overridden in the subblock config)
        for field_name in field_names:
            # note that MLA fields are also not in the main_config
            if field_name not in subblock_config_dict and hasattr(main_config, field_name):
                subblock_config_dict[field_name] = getattr(main_config, field_name)
        return cls(**subblock_config_dict)

    def __post_init__(self) -> None:
        assert not (self.no_op and self.replace_with_linear), (
            "at most one of no_op, replace_with_linear can be True"
        )


@dataclass
class PuzzletronAttentionConfig(PuzzletronSubblockConfig):
    """Configuration parameters for the self-attention part of a Puzzletron transformer block.

    This class extends the original Megatron-LM AttentionConfig with support for
    Mamba layers and enhanced Multi-Latent Attention (MLA) configurations.

    Key differences from the original AttentionConfig:
    - **Mamba Support**: Adds `is_mamba` flag and Mamba-specific parameters
    - **Enhanced MLA**: Extended MLA parameters with LoRA ranks and head dimensions
    - **Context Parallelism**: Adds `cp_comm_type` for attention context parallelism
    - **Validation**: Enhanced `__post_init__()` with Mamba-MLA mutual exclusivity check
    - **Flexibility**: Supports MHA, MLA, and Mamba attention types in one config

    Attributes:
        # MHA (Multi-Head Attention) parameters
        num_attention_heads: Number of attention heads
        num_query_groups: Number of query groups for grouped query attention
        kv_channels: Key-value projection dimension
        window_size: Sliding window size for local attention

        # MLA (Multi-Latent Attention) parameters
        multi_latent_attention: Whether to use MLA instead of MHA
        q_lora_rank: LoRA rank for query projections
        kv_lora_rank: LoRA rank for key-value projections
        qk_head_dim: Query-key head dimension
        qk_pos_emb_head_dim: Query-key positional embedding head dimension
        v_head_dim: Value head dimension

        # Context parallelism
        cp_comm_type: Communication type for context parallelism

        # Mamba parameters
        is_mamba: Whether to use Mamba instead of attention
        mamba_state_dim: Mamba state dimension
        mamba_head_dim: Mamba head dimension
        mamba_num_groups: Number of groups in Mamba
        mamba_num_heads: Number of heads in Mamba (auto-calculated if None)
    """

    # all attributes, except for is_mamba are part of TransformerConfig/MLATransformerConfig
    # MHA
    num_attention_heads: Optional[int] = None
    num_query_groups: Optional[int] = None
    kv_channels: Optional[int] = None
    window_size: Optional[Tuple[int, int]] = None
    # MLA (Note that for MLA we have to instantiate a MLATransformerConfig, since there is a isinstance in attention.py)
    multi_latent_attention: bool = False
    q_lora_rank: int = 512
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    # for attention context parallelism (ignored for mamba)
    cp_comm_type: str = "p2p"
    # Mamba
    is_mamba: bool = False  # new
    mamba_state_dim: int = 128
    mamba_head_dim: int = 64
    mamba_num_groups: int = 8
    mamba_num_heads: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.no_op or self.replace_with_linear:
            self.is_mamba = False
            self.num_attention_heads = 8
            self.multi_latent_attention = False
        if self.is_mamba:
            if self.num_attention_heads is None or self.num_attention_heads == 0:
                self.num_attention_heads = 8  # to avoid division by zero
        assert not (self.is_mamba and self.multi_latent_attention), (
            "Mamba and MLA cannot be used together"
        )


@dataclass
class PuzzletronMLPConfig(PuzzletronSubblockConfig):
    """Configuration parameters for the MLP part of a Puzzletron transformer block.

    This class extends the original Megatron-LM MLPConfig with enhanced
    Mixture of Experts (MoE) support and improved configuration building.

    Key differences from the original MLPConfig:
    - **Enhanced MoE**: Extended MoE parameters with shared expert support
    - **Validation**: Includes `__post_init__()` validation for no-op/linear modes
    - **Building**: Uses `build_config_from_dict()` with main config fallback
    - **Flexibility**: Supports standard MLP, MoE, no-op, and linear replacement modes

    Attributes:
        # Standard MLP parameters
        ffn_hidden_size: MLP intermediate size (hidden dimension)

        # MoE (Mixture of Experts) parameters
        num_moe_experts: Number of expert networks in MoE
        moe_shared_expert_intermediate_size: Size of shared expert intermediate layer
        moe_ffn_hidden_size: Hidden size for MoE expert networks
        moe_router_topk: Number of top-k experts to route tokens to
    """

    # all attributes are part of TransformerConfig
    ffn_hidden_size: Optional[int] = None
    # MoE
    num_moe_experts: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None
    moe_router_topk: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.no_op or self.replace_with_linear:
            self.ffn_hidden_size = None
            self.num_moe_experts = None
            self.moe_ffn_hidden_size = None


# NOTE: based on https://github.com/NVIDIA/Megatron-LM/blob/aacc3b8aa5f0d3071431a94503d6233802fbaedd/megatron/core/transformer/heterogeneous/heterogeneous_config.py#L134
@dataclass
class PuzzletronTransformerBlockConfig:
    """Configuration for a single Puzzletron transformer block in a heterogeneous model.

    This class represents the configuration for one transformer block, containing
    both attention and MLP subblock configurations. It's based on the original
    Megatron-LM TransformerBlockConfig but uses Puzzletron-specific subblock configs.

    Key differences from the original TransformerBlockConfig:
    - **Puzzletron Subblocks**: Uses `PuzzletronAttentionConfig` and `PuzzletronMLPConfig`
    - **Enhanced Building**: Uses `build_from_dict()` with main config fallback
    - **Mamba Support**: Supports Mamba layers through attention config
    - **MoE Support**: Enhanced MoE support through MLP config
    - **Flexibility**: Supports all Puzzletron attention and MLP variants

    Attributes:
        attention: Configuration for the attention subblock (MHA, MLA, or Mamba)
        mlp: Configuration for the MLP subblock (standard MLP or MoE)
    """

    attention: PuzzletronAttentionConfig
    mlp: PuzzletronMLPConfig

    @classmethod
    def build_from_dict(
        cls, block: dict[str, Any], main_config: "PuzzletronHeterogeneousTransformerConfig"
    ):
        if "mlp" in block:
            mlp = block["mlp"]
        elif "ffn" in block:
            mlp = block["ffn"]
        else:
            raise ValueError(f"mlp/ffn not found in block: {block}")

        return cls(
            attention=PuzzletronAttentionConfig.build_config_from_dict(
                subblock_config_dict=block["attention"], main_config=main_config
            ),
            mlp=PuzzletronMLPConfig.build_config_from_dict(
                subblock_config_dict=mlp, main_config=main_config
            ),
        )


@dataclass
class PuzzletronMambaTransformerConfig(TransformerConfig):
    """Configuration for Puzzletron Mamba-only transformer models.

    This class extends the base TransformerConfig for models that use
    Mamba layers exclusively instead of attention mechanisms. It inherits
    all standard transformer configuration parameters from TransformerConfig.

    Key differences from standard TransformerConfig:
    - **Mamba Focus**: Designed specifically for Mamba-based architectures
    - **Inheritance**: Inherits all standard transformer parameters
    - **Simplicity**: Currently a pass-through class for future Mamba-specific extensions

    Note: This class is currently minimal and inherits all functionality
    from the base TransformerConfig. Future versions may add Mamba-specific
    configuration parameters as needed.
    """


# NOTE: based on https://github.com/NVIDIA/Megatron-LM/blob/aacc3b8aa5f0d3071431a94503d6233802fbaedd/megatron/core/transformer/heterogeneous/heterogeneous_config.py#L147
@dataclass
class PuzzletronHeterogeneousTransformerConfig(TransformerConfig):
    """Configuration object for Puzzletron heterogeneous transformers.

    This class extends the original Megatron-LM HeterogeneousTransformerConfig with
    enhanced support for Mamba layers and improved configuration management.

    Key differences from the original HeterogeneousTransformerConfig:
    - **Mamba Support**: Adds Mamba-specific parameters for state-space models
    - **Enhanced Block Configs**: Uses `PuzzletronTransformerBlockConfig` with Mamba support
    - **Improved Building**: Enhanced `__post_init__()` with better config validation
    - **Flexibility**: Supports all Puzzletron attention and MLP variants

    Heterogeneous models refer to transformer architectures where individual layers can differ
    in configuration. Specifically:
        - Attention layers can be MHA, MLA, Mamba, Linear, or No-op (all with their own config)
        - MLP layers can be MoE, MLP, Linear, or No-op (all with their own config)
        - Layers can have parallel blocks that run simultaneously and sum their outputs

    Mamba Parameters (shared across all Mamba layers):
        d_conv: Convolution dimension for Mamba
        expand: Expansion factor for Mamba hidden dimension
        D_has_hdim: Whether D matrix has hidden dimension
        rmsnorm: Whether to use RMS normalization
        norm_before_gate: Whether to normalize before gating
        dt_min/max/scale: Delta time parameters for Mamba
        bias/conv_bias: Bias settings for Mamba layers
        chunk_size: Chunk size for Mamba processing
    """

    heterogeneous_layers_config_path: str = ""
    """Path to the json file containing the heterogeneous block specs."""

    heterogeneous_layers_config_encoded_json: str = ""
    """The contents of the json file containing the heterogeneous block specs. It will be read from
    heterogeneous_layers_config_path at first, then saved forever inside the model checkpoint."""

    per_block_parameters: list[PuzzletronTransformerBlockConfig] = field(init=False)
    """Configuration parameters for each of the transformer blocks in a
    heterogeneous transformer."""

    # all of these can be used to instantiate a MambaMixer, they are shared for all Mamba layers
    d_conv: int = 4
    expand: int = 2
    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_scale: float = 1.0
    bias: bool = False
    conv_bias: bool = True
    chunk_size: int = 128

    def __post_init__(self) -> None:
        if self.kv_channels is None and self.num_attention_heads == 0:
            self.num_attention_heads = 8  # to avoid division by zero
        # Type assertion to help mypy understand the type after the check
        assert isinstance(self.num_attention_heads, int), "num_attention_heads must be an integer"
        if self.heterogeneous_layers_config_encoded_json in ("", None):
            assert self.heterogeneous_layers_config_path not in (
                None,
                "",
            ), (
                "heterogeneous_layers_config_path is required, if heterogeneous_layers_config_encoded_json is not provided"
            )
            self.heterogeneous_layers_config_encoded_json = Path(
                self.heterogeneous_layers_config_path
            ).read_text()
        hf_config_dict: dict[str, Any] = json.loads(self.heterogeneous_layers_config_encoded_json)
        block_list = hf_config_dict["block_configs"]
        # TODO: should we change the definition of num_layers? it can be sum(mlp/attention) rather than uneven blocks
        if self.num_layers is None or self.num_layers == 0:
            self.num_layers = len(block_list)
        # Type assertion to help mypy understand the type after the check
        assert isinstance(self.num_layers, int), "num_layers must be an integer"
        assert self.num_layers == len(block_list), (
            "num_layers must match the number of blocks in the json file"
        )
        super().__post_init__()
        self.heterogeneous_block_specs = True
        self.heterogeneous_dist_checkpoint = True  # TODO: check if this is correct/needed
        self.per_block_parameters = [
            PuzzletronTransformerBlockConfig.build_from_dict(block=block, main_config=self)
            for block in block_list
        ]

    # TODO add parallel blocks support
    def get_config_for_layer(
        self, layer_number: int
    ) -> TransformerConfig | MLATransformerConfig | PuzzletronMambaTransformerConfig:
        """
        Get the config for the given layer number.
        Based on the layer number, the corresponding block config is returned,
        overriding the main config's value.

        Returns:
            TransformerConfig: For standard transformer layers
            MLATransformerConfig: For MLA layers
            PuzzletronMambaTransformerConfig: For Mamba layers
        """
        layer_idx = layer_number - 1  # layer number starts from 1
        if layer_idx < 0 or layer_idx >= len(self.per_block_parameters):
            raise ValueError(
                f"Invalid layer number: {layer_number}. Should be in "
                f"range [1, {len(self.per_block_parameters)}]."
            )
        block_config = self.per_block_parameters[layer_idx]

        # Determine which config class to use based on the block configuration
        if block_config.attention.is_mamba:
            config_class = PuzzletronMambaTransformerConfig
        elif block_config.attention.multi_latent_attention:
            config_class = MLATransformerConfig
        else:
            config_class = TransformerConfig

        # Get all available fields from the attention and MLP configs
        attention_fields = {f.name for f in fields(block_config.attention)}
        mlp_fields = {f.name for f in fields(block_config.mlp)}

        # Get all available fields from the target config class
        target_config_fields = {f.name for f in fields(config_class)}

        # Start with the base config
        transformer_config_dict = asdict(self)

        # Remove keys that are not in the target config class
        transformer_config_dict = {
            k: v for k, v in transformer_config_dict.items() if k in target_config_fields
        }

        # Update with all available attention config values (if they exist in target config)
        for field_name in attention_fields:
            if field_name in target_config_fields:
                transformer_config_dict[field_name] = getattr(block_config.attention, field_name)

        # Update with all available MLP config values (if they exist in target config)
        for field_name in mlp_fields:
            if field_name in target_config_fields:
                transformer_config_dict[field_name] = getattr(block_config.mlp, field_name)

        if transformer_config_dict["num_moe_experts"] is None:
            # to pass __post_init__ of config_class
            transformer_config_dict["expert_model_parallel_size"] = 1
        config = config_class(**transformer_config_dict)

        return config


# NOTE: based on https://github.com/NVIDIA/Megatron-LM/blob/ba97a7e282a8478a02d012bc9b9e45f3a6be216e/megatron/core/extensions/transformer_engine.py#L449
class WrappedTENormLinear(TELayerNormColumnParallelLinear):
    """A wrapper around TELayerNormColumnParallelLinear with simplified interface and forced configurations.

    This wrapper simplifies the interface of TELayerNormColumnParallelLinear by:
    1. Taking only a config object instead of individual parameters
    2. Forcing specific configurations (tp_group=None, tp_size=1, etc.) for compatibility
    3. Adding version checks for Transformer Engine features
    4. Providing a cleaner interface for heterogeneous transformer models

    Key differences from TELayerNormColumnParallelLinear:
    - Simplified constructor: only requires config and optional unused parameters
    - Forces tensor parallel settings: tp_group=None, tp_size=1, tp_rank=0
    - Automatically sets input_size=output_size=config.hidden_size
    - Adds version checks for TE features (delay_wgrad_compute, normalization, symmetric_ar_type)
    - Forces bias=False, skip_bias_add=False for consistency
    - Disables gather_output (raises error if True)
    - Uses simplified init_method=lambda w: None

    This wrapper is designed for use in heterogeneous transformer architectures where
    individual layers may have different configurations but need a consistent interface.
    """

    def __init__(
        self,
        config,
        layer_number=None,  # unused
        model_comm_pgs=None,  # unused
        cp_comm_type=None,  # unused
        tp_group=None,  # unused
        tp_comm_buffer_name=None,
        gather_output=False,  # unused
    ):
        # unfortunately, TELayerNormColumnParallelLinear sets tp_group and forcing it to be None requires to copy/paste __init__
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        self.config = config

        if gather_output:
            raise ValueError("Transformer Engine linear layers do not support gather_output = True")

        skip_bias_add = False
        bias = False

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        extra_kwargs = _get_extra_te_kwargs(config)
        self.tp_size = 1
        self.tp_rank = 0

        if self.config.delay_wgrad_compute:
            if is_te_min_version("2.3.0"):
                extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
            else:
                raise RuntimeError("Only TE with version >=2.3.0 supports delay_wgrad_compute now.")

        # Only Transformer-Engine version >= 0.11.0 supports `RMSNorm`
        if is_te_min_version("0.11.0"):
            extra_kwargs["normalization"] = self.config.normalization
        elif self.config.normalization != "LayerNorm":
            te_version = get_te_version()
            raise ValueError(
                f"Transformer Engine v{te_version} does not support {self.config.normalization}."
            )

        if self.config.symmetric_ar_type is not None:
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"
            extra_kwargs["symmetric_ar_type"] = self.config.symmetric_ar_type

        output_size = config.hidden_size
        input_size = config.hidden_size
        # calling te.pytorch.LayerNormLinear's __init__
        super(TELayerNormColumnParallelLinear, self).__init__(
            in_features=input_size,
            out_features=output_size,
            eps=self.config.layernorm_epsilon,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            tp_group=None,
            tp_size=1,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=lambda w: None,
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=None,
            return_layernorm_output=False,
            zero_centered_gamma=self.config.layernorm_zero_centered_gamma,
            **extra_kwargs,
        )

        if config.use_cpu_initialization:
            output_size_per_partition = output_size
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                output_size_per_partition,
                0,
                init_method=lambda w: None,
                stride=1,
                return_master_weight=False,
                rank=self.tp_rank,
                world_size=self.tp_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(output_size_per_partition, dtype=config.params_dtype)
                )
                with torch.no_grad():
                    self.bias.zero_()

    def forward(self, x, *args, **kwargs):
        return super().forward(x)


class WrappedLinear(Linear):
    def __init__(
        self,
        config,
        layer_number=None,
        model_comm_pgs=None,
        cp_comm_type=None,
        tp_group=None,
        tp_comm_buffer_name=None,
        gather_output=False,
    ):
        super().__init__(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=gather_output,
            skip_bias_add=False,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
        )

    def forward(self, x, *args, **kwargs):
        return super().forward(x)


class WrappedTELinear(TELinear):
    # TODO: docstring
    def __init__(
        self,
        config,
        layer_number=None,  # unused
        model_comm_pgs=None,  # unused
        cp_comm_type=None,  # unused
        tp_group=None,  # unused
        tp_comm_buffer_name=None,
        gather_output=False,  # unused
    ):
        super().__init__(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            parallel_mode="duplicated",
            # parallel_mode=None,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=False,
        )

    def forward(self, x, *args, **kwargs):
        return super().forward(x)


class WrappedMambaMixer(MambaMixer):
    def __init__(self, *args, cp_comm_type: Optional[str] = None, **kwargs):
        # ignoring cp_comm_type
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        result = super().forward(hidden_states, inference_context=inference_context)
        # Ensure we return a tuple of two tensors
        assert isinstance(result, tuple) and len(result) == 2
        return result


# NOTE: new method
def get_layer_spec_for_layer(
    block_params: PuzzletronTransformerBlockConfig,
    config: PuzzletronHeterogeneousTransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
) -> ModuleSpec:
    # this part is copied from megatron.core.models.gpt.gpt_layer_specs.get_gpt_decoder_block_spec()
    if use_transformer_engine:
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=block_params.mlp.num_moe_experts,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=block_params.attention.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            # use_te_activation_func=config.use_te_activation_func, #TODO: part of megatron 0.14 version. check if this is needed now.
        )
    else:
        layer_spec = get_gpt_layer_local_spec(
            num_experts=block_params.mlp.num_moe_experts,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=block_params.attention.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    if block_params.attention.no_op:
        layer_spec.submodules.input_layernorm = IdentityOp
        layer_spec.submodules.self_attn_bda = IdentityFuncOp
        layer_spec.submodules.self_attention = ModuleSpec(module=IdentityOp)
    elif block_params.attention.replace_with_linear:
        layer_spec.submodules.self_attention = ModuleSpec(
            module=WrappedTENormLinear if use_transformer_engine else WrappedLinear,
            params={"tp_comm_buffer_name": "linear_attn"},
        )
    elif block_params.attention.is_mamba:
        mamba_mixer_params = dict(
            d_model=config.hidden_size,
            d_conv=config.d_conv,
            expand=config.expand,
            D_has_hdim=config.D_has_hdim,
            rmsnorm=config.rmsnorm,
            norm_before_gate=config.norm_before_gate,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_scale=config.dt_scale,
            bias=config.bias,
            conv_bias=config.conv_bias,
            chunk_size=config.chunk_size,
        )
        layer_spec.submodules.self_attention = ModuleSpec(
            module=WrappedMambaMixer,
            params=mamba_mixer_params,
            submodules=MambaMixerSubmodules(
                in_proj=(
                    TELayerNormColumnParallelLinear
                    if use_transformer_engine
                    else ColumnParallelLinear
                ),
                out_proj=TERowParallelLinear if use_transformer_engine else RowParallelLinear,
            ),
        )

    if block_params.mlp.no_op:
        layer_spec.submodules.pre_mlp_layernorm = IdentityOp
        layer_spec.submodules.mlp_bda = IdentityFuncOp
        layer_spec.submodules.mlp = ModuleSpec(module=IdentityOp)
    elif block_params.mlp.replace_with_linear:
        layer_spec.submodules.mlp = ModuleSpec(
            module=WrappedTENormLinear if use_transformer_engine else WrappedLinear,
            params={"tp_comm_buffer_name": "linear_mlp"},
        )

    layer_spec.submodules.sharded_state_dict_keys_map = _get_sharded_state_dict_keys_map(
        block_params, use_transformer_engine
    )
    return layer_spec


# NOTE: based on https://github.com/NVIDIA/Megatron-LM/blob/aacc3b8aa5f0d3071431a94503d6233802fbaedd/megatron/core/models/gpt/heterogeneous/heterogeneous_layer_specs.py#L168
def get_gpt_heterogeneous_layer_spec_puzzletron(
    config: PuzzletronHeterogeneousTransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Generate heterogeneous layer specifications for Puzzletron transformer models.

    This function is a specialized version of the original Megatron Core
    `get_gpt_heterogeneous_layer_spec` function, adapted for Puzzletron's specific
    heterogeneous transformer architecture requirements.

    Key differences from the original:
    - **Signature**: Adds `normalization` and `qk_l2_norm` parameters, removes `pp_rank`
    - **Architecture**: Uses `get_layer_spec_for_layer()` helper for modular layer creation
    - **Pipeline Parallel**: Enhanced with `pipeline_model_parallel_layout` support
    - **Configuration**: Uses `PuzzletronHeterogeneousTransformerConfig` with Mamba parameters
    - **Layer Norm**: Simplified to `TENorm` vs `LNImpl` (removes `WrappedTorchNorm` complexity)
    - **Features**: Supports Mamba layers, custom attention types, and advanced parallelization

    Args:
        config: Puzzletron heterogeneous transformer configuration
        use_transformer_engine: Whether to use Transformer Engine optimizations
        normalization: Optional normalization type override
        qk_l2_norm: Whether to apply L2 normalization to QK matrices
        vp_stage: Virtual pipeline stage for advanced parallelization

    Returns:
        TransformerBlockSubmodules: Complete layer specification for the heterogeneous model
    """
    # Create the layer specs for the model.
    layer_specs = [
        get_layer_spec_for_layer(
            block_params, config, use_transformer_engine, normalization, qk_l2_norm
        )
        for block_params in config.per_block_parameters
    ]

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    if use_transformer_engine:
        layer_norm_impl = TENorm
    else:
        layer_norm_impl = LNImpl

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs, layer_norm=layer_norm_impl
    )

    return block_spec


# NOTE: based on https://github.com/NVIDIA/Megatron-LM/blob/aacc3b8aa5f0d3071431a94503d6233802fbaedd/gpt_builders.py#L23
def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None):
    """Build a GPT model with Puzzletron's heterogeneous transformer architecture.

    This function is a specialized version of the original Megatron-LM `gpt_builder` function,
    adapted for Puzzletron's heterogeneous transformer architecture requirements.

    Key differences from the original:
    - **Simplified**: Focuses exclusively on heterogeneous models (rejects legacy, spec-based, MoE, MTP)
    - **Configuration**: Only supports args-based config (removes YAML complexity)
    - **Layer Spec**: Uses single `get_gpt_heterogeneous_layer_spec_puzzletron` function
    - **Error Handling**: Explicit error messages for unsupported features
    - **Logging**: Removes debug logging for cleaner implementation

    Args:
        args: Command-line arguments namespace containing model configuration parameters
        pre_process: Whether to include pre-processing layers
        post_process: Whether to include post-processing layers
        vp_stage: Virtual pipeline stage for advanced parallelization
        config: Optional pre-configured transformer config (if None, created from args)

    Returns:
        GPTModel: Configured GPT model with heterogeneous transformer architecture

    Raises:
        ValueError: If legacy models, spec-based models, or MTP are requested
    """
    assert config is not None, "config is required"
    if args.use_legacy_models:
        raise ValueError("Legacy models are not supported")
    if args.spec is not None:
        raise ValueError("Spec is not supported")
    use_te = args.transformer_impl == "transformer_engine"
    transformer_layer_spec = get_gpt_heterogeneous_layer_spec_puzzletron(
        config,
        use_te,
        normalization=args.normalization,
        qk_l2_norm=args.qk_l2_norm,
        vp_stage=vp_stage,
    )
    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        raise ValueError("MTP is not supported")
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )

    return model
