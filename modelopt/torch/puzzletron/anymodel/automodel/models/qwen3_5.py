# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from ....block_config import AttentionConfig, FFNConfig, MambaConfig, MoEConfig
from ...puzzformer.no_op import MatchingZeros, Same
from ..auto_model_descriptor import AutoModelDescriptor, AutoModelDescriptorFactory

__all__ = ["Qwen3_5AutoModelDescriptor", "Qwen3_5MoeAutoModelDescriptor"]


@AutoModelDescriptorFactory.register_decorator(
    "qwen3_5_text", "qwen3_6_text", "qwen3_5", "qwen3_6"
)
class Qwen3_5AutoModelDescriptor(AutoModelDescriptor):
    """Native AutoModel descriptor shared by Qwen3.5/3.6 text and VLM models."""

    @staticmethod
    def decoder_layer_cls():
        from nemo_automodel.components.models.qwen3_5.model import Qwen3_5DenseBlock

        return Qwen3_5DenseBlock

    @staticmethod
    def block_config_to_config_overrides(block_config):
        overrides = AutoModelDescriptor.block_config_to_config_overrides(block_config)
        mamba = block_config.get_subblock("mamba")
        if isinstance(mamba, MambaConfig):
            overrides.pop("mamba_num_heads", None)
            overrides.pop("mamba_head_dim", None)
            overrides.pop("ssm_state_size", None)
            if mamba.num_groups is not None:
                overrides["linear_num_key_heads"] = mamba.num_groups
            if mamba.num_heads is not None:
                overrides["linear_num_value_heads"] = mamba.num_heads
            if mamba.state_dim is not None:
                overrides["linear_key_head_dim"] = mamba.state_dim
            if mamba.head_dim is not None:
                overrides["linear_value_head_dim"] = mamba.head_dim
        return overrides

    @staticmethod
    def attn_no_op_post_init(layer: Any) -> None:
        layer.input_layernorm = Same()
        if layer.layer_type == "linear_attention":
            layer.linear_attn = MatchingZeros()
            # Native Qwen's linear-attention initializer directly touches GDN
            # tensors instead of delegating to ``init_weights``.  Temporarily
            # hide the pruned branch only during initialization; forward keeps
            # the original layer type and therefore calls MatchingZeros.
            original_init_weights = layer.init_weights

            def _init_weights_without_pruned_linear_attention(buffer_device):
                original_layer_type = layer.layer_type
                layer.layer_type = "puzzletron_no_op"
                try:
                    return original_init_weights(buffer_device)
                finally:
                    layer.layer_type = original_layer_type

            layer.init_weights = _init_weights_without_pruned_linear_attention
        else:
            layer.self_attn = MatchingZeros()

    @staticmethod
    def mlp_no_op_post_init(layer: Any) -> None:
        layer.post_attention_layernorm = Same()
        layer.mlp = MatchingZeros()

        # Native Qwen routes every FFN call through a typed helper that only
        # accepts its dense MLP or MoE classes.  A realized Puzzletron no-op is
        # intentionally neither, so bypass that dispatch while preserving the
        # block's residual connection (``x + 0``).
        def _no_op_mlp(*, x, padding_mask=None):
            del padding_mask
            return layer.mlp(x)

        layer._mlp = _no_op_mlp

    @classmethod
    def make_patched_init(cls, orig_init, block_configs):
        """Adapt the native ``(layer_idx, config, moe_config, backend)`` signature."""

        def _patched(self, layer_idx, config, moe_config, backend, *args, **kwargs):
            block_config = (
                block_configs[layer_idx]
                if block_configs and 0 <= layer_idx < len(block_configs)
                else None
            )
            config = cls._apply_overrides(config, block_config)
            if block_configs and block_config is not None:
                config.block_configs = list(block_configs)
                config.layer_types = [
                    "linear_attention"
                    if candidate.get_subblock("mamba") is not None
                    else "full_attention"
                    for candidate in block_configs
                ]
            orig_init(self, layer_idx, config, moe_config, backend, *args, **kwargs)
            if block_config is None:
                return
            attn = block_config.get_subblock("attention")
            ffn = block_config.get_subblock("ffn")
            mamba = block_config.get_subblock("mamba")
            moe = block_config.get_subblock("moe")
            if (
                (isinstance(attn, AttentionConfig) and attn.no_op)
                or (isinstance(mamba, MambaConfig) and mamba.no_op)
            ):
                cls.attn_no_op_post_init(self)
            if isinstance(ffn, FFNConfig) and ffn.no_op:
                cls.mlp_no_op_post_init(self)
            if isinstance(moe, MoEConfig) and moe.no_op:
                cls.mlp_no_op_post_init(self)

        return _patched


@AutoModelDescriptorFactory.register_decorator(
    "qwen3_5_moe_text", "qwen3_6_moe_text", "qwen3_5_moe", "qwen3_6_moe"
)
class Qwen3_5MoeAutoModelDescriptor(Qwen3_5AutoModelDescriptor):
    """Native Qwen3.5/3.6 MoE bridge using the shared Qwen axis mapping."""

    @staticmethod
    def decoder_layer_cls():
        from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeBlock

        return Qwen3_5MoeBlock
