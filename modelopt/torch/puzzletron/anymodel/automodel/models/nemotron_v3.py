# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from ...puzzformer.no_op import MatchingZeros, Same
from ..auto_model_descriptor import AutoModelDescriptor, AutoModelDescriptorFactory

__all__ = ["NemotronV3AutoModelDescriptor"]


@AutoModelDescriptorFactory.register_decorator("nemotron_h", "nemotron_v3")
class NemotronV3AutoModelDescriptor(AutoModelDescriptor):
    """Native AutoModel descriptor for Nemotron3/NemotronH custom blocks."""

    @staticmethod
    def decoder_layer_cls():
        from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block

        return NemotronV3Block

    @staticmethod
    def block_config_to_config_overrides(block_config):
        overrides = AutoModelDescriptor.block_config_to_config_overrides(block_config)
        moe = block_config.get_subblock("moe")
        mamba = block_config.get_subblock("mamba")
        if moe is not None:
            overrides.pop("num_experts", None)
            if moe.num_experts is not None:
                overrides["n_routed_experts"] = moe.num_experts
            if moe.top_k is not None:
                overrides["num_experts_per_tok"] = moe.top_k
            if moe.expert_intermediate_size is not None:
                overrides["moe_intermediate_size"] = moe.expert_intermediate_size
            if moe.shared_expert_intermediate_size is not None:
                overrides["moe_shared_expert_intermediate_size"] = (
                    moe.shared_expert_intermediate_size
                )
            if moe.latent_dim is not None:
                overrides["moe_latent_size"] = moe.latent_dim
        if mamba is not None:
            if mamba.num_heads is not None:
                overrides["mamba_num_heads"] = mamba.num_heads
            if mamba.head_dim is not None:
                overrides["mamba_head_dim"] = mamba.head_dim
            if mamba.state_dim is not None:
                overrides["ssm_state_size"] = mamba.state_dim
        return overrides

    @staticmethod
    def _block_no_op_post_init(layer: Any) -> None:
        block_config = layer.config.block_configs[layer.layer_idx]
        attn = block_config.get_subblock("attention")
        ffn = block_config.get_subblock("ffn")
        mamba = block_config.get_subblock("mamba")
        moe = block_config.get_subblock("moe")
        if (
            (attn is None or attn.no_op)
            and (ffn is None or ffn.no_op)
            and (mamba is None or mamba.no_op)
            and (moe is None or moe.no_op)
        ):
            layer.norm = Same()
            layer.mixer = MatchingZeros()
            if hasattr(layer, "self_attn"):
                layer.self_attn = layer.mixer

    @staticmethod
    def attn_no_op_post_init(layer: Any) -> None:
        NemotronV3AutoModelDescriptor._block_no_op_post_init(layer)

    @staticmethod
    def mlp_no_op_post_init(layer: Any) -> None:
        NemotronV3AutoModelDescriptor._block_no_op_post_init(layer)
