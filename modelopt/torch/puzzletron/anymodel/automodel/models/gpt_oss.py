# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from ...models.gpt_oss.gpt_oss_model_descriptor import GptOssModelDescriptor
from ...puzzformer.no_op import MatchingZeros, Same
from ..auto_model_descriptor import AutoModelDescriptorFactory, ContractAutoModelDescriptor

__all__ = ["GptOssAutoModelDescriptor"]


@AutoModelDescriptorFactory.register_decorator("gpt_oss")
class GptOssAutoModelDescriptor(ContractAutoModelDescriptor):
    """Native GPT-OSS bridge using the shared GQA/window/MoE contracts."""

    STRUCTURAL_DESCRIPTOR = GptOssModelDescriptor

    @staticmethod
    def decoder_layer_cls():
        from nemo_automodel.components.models.gpt_oss.model import Block

        return Block

    @staticmethod
    def attn_no_op_post_init(layer: Any) -> None:
        layer.input_layernorm = Same()
        layer.self_attn = MatchingZeros()

    @staticmethod
    def mlp_no_op_post_init(layer: Any) -> None:
        layer.post_attention_layernorm = Same()
        layer.mlp = MatchingZeros()

        def _no_op_mlp(x, padding_mask=None):
            del padding_mask
            return layer.mlp(x)

        layer._mlp = _no_op_mlp
