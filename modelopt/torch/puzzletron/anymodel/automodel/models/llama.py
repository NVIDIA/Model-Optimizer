# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from ..auto_model_descriptor import AutoModelDescriptor, AutoModelDescriptorFactory

__all__ = ["LlamaAutoModelDescriptor"]


@AutoModelDescriptorFactory.register_decorator("llama", "nemotron_v3_dense")
class LlamaAutoModelDescriptor(AutoModelDescriptor):
    """Native AutoModel descriptor for Llama-style dense decoder blocks."""

    @staticmethod
    def decoder_layer_cls():
        from nemo_automodel.components.models.llama.model import LlamaDecoderLayer

        return LlamaDecoderLayer

    @staticmethod
    def attn_no_op_post_init(layer: Any) -> None:
        from ...puzzformer.no_op import MatchingZeros, Same, return_tuple_of_size

        layer.input_layernorm = Same()
        layer.self_attn = return_tuple_of_size(MatchingZeros, size=2)()

    @staticmethod
    def mlp_no_op_post_init(layer: Any) -> None:
        from ...puzzformer.no_op import MatchingZeros, Same

        layer.post_attention_layernorm = Same()
        layer.mlp = MatchingZeros()

    @classmethod
    def make_patched_init(cls, orig_init, block_configs):
        """NeMo LlamaDecoderLayer uses ``(config, layer_idx, backend=None)``."""

        def _patched(self, config, layer_idx, backend=None, *args, **kwargs):
            # AutoModel may construct auxiliary decoder layers (for example MTP)
            # after the main decoder stack. Those layers are intentionally not
            # described by Puzzletron's main block_configs.
            block_config = (
                block_configs[layer_idx]
                if block_configs and 0 <= layer_idx < len(block_configs)
                else None
            )
            config = cls._apply_overrides(config, block_config)
            orig_init(self, config, layer_idx, backend, *args, **kwargs)
            if block_config is None:
                return
            attn = block_config.get_subblock("attention")
            ffn = block_config.get_subblock("ffn")
            if attn is not None and attn.no_op:
                cls.attn_no_op_post_init(self)
            if ffn is not None and ffn.no_op:
                cls.mlp_no_op_post_init(self)

        return _patched
