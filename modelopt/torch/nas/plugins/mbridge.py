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

"""Plugin to add NAS/Pruning support for Megatron-Bridge model-specific layers.

Megatron-Bridge defines model-specific subclasses on top of megatron-core modules (handled in
``megatron.py``). These subclasses override ``forward``/``__init__`` so they are not matched to
their megatron-core parents by ``DMRegistry`` and must be registered explicitly here.

Currently covers Gemma3, whose ``gemma3_layer_spec`` uses:
  - ``Gemma3LanguageModelEmbedding`` (scales embeddings by sqrt(hidden_size)),
  - ``Gemma3SelfAttention`` / ``Gemma3TEDotProductAttention`` (local/global sliding-window rope),
  - ``TERowParallelLinearLayerNorm`` (a post-attention / post-MLP RMSNorm fused into the output
    row-parallel linear) for both ``self_attention.linear_proj`` and ``mlp.linear_fc2``.
"""

import torch
from megatron.bridge.models.gemma.gemma3_provider import (
    Gemma3LanguageModelEmbedding,
    Gemma3SelfAttention,
    TERowParallelLinearLayerNorm,
)

from ..registry import DMRegistry
from ..traced_hp import TracedHp
from .megatron import (
    NumAttentionHeadsHp,
    _DynamicLanguageModelEmbedding,
    _DynamicMCoreLanguageModel,
    _DynamicSelfAttention,
    _DynamicTEProjRowParallelLinear,
    _DynamicTERowParallelLinear,
)

__all__ = []


# Gemma3 ##########################################################################################
# Gemma3LanguageModelEmbedding only overrides forward (embedding scaling); the dynamic setup is
# identical to a plain LanguageModelEmbedding, so reuse the existing dynamic class.
DMRegistry.register(
    {
        Gemma3LanguageModelEmbedding: (
            "megatron.bridge.models.gemma.gemma3_provider.Gemma3LanguageModelEmbedding"
        )
    }
)(_DynamicLanguageModelEmbedding)


# TERowParallelLinearLayerNorm (post-LN row-parallel linear)
def _convert_post_layernorm(mod) -> None:
    """Convert the fused post-LN to dynamic, sliced by the linear's (pruned) output_size."""
    # post_layernorm normalizes over output_size (== hidden_size for both linear_proj and
    # linear_fc2), so it must shrink together with the linear's output_size hparam.
    DMRegistry.convert(mod.post_layernorm, num_features=mod.get_hparam("output_size"))


@DMRegistry.register(
    {
        TERowParallelLinearLayerNorm: (
            "megatron.bridge.models.gemma.gemma3_provider.TERowParallelLinearLayerNorm"
        )
    }
)
class _DynamicTERowParallelLinearLayerNorm(_DynamicTERowParallelLinear):
    """``TERowParallelLinearLayerNorm`` with dynamic hyperparams (used for ``mlp.linear_fc2``).

    Adds dynamic handling of the fused ``post_layernorm`` on top of the row-parallel linear.
    """

    def _setup(self, *, input_size: TracedHp | None = None, output_size: TracedHp | None = None):
        super()._setup(input_size=input_size, output_size=output_size)
        _convert_post_layernorm(self)

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a torch.nn.Module."""
        self.post_layernorm.export()
        return super().export()


# Gemma3 SelfAttention
# NOTE: Not registered to DMRegistry; converted directly inside _DynamicGemma3SelfAttention since
# it uses the num_attention_heads/hidden_size setup (like _DynamicTEProjRowParallelLinear).
class _DynamicTEProjRowParallelLinearLayerNorm(
    _DynamicTEProjRowParallelLinear, TERowParallelLinearLayerNorm
):
    """``TERowParallelLinearLayerNorm`` for attention output projection (``linear_proj``).

    Adds dynamic handling of the fused ``post_layernorm`` on top of the attention output projection.
    """

    def _setup(self, *, num_attention_heads: NumAttentionHeadsHp, hidden_size: TracedHp):
        super()._setup(num_attention_heads=num_attention_heads, hidden_size=hidden_size)
        _convert_post_layernorm(self)

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a torch.nn.Module."""
        self.post_layernorm.export()
        return super().export()


@DMRegistry.register(
    {Gemma3SelfAttention: "megatron.bridge.models.gemma.gemma3_provider.Gemma3SelfAttention"}
)
class _DynamicGemma3SelfAttention(_DynamicSelfAttention):
    """A Gemma3SelfAttention layer with dynamic hyperparams.

    Identical to ``_DynamicSelfAttention`` except its output projection is the post-LN
    ``TERowParallelLinearLayerNorm`` instead of a plain ``TERowParallelLinear``.
    """

    def _convert_linear_proj(
        self, *, num_attention_heads: NumAttentionHeadsHp, hidden_size: TracedHp
    ) -> None:
        _DynamicTEProjRowParallelLinearLayerNorm.convert(
            self.linear_proj,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
        )


# Qwen3-VL / Qwen3.5-VL ###########################################################################
# The VLM wraps the language model as ``Qwen3VLModel.language_model`` (a ``Qwen3VLGPTModel``); prune
# the language model by passing ``model.language_model`` to ``mtp.prune``. Both the LM class and its
# full-attention class override ``forward`` (absolute mRoPE), so they need explicit registration.
# GatedDeltaNet (Qwen3.5) and gated attention are already handled in ``megatron.py``.
# Guarded import — these classes exist only in newer Megatron-Bridge builds with the Qwen3-VL bridge.
try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel

    _QWEN3VL_PREFIX = "megatron.bridge.models.qwen_vl.modelling_qwen3_vl"

    # Both subclasses only override forward; their dynamic setup is identical to the megatron-core
    # base classes, so reuse the existing dynamic classes directly (no behavioral subclass needed).
    DMRegistry.register({Qwen3VLGPTModel: f"{_QWEN3VL_PREFIX}.text_model.Qwen3VLGPTModel"})(
        _DynamicMCoreLanguageModel
    )
    DMRegistry.register(
        {Qwen3VLSelfAttention: f"{_QWEN3VL_PREFIX}.attention.Qwen3VLSelfAttention"}
    )(_DynamicSelfAttention)

except ImportError:
    pass
