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

"""Plugin to add NAS/Pruning support for megatron-core Language models like GPT and Mamba."""

import types
from abc import ABC
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.gpt import GPTModel
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_layer import TransformerLayer

from modelopt.torch.nas.modules import DynamicModuleList
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.opt.hparam import HPType
from modelopt.torch.opt.searcher import ConstraintsDict
from modelopt.torch.trace import Symbol
from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import (
    get_module_device,
    make_divisible,
    param_num_from_forward,
    print_rank_0,
    random,
)

from ..algorithms import (
    MODULE_TYPE_TO_CONSTRAINTS_FUNC,
    ConstraintEvalFunc,
    ConstraintInterpolator,
    ConstraintsFunc,
    ConstraintsRes,
)
from ..hparams.concat import build_concat_hp
from ..modules import _DynamicLayerNorm
from ..modules.utils import get_sliced_tensor, get_sliced_tensor_by_slices
from ..registry import DMRegistry
from ..search_space import SampleFunc
from ..traced_hp import TracedHp

SUPPORTED_MODELS = {GPTModel: "megatron.core.models.gpt.GPTModel"}

try:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    HAS_TE = True
except ImportError:
    HAS_TE = False

try:
    import mamba_ssm  # noqa: F401
    from megatron.core.models.mamba import MambaModel
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.ssm.mamba_mixer import ExtendedRMSNorm, MambaMixer

    SUPPORTED_MODELS[MambaModel] = "megatron.core.models.mamba.MambaModel"

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

__all__ = []


class _DynamicParallelLinear(DynamicModule):
    """A parallel linear layer with dynamic hyperparams."""

    def _setup(self, *, input_size: TracedHp | None = None, output_size: TracedHp | None = None):
        # register hyperparameters
        if input_size is None:
            input_size = TracedHp(list(range(1, self.input_size + 1)))
        self._register_hparam("input_size", input_size)

        if output_size is None:
            output_size = TracedHp(list(range(1, self.output_size + 1)))
        self._register_hparam("output_size", output_size)

        # register dynamic attributes of the class
        self._register_dynamic_attribute("weight", self._get_weight)
        self._register_dynamic_attribute("bias", self._get_bias)

    @staticmethod
    def _get_weight(mod: "_DynamicParallelLinear", weight: torch.Tensor) -> torch.Tensor:
        return get_sliced_tensor(mod, weight, "output_size", "input_size")

    @staticmethod
    def _get_bias(mod: "_DynamicParallelLinear", bias: torch.Tensor | None) -> torch.Tensor | None:
        return get_sliced_tensor(mod, bias, "output_size")


@DMRegistry.register(
    {ColumnParallelLinear: "megatron.core.tensor_parallel.layers.ColumnParallelLinear"}
)
class _DynamicColumnParallelLinear(_DynamicParallelLinear):
    """A ColumnParallelLinear layer with dynamic hyperparams."""

    def _setup(self, *, input_size: TracedHp | None = None, output_size: TracedHp | None = None):
        super()._setup(input_size=input_size, output_size=output_size)
        self._register_dynamic_attribute(
            "output_size_per_partition", lambda mod, val: mod.output_size
        )


@DMRegistry.register({RowParallelLinear: "megatron.core.tensor_parallel.layers.RowParallelLinear"})
class _DynamicRowParallelLinear(_DynamicParallelLinear):
    """A RowParallelLinear layer with dynamic hyperparams."""

    def _setup(self, *, input_size: TracedHp | None = None, output_size: TracedHp | None = None):
        super()._setup(input_size=input_size, output_size=output_size)
        self._register_dynamic_attribute(
            "input_size_per_partition", lambda mod, val: mod.input_size
        )


# Embedding DynamicModule ##########################################################################
@DMRegistry.register(
    {
        VocabParallelEmbedding: "megatron.core.tensor_parallel.layers.VocabParallelEmbedding",
        nn.Embedding: "nn.Embedding",
    },
)
class _DynamicEmbedding(DynamicModule):
    """A Embedding layer with dynamic hyperparams."""

    def _setup(self, *, embedding_dim: TracedHp | None = None):
        if embedding_dim is None:
            embedding_dim = TracedHp(list(range(1, self.embedding_dim + 1)))
        self._register_hparam("embedding_dim", embedding_dim)
        self._register_dynamic_attribute("weight", self._get_weight)

    @staticmethod
    def _get_weight(mod: "_DynamicEmbedding", weight: torch.Tensor) -> torch.Tensor:
        """Return the weight tensor of the embedding layer."""
        return get_sliced_tensor(mod, weight, None, "embedding_dim")


@DMRegistry.register(
    {
        LanguageModelEmbedding: "megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding"
    }
)
class _DynamicLanguageModelEmbedding(DynamicModule):
    """A LanguageModelEmbedding layer with dynamic hyperparams."""

    def _setup(self):
        # Use same embedding_dim hparam for position and tokentype embeddings
        DMRegistry.convert(self.word_embeddings)
        hp_hidden_size = self.word_embeddings.get_hparam("embedding_dim")
        if hasattr(self, "position_embeddings") and self.position_embeddings is not None:
            DMRegistry.convert(self.position_embeddings, embedding_dim=hp_hidden_size)
        if hasattr(self, "tokentype_embeddings") and self.tokentype_embeddings is not None:
            DMRegistry.convert(self.tokentype_embeddings, embedding_dim=hp_hidden_size)

    def export(self) -> torch.nn.Module:
        self.word_embeddings.export()
        if hasattr(self, "position_embeddings") and self.position_embeddings is not None:
            self.position_embeddings.export()
        if hasattr(self, "tokentype_embeddings") and self.tokentype_embeddings is not None:
            self.tokentype_embeddings.export()
        return super().export()


# Normalization DynamicModule ######################################################################
@DMRegistry.register({FusedLayerNorm: "megatron.core.fusions.fused_layer_norm.FusedLayerNorm"})
class _DynamicFusedLayerNorm(_DynamicLayerNorm):
    """A FusedLayerNorm layer with dynamic hyperparams."""

    def _setup(self, *, num_features: TracedHp):
        """Setup the FusedLayerNorm dynamic module with pre-defined num_features hparam."""
        self._register_hparam("num_features", num_features)

        # register dynamic attributes
        self._register_dynamic_attribute("weight", self._cut_to_active_features)
        self._register_dynamic_attribute("bias", self._cut_to_active_features)
        self._register_dynamic_attribute("hidden_size", self._get_normalized_shape)


# MLP DynamicModule ################################################################################
@DMRegistry.register(
    {
        MLP: "megatron.core.transformer.mlp.MLP",
        SharedExpertMLP: "megatron.core.transformer.moe.shared_experts.SharedExpertMLP",
    }
)
class _DynamicMLP(DynamicModule):
    """An MLP layer with dynamic hyperparams.

    Use for standard MLP and inside MoE layers (SequentialMLP and SharedExpertMLP).
    """

    def _setup(self, *, hidden_size: TracedHp):
        """Setup the MLP dynamic module with global hidden_size hparam."""
        assert self.input_size == self.config.hidden_size, (
            "MLP input_size must be equal to hidden_size"
        )
        if isinstance(self, SharedExpertMLP):
            self.hparam_name = "moe_shared_expert_intermediate_size"
        elif self.config.num_moe_experts is not None:
            self.hparam_name = "moe_ffn_hidden_size"
        else:
            self.hparam_name = "ffn_hidden_size"

        ffn_hidden_size = TracedHp(list(range(1, self.config.ffn_hidden_size + 1)))
        self._register_hparam(self.hparam_name, ffn_hidden_size)

        linear_fc1_output_size = (
            build_concat_hp([ffn_hidden_size] * 2)
            if self.config.gated_linear_unit
            else ffn_hidden_size
        )
        DMRegistry.convert(
            self.linear_fc1, input_size=hidden_size, output_size=linear_fc1_output_size
        )

        DMRegistry.convert(self.linear_fc2, input_size=ffn_hidden_size, output_size=hidden_size)

        self._register_dynamic_attribute("input_size", lambda mod, val: mod.linear_fc1.input_size)

    def modify(self, ffn_hidden_size_divisor: int, **kwargs) -> None:
        """Modify the ffn_hidden_size hparam choices based on search space config."""
        hp_mlp = self.get_hparam(self.hparam_name)
        choices = {int(make_divisible(c, ffn_hidden_size_divisor)) for c in hp_mlp.choices}  # type: ignore[arg-type]
        hp_mlp.choices = list(set(hp_mlp.choices) & choices | {hp_mlp.original})

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a torch.nn.Module."""
        self.linear_fc1.export()
        self.linear_fc2.export()
        return super().export()


# SelfAttention DynamicModules #####################################################################
def expand_head_indices(heads: torch.LongTensor, hidden_size_per_head: int) -> torch.LongTensor:
    """Expand each head index to hidden_size_per_head indices and offset by head * hidden_size_per_head."""
    return (
        heads[:, None].repeat(1, hidden_size_per_head) * hidden_size_per_head
        + torch.arange(hidden_size_per_head, device=heads.device)[None, :]
    ).flatten()


class NumAttentionHeadsHp(TracedHp):
    """Configurable hparam for total number of attention heads.

    Choices are constrained to be multiples of num_query_groups to maintain valid GQA configurations.
    Provides group-aware sorting and slicing through active_slice property.
    """

    def __init__(self, num_attention_heads: int, num_query_groups: int) -> None:
        """Initialize with choices that are multiples of num_query_groups."""
        choices = [
            h * num_query_groups for h in range(1, num_attention_heads // num_query_groups + 1)
        ]
        super().__init__(choices)
        self._num_query_groups = num_query_groups

    @property
    def num_heads_per_group(self) -> int:
        """Return the active number of heads per group."""
        active = self.active
        assert isinstance(active, int)
        return active // self._num_query_groups

    @property
    def active_slice(self) -> torch.LongTensor:
        """Return sorted indices for attention heads with group-aware sorting.

        Heads are sorted within each query group to maintain GQA structure.
        """
        max_heads_per_group = self.max // self._num_query_groups

        if self._slice_order is None:
            all_head_ranking = torch.arange(self.max).view(self._num_query_groups, -1)
        else:
            # Reshape to 2D if needed: [num_query_groups, max_heads_per_group]
            all_head_ranking = self._slice_order.view(self._num_query_groups, max_heads_per_group)

        # Select active heads per group (keep all query groups, trim heads within each group)
        selected_attn_heads = all_head_ranking[:, : self.num_heads_per_group].flatten()

        return selected_attn_heads


# NOTE: We provide a parent class since we do not register to DMRegistry.
class _DynamicQKVColumnParallelLinear(DynamicModule, ColumnParallelLinear):
    """An mcore ColumnParallelLinear layer for linear_qkv with dynamic attributes."""

    def _setup(self, *, num_attention_heads: NumAttentionHeadsHp, hidden_size: TracedHp):
        """Setup the _DynamicQKVColumnParallelLinear dynamic module with global hidden_size hparam."""
        self._register_hparam("input_size", hidden_size)
        self._register_hparam("num_attention_heads", num_attention_heads)
        self._register_dynamic_attribute(
            "output_size",
            lambda mod, val: (num_attention_heads.active + 2 * mod.config.num_query_groups)
            * mod.config.kv_channels,
        )
        self._register_dynamic_attribute(
            "output_size_per_partition", lambda mod, val: mod.output_size
        )
        self._register_dynamic_attribute("weight", self._get_weight)
        self._register_dynamic_attribute("bias", self._get_bias)

    def _get_output_size_indices(self) -> torch.LongTensor:
        """Get the indices of the output size based on sorted + pruned attention heads.

        QKV layout: For each query group: [Q_head_0, ..., Q_head_n, K_head, V_head]

        Example: 32 attention heads (8 groups x 4 heads/group) → 48 QKV heads (8 groups x 6)

        Attention layout (8 groups x 4 heads/group):
            Group 0: [0, 1, 2, 3],  Group 1: [4, 5, 6, 7],  ...,  Group 7: [28, 29, 30, 31]

        QKV layout (8 groups x 6 heads/group):
            Group 0: [0, 1, 2, 3, 4, 5]    (Q0, Q1, Q2, Q3, K, V)
            Group 1: [6, 7, 8, 9, 10, 11]  (Q0, Q1, Q2, Q3, K, V)
            ...
            Group 7: [42, 43, 44, 45, 46, 47]

        If active_slice returns top-2 ranked Q heads per group (by importance):
            E.g., [3, 1, 7, 4, 9, 11, ...] (ranked by importance within each group)
            Reshaped per-group: [[3, 1], [7, 4], [9, 11], ...]
            Group IDs: [[0, 0], [1, 1], [2, 2], ...]
            Local positions: [[3, 1], [3, 0], [1, 3], ...]
            After mapping to QKV: [[3, 1], [9, 6], [13, 15], ...]
            With K,V added: [[3, 1, 4, 5], [9, 6, 10, 11], [13, 15, 16, 17], ...]
        """
        nheads_hp: NumAttentionHeadsHp = self.get_hparam("num_attention_heads")
        nquery_groups = self.config.num_query_groups
        max_nheads_per_group = nheads_hp.max // nquery_groups
        nheads_per_group = nheads_hp.num_heads_per_group
        qkv_heads_per_group = max_nheads_per_group + 2  # Q heads + K head + V head

        if nheads_hp._slice_order is None and nheads_hp.active == nheads_hp.max:
            return slice((max_nheads_per_group + 2) * nquery_groups * self.config.kv_channels)

        # Get sorted and pruned Q head indices: [nquery_groups * active_num_heads_per_group]
        q_head_indices = nheads_hp.active_slice
        assert isinstance(q_head_indices, torch.LongTensor)

        q_head_indices_per_group = q_head_indices.view(nquery_groups, nheads_per_group).cpu()

        # Map from attention layout to QKV layout
        # Determine which group each Q head belongs to
        group_ids = q_head_indices_per_group // max_nheads_per_group

        # Subtract old offset to get local position within attention group
        local_pos_in_attn = q_head_indices_per_group - group_ids * max_nheads_per_group

        # Add new offset to map to QKV layout (same local position, different group stride)
        q_head_indices = group_ids * qkv_heads_per_group + local_pos_in_attn

        # Add K,V head indices for each group (always at fixed positions within each group)
        # K head is at position max_nheads_per_group, V head at max_nheads_per_group + 1
        kv_head_indices = (
            torch.arange(nquery_groups)[:, None] * qkv_heads_per_group
            + torch.arange(max_nheads_per_group, qkv_heads_per_group)[None, :]
        )

        # Concatenate Q and K,V heads: [nquery_groups, active_num_heads_per_group + 2]
        selected_qkv_heads = torch.cat([q_head_indices, kv_head_indices], dim=1).flatten()

        # Expand to dimension indices
        selected_indices = expand_head_indices(selected_qkv_heads, self.config.kv_channels)

        return selected_indices.cpu()

    @staticmethod
    def _get_weight(mod: "_DynamicQKVColumnParallelLinear", weight: torch.Tensor) -> torch.Tensor:
        """Return the weight tensor of the linear layer."""
        return get_sliced_tensor_by_slices(
            weight, [mod._get_output_size_indices(), mod.get_hparam("input_size").active_slice]
        )

    @staticmethod
    def _get_bias(
        mod: "_DynamicQKVColumnParallelLinear", bias: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Return the bias tensor of the linear layer."""
        if bias is None:
            return bias
        return get_sliced_tensor_by_slices(bias, [mod._get_output_size_indices()])


# NOTE: We provide a parent class since we do not register to DMRegistry.
class _DynamicProjRowParallelLinear(DynamicModule, RowParallelLinear):
    """An mcore RowParallelLinear layer for linear_qkv with dynamic attributes."""

    def _setup(self, *, num_attention_heads: NumAttentionHeadsHp, hidden_size: TracedHp):
        """Setup the _DynamicProjRowParallelLinear dynamic module with global hidden_size hparam."""
        self._register_hparam("output_size", hidden_size)
        self._register_hparam("num_attention_heads", num_attention_heads)
        self._register_dynamic_attribute(
            "input_size", lambda mod, val: num_attention_heads.active * mod.config.kv_channels
        )
        self._register_dynamic_attribute(
            "input_size_per_partition", lambda mod, val: mod.input_size
        )
        self._register_dynamic_attribute("weight", self._get_weight)
        self._register_dynamic_attribute("bias", self._get_bias)

    def _get_input_size_indices(self) -> torch.LongTensor:
        """Get the indices of the input size based on sorted + pruned heads and query groups."""
        nheads_hp = self.get_hparam("num_attention_heads")
        if nheads_hp._slice_order is None and nheads_hp.active == nheads_hp.max:
            return slice(nheads_hp.max * self.config.kv_channels)

        selected_attn_heads = nheads_hp.active_slice
        assert isinstance(selected_attn_heads, torch.LongTensor)
        selected_indices = expand_head_indices(selected_attn_heads, self.config.kv_channels)

        return selected_indices.cpu()

    @staticmethod
    def _get_weight(mod: "_DynamicProjRowParallelLinear", weight: torch.Tensor) -> torch.Tensor:
        """Return the weight tensor of the linear layer."""
        return get_sliced_tensor_by_slices(
            weight, [mod.get_hparam("output_size").active_slice, mod._get_input_size_indices()]
        )

    @staticmethod
    def _get_bias(
        mod: "_DynamicProjRowParallelLinear", bias: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Return the bias tensor of the linear layer."""
        return get_sliced_tensor(mod, bias, "output_size")


@DMRegistry.register({SelfAttention: "megatron.core.transformer.attention.SelfAttention"})
class _DynamicSelfAttention(DynamicModule):
    """A SelfAttention layer with dynamic hyperparams.

    NOTE: Layernorms apply on hidden_size_per_attention_head hence no need to convert to dynamic
    """

    def _setup(self, *, hidden_size: TracedHp):
        """Setup the SelfAttention dynamic module with global hidden_size hparam."""
        # Register hparams
        num_attention_heads = NumAttentionHeadsHp(
            self.num_attention_heads_per_partition, self.num_query_groups_per_partition
        )
        self._register_hparam("num_attention_heads", num_attention_heads)

        # Register dynamic attributes
        self._register_dynamic_attribute(
            "num_attention_heads_per_partition", lambda mod, val: self.num_attention_heads
        )

        # Convert the Dot Product Attention to dynamic module
        if isinstance(self.core_attention, DotProductAttention):
            _DynamicDotProductAttention: DynamicModule = type(  # noqa: N806
                "_DynamicDotProductAttention",
                (DynamicModule, DotProductAttention),
                {"_setup": lambda self: None},
            )

            _DynamicDotProductAttention.convert(self.core_attention)
            self.core_attention._register_dynamic_attribute(
                "hidden_size_per_partition",
                lambda mod, val: self.config.kv_channels * self.num_attention_heads_per_partition,
            )
            self.core_attention._register_dynamic_attribute(
                "num_attention_heads_per_partition",
                lambda mod, val: self.num_attention_heads_per_partition,
            )
        else:
            assert HAS_TE and isinstance(self.core_attention, TEDotProductAttention)

            _DynamicTEDotProductAttention: DynamicModule = type(  # noqa: N806
                "_DynamicTEDotProductAttention",
                (DynamicModule, TEDotProductAttention),
                {"_setup": lambda self: None},
            )

            _DynamicTEDotProductAttention.convert(self.core_attention)
            self.core_attention._register_dynamic_attribute(
                "num_attention_heads", lambda mod, val: self.num_attention_heads_per_partition
            )

        # Convert the fused qkv and output projection linear layer to dynamic module
        _DynamicQKVColumnParallelLinear.convert(
            self.linear_qkv, num_attention_heads=num_attention_heads, hidden_size=hidden_size
        )
        _DynamicProjRowParallelLinear.convert(
            self.linear_proj, num_attention_heads=num_attention_heads, hidden_size=hidden_size
        )

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a torch.nn.Module."""
        self.core_attention.export()
        self.linear_qkv.export()
        self.linear_proj.export()
        return super().export()


# MoE DynamicModules ###############################################################################
# Add ABC to avoid TypeError: object layout differs (because parent if TopKRouter inherits from ABC)
@DMRegistry.register({TopKRouter: "megatron.core.transformer.moe.router.TopKRouter"})
class _DynamicTopKRouter(ABC, DynamicModule):
    """A TopKRouter with dynamic hyperparams."""

    def _setup(self, *, hidden_size: TracedHp, num_experts: TracedHp):
        """Setup the TopKRouter dynamic module with global hidden_size hparam."""
        # Register hparams for router weight dimensions
        # Router weight shape: [num_experts, hidden_size]
        self._register_hparam("num_experts", num_experts)
        self._register_hparam("hidden_size", hidden_size)

        # Register dynamic attributes
        self._register_dynamic_attribute("weight", self._get_router_weight)
        if self.config.add_bias_linear:
            self._register_dynamic_attribute("bias", self._get_slice_by_num_experts)
        if self.enable_expert_bias:
            self._register_dynamic_attribute(
                "local_tokens_per_expert", self._get_slice_by_num_experts
            )
            self._register_dynamic_attribute("expert_bias", self._get_slice_by_num_experts)

    @staticmethod
    def _get_router_weight(mod: "_DynamicTopKRouter", weight: torch.Tensor) -> torch.Tensor:
        return get_sliced_tensor(mod, weight, "num_experts", "hidden_size")

    @staticmethod
    def _get_slice_by_num_experts(mod: "_DynamicTopKRouter", val: torch.Tensor) -> torch.Tensor:
        return get_sliced_tensor(mod, val, "num_experts")


@DMRegistry.register({SequentialMLP: "megatron.core.transformer.moe.experts.SequentialMLP"})
class _DynamicSequentialMLP(DynamicModule):
    """A SequentialMLP with dynamic hyperparams."""

    def _setup(self, *, hidden_size: TracedHp):
        """Setup the SequentialMLP dynamic module with global hidden_size hparam."""
        # Register hparam for number of active experts (will be shared with _DynamicTopKRouter's hp)
        num_moe_experts = TracedHp(list(range(1, self.num_local_experts + 1)))
        self._register_hparam("num_local_experts", num_moe_experts)

        # Convert local_experts list and each individual expert MLP to dynamic modules
        DynamicModuleList.convert(self.local_experts)
        self.local_experts.depth = num_moe_experts  # Reuse same hparam for depth
        for expert in self.local_experts:
            DMRegistry.convert(expert, hidden_size=hidden_size)

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a standard SequentialMLP."""
        for expert in self.local_experts:
            expert.export()
        self.local_experts.export()
        return super().export()


@DMRegistry.register({MoELayer: "megatron.core.transformer.moe.moe_layer.MoELayer"})
class _DynamicMoELayer(DynamicModule):
    """A MoELayer with dynamic hyperparams."""

    def _setup(self, *, hidden_size: TracedHp):
        """Setup the MoELayer dynamic module with global hidden_size hparam."""
        # Convert to dynamic modules
        # Reuse _DynamicSequentialMLP's num_moe_experts hparam for _DynamicTopKRouter's hparam so
        #   importance estimator and depth hparam is retained.
        DMRegistry.convert(self.experts, hidden_size=hidden_size)
        num_moe_experts_hp = self.experts.get_hparam("num_local_experts")
        DMRegistry.convert(self.router, hidden_size=hidden_size, num_experts=num_moe_experts_hp)

        # NOTE: Use num_moe_experts hparam name in top-level module to match TransformerConfig's name
        self._register_hparam("num_moe_experts", num_moe_experts_hp)
        self._register_dynamic_attribute(
            "num_local_experts",
            lambda mod, val: num_moe_experts_hp.active,  # EP = 1
        )
        if self.use_shared_expert:
            DMRegistry.convert(self.shared_experts, hidden_size=hidden_size)

    def forward(self, *args, **kwargs):
        """Forward pass for the MoE layer."""
        # Dont allow forward if model is sorted / trimmed unless exported (reinitializing token dispatcher correctly)
        if isinstance(self, DynamicModule) and (
            self.get_hparam("num_moe_experts")._slice_order is not None
            or self.get_hparam("num_moe_experts").active != self.get_hparam("num_moe_experts").max
        ):
            raise RuntimeError("Only run forward after exporting the pruned model")
        return super().forward(*args, **kwargs)

    def modify(
        self, *, num_moe_experts_divisor: int = 1, ffn_hidden_size_divisor: int = 1, **kwargs
    ):
        """Modify MoE hparam choices based on search space config."""
        # Modify num_moe_experts hparam choices (applies to both router and experts)
        expert_hp = self.get_hparam("num_moe_experts")
        choices = {int(make_divisible(c, num_moe_experts_divisor)) for c in expert_hp.choices}  # type: ignore[arg-type]
        expert_hp.choices = list(set(expert_hp.choices) & choices | {expert_hp.original})

        # Modify expert FFN hparam choices
        for expert in self.experts.local_experts:
            expert.modify(ffn_hidden_size_divisor=ffn_hidden_size_divisor)
        if self.use_shared_expert:
            self.shared_experts.modify(ffn_hidden_size_divisor)

    def _export_reinit_token_dispatcher(self) -> None:
        """Reinitialize the token dispatcher after pruning."""
        print_rank_0("Reinitializing token dispatcher after pruning")
        if hasattr(moe_utils, "get_default_model_comm_pgs"):
            model_comm_pgs = moe_utils.get_default_model_comm_pgs()
        else:
            model_comm_pgs = moe_utils.get_default_pg_collection()
        # NOTE: Update config.num_moe_experts for correct router initialization.
        self.config.num_moe_experts = self.num_moe_experts
        self.token_dispatcher = type(self.token_dispatcher)(
            self.num_local_experts, list(range(self.num_local_experts)), self.config, model_comm_pgs
        )

        if self.use_shared_expert and self.shared_expert_overlap:
            self.token_dispatcher.set_shared_experts(self.shared_experts)

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a standard MoELayer."""
        self.router.export()
        self.experts.export()
        if self.use_shared_expert:
            self.shared_experts.export()
        self._export_reinit_token_dispatcher()
        return super().export()


# TransformerLayer DynamicModule ###################################################################
@DMRegistry.register(
    {TransformerLayer: "megatron.core.transformer.transformer_layer.TransformerLayer"}
)
class _DynamicTransformerLayer(DynamicModule):
    """A TransformerLayer layer with dynamic hyperparams."""

    def _setup(self, *, hidden_size: TracedHp):
        """Setup the TransformerLayer dynamic module with global hidden_size hparam."""
        # Convert the layernorms, self-attention, and mlp/moe layers to dynamic modules
        # NOTE: Mamba stack layers have either Attention or MLP, not both unlike GPT models
        if isinstance(self.self_attention, SelfAttention):
            DMRegistry.convert(self.input_layernorm, num_features=hidden_size)
            DMRegistry.convert(self.self_attention, hidden_size=hidden_size)

        if isinstance(self.mlp, (MLP, MoELayer)):
            DMRegistry.convert(self.pre_mlp_layernorm, num_features=hidden_size)
            DMRegistry.convert(self.mlp, hidden_size=hidden_size)

    def modify(
        self,
        *,
        ffn_hidden_size_divisor: int = 1,
        num_moe_experts_divisor: int = 1,
        **kwargs,  # Unused hparams
    ) -> None:
        """Modify TransformerLayer hparam choices based on search space config."""
        # Modify MLP hparam (regular or MoE)
        if isinstance(self.mlp, (MLP, MoELayer)):
            self.mlp.modify(
                ffn_hidden_size_divisor=ffn_hidden_size_divisor,
                num_moe_experts_divisor=num_moe_experts_divisor,
            )

    def export(self):
        """Export the dynamic module to a torch.nn.Module."""
        if isinstance(self.self_attention, SelfAttention):
            self.input_layernorm.export()
            self.self_attention.export()
        if isinstance(self.mlp, (MLP, MoELayer)):
            self.pre_mlp_layernorm.export()
            self.mlp.export()
        return super().export()


# Mamba DynamicModules #############################################################################
class MambaNumHeadsHp(TracedHp):
    """An hparam for Mamba's num_heads.

    Need special handling for active_slice property to trim heads within each group.
    """

    def __init__(self, nheads: int, ngroups: int = 1) -> None:
        """Initialize choices as multiples of ngroups."""
        choices = [h * ngroups for h in range(1, nheads // ngroups + 1)]
        super().__init__(choices)
        self._ngroups = ngroups

    @property
    def active_slice(self) -> TracedHp.ActiveSlice:
        """Return the currently active sorted indices by trimming heads within each group."""
        if self._slice_order is None:
            if self.active == self.max:
                return slice(self.max)
            slice_order = torch.arange(self.max)
        else:
            slice_order = self._slice_order
        target_nheads_per_group = self.active // self._ngroups
        return slice_order.view(self._ngroups, -1)[:, :target_nheads_per_group].flatten()  # type: ignore[misc]


class MambaDInnerHp(TracedHp):
    """An hparam for Mamba's d_inner.

    Mamba's d_inner is a multiplication of mamba_num_heads and mamba_head_dim hparams.
    """

    def __init__(self, mamba_num_heads: MambaNumHeadsHp, mamba_head_dim: TracedHp) -> None:
        """Initialize the Mamba d_inner hparam."""
        self._mamba_num_heads = mamba_num_heads
        self._mamba_head_dim = mamba_head_dim
        choices = self._get_choices()
        original = mamba_num_heads.original * mamba_head_dim.original
        super().__init__(choices, original)
        self._is_configurable = False
        self._importance_estimators = None

    @property  # type: ignore[misc]
    def active(self) -> int:
        """Return the active value of the hparam."""
        assert isinstance(self._mamba_num_heads.active, int)
        assert isinstance(self._mamba_head_dim.active, int)
        return self._mamba_num_heads.active * self._mamba_head_dim.active

    @property
    def active_slice(self) -> TracedHp.ActiveSlice:
        """Return the currently active sorted indices or slice corresponding to the active value."""
        num_heads_active_slice = self._mamba_num_heads.active_slice
        head_dim_active_slice = self._mamba_head_dim.active_slice
        if isinstance(num_heads_active_slice, slice):
            num_heads_active_slice = torch.LongTensor(range(num_heads_active_slice.stop))
        if isinstance(head_dim_active_slice, slice):
            head_dim_active_slice = torch.LongTensor(range(head_dim_active_slice.stop))

        indices = torch.arange(self.max).view(self._mamba_num_heads.max, self._mamba_head_dim.max)
        active_slice = indices[num_heads_active_slice, :][:, head_dim_active_slice].flatten()

        # check if active_slice corresponds to the vanilla slice
        if torch.equal(active_slice, torch.arange(self.max)):
            return slice(self.max)

        return active_slice

    def _get_choices(self) -> Sequence[HPType]:
        return sorted(
            {
                num_heads * head_dim
                for num_heads in self._mamba_num_heads.choices
                for head_dim in self._mamba_head_dim.choices
            }
        )

    def reset_choices(self) -> None:
        """Reset the choices of the Mamba d_inner hparam using updated choices of mamba_num_heads and mamba_head_dim."""
        self._choices = self._get_choices()

    @property  # type: ignore[misc]
    def choices(self) -> Sequence[HPType]:
        """Return available choices."""
        return self._get_choices()

    def _resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], TracedHp]
    ) -> dict[Symbol, TracedHp]:
        raise NotImplementedError("MambaDInnerHp does not support `_resolve_dependencies`!")


class _DynamicExtendedRMSNorm(DynamicModule):
    """An ExtendedRMSNorm (GroupNorm) layer with dynamic hyperparams.

    Very similar to _DynamicGroupNorm but with group_size dynamic attribute instead of num_groups.
    Will be registered to DMRegistry if Mamba is available.
    """

    def _setup(self):
        # register hidden_size as hyperparameter
        orig_hidden_size = self.weight.shape[0]
        num_groups = orig_hidden_size // self.group_size
        choices = [
            c
            for c in range(num_groups, orig_hidden_size + 1)
            if c % num_groups == 0 and c % self.group_size == 0
        ]
        self._register_hparam("hidden_size", TracedHp(choices, original=orig_hidden_size))

        # register num_groups as a dynamic attribute so group size is same
        self._register_temp_attribute("_num_groups", num_groups)
        self._register_dynamic_attribute("group_size", self._get_group_size)

        # register dynamic attributes
        dyn_attrs = ["weight", "bias"]
        for attr in dyn_attrs:
            self._register_dynamic_attribute(attr, self._cut_to_active_hidden_size)

    @staticmethod
    def _get_group_size(mod: "_DynamicExtendedRMSNorm", value: int) -> int:
        return mod.hidden_size // mod._num_groups

    @staticmethod
    def _cut_to_active_hidden_size(mod: "_DynamicExtendedRMSNorm", value: torch.Tensor | None):
        return get_sliced_tensor(mod, value, "hidden_size")


class _MambaContextParallelProxy:
    """A proxy for the MambaContextParallel class.

    This is used to return dynamic values for specific attributes of the MambaContextParallel class.
    """

    def __init__(self, mixer, cp):
        """Initialize the proxy."""
        object.__setattr__(self, "_mixer", mixer)
        object.__setattr__(self, "_cp", cp)

    def __getattribute__(self, name):
        """Return the dynamic value for the given attribute."""
        mixer = object.__getattribute__(self, "_mixer")
        if name in ("d_inner_local_tp", "d_inner_local_tpcp"):
            return mixer.d_inner
        if name in ("nheads_local_tp", "nheads_local_tpcp"):
            return mixer.nheads
        if name == "conv1d_cp1":
            return mixer.conv1d
        if name == "dt_bias_cp1":
            return mixer.dt_bias
        if name == "A_log_cp1":
            return mixer.A_log
        if name == "D_cp1":
            return mixer.D
        # Delegate to the underlying cp object for everything else, but
        # rebind bound methods so that `self` inside them is this proxy.
        cp = object.__getattribute__(self, "_cp")
        attr = getattr(cp, name)
        # Avoid meddling with dunder/special attributes
        if isinstance(name, str) and name.startswith("__"):
            return attr
        # Rebind methods originally bound to the underlying cp instance
        if (
            callable(attr)
            and hasattr(attr, "__self__")
            and getattr(attr, "__self__", None) is cp
            and hasattr(attr, "__func__")
        ):
            return types.MethodType(attr.__func__, self)
        return attr

    def __setattr__(self, name, value):
        if name in ("_mixer", "_cp"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_cp"), name, value)


class _DynamicMambaMixer(DynamicModule):
    """A MambaMixer layer with dynamic hyperparams.

    Will be registered to DMRegistry if Mamba is available.
    """

    def _setup(self, *, hidden_size: TracedHp):
        """Setup the MambaMixer dynamic module with global hidden_size hparam."""
        assert self.d_inner == self.nheads * self.headdim, "d_inner must be nheads * headdim"

        # Register hyperparameters for Mamba heads and head dimensions
        mamba_num_heads = MambaNumHeadsHp(self.nheads, self.ngroups)
        mamba_head_dim = TracedHp(list(range(1, self.headdim + 1)))
        d_inner = MambaDInnerHp(mamba_num_heads, mamba_head_dim)
        bc = TracedHp([2 * self.ngroups * self.d_state])  # not configurable

        self._register_hparam("d_model", hidden_size)
        self._register_hparam("d_inner", d_inner)
        self._register_hparam("mamba_num_heads", mamba_num_heads)
        self._register_hparam("mamba_head_dim", mamba_head_dim)
        self._register_hparam("bc", bc)
        self._register_dynamic_attribute("d_inner_local_tp", lambda mod, val: self.d_inner)

        # Register dynamic attributes
        self._register_dynamic_attribute("nheads", lambda mod, val: self.mamba_num_heads)
        self._register_dynamic_attribute("nheads_local_tp", lambda mod, val: self.nheads)
        self._register_dynamic_attribute("headdim", lambda mod, val: self.mamba_head_dim)

        # Convert to dynamic modules
        in_proj_output_size = build_concat_hp(
            [d_inner, d_inner, bc, mamba_num_heads]
        )  # z, x, B, C, dt
        DMRegistry.convert(self.in_proj, input_size=hidden_size, output_size=in_proj_output_size)

        conv_dim = build_concat_hp([d_inner, bc])  # z, B, C
        DMRegistry.convert(self.conv1d)
        self.conv1d.in_channels = conv_dim
        self.conv1d.out_channels = conv_dim
        ks = self.conv1d.get_hparam("kernel_size")
        ks.choices = [ks.original]

        if self.rmsnorm:
            DMRegistry.convert(self.norm)
            self.norm.hidden_size = d_inner

        DMRegistry.convert(self.out_proj, input_size=d_inner, output_size=hidden_size)

        # Register dynamic attributes for Mamba-specific parameters
        self._register_dynamic_attribute("dt_bias", self._get_dt_bias_A_log_D)
        self._register_dynamic_attribute("A_log", self._get_dt_bias_A_log_D)
        self._register_dynamic_attribute("D", self._get_dt_bias_A_log_D)
        assert not self.D_has_hdim, "D_has_hdim is not supported yet"

        self.cp = _MambaContextParallelProxy(self, self.cp)

    @staticmethod
    def _get_dt_bias_A_log_D(mod: "_DynamicMambaMixer", data: torch.Tensor) -> torch.Tensor:  # noqa: N802
        """Return the sliced data based on mamba_num_heads's active_slice."""
        return get_sliced_tensor(mod, data, "mamba_num_heads")

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a torch.nn.Module."""
        self.in_proj.export()
        self.out_proj.export()
        self.conv1d.export()
        if self.rmsnorm:
            self.norm.export()
        return super().export()


class _DynamicMambaLayer(DynamicModule):
    """A MambaLayer layer with dynamic hyperparams.

    Will be registered to DMRegistry if Mamba is available.
    """

    def _setup(self, *, hidden_size: TracedHp):
        """Setup the MambaLayer dynamic module with global hidden_size hparam."""
        # Convert to dynamic module
        DMRegistry.convert(self.mixer, hidden_size=hidden_size)

        DMRegistry.convert(self.norm, num_features=hidden_size)

    def modify(
        self,
        *,
        mamba_head_dim_divisor: int = 1,
        **kwargs,  # Unused hparams
    ) -> None:
        """Modify Mamba hyperparameters."""
        # Modify MambaMixer hparams
        hp = self.mixer.get_hparam("mamba_head_dim")
        choices = {int(make_divisible(c, mamba_head_dim_divisor)) for c in hp.choices}
        hp.choices = list(set(hp.choices) & choices | {hp.original})

    def export(self):
        """Export the dynamic module to a torch.nn.Module."""
        self.mixer.export()
        self.norm.export()
        return super().export()


if HAS_MAMBA:
    DMRegistry.register({ExtendedRMSNorm: "megatron.core.ssm.mamba_mixer.ExtendedRMSNorm"})(
        _DynamicExtendedRMSNorm
    )

    DMRegistry.register({MambaMixer: "megatron.core.ssm.mamba_mixer.MambaMixer"})(
        _DynamicMambaMixer
    )

    DMRegistry.register({MambaLayer: "megatron.core.ssm.mamba_layer.MambaLayer"})(
        _DynamicMambaLayer
    )


# GPTModel / MambaModel DynamicModule ##############################################################
@DMRegistry.register(SUPPORTED_MODELS)
class _DynamicMCoreLanguageModel(DynamicModule):
    """A GPTModel / MambaModel with dynamic hyperparams."""

    def _setup(self):
        assert self.config.tensor_model_parallel_size == 1, "Only TP=1 is supported."
        assert self.config.virtual_pipeline_model_parallel_size is None, (
            "Virtual pipeline parallel is not supported."
        )
        assert not self.config.sequence_parallel, "Sequence parallel is not supported."
        assert self.config.context_parallel_size == 1, "Context parallel is not supported."
        assert self.config.expert_model_parallel_size == 1, "Expert parallel is not supported."
        assert self.pre_process == is_pipeline_first_stage()
        assert self.post_process == is_pipeline_last_stage()

        # Register num_layers hparam for depth pruning
        self._register_hparam("num_layers", TracedHp(list(range(1, self.config.num_layers + 1))))

        # Convert layers to dynamic modules and set the shared hidden_size hparam for all layers
        if is_pipeline_first_stage():
            DMRegistry.convert(self.embedding)
            hidden_size = self.embedding.word_embeddings.get_hparam("embedding_dim")
        else:
            hidden_size = None
        hidden_size = dist.broadcast(hidden_size, src=0)
        self._register_hparam("hidden_size", hidden_size)

        for i in range(len(self.decoder.layers)):
            DMRegistry.convert(self.decoder.layers[i], hidden_size=hidden_size)

        if is_pipeline_last_stage():
            # NOTE: GPTModel has final_layernorm, MambaModel has final_norm
            DMRegistry.convert(
                getattr(
                    self.decoder,
                    "final_layernorm" if hasattr(self.decoder, "final_layernorm") else "final_norm",
                ),
                num_features=hidden_size,
            )
            DMRegistry.convert(self.output_layer, input_size=hidden_size)
            self.output_layer.get_hparam("output_size").choices = [self.output_layer.output_size]

    def modify(
        self,
        *,
        hidden_size_divisor: int = 1,
        ffn_hidden_size_divisor: int = 1,
        mamba_num_heads_divisor: int = 1,
        mamba_head_dim_divisor: int = 1,
        num_moe_experts_divisor: int = 1,
    ):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            hidden_size_divisor: The divisor of the hidden_size.
            ffn_hidden_size_divisor: The divisor of the mlp ffn_hidden_size.
            mamba_num_heads_divisor: The divisor of the mamba num_heads.
            mamba_head_dim_divisor: The divisor of the mamba head_dim.
            num_moe_experts_divisor: The divisor of the number of MoE experts.
        """
        hp = self.get_hparam("hidden_size")
        choices = {int(make_divisible(c, hidden_size_divisor)) for c in hp.choices}  # type: ignore[arg-type]
        hp.choices = list(set(hp.choices) & choices | {hp.original})

        for layer in self.decoder.layers:
            layer.modify(
                ffn_hidden_size_divisor=ffn_hidden_size_divisor,
                mamba_num_heads_divisor=mamba_num_heads_divisor,
                mamba_head_dim_divisor=mamba_head_dim_divisor,
                num_moe_experts_divisor=num_moe_experts_divisor,
            )

    def export(self) -> torch.nn.Module:
        """Export the dynamic module to a torch.nn.Module."""
        # Drop layers if depth pruning is enabled - handled by mcore_minitron.py
        if is_pipeline_first_stage():
            self.embedding.export()
        for layer in self.decoder.layers:
            layer.export()
        if is_pipeline_last_stage():
            getattr(
                self.decoder,
                "final_layernorm" if hasattr(self.decoder, "final_layernorm") else "final_norm",
            ).export()
            self.output_layer.export()
        return super().export()


class MegatronConstraintsFunc(ConstraintsFunc):
    """A Functor class to check if sub-net satisfied all provided constraints.

    We intentionally expose some attributes like `limits` s.t. we can modify it manually.
    """

    _sample_points_dict: dict[tuple[str, ...], dict[str, SampleFunc]] = {
        ("params",): {"min": min, "centroid": random.centroid, "max": max},
    }

    def __init__(
        self,
        model: MegatronModule,
        constraints: ConstraintsDict,
        dummy_input: Any | tuple[Any, ...],
        deployment: dict | None = None,
        fast_eval: bool = True,
    ):
        """Initialize with additional data parallel group info from megatron."""
        for key in constraints:
            if key != "params":
                raise ValueError("Only params constraints is supported for MegatronModule!")

        self.model = model
        self.dummy_input = dummy_input
        self.deployment = deployment
        self._fast_eval = fast_eval

        # Getting data parallel group for
        self.dp_group = get_data_parallel_group()

        # initialize latency interpolator
        keys_for_interpolation = ("params",)
        if ConstraintsFunc.is_configurable(self.model, "depth"):
            keys_for_interpolation += ("flops_min_depth",)
        self._latency_interpolator = ConstraintInterpolator(
            self.model,
            points_funcs={k: self.constraint_eval_funcs[k] for k in keys_for_interpolation},
            value_func=self._get_true_latency,
        )
        # set fast/regular mode for latency interpolator
        self._latency_interpolator.collect_mode = not self.fast_eval

        # set limit at the end with setter to use sanity checks on constraints
        self._limits = {}
        self.limits = constraints

    @property
    def constraint_eval_funcs(self) -> dict[str, ConstraintEvalFunc]:
        """Get constraint eval fns."""
        return {
            "params": self._get_params,
        }

    def _get_params(self, _: ConstraintsRes | None = None) -> float:
        """Get number of model parameters from forward pass."""
        params = param_num_from_forward(self.model, args=self.dummy_input, unit=1.0)
        reduced_params = torch.Tensor([params]).to(device=get_module_device(self.model))
        torch.distributed.all_reduce(reduced_params, group=get_pipeline_model_parallel_group())
        torch.distributed.all_reduce(reduced_params, group=get_tensor_model_parallel_group())
        return reduced_params.item()

    def _get_flops(self, _: ConstraintsRes | None = None) -> float:
        """Get inference FLOPs."""
        raise NotImplementedError

    def _get_flops_min_depth(self, _: ConstraintsRes | None = None) -> float:
        """Get inference FLOPs with depth set to minimum."""
        raise NotImplementedError

    def _get_true_latency(self, _: ConstraintsRes | None = None) -> float:
        """Get true inference latency."""
        raise NotImplementedError

    def _get_latency(self, precomputed: ConstraintsRes | None = None) -> float:
        """Get inference latency from interpolator."""
        raise NotImplementedError


# Clear the mapping and reinsert.
MODULE_TYPE_TO_CONSTRAINTS_FUNC[MegatronModule] = MegatronConstraintsFunc
