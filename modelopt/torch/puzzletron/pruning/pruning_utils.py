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

import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers import PretrainedConfig

from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptor
from modelopt.torch.puzzletron.pruning.pruning_mixin import PruningMixIn


class GQAInitMode(Enum):
    RandomKV = "RandomKV"
    AverageKV = "AverageKV"
    FirstKV = "FirstKV"
    RandomBlock = "RandomBlock"
    CopyAsIs = "CopyAsIs"
    Degrouping = "Degrouping"
    PruneKVHeads = "PruneKVHeads"


class MlpInitMode(Enum):
    Random = "Random"
    Truncate = "Truncate"
    CopyAsIs = "CopyAsIs"
    PruneByActivationsLog = "PruneByActivationsLog"
    ExpertRemoval = "ExpertRemoval"
    ConcatExpertsIntoDenseFFN = "ConcatExpertsIntoDenseFFN"


class LinearInitMode(Enum):
    Random = "Random"
    FromTeacher = "FromTeacher"


class HiddenSizeInitMode(Enum):
    Random = "Random"
    Truncate = "Truncate"
    PruneByChannelRanking = "PruneByChannelRanking"
    CopyAsIs = "CopyAsIs"


def resolve_pruning_mixin(
    pruning_mixin, descriptor: Type[ModelDescriptor]
) -> PruningMixIn | List[PruningMixIn]:
    """
    Convert pruning_mixin argument to PruningMixIn instance(s).

    Args:
        pruning_mixin: Can be a string identifier, PruningMixIn instance,
                      or a list of any of those types.
        descriptor: ModelDescriptor class that provides the pruning_mixins() mapping.

    Returns:
        PruningMixIn or List[PruningMixIn] depending on input type.
    """
    # Handle list of values recursively
    if isinstance(pruning_mixin, list):
        return [resolve_pruning_mixin(item, descriptor) for item in pruning_mixin]

    # Handle single value
    # If it's already a PruningMixIn, return as is
    if isinstance(pruning_mixin, PruningMixIn):
        return pruning_mixin

    # Get the pruning mixins mapping from the descriptor
    mixins_dict = descriptor.pruning_mixins()

    if isinstance(pruning_mixin, str):
        if pruning_mixin not in mixins_dict:
            available_methods = list(mixins_dict.keys())
            raise ValueError(
                f"Pruning method '{pruning_mixin}' is not supported by {descriptor.__name__}. "
                f"Available methods: {available_methods}"
            )
        return mixins_dict[pruning_mixin]

    raise ValueError(f"Unsupported pruning_mixin type: {type(pruning_mixin)}")


def _init_mlp_module(
    mlp_init_mode: Union[MlpInitMode, str],
    mlp_prefix: str,
    expanded_dim: int,
    layer_idx: int,
    new_item: torch.Tensor,
    new_config: PretrainedConfig,
    orig_item: torch.Tensor,
    original_config: PretrainedConfig,
    mlp_init_config: Optional[dict[str, Any]],
    pruned_filters: Optional[torch.Tensor] = None,
    projection_matrix: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict[str, torch.Tensor]]]:
    if isinstance(mlp_init_mode, str):
        mlp_init_mode = MlpInitMode(mlp_init_mode)
    assert orig_item.ndim == 2, f"{orig_item.ndim=}"
    assert new_item.ndim == 2, f"{new_item.ndim=}"

    assert new_config.num_hidden_layers == original_config.num_hidden_layers, (
        f"({new_config.num_hidden_layers=}) != ({original_config.num_hidden_layers=})"
    )

    new_intermediate_size = new_config.block_configs[layer_idx].ffn.intermediate_size
    original_intermediate_size = original_config.block_configs[layer_idx].ffn.intermediate_size

    if mlp_init_mode == MlpInitMode.CopyAsIs:
        assert new_intermediate_size == original_intermediate_size, (
            f"({new_intermediate_size=}) != ({original_intermediate_size=}), can't be copied as is."
        )
        mlp_module_weight = orig_item

    elif mlp_init_mode == MlpInitMode.Random:
        mlp_module_weight = new_item

    elif new_intermediate_size == original_intermediate_size:
        mlp_module_weight = orig_item

    elif mlp_init_mode in (
        MlpInitMode.Truncate,
        MlpInitMode.PruneByActivationsLog,
    ):
        assert original_intermediate_size >= new_intermediate_size, (
            f"({original_intermediate_size=}) < ({new_intermediate_size=}), can't be truncated."
        )
        orig_ffn_size = orig_item.shape[expanded_dim]
        new_ffn_size = new_item.shape[expanded_dim]

        if mlp_init_mode == MlpInitMode.Truncate:
            truncated_weight = torch.narrow(
                orig_item, dim=expanded_dim, start=0, length=new_ffn_size
            )
            mlp_module_weight = truncated_weight

        elif mlp_init_mode == MlpInitMode.PruneByActivationsLog:
            if pruned_filters is None:
                filter_importance = _load_activations_log(
                    mlp_init_config, module_name=f"{mlp_prefix}.down_proj"
                )
                filters_sorted_by_importance = torch.argsort(filter_importance, descending=True)
                pruned_filters = filters_sorted_by_importance[:new_ffn_size].to(orig_item.device)

            pruned_weight = torch.index_select(orig_item, dim=expanded_dim, index=pruned_filters)
            if mlp_init_config.get("scale_pruned_weights", False) and expanded_dim == 1:
                pruned_weight = pruned_weight * (orig_ffn_size / new_ffn_size)
            mlp_module_weight = pruned_weight

    elif (
        mlp_init_mode == MlpInitMode.ExpertRemoval
    ):  # the case of mlp layers of maverick. for now we only support copy as is
        assert new_intermediate_size == original_intermediate_size, (
            f"({new_intermediate_size=}) != ({original_intermediate_size=}), can't be copied as is."
        )
        mlp_module_weight = orig_item

    else:
        raise ValueError(f"Unsupported {mlp_init_mode=}")

    return mlp_module_weight, pruned_filters, projection_matrix


def _load_activations_log(mlp_init_config: dict[str, Any], module_name: str) -> torch.Tensor:
    _cache_activations_log(mlp_init_config)
    module_log = ACTIVATIONS_LOG[module_name]
    filter_importance = module_log["score"]
    return filter_importance


ACTIVATIONS_LOG = dict()


def _cache_activations_log(mlp_init_config: dict[str, Any]) -> None:
    if len(ACTIVATIONS_LOG) == 0:
        assert "activations_log_dir" in mlp_init_config
        activations_log_dir = mlp_init_config["activations_log_dir"]
        print(f"Loading activations_log from {activations_log_dir}")
        # Only load rank_*.pth files to avoid loading hook_states_*.pth checkpoint files
        ACTIVATIONS_LOG.update(
            {
                module_name: module_log
                for p in Path(activations_log_dir).glob("rank_*.pth")
                for module_name, module_log in torch.load(p).items()
            }
        )


def _init_attention_weights(
    gqa_init_mode,
    layer_idx,
    new_state_dict,
    new_config,
    original_state_dict,
    q_key,
    k_key,
    v_key,
    o_key,
    original_config,
    is_original_mha,
    head_size,
    mlp_init_config,
):
    assert new_config.num_attention_heads == original_config.num_attention_heads, (
        f"({new_config.num_attention_heads=}) != ({original_config.num_attention_heads=})"
    )
    num_q_heads = new_config.num_attention_heads
    num_kv_heads = new_config.block_configs[layer_idx].attention.num_key_value_heads
    orig_num_kv_heads = original_config.block_configs[layer_idx].attention.num_key_value_heads

    # new_w* are typically randomly initialized
    new_wq = new_state_dict[q_key]
    new_wk = new_state_dict[k_key]
    new_wv = new_state_dict[v_key]
    new_wo = new_state_dict[o_key]

    # w* are from the parent model
    wq = original_state_dict[q_key]
    wk = original_state_dict[k_key]
    wv = original_state_dict[v_key]
    wo = original_state_dict[o_key]

    if "bias" in k_key:
        for tensor in [wq, wk, wv, wo, new_wq, new_wk, new_wv, new_wo]:
            assert tensor.ndim == 1
            tensor.unsqueeze_(1)
    dim1 = wk.shape[1]  # this is the hidden_size in case of matrix weights, and 1 in case of biases

    if gqa_init_mode in (GQAInitMode.RandomKV, GQAInitMode.RandomBlock):
        wk, wv = new_wk, new_wv
    elif gqa_init_mode in (GQAInitMode.AverageKV, GQAInitMode.FirstKV):
        assert orig_num_kv_heads % num_kv_heads == 0, (
            f"({orig_num_kv_heads=}) % ({num_kv_heads=}) != 0"
        )
        n_heads_to_aggregate = orig_num_kv_heads // num_kv_heads

        wk = wk.view(-1, n_heads_to_aggregate, head_size, dim1)
        wv = wv.view(-1, n_heads_to_aggregate, head_size, dim1)

        if gqa_init_mode == GQAInitMode.AverageKV:
            wk = wk.mean(dim=1)
            wv = wv.mean(dim=1)
        else:
            wk = wk[:, 0]
            wv = wv[:, 0]
    elif gqa_init_mode == GQAInitMode.CopyAsIs:
        assert new_wk.shape == wk.shape, f"({new_wk.shape=}) != ({wk.shape=})"
        assert new_wv.shape == wv.shape, f"({new_wv.shape=}) != ({wv.shape=})"
        assert new_wq.shape == wq.shape, f"({new_wq.shape=}) != ({wq.shape=})"
        assert new_wo.shape == wo.shape, f"({new_wo.shape=}) != ({wo.shape=})"

    elif gqa_init_mode == GQAInitMode.Degrouping:
        assert not is_original_mha, (
            "Degrouping can only be done on original models that are GQA themselves."
        )
        n_groups = num_kv_heads
        orig_n_groups = orig_num_kv_heads
        assert n_groups % orig_n_groups == 0, f"{n_groups=} must be a divisor of {orig_n_groups=}"
        n_repeats = n_groups // orig_n_groups
        if n_repeats > 1:
            print(f"Degrouping {orig_n_groups} into {n_groups}")

        def degroup_w(w):
            w = w.view(orig_n_groups, head_size, dim1)
            w = torch.repeat_interleave(w, repeats=n_repeats, dim=0)
            w = w.reshape(n_groups * head_size, dim1)
            return w

        wk = degroup_w(wk)
        wv = degroup_w(wv)

    elif gqa_init_mode == GQAInitMode.PruneKVHeads:
        wk = wk.view(orig_num_kv_heads, head_size, dim1)
        wv = wv.view(orig_num_kv_heads, head_size, dim1)
        wq = wq.view(orig_num_kv_heads, num_q_heads // orig_num_kv_heads, head_size, dim1)
        wo = wo.view(dim1, orig_num_kv_heads, num_q_heads // orig_num_kv_heads, head_size)

        o_proj_module_name = o_key.replace(".weight", "")
        kv_head_importance = _load_activations_log(mlp_init_config, module_name=o_proj_module_name)
        kv_heads_sorted_by_importance = torch.argsort(kv_head_importance, descending=True)
        kv_heads_to_keep = kv_heads_sorted_by_importance[:num_kv_heads]
        kv_heads_to_remove = kv_heads_sorted_by_importance[num_kv_heads:]

        wk = wk[kv_heads_to_keep]
        wv = wv[kv_heads_to_keep]

        reduction_factor = orig_num_kv_heads // num_kv_heads

        prune_via_duplication = False
        if prune_via_duplication:
            ## Wq option 1 - replicate the query groups to match the total number of attention heads. Queries work with familiar kv heads.
            wq = wq[kv_heads_to_keep]
            wq = torch.repeat_interleave(wq, repeats=reduction_factor, dim=0)

            ## Wo option 1 - replicate the groups of the original Wo. Multiple by the reduction factor to mimic pruning of the other groups.
            ## This makes sense with Wq option 1, but it will not be more expressive than true pruning due to symmetry, unless we add noise.
            wo = wo[:, kv_heads_to_keep]
            wo = torch.repeat_interleave(wo, repeats=reduction_factor, dim=1)
            wo = wo / reduction_factor

        else:  # prune via zeroing out
            ## Wq option 2 - keep the original queries. At init they will not be used (see the Wo zeroing), during training they can adapt to new kv heads like in variable GQA.
            ## We need to interleave them to keep the matching between queries and kv heads.
            kv_heads_to_keep = kv_heads_to_keep.tolist()
            kv_heads_to_remove = kv_heads_to_remove.tolist()
            kv_head_ordering = []
            zero_out_mask = []
            for i_head in range(orig_num_kv_heads):
                if i_head % reduction_factor == 0:
                    kv_head_ordering.append(kv_heads_to_keep.pop(0))
                    zero_out_mask.append(False)
                else:
                    kv_head_ordering.append(kv_heads_to_remove.pop(0))
                    zero_out_mask.append(True)

            wq = wq[kv_head_ordering]

            ## Wo option 2 - zero-out the contribution of queries that do not belong to chosen kv heads.
            ## At initialization it's exactly like pruning, but the extra weights will have the chance to adapt to new kv heads if we train the model.
            ## Even though the weight is 0 it can still train, like initializing biases to 0 does not prevent them from training.
            ## Matmul backprop: if Y = AB and dY is the gradient of Y, then dA = dY @ B.T and dB = A.T @ dY, so the gradient of the zeroed-out weights depends on the gradient of what multiplies them.
            wo = wo[:, kv_head_ordering]
            wo[:, zero_out_mask] = 0.0

    else:
        raise ValueError(f"{gqa_init_mode=} not supported")

    wk = wk.reshape(-1, dim1)
    wv = wv.reshape(-1, dim1)
    wq = wq.reshape(-1, dim1)
    wo = wo.reshape(dim1, -1)
    return wq, wk, wv, wo


def _init_attention_biases(
    gqa_init_mode,
    layer_idx,
    new_state_dict,
    new_config,
    original_state_dict,
    q_key,
    k_key,
    v_key,
    o_key,
    original_config,
    is_original_mha,
    head_size,
    mlp_init_config,
):
    assert new_config.num_attention_heads == original_config.num_attention_heads, (
        f"({new_config.num_attention_heads=}) != ({original_config.num_attention_heads=})"
    )
    num_q_heads = new_config.num_attention_heads
    n_heads_in_group = new_config.block_configs[layer_idx].attention.n_heads_in_group
    orig_n_heads_in_group = original_config.block_configs[layer_idx].attention.n_heads_in_group
    num_kv_heads = num_q_heads // n_heads_in_group
    orig_num_kv_heads = num_q_heads // orig_n_heads_in_group

    o_proj_bias = new_config.o_proj_bias
    attention_bias = new_config.attention_bias

    # If no biases
    if not (o_proj_bias or attention_bias):
        return {}

    new_bias_sd = {}
    bias_sd = {}
    # new_w* are typically randomly initialized
    if o_proj_bias:
        new_bias_sd["o"] = new_state_dict[o_key]
        bias_sd["o"] = original_state_dict[o_key]
    if attention_bias:
        for bias_key, key in zip("qkv", [q_key, k_key, v_key]):
            new_bias_sd[bias_key] = new_state_dict[key]
            bias_sd[bias_key] = original_state_dict[key]

    # maybe unsqueeze all tensors
    for tensor in list(new_bias_sd.values()) + list(bias_sd.values()):
        assert tensor.ndim == 1
        tensor.unsqueeze_(1)

    dim1 = 1  # this is the hidden_size in case of matrix weights, and 1 in case of biases
    if gqa_init_mode in (GQAInitMode.RandomKV, GQAInitMode.RandomBlock) and attention_bias:
        bias_sd["k"] = torch.zeros(
            new_bias_sd["k"].shape, dtype=bias_sd["k"].dtype, device=bias_sd["k"].device
        )
        bias_sd["v"] = torch.zeros(
            new_bias_sd["v"].shape, dtype=bias_sd["v"].dtype, device=bias_sd["v"].device
        )
    elif gqa_init_mode in (GQAInitMode.AverageKV, GQAInitMode.FirstKV) and attention_bias:
        assert n_heads_in_group % orig_n_heads_in_group == 0, (
            f"({n_heads_in_group=}) % ({orig_n_heads_in_group=}) != 0"
        )
        n_heads_to_aggregate = n_heads_in_group // orig_n_heads_in_group

        bias_sd["k"] = bias_sd["k"].view(-1, n_heads_to_aggregate, head_size, dim1)
        bias_sd["v"] = bias_sd["v"].view(-1, n_heads_to_aggregate, head_size, dim1)

        if gqa_init_mode == GQAInitMode.AverageKV:
            bias_sd["k"] = bias_sd["k"].mean(dim=1)
            bias_sd["v"] = bias_sd["v"].mean(dim=1)
        else:
            bias_sd["k"] = bias_sd["k"][:, 0]
            bias_sd["v"] = bias_sd["v"][:, 0]
    elif gqa_init_mode == GQAInitMode.CopyAsIs:
        for key in bias_sd.keys():
            assert new_bias_sd[key].shape == bias_sd[key].shape, (
                f"({new_bias_sd[key].shape=}) != ({bias_sd[key].shape=})"
            )

    elif gqa_init_mode == GQAInitMode.Degrouping and attention_bias:
        assert not is_original_mha, (
            "Degrouping can only be done on original models that are GQA themselves."
        )
        n_groups = new_config.num_attention_heads // n_heads_in_group
        orig_n_groups = original_config.num_attention_heads // orig_n_heads_in_group
        assert n_groups % orig_n_groups == 0, f"{n_groups=} must be a divisor of {orig_n_groups=}"
        n_repeats = n_groups // orig_n_groups
        if n_repeats > 1:
            print(f"Degrouping {orig_n_groups} into {n_groups}")

        def degroup_w(w):
            w = w.view(orig_n_groups, head_size, dim1)
            w = torch.repeat_interleave(w, repeats=n_repeats, dim=0)
            w = w.reshape(n_groups * head_size, dim1)
            return w

        bias_sd["k"] = degroup_w(bias_sd["k"])
        bias_sd["v"] = degroup_w(bias_sd["v"])

    elif gqa_init_mode == GQAInitMode.PruneKVHeads:
        if o_proj_bias:
            o_proj_module_name = o_key.rsplit(".", 1)[0]
        else:
            # Here we assume that the o_proj layer is called "o_proj"
            o_proj_module_name = k_key.rsplit(".", 2)[0] + ".o_proj"

        kv_head_importance = _load_activations_log(mlp_init_config, module_name=o_proj_module_name)
        kv_heads_sorted_by_importance = torch.argsort(kv_head_importance, descending=True)
        kv_heads_to_keep = kv_heads_sorted_by_importance[:num_kv_heads]
        kv_heads_to_remove = kv_heads_sorted_by_importance[num_kv_heads:]

        # view as KV groups
        if attention_bias:
            bias_sd["k"] = bias_sd["k"].view(orig_num_kv_heads, head_size, dim1)
            bias_sd["v"] = bias_sd["v"].view(orig_num_kv_heads, head_size, dim1)
            bias_sd["q"] = bias_sd["q"].view(
                orig_num_kv_heads, orig_n_heads_in_group, head_size, dim1
            )
            # Keep important KV heads and prune the others
            bias_sd["k"] = bias_sd["k"][kv_heads_to_keep]
            bias_sd["v"] = bias_sd["v"][kv_heads_to_keep]
        if o_proj_bias:
            bias_sd["o"] = bias_sd["o"].view(
                dim1, orig_num_kv_heads, orig_n_heads_in_group, head_size
            )

        reduction_factor = orig_num_kv_heads // num_kv_heads

        prune_via_duplication = False
        if prune_via_duplication:
            if attention_bias:
                ## Wq option 1 - replicate the query groups to match the total number of attention heads. Queries work with familiar kv heads.
                bias_sd["q"] = bias_sd["q"][kv_heads_to_keep]
                bias_sd["q"] = torch.repeat_interleave(
                    bias_sd["q"], repeats=reduction_factor, dim=0
                )

            if o_proj_bias:
                ## Wo option 1 - replicate the groups of the original Wo. Multiple by the reduction factor to mimic pruning of the other groups.
                ## This makes sense with Wq option 1, but it will not be more expressive than true pruning due to symmetry, unless we add noise.
                bias_sd["o"] = bias_sd["o"][:, kv_heads_to_keep]
                bias_sd["o"] = torch.repeat_interleave(
                    bias_sd["o"], repeats=reduction_factor, dim=1
                )
                bias_sd["o"] = bias_sd["o"] / reduction_factor

        else:  # prune via zeroing out
            ## Wq option 2 - keep the original queries. At init they will not be used (see the Wo zeroing), during training they can adapt to new kv heads like in variable GQA.
            ## We need to interleave them to keep the matching between queries and kv heads.
            kv_heads_to_keep = kv_heads_to_keep.tolist()
            kv_heads_to_remove = kv_heads_to_remove.tolist()
            kv_head_ordering = []
            zero_out_mask = []
            for i_head in range(orig_num_kv_heads):
                if i_head % reduction_factor == 0:
                    kv_head_ordering.append(kv_heads_to_keep.pop(0))
                    zero_out_mask.append(False)
                else:
                    kv_head_ordering.append(kv_heads_to_remove.pop(0))
                    zero_out_mask.append(True)

            if attention_bias:
                bias_sd["q"] = bias_sd["q"][kv_head_ordering]

            if o_proj_bias:
                ## Wo option 2 - zero-out the contribution of queries that do not belong to chosen kv heads.
                ## At initialization it's exactly like pruning, but the extra weights will have the chance to adapt to new kv heads if we train the model.
                ## Even though the weight is 0 it can still train, like initializing biases to 0 does not prevent them from training.
                ## Matmul backprop: if Y = AB and dY is the gradient of Y, then dA = dY @ B.T and dB = A.T @ dY, so the gradient of the zeroed-out weights depends on the gradient of what multiplies them.
                bias_sd["o"] = bias_sd["o"][:, kv_head_ordering]
                bias_sd["o"][:, zero_out_mask] = 0.0

    else:
        raise ValueError(f"{gqa_init_mode=} not supported")

    if attention_bias:
        for bias_key in "qkv":
            bias_sd[bias_key] = bias_sd[bias_key].reshape(-1)
    if o_proj_bias:
        bias_sd["o"] = bias_sd["o"].reshape(-1)
    return bias_sd


def _init_moe_module(
    mlp_init_mode: Union[MlpInitMode, str],
    mlp_init_config: Optional[Dict[str, Any]],
    layer_idx: int,
    orig_router_weights: Dict[str, List[torch.Tensor]],
    orig_experts_weights: Dict[str, List[torch.Tensor]],
    new_router_weights: Dict[str, List[torch.Tensor]],
    new_experts_weights: Dict[str, List[torch.Tensor]],
    orig_num_experts: int,
    new_num_experts: int,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[torch.Tensor]]]:
    if isinstance(mlp_init_mode, str):
        mlp_init_mode = MlpInitMode(mlp_init_mode)

    if mlp_init_mode != MlpInitMode.ExpertRemoval:
        raise ValueError(f"Unsupported {mlp_init_mode=}")

    selected_experts = _select_expert_indices(
        mlp_init_config=mlp_init_config,
        layer_idx=layer_idx,
        orig_num_experts=orig_num_experts,
        new_num_experts=new_num_experts,
    )

    # Router: prefer parent tensors when available; if child has bias only, slice from child
    result_router_weights: dict[str, list[torch.Tensor]] = {}
    for name, new_list in new_router_weights.items():
        result_router_weights[name] = [
            tensor_to_slice[selected_experts] for tensor_to_slice in orig_router_weights[name]
        ]

    # Experts: for each name present in the child, take from parent if available, else from child
    result_experts_weights: dict[str, list[torch.Tensor]] = {}
    for name, new_list in new_experts_weights.items():
        if name in orig_experts_weights:
            src_list = orig_experts_weights[name]
        else:
            src_list = new_list
        result_experts_weights[name] = [src_list[i] for i in selected_experts]

    # Validate shapes
    assert result_router_weights.keys() == new_router_weights.keys(), (
        "result_router_weights and new_router_weights must have the same keys"
    )
    for name in new_router_weights.keys():
        assert len(new_router_weights[name]) == len(result_router_weights[name])
        for new_router_weight, result_router_weight in zip(
            new_router_weights[name], result_router_weights[name]
        ):
            assert new_router_weight.shape == result_router_weight.shape

    assert result_experts_weights.keys() == new_experts_weights.keys(), (
        "result_experts_weights and new_experts_weights must have the same keys"
    )
    for name in result_experts_weights.keys():
        assert len(new_experts_weights[name]) == len(result_experts_weights[name])
        for new_expert_weight, result_expert_weight in zip(
            new_experts_weights[name], result_experts_weights[name]
        ):
            assert new_expert_weight.shape == result_expert_weight.shape

    return result_router_weights, result_experts_weights


def _select_expert_indices(
    *, mlp_init_config: dict[str, Any], layer_idx: int, orig_num_experts: int, new_num_experts: int
) -> list[int]:
    expert_scores = _load_expert_scores(mlp_init_config, layer_idx)
    assert len(expert_scores) == orig_num_experts
    selected_experts = sorted(
        range(orig_num_experts),
        key=lambda i: expert_scores[i] if not math.isnan(expert_scores[i]) else float("inf"),
        reverse=mlp_init_config.get("higher_is_better", True),
    )[:new_num_experts]
    return selected_experts


def _load_expert_scores(
    mlp_init_config: Optional[dict[str, Any]], layer_idx: int
) -> list[list[int | float]]:
    assert mlp_init_config is not None
    if "expert_scores_file" in mlp_init_config:
        expert_scores_file = mlp_init_config["expert_scores_file"]
        with open(expert_scores_file, "r") as f:
            expert_scores = json.load(f)
    elif "activations_log_dir" in mlp_init_config:
        _cache_activations_log(mlp_init_config)
        # Use layer_prefix_template from pruning config, or fall back to legacy nemotron_h format
        # TODO - get from descriptors
        layer_prefix_template = mlp_init_config.get(
            "layer_prefix_template", "backbone.layers.{layer_idx}."
        )
        layer_prefix = layer_prefix_template.format(layer_idx=layer_idx)
        candidate_layer_keys = [
            key for key in ACTIVATIONS_LOG.keys() if key.startswith(layer_prefix)
        ]
        if len(candidate_layer_keys) == 0:
            raise ValueError(f"No layer keys found for {layer_prefix=}. {ACTIVATIONS_LOG.keys()=}")
        elif len(candidate_layer_keys) > 1:
            if "layer_suffix" not in mlp_init_config:
                raise ValueError(
                    f"Multiple candidate layer keys found for {layer_prefix=}, you must specify a layer_suffix in the mlp_init_config. {candidate_layer_keys=}"
                )
            layer_suffix = mlp_init_config["layer_suffix"]
            layer_key = f"{layer_prefix}{layer_suffix}"
        else:
            layer_key = candidate_layer_keys[0]
        layer_log = ACTIVATIONS_LOG[layer_key]

        expert_scores_key = mlp_init_config.get("expert_scores_key", "expert_ranks")
        if expert_scores_key not in layer_log:
            raise ValueError(
                f"Expert scores key {expert_scores_key=} not found in {layer_log.keys()=}"
            )
        expert_scores = layer_log[expert_scores_key]
    else:
        raise ValueError(f"Unsupported {mlp_init_config=}")
    return expert_scores
