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

"""Constrains the search space for the MIP optimization."""

import traceback

from modelopt.torch._compress.decilm.deci_lm_hf_code.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
)
from modelopt.torch._compress.utils.utils import load_json


def drop_attentions_only(gathered_metrics, teacher_intermediate_size, teacher_n_heads_in_group):
    """
    changes the search space such that puzzle is not allowed to change the ffns
    but is only allowed to drop or reduce attention.

    Usage example:
    add the following flags to your run_puzzle command:

    --constrain_search_func drop_attentions_only --constrain_search_args {\"teacher_intermediate_size\": 14336, \"teacher_n_heads_in_group\": 16, \"above_layer\": 60}

    """

    for block_name, block_variants in gathered_metrics.items():
        to_delete = []  # Collect keys to delete after the loop
        for variant_config, variant_metrics in block_variants.items():
            block_intermediate_size = variant_config.ffn.intermediate_size
            block_attn_n_heads = variant_config.attention.n_heads_in_group
            if (
                (
                    block_intermediate_size is not None
                    and block_intermediate_size != teacher_intermediate_size
                )
                or variant_config.ffn.replace_with_linear
                or variant_config.ffn.no_op  ## uncomment this line if you want to drop only attns
                or variant_config.attention.replace_with_linear
                or (
                    block_attn_n_heads is not None
                    and block_attn_n_heads != teacher_n_heads_in_group
                )
            ):
                print(f"Marking for deletion: {block_name}-{variant_config}")
                to_delete.append(variant_config)
        for key in to_delete:
            del block_variants[key]

    print(f"new search space in block 0 {gathered_metrics['block_0']}")
    return gathered_metrics


def reduce_only_ffns(
    gathered_metrics,
    teacher_intermediate_size: int,
    teacher_n_heads_in_group: int,
    above_layer: int,
    allow_no_ops: bool,
):
    """
    only allows to reduce FFNs but not to completely drop them from layer 60 onwards
    attention is only allowed to be like uniform teacher

    Usage example:
    add the following flags to your run_puzzle command:
    constrain_search_args='{"teacher_intermediate_size": 14336, "teacher_n_heads_in_group": 16, "above_layer": 60, "allow_no_ops": false}'

    sbatch puzzle/cli/run_puzzle ... --constrain_search_func reduce_only_ffns --constrain_search_args="$(echo "$constrain_search_args" | jq -c .)"
    """
    print(f"{teacher_n_heads_in_group=}")
    for block_name, block_variants in gathered_metrics.items():
        to_delete = []  # Collect keys to delete after the loop
        block_id = int(block_name.split("_")[1])

        for variant_config, variant_metrics in block_variants.items():
            block_intermediate_size = variant_config.ffn.intermediate_size
            block_attn_n_heads = variant_config.attention.n_heads_in_group

            attn_no_op = variant_config.attention.no_op
            attn_linear = variant_config.attention.replace_with_linear
            if (
                attn_no_op
                or attn_linear
                or (block_attn_n_heads != teacher_n_heads_in_group)  # keep attention as the teacher
                or (
                    block_id <= above_layer
                    and (block_intermediate_size != teacher_intermediate_size)
                )
                or ((not allow_no_ops) and variant_config.ffn.no_op)
            ):
                # print(f"Marking for deletion: {block_name}-{variant_config}")
                to_delete.append(variant_config)  # Add key to delete list

        for key in to_delete:
            del block_variants[key]

    print(f"new search space in block 0 {gathered_metrics['block_0']}")
    return gathered_metrics


def drop_entire_blocks_only(gathered_metrics):
    teacher_block_config = _infer_teacher_config(gathered_metrics)
    for block_name, block_variants in gathered_metrics.items():
        to_delete = []  # Collect keys to delete after the loop
        for variant_config, variant_metrics in block_variants.items():
            is_no_op_block = (
                variant_config.ffn.no_op
                and variant_config.attention.no_op
                and getattr(variant_config, "parallel_blocks", None) is None
            )
            is_teacher = variant_config == teacher_block_config
            if not is_no_op_block and not is_teacher:
                to_delete.append(variant_config)
        for key in to_delete:
            del block_variants[key]

    print(f"new search space in block 0 {gathered_metrics['block_0']}")
    return gathered_metrics


def css_to_reference_attention(gathered_metrics, attention_pruned_arch):
    """
    given a reference architecture we fix the search space to only include options that change the FFNs
    but to never change the Attentions from the reference arch's Attentions.
    """

    attention_pruned_arch = load_json(attention_pruned_arch)[0]
    attention_dropped_blocks = [
        block_name
        for block_name, block_config in attention_pruned_arch["chosen_items"].items()
        if block_config["attention"]["no_op"]
    ]

    for block_name, block_variants in gathered_metrics.items():
        to_delete = []  # Collect keys to delete after the loop
        for variant_config, _ in block_variants.items():
            # Uncomment and adjust this block if needed
            # does drop only attention
            block_attn_n_heads = variant_config.attention.n_heads_in_group

            reference_arch_attn = attention_pruned_arch["chosen_items"][block_name]["attention"][
                "n_heads_in_group"
            ]
            if (  # we reduce the search space by keeping the reference arch attention as is
                (block_name in attention_dropped_blocks and not variant_config.attention.no_op)
                or (
                    block_name not in attention_dropped_blocks
                    and block_attn_n_heads != reference_arch_attn
                )
            ):
                print(f"Marking for deletion: {block_name}-{variant_config}")
                to_delete.append(variant_config)

        # Delete marked keys outside the loop
        for key in to_delete:
            del block_variants[key]

    print(f"new search space in block 0 {gathered_metrics['block_0']}")
    return gathered_metrics


def css_to_reference_ffn(gathered_metrics, ffn_pruned_arch, allow_linear_attn=True):
    """
    given a reference architecture we fix the search space to only include options that change the Attentions
    but to never change the FFNs from the reference arch's FFNs.
    """

    ffn_pruned_arch = load_json(ffn_pruned_arch)[0]

    for block_name, block_variants in gathered_metrics.items():
        to_delete = []  # Collect keys to delete after the loop
        for variant_config, _ in block_variants.items():
            block_ffn = variant_config.ffn
            is_linear_attn = variant_config.attention.replace_with_linear

            reference_arch_ffn = ffn_pruned_arch["chosen_items"][block_name]["ffn"]
            reference_arch_ffn = FFNConfig(**reference_arch_ffn)

            if (  # we reduce the search space by keeping the reference arch ffn as is
                (block_ffn != reference_arch_ffn) or (not allow_linear_attn and is_linear_attn)
            ):
                # print(f"Marking for deletion: {block_name}-{variant_config}")
                to_delete.append(variant_config)

        # Delete marked keys outside the loop
        for key in to_delete:
            del block_variants[key]

    print(f"new search space in block 0 {gathered_metrics['block_0']}")
    return gathered_metrics


def avoid_variable_gqa(
    gathered_metrics,
    allow_no_op_attn: bool = True,
    allow_linear_attn: bool = False,
    target_n_heads_in_group: int = None,
):
    """
    Allow only the teacher n_heads_in_group,
    and optionally also attention no-op (default allow)
    and attention linear (default avoid).

    This reducer affects only the attention layers: FFNs are allowed their entire search space.
    """
    teacher_block_config = infer_teacher_replacement_config(gathered_metrics)

    if target_n_heads_in_group is None:
        target_n_heads_in_group = teacher_block_config.attention.n_heads_in_group

    to_delete = []  # Collect keys to delete after the loop
    for replacement_id, replacement in gathered_metrics.items():
        variant_config = replacement["block_config"]
        if not (
            (variant_config.attention.n_heads_in_group == target_n_heads_in_group)
            or (variant_config.attention.no_op and allow_no_op_attn)
            or (variant_config.attention.replace_with_linear and allow_linear_attn)
        ):
            to_delete.append(replacement_id)

    for key in to_delete:
        del gathered_metrics[key]

    parent_layer_idx = 0
    print(
        f"new search space in block {parent_layer_idx}",
        [
            replacement["block_config"]
            for replacement_id, replacement in gathered_metrics.items()
            if replacement["parent_layer_indices"][0] == parent_layer_idx
        ],
    )
    return gathered_metrics


def reduce_in_range(
    gathered_metrics,
    layer_start: int,
    layer_end: int,
):
    """
    Allow only reduction of layers between layer_start and layer_end. Leyers before layers start, and after layer_end are kept as is (the teacher).

    """
    assert layer_start < layer_end, (
        f"Wrong input arguments: {layer_start=} must be less than {layer_end=}"
    )
    teacher_block_config = infer_teacher_replacement_config(gathered_metrics)

    to_delete = []  # Collect keys to delete after the loop
    for replacement_id, replacement in gathered_metrics.items():
        block_id = max(replacement["parent_layer_indices"])
        variant_config = replacement["block_config"]
        is_teacher = variant_config == teacher_block_config
        if (block_id < layer_start or block_id > layer_end) and not is_teacher:
            to_delete.append(replacement_id)

    for key in to_delete:
        del gathered_metrics[key]

    parent_layer_idx = 0
    print(
        f"new search space in block {parent_layer_idx}",
        [
            replacement["block_config"]
            for replacement_id, replacement in gathered_metrics.items()
            if replacement["parent_layer_indices"][0] == parent_layer_idx
        ],
    )
    return gathered_metrics


#############################################################################################


# automatically builds a dictionary mapping method names in this module to their functions
# this dictionary is used to dynamically dispatch functions
dispatcher = {
    method_name: method_callable
    for method_name, method_callable in globals().items()
    if callable(method_callable)
}


def _infer_teacher_config(gathered_metrics) -> BlockConfig:
    n_heads_in_group, intermediate_size = zip(
        *[
            (variant_config.attention.n_heads_in_group, variant_config.ffn.intermediate_size)
            for block_name, block_variants in gathered_metrics.items()
            for variant_config, variant_metrics in block_variants.items()
        ]
    )
    teacher_n_heads_in_group = min(filter(None, n_heads_in_group))
    teacher_intermediate_size = max(filter(None, intermediate_size))

    unique_teacher_candidates = set()
    for block_name, block_variants in gathered_metrics.items():
        for variant_config, variant_metrics in block_variants.items():
            if (
                variant_config.ffn.intermediate_size == teacher_intermediate_size
                and variant_config.attention.n_heads_in_group == teacher_n_heads_in_group
            ):
                unique_teacher_candidates.add(variant_config)

    assert len(unique_teacher_candidates) == 1, (
        f"Woops, expected example one candidate to be the teacher block config, instead found: {unique_teacher_candidates=}"
    )

    teacher_block_config = unique_teacher_candidates.pop()
    return teacher_block_config


def infer_teacher_replacement_config(gathered_metrics) -> BlockConfig:
    n_heads_in_group, intermediate_size = zip(
        *[
            (
                replacement["block_config"].attention.n_heads_in_group,
                replacement["block_config"].ffn.intermediate_size,
            )
            for replacement_id, replacement in gathered_metrics.items()
        ]
    )
    teacher_intermediate_size = max(filter(None, intermediate_size))
    teacher_n_heads_in_group = min(filter(None, n_heads_in_group))
    unique_teacher_candidates = set()
    for replacement_id, replacement in gathered_metrics.items():
        if (
            replacement["block_config"].ffn.intermediate_size == teacher_intermediate_size
            and replacement["block_config"].attention.n_heads_in_group == teacher_n_heads_in_group
        ):
            unique_teacher_candidates.add(replacement["block_config"])

    assert len(unique_teacher_candidates) == 1, (
        f"Woops, expected example one candidate to be the teacher block config, instead found: {unique_teacher_candidates=}"
    )

    teacher_replacement_config = unique_teacher_candidates.pop()
    return teacher_replacement_config


def apply(css_func_name, gathered_metrics, method_kwargs):
    search_space_reducer = dispatcher.get(css_func_name)
    if search_space_reducer is None:
        raise ValueError(
            f"could not find a function called `{css_func_name}` in {__name__}.py to reduce search space "
        )

    try:
        gathered_metrics = search_space_reducer(gathered_metrics, **method_kwargs)
    except Exception as e:
        traceback.print_exc()
        raise ValueError(
            f"something went wrong when trying to apply the following search space reducer `{css_func_name}` \
                         with the folloing args: {method_kwargs}, here's the exception: {e}"
        )

    return gathered_metrics
