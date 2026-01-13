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

"""TODO Add description. Analyze this code, why is it so long and complex? Can it be simplified?"""

import concurrent.futures
import dataclasses
import json
import os
import re
import time
from collections.abc import Callable
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import torch
from typeguard import check_type

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import (
    SUBBLOCK_CLS_DICT,
    BlockConfig,
    _get_dataclass_type,
    _is_dataclass_type,
)
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.configuration_decilm import DeciLMConfig
from modelopt.torch.puzzletron.tools.logger import aprint, mprint


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
    MoEChannelPruning = "MoEChannelPruning"


class LinearInitMode(Enum):
    Random = "Random"
    FromTeacher = "FromTeacher"


class HiddenSizeInitMode(Enum):
    Random = "Random"
    Truncate = "Truncate"
    PruneByChannelRanking = "PruneByChannelRanking"
    CopyAsIs = "CopyAsIs"


IgnoreFn = Callable[[str], bool]

default_ignore_fn: IgnoreFn = lambda _: False


class Printer:
    @staticmethod
    def print(s: str) -> None:
        print(s)


def _process_single_layer(
    layer_idx: int,
    parent_state_dict: dict,
    new_state_dict: dict,
    original_config: DeciLMConfig,
    new_config: DeciLMConfig,
    gqa_init_mode: GQAInitMode,
    mlp_init_mode: MlpInitMode,
    mlp_init_config: dict[str, Any] | None,
    linear_init_mode: LinearInitMode,
    ignored_keys: set,
    keys: dict,
    is_original_mha: bool,
    head_size: int,
    hidden_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Process a single layer in parallel. Returns (layer_state_dict, keys_to_remove).
    Thread-safe function for parallel layer processing.
    """
    layer_out_state_dict = {}
    keys_to_remove = {}

    parent_block_config = original_config.block_configs[layer_idx]
    child_block_config = new_config.block_configs[layer_idx]

    # Attention processing
    for part in ["weight", "bias"]:
        attn_prefix = f"model.layers.{layer_idx}.self_attn"
        q_key = f"{attn_prefix}.q_proj.{part}"
        k_key = f"{attn_prefix}.k_proj.{part}"
        v_key = f"{attn_prefix}.v_proj.{part}"
        o_key = f"{attn_prefix}.o_proj.{part}"
        attn_keys = [q_key, k_key, v_key, o_key]
        # Drop attn keys that don't exist and required to be in the new state_dict
        attn_keys = [key for key in attn_keys if key in new_state_dict]
        if len(attn_keys) > 0 and all(key in keys for key in attn_keys):
            for key in attn_keys:
                keys_to_remove[key] = keys[key]
            if all(key not in ignored_keys for key in attn_keys):
                is_student_and_teacher_have_same_attention_implementation = all(
                    key in new_state_dict for key in attn_keys
                )
                if is_student_and_teacher_have_same_attention_implementation:
                    if part == "weight":
                        wq, wk, wv, wo = _init_attention_weights(
                            gqa_init_mode=gqa_init_mode,
                            layer_idx=layer_idx,
                            new_state_dict=new_state_dict,
                            new_config=new_config,
                            original_state_dict=parent_state_dict,
                            original_config=original_config,
                            q_key=q_key,
                            k_key=k_key,
                            v_key=v_key,
                            o_key=o_key,
                            is_original_mha=is_original_mha,
                            head_size=head_size,
                            mlp_init_config=mlp_init_config,
                        )
                        layer_out_state_dict[q_key], layer_out_state_dict[k_key] = wq, wk
                        layer_out_state_dict[v_key], layer_out_state_dict[o_key] = wv, wo
                    else:
                        bias_sd = _init_attention_biases(
                            gqa_init_mode=gqa_init_mode,
                            layer_idx=layer_idx,
                            new_state_dict=new_state_dict,
                            new_config=new_config,
                            original_state_dict=parent_state_dict,
                            original_config=original_config,
                            q_key=q_key,
                            k_key=k_key,
                            v_key=v_key,
                            o_key=o_key,
                            is_original_mha=is_original_mha,
                            head_size=head_size,
                            mlp_init_config=mlp_init_config,
                        )
                        for bias_key, sd_key in zip("qkvo", [q_key, k_key, v_key, o_key]):
                            if bias_key in bias_sd.keys():
                                layer_out_state_dict[sd_key] = bias_sd[bias_key]

                else:
                    linear_attn_key = f"{attn_prefix}.linear_attn.weight"
                    is_student_attn_replaced_with_linear = linear_attn_key in new_state_dict
                    if is_student_attn_replaced_with_linear:
                        if linear_init_mode == LinearInitMode.Random:
                            layer_out_state_dict[linear_attn_key] = new_state_dict[linear_attn_key]
                        elif linear_init_mode == LinearInitMode.FromTeacher:
                            layer_out_state_dict[linear_attn_key] = _init_linear_attn(
                                parent_state_dict, original_config, layer_idx, v_key, o_key
                            )
                        else:
                            raise ValueError(f"Unknown {linear_init_mode=}")
                    else:
                        # student attn random init
                        for new_key in new_state_dict:
                            if attn_prefix in new_key:
                                layer_out_state_dict[new_key] = new_state_dict[new_key]

    # MLP/MoE processing
    is_parent_moe = parent_block_config.ffn.is_moe
    if not is_parent_moe:  # not MoE, init the MLP
        mlp_prefix = f"model.layers.{layer_idx}.mlp"
        linear_mlp_key = f"{mlp_prefix}.linear_mlp.weight"

        is_student_mlp_replaced_with_linear = linear_mlp_key in new_state_dict
        if is_student_mlp_replaced_with_linear:
            if linear_init_mode == LinearInitMode.Random:
                layer_out_state_dict[linear_mlp_key] = new_state_dict[linear_mlp_key]
            elif linear_init_mode == LinearInitMode.FromTeacher:
                teacher_mlp_state_dict = {
                    k.split(mlp_prefix + ".")[1]: v
                    for k, v in parent_state_dict.items()
                    if mlp_prefix in k
                }
                layer_out_state_dict[linear_mlp_key] = _init_linear_mlp(teacher_mlp_state_dict)
            else:
                raise ValueError(f"Unknown {linear_init_mode=}")
        else:
            layer_out_state_dict.update(
                _init_mlp(
                    mlp_init_mode=mlp_init_mode,
                    layer_idx=layer_idx,
                    original_config=original_config,
                    mlp_init_config=mlp_init_config,
                    original_state_dict=parent_state_dict,
                    new_state_dict=new_state_dict,
                    new_config=new_config,
                    keys=keys,
                    ignored_keys=ignored_keys,
                )
            )
    else:
        is_child_moe = child_block_config.ffn.is_moe
        if is_child_moe:
            parent_moe_config = original_config.block_configs[layer_idx].ffn.moe
            child_moe_config = new_config.block_configs[layer_idx].ffn.moe
            if parent_moe_config == child_moe_config:
                pass  # copy the MoE as is
            elif mlp_init_mode == MlpInitMode.MoEChannelPruning:
                for expert_idx in range(parent_moe_config.num_local_experts):
                    layer_out_state_dict.update(
                        _init_mlp(
                            mlp_init_mode=mlp_init_mode,
                            layer_idx=layer_idx,
                            original_config=original_config,
                            mlp_init_config=mlp_init_config,
                            original_state_dict=parent_state_dict,
                            new_state_dict=new_state_dict,
                            new_config=new_config,
                            keys=keys,
                            ignored_keys=ignored_keys,
                            expert_idx=expert_idx,
                        )
                    )

            elif mlp_init_mode == MlpInitMode.ExpertRemoval:  # remove some of the routed experts
                router_key, new_experts_keys = _generate_moe_keys(
                    layer_idx, child_block_config.ffn.moe.num_local_experts
                )
                _, orig_experts_keys = _generate_moe_keys(
                    layer_idx, parent_block_config.ffn.moe.num_local_experts
                )
                keys_to_remove[router_key] = keys.get(router_key)
                for key in sum(orig_experts_keys.values(), []):
                    keys_to_remove[key] = keys.get(key)

                orig_experts_weights = {
                    name: [parent_state_dict[key] for key in orig_experts_module_keys]
                    for name, orig_experts_module_keys in orig_experts_keys.items()
                }
                new_experts_weights = {
                    name: [new_state_dict[key] for key in new_experts_module_keys]
                    for name, new_experts_module_keys in new_experts_keys.items()
                }
                out_router_weights, out_experts_weights = _init_moe_module(
                    layer_idx=layer_idx,
                    mlp_init_mode=mlp_init_mode,
                    mlp_init_config=mlp_init_config,
                    orig_router_weight=parent_state_dict[router_key],
                    orig_experts_weights=orig_experts_weights,
                    new_router_weight=new_state_dict[router_key],
                    new_experts_weights=new_experts_weights,
                )
                layer_out_state_dict[router_key] = out_router_weights
                for name in new_experts_keys.keys():
                    layer_out_state_dict.update(
                        zip(new_experts_keys[name], out_experts_weights[name])
                    )
        elif child_block_config.ffn.no_op:  # no-op, drop this layer
            parent_mlp_prefix = f"model.layers.{layer_idx}.mlp"
            for key in list(keys.keys()):
                if key.startswith(parent_mlp_prefix):
                    keys_to_remove[key] = keys[key]
        else:
            assert mlp_init_mode == MlpInitMode.ConcatExpertsIntoDenseFFN, (
                "The parent layer is MoE and the child layer is a normal FFN. The only supported mode is ConcatExpertsAsMLP."
            )

            child_ffn_state_dict = _concatenate_experts_into_dense_ffn(
                parent_state_dict,
                mlp_init_config,
                hidden_size,
                layer_idx,
                child_block_config,
                parent_block_config,
            )
            layer_out_state_dict.update(child_ffn_state_dict)

            for key in list(keys.keys()):
                if key.startswith(f"model.layers.{layer_idx}.mlp"):
                    keys_to_remove[key] = keys[key]

    # Handle missing keys
    for key_possibly_missing_in_student in [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ]:
        key_possibly_missing_in_student = f".{layer_idx}.{key_possibly_missing_in_student}"
        is_key_missing_from_student = (
            len([k for k in new_state_dict if key_possibly_missing_in_student in k]) == 0
        )
        if is_key_missing_from_student:
            for k in list(keys.keys()):
                if key_possibly_missing_in_student in k:
                    keys_to_remove[k] = keys[k]

    return layer_out_state_dict, keys_to_remove


@torch.no_grad()
def create_child_state_dict(
    original_state_dict: dict,
    new_state_dict: dict,
    original_config: DeciLMConfig,
    new_config: DeciLMConfig,
    gqa_init_mode: GQAInitMode,
    ignore_fn: IgnoreFn = default_ignore_fn,
    mlp_init_mode: MlpInitMode = MlpInitMode.CopyAsIs,
    mlp_init_config: dict[str, Any] | None = None,
    owned_block_indexes: set[int] | None = None,
    linear_init_mode: LinearInitMode = LinearInitMode.Random,
    hidden_size_init_mode: HiddenSizeInitMode = HiddenSizeInitMode.CopyAsIs,
    channel_importance_path: str | None = None,
    max_layer_workers: int | None = None,  # Now optional - will auto-calculate if None
):
    mprint("=== Starting create_child_state_dict with optimizations ===")
    total_start_time = time.time()

    # Phase 1: Initial setup and validation
    setup_start_time = time.time()
    if owned_block_indexes is None:
        owned_block_indexes = set(range(new_config.num_hidden_layers))

    # Auto-calculate optimal layer workers: min(cpu_count, num_layers)
    if max_layer_workers is None:
        cpu_count = os.cpu_count() or 1
        num_layers = len(owned_block_indexes)
        max_layer_workers = min(cpu_count, num_layers)
        mprint(
            f"Auto-calculated layer workers: min({cpu_count} CPUs, {num_layers} layers) = {max_layer_workers}"
        )
    else:
        mprint(f"Using specified layer workers: {max_layer_workers}")

    # Memory optimization: Pre-allocate output state dict with known shapes
    expected_keys_and_shapes = {k: v.shape for k, v in new_state_dict.items()}
    out_state_dict = {}

    # Pre-allocate tensors where possible to reduce memory fragmentation
    for key, shape in expected_keys_and_shapes.items():
        if key in new_state_dict:
            tensor = new_state_dict[key]
            # Only make contiguous if necessary (memory optimization)
            if not tensor.is_contiguous():
                out_state_dict[key] = tensor.contiguous()
            else:
                out_state_dict[key] = tensor

    original_n_heads_in_group_per_layer = [
        b.attention.n_heads_in_group for b in original_config.block_configs
    ]
    is_original_mha = set(original_n_heads_in_group_per_layer) == {1}
    is_same_hidden_size = original_config.hidden_size == new_config.hidden_size
    head_size = new_config.head_dim
    orig_head_size = original_config.head_dim
    assert head_size == orig_head_size, f"head_size {head_size} != orig_head_size {orig_head_size}"

    # Allow different hidden sizes for pruning
    if not is_same_hidden_size:
        assert new_config.hidden_size <= original_config.hidden_size, (
            f"New hidden size ({new_config.hidden_size}) must be <= original ({original_config.hidden_size})"
        )
        assert hidden_size_init_mode != HiddenSizeInitMode.CopyAsIs, (
            "Cannot copy as is when hidden sizes differ"
        )

    hidden_size = original_config.hidden_size

    ignored_keys = set([key for key in original_state_dict if ignore_fn(key)])
    for key in ignored_keys:
        aprint(f"Ignoring key {key} and taking its init from new_state_dict")
        out_state_dict[key] = new_state_dict[key]

    keys = {
        match.group(1) if (match := re.search(r"(h\.\d+\..*)", key)) is not None else key: key
        for key in original_state_dict
    }
    setup_time = time.time() - setup_start_time
    mprint(f"Phase 1 - Setup and memory pre-allocation: {setup_time:.2f}s")

    # Phase 2: Parallel layer processing
    layer_processing_start_time = time.time()

    # Prepare arguments for parallel processing
    process_layer_partial = partial(
        _process_single_layer,
        parent_state_dict=original_state_dict,
        new_state_dict=new_state_dict,
        original_config=original_config,
        new_config=new_config,
        gqa_init_mode=gqa_init_mode,
        mlp_init_mode=mlp_init_mode,
        mlp_init_config=mlp_init_config,
        linear_init_mode=linear_init_mode,
        ignored_keys=ignored_keys,
        keys=keys,
        is_original_mha=is_original_mha,
        head_size=head_size,
        hidden_size=hidden_size,
    )

    # Process layers in parallel with optimal worker count
    mprint(
        f"Processing {len(owned_block_indexes)} layers in parallel with {max_layer_workers} workers..."
    )
    layer_results = []
    all_keys_to_remove = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_layer_workers) as executor:
        future_to_layer = {
            executor.submit(process_layer_partial, layer_idx): layer_idx
            for layer_idx in owned_block_indexes
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_layer):
            layer_idx = future_to_layer[future]
            try:
                layer_state_dict, keys_to_remove = future.result()
                layer_results.append((layer_idx, layer_state_dict))
                all_keys_to_remove.update(keys_to_remove)

                completed += 1
                if completed % 20 == 0 or completed == len(
                    owned_block_indexes
                ):  # More frequent progress updates
                    mprint(f"Completed {completed}/{len(owned_block_indexes)} layers")
            except Exception as exc:
                mprint(f"Layer {layer_idx} generated an exception: {exc}")
                raise exc

    # Merge layer results into main state dict (memory efficient)
    for layer_idx, layer_state_dict in layer_results:
        out_state_dict.update(layer_state_dict)

    # Remove processed keys from the keys dict
    for key_to_remove in all_keys_to_remove:
        keys.pop(key_to_remove, None)

    layer_processing_time = time.time() - layer_processing_start_time
    mprint(
        f"Phase 2 - Parallel layer processing: {layer_processing_time:.2f}s ({max_layer_workers} workers)"
    )

    # Phase 3: Copy remaining keys from original model
    copy_start_time = time.time()
    keys_to_copy_from_orig_model = set(keys.values()) - ignored_keys
    for key in keys_to_copy_from_orig_model:
        # Memory optimization: avoid unnecessary copies
        tensor = original_state_dict[key]
        if not tensor.is_contiguous():
            out_state_dict[key] = tensor.contiguous()
        else:
            out_state_dict[key] = tensor
    copy_time = time.time() - copy_start_time
    mprint(
        f"Phase 3 - Copy remaining keys: {copy_time:.2f}s ({len(keys_to_copy_from_orig_model)} keys)"
    )

    # Handle hidden size pruning for remaining keys
    if not is_same_hidden_size:
        out_state_dict = _apply_hidden_size_pruning(
            out_state_dict,
            original_state_dict,
            new_config,
            original_config,
            hidden_size_init_mode,
            channel_importance_path,
            owned_block_indexes,
        )

    # Phase 4: Verification
    verify_start_time = time.time()
    _verify_state_dicts_match(out_state_dict, expected_keys_and_shapes)
    verify_time = time.time() - verify_start_time
    mprint(f"Phase 4 - Verification: {verify_time:.2f}s")

    total_time = time.time() - total_start_time
    mprint(f"=== create_child_state_dict completed in {total_time:.2f}s ===")
    mprint(
        f"Breakdown: Setup {setup_time:.1f}s + ParallelProcessing {layer_processing_time:.1f}s + Copy {copy_time:.1f}s + Verify {verify_time:.1f}s"
    )
    mprint(
        f"Speedup: Used {max_layer_workers} workers for {len(owned_block_indexes)} layers (CPU utilization: {max_layer_workers}/{os.cpu_count() or 1})"
    )

    return out_state_dict


def _generate_moe_keys(layer_idx: int, num_experts: int) -> tuple[str, dict[str, list[str]]]:
    mlp_prefix = f"model.layers.{layer_idx}.mlp"
    router_key = f"{mlp_prefix}.router.weight"
    names = ["gate_proj", "up_proj", "down_proj"]
    experts_module_names = {
        name: f"{mlp_prefix}.experts.{{expert_idx}}.{name}.weight" for name in names
    }
    return router_key, {
        name: [module_name.format(expert_idx=expert_idx) for expert_idx in range(num_experts)]
        for name, module_name in experts_module_names.items()
    }


def _concatenate_experts_into_dense_ffn(
    original_state_dict: dict[str, torch.Tensor],
    mlp_init_config: dict | None,
    hidden_size: int,
    layer_idx: int,
    child_block_config: BlockConfig,
    parent_block_config: BlockConfig,
) -> dict[str, torch.Tensor]:
    assert child_block_config.ffn.gated and child_block_config.ffn.hidden_act == "silu", (
        "Llama4 experts use SwiGLU."
    )

    # verify sizes
    child_intermediate_size = child_block_config.ffn.intermediate_size
    parent_moe_config = parent_block_config.ffn.moe
    shared_expert_intermediate_dim = parent_moe_config.shared_expert_intermediate_dim
    routed_expert_intermediate_dim = parent_moe_config.expert_intermediate_dim
    total_concatenated_routed_experts_size = (
        child_intermediate_size - shared_expert_intermediate_dim
    )
    assert total_concatenated_routed_experts_size % routed_expert_intermediate_dim == 0, (
        f"{child_intermediate_size=}  "
        f"{shared_expert_intermediate_dim=}  "
        f"{routed_expert_intermediate_dim=}  "
        f"{total_concatenated_routed_experts_size=}  "
        f"{total_concatenated_routed_experts_size % routed_expert_intermediate_dim=} != 0"
    )
    num_concatenated_routed_experts = (
        total_concatenated_routed_experts_size // routed_expert_intermediate_dim
    )

    # if needed, concatenate some of the routed experts
    if num_concatenated_routed_experts == 0:
        print(
            f"Removing all routed experts from layer {layer_idx}, turning the shared expert into a dense FFN."
        )
        concat_routed_state_dict = dict()
    else:
        print(
            f"Concatenating {num_concatenated_routed_experts} routed experts to the shared expert in layer {layer_idx}"
        )
        router_key, orig_experts_keys = _generate_moe_keys(
            layer_idx, parent_moe_config.num_local_experts
        )
        orig_experts_weights = {
            name: [original_state_dict[key] for key in orig_experts_module_keys]
            for name, orig_experts_module_keys in orig_experts_keys.items()
        }
        _, experts_weights = _prune_experts_by_score(
            mlp_init_config=mlp_init_config,
            layer_idx=layer_idx,
            orig_router_weight=original_state_dict[router_key],
            orig_experts_weights=orig_experts_weights,
            new_num_experts=num_concatenated_routed_experts,
        )
        concat_dims = {"gate_proj": 0, "up_proj": 0, "down_proj": 1}
        assert list(concat_dims) == list(experts_weights), (
            "concat_dims and experts_weights must have the same keys"
        )
        concat_routed_state_dict = {
            name: torch.cat(experts_weights[name], dim=concat_dims[name]) for name in concat_dims
        }

    # turn the shared expert into a normal FFN. concatenate the pruned routed experts if needed.
    parent_shared_expert_prefix = f"model.layers.{layer_idx}.mlp.shared_expert"
    child_ffn_prefix = f"model.layers.{layer_idx}.mlp"
    child_ffn_state_dict = dict()

    for module_name in [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]:
        shared_expert_key = f"{parent_shared_expert_prefix}.{module_name}.weight"
        child_ffn_key = f"{child_ffn_prefix}.{module_name}.weight"
        shared_expert_weight = original_state_dict[shared_expert_key]
        concat_routed_weight = concat_routed_state_dict.get(module_name)

        if concat_routed_weight is None:
            child_weight = shared_expert_weight
        else:
            child_weight = torch.cat(
                [shared_expert_weight, concat_routed_weight],
                dim=1 if module_name == "down_proj" else 0,
            )
        child_ffn_state_dict[child_ffn_key] = child_weight

    return child_ffn_state_dict


def _verify_state_dicts_match(
    state_dict: dict[str, torch.Tensor],
    expected_keys_and_shapes: dict[str, torch.Size],
) -> None:
    # Verify keys match
    expected_keys = expected_keys_and_shapes.keys()
    missing_keys = set(expected_keys) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(expected_keys)
    assert len(missing_keys) == 0 and len(unexpected_keys) == 0, (
        f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}"
    )

    # Verify shapes match
    shape_mismatches = []
    for key in expected_keys:
        expected_shape = expected_keys_and_shapes[key]
        actual_shape = state_dict[key].shape
        if expected_shape != actual_shape:
            shape_mismatches.append(f"{key}: expected {expected_shape}, got {actual_shape}")

    assert len(shape_mismatches) == 0, "Shape mismatches found:\n" + "\n".join(shape_mismatches)
    print("""
############################
create_child_state_dict: all keys and shapes matched successfully.
############################
""")


def _init_mlp(
    *,
    mlp_init_mode: MlpInitMode | str,
    layer_idx: int,
    original_config: DeciLMConfig,
    mlp_init_config: dict[str, Any] | None,
    original_state_dict: dict,
    new_state_dict: dict,
    new_config: DeciLMConfig,
    keys: dict[str, str],
    ignored_keys: set[str],
    expert_idx: int | None = None,
) -> dict[str, torch.Tensor]:
    out_state_dict = {}

    if mlp_init_mode == MlpInitMode.MoEChannelPruning:
        if expert_idx is None:
            return {}
        mlp_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
    else:
        mlp_prefix = f"model.layers.{layer_idx}.mlp"

    key = f"{mlp_prefix}.down_proj.weight"
    if key not in keys:
        return {}

    mlp_c_proj_key = keys[key]
    if mlp_c_proj_key not in ignored_keys:
        mlp_keys = [
            keys.pop(f"{mlp_prefix}.{module_name}.weight")
            for module_name in ["down_proj", "gate_proj", "up_proj"]
        ]
        pruned_filters = None
        projection_matrix = None
        for mlp_key in mlp_keys:
            expanded_dim = 1 if "down_proj" in mlp_key else 0
            if mlp_key in new_state_dict:
                mlp_module_weight, pruned_filters, projection_matrix = _init_mlp_module(
                    mlp_init_mode,
                    expanded_dim,
                    new_state_dict[mlp_key],
                    new_config,
                    original_state_dict[mlp_key],
                    original_config,
                    mlp_init_config,
                    pruned_filters,
                    projection_matrix,
                    mlp_prefix,
                )
                out_state_dict[mlp_key] = mlp_module_weight
            else:
                mprint(f"mlp_key {mlp_key} not in new_state_dict")
    return out_state_dict


def _init_mlp_module(
    mlp_init_mode: MlpInitMode | str,
    expanded_dim: int,
    new_item: torch.Tensor,
    new_config: DeciLMConfig,
    orig_item: torch.Tensor,
    original_config: DeciLMConfig,
    mlp_init_config: dict[str, Any] | None,
    pruned_filters: torch.Tensor | None = None,
    projection_matrix: dict[str, torch.Tensor] | None = None,
    mlp_prefix: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
    if isinstance(mlp_init_mode, str):
        mlp_init_mode = MlpInitMode(mlp_init_mode)
    assert orig_item.ndim == 2, f"{orig_item.ndim=}"
    assert new_item.ndim == 2, f"{new_item.ndim=}"

    assert new_config.num_hidden_layers == original_config.num_hidden_layers, (
        f"({new_config.num_hidden_layers=}) != ({original_config.num_hidden_layers=})"
    )

    orig_ffn_size = orig_item.shape[expanded_dim]
    new_ffn_size = new_item.shape[expanded_dim]

    if mlp_init_mode == MlpInitMode.CopyAsIs:
        assert new_ffn_size == orig_ffn_size, (
            f"({new_ffn_size=}) != ({orig_ffn_size=}), can't be copied as is."
        )
        mlp_module_weight = orig_item

    elif mlp_init_mode == MlpInitMode.Random:
        mlp_module_weight = new_item

    elif new_ffn_size == orig_ffn_size:
        mlp_module_weight = orig_item

    elif mlp_init_mode in (
        MlpInitMode.Truncate,
        MlpInitMode.PruneByActivationsLog,
        MlpInitMode.MoEChannelPruning,
    ):
        assert new_ffn_size <= orig_ffn_size, (
            f"({new_ffn_size=}) > ({orig_ffn_size=}), can't be truncated."
        )

        if mlp_init_mode == MlpInitMode.Truncate:
            truncated_weight = torch.narrow(
                orig_item, dim=expanded_dim, start=0, length=new_ffn_size
            )
            mlp_module_weight = truncated_weight

        elif mlp_init_mode in (MlpInitMode.PruneByActivationsLog, MlpInitMode.MoEChannelPruning):
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
        assert new_ffn_size == orig_ffn_size, (
            f"({new_ffn_size=}) != ({orig_ffn_size=}), can't be copied as is."
        )
        mlp_module_weight = orig_item

    else:
        raise ValueError(f"Unsupported {mlp_init_mode=}")

    return mlp_module_weight, pruned_filters, projection_matrix


def _init_moe_module(
    *,
    mlp_init_mode: MlpInitMode | str,
    mlp_init_config: dict[str, Any] | None,
    layer_idx: int,
    orig_router_weight: torch.Tensor,
    orig_experts_weights: dict[str, list[torch.Tensor]],
    new_router_weight: torch.Tensor,
    new_experts_weights: dict[str, list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
    if isinstance(mlp_init_mode, str):
        mlp_init_mode = MlpInitMode(mlp_init_mode)

    if mlp_init_mode == MlpInitMode.ExpertRemoval:
        result_router_weight, result_experts_weights = _prune_experts_by_score(
            mlp_init_config=mlp_init_config,
            layer_idx=layer_idx,
            orig_router_weight=orig_router_weight,
            orig_experts_weights=orig_experts_weights,
            new_num_experts=new_router_weight.shape[0],
        )
    else:
        raise ValueError(f"Unsupported {mlp_init_mode=}")

    assert result_router_weight.shape == new_router_weight.shape
    assert result_experts_weights.keys() == new_experts_weights.keys(), (
        "result_experts_weights and new_experts_weights must have the same keys"
    )
    assert all(
        len(new_experts_weights[name]) == len(result_experts_weights[name])
        for name in result_experts_weights.keys()
    )
    assert all(
        all(
            new_expert_weight.shape == result_expert_weight.shape
            for new_expert_weight, result_expert_weight in zip(
                new_experts_weights[name], result_experts_weights[name]
            )
        )
        for name in result_experts_weights.keys()
    )
    return result_router_weight, result_experts_weights


def _prune_experts_by_score(
    *,
    mlp_init_config: dict[str, Any],
    layer_idx: int,
    orig_router_weight: torch.Tensor,
    orig_experts_weights: dict[str, list[torch.Tensor]],
    new_num_experts: int,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    orig_num_experts = orig_router_weight.shape[0]
    assert all(
        len(orig_experts_module_weights) == orig_num_experts
        for orig_experts_module_weights in orig_experts_weights.values()
    )
    expert_scores = _load_expert_scores(mlp_init_config)[layer_idx]
    assert len(expert_scores) == orig_num_experts
    selected_experts = sorted(
        range(orig_num_experts),
        key=lambda i: expert_scores[i],
        reverse=mlp_init_config.get("higher_is_better", True),
    )[:new_num_experts]
    result_router_weight = orig_router_weight[selected_experts]
    result_experts_weights = {
        name: [orig_experts_module_weights[i] for i in selected_experts]
        for name, orig_experts_module_weights in orig_experts_weights.items()
    }
    return result_router_weight, result_experts_weights


def _load_expert_scores(mlp_init_config: dict[str, Any] | None) -> list[list[int | float]]:
    assert mlp_init_config is not None
    if "expert_scores_file" in mlp_init_config:
        expert_scores_file = mlp_init_config["expert_scores_file"]
        with open(expert_scores_file) as f:
            expert_scores = json.load(f)
    elif "activations_log_dir" in mlp_init_config:
        _cache_activations_log(mlp_init_config)
        num_layers = len(ACTIVATIONS_LOG)
        expert_scores = []
        for layer_idx in range(num_layers):
            router_name = f"model.layers.{layer_idx}.mlp.router"
            expert_scores.append(ACTIVATIONS_LOG[router_name]["expert_ranks"])
        expert_scores = torch.stack(expert_scores)
        expert_scores = expert_scores.tolist()
    else:
        raise ValueError(f"Unsupported {mlp_init_config=}")
    return expert_scores


ACTIVATIONS_LOG = dict()


def _cache_activations_log(mlp_init_config: dict[str, Any]) -> None:
    if len(ACTIVATIONS_LOG) == 0:
        assert "activations_log_dir" in mlp_init_config
        activations_log_dir = mlp_init_config["activations_log_dir"]
        ACTIVATIONS_LOG.update(
            {
                module_name: module_log
                for p in Path(activations_log_dir).glob("rank*.pth")
                for module_name, module_log in torch.load(p).items()
            }
        )


def _load_activations_log(mlp_init_config: dict[str, Any], module_name: str) -> torch.Tensor:
    _cache_activations_log(mlp_init_config)
    module_log = ACTIVATIONS_LOG[module_name]
    filter_importance = module_log["score"]
    return filter_importance


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
    n_heads_in_group = new_config.block_configs[layer_idx].attention.n_heads_in_group
    orig_n_heads_in_group = original_config.block_configs[layer_idx].attention.n_heads_in_group
    num_kv_heads = num_q_heads // n_heads_in_group
    orig_num_kv_heads = num_q_heads // orig_n_heads_in_group

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
        assert n_heads_in_group % orig_n_heads_in_group == 0, (
            f"({n_heads_in_group=}) % ({orig_n_heads_in_group=}) != 0"
        )
        n_heads_to_aggregate = n_heads_in_group // orig_n_heads_in_group

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

        wk = degroup_w(wk)
        wv = degroup_w(wv)

    elif gqa_init_mode == GQAInitMode.PruneKVHeads:
        wk = wk.view(orig_num_kv_heads, head_size, dim1)
        wv = wv.view(orig_num_kv_heads, head_size, dim1)
        wq = wq.view(orig_num_kv_heads, orig_n_heads_in_group, head_size, dim1)
        wo = wo.view(dim1, orig_num_kv_heads, orig_n_heads_in_group, head_size)

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
    new_config: DeciLMConfig,
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
        for key in bias_sd:
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


def _init_linear_attn(
    parent_state_dict: dict[str, torch.Tensor],
    parent_config: DeciLMConfig,
    layer_idx: int,
    v_key: str,
    o_key: str,
) -> torch.Tensor:
    """Init a linear layer that operates like an attention layer that assigns score 1 to the current token
    and score 0 to all others: out = (Wo @ Wv) @ x
    """
    n_embd = parent_config.hidden_size
    head_size = parent_config.head_dim
    n_heads_in_group = parent_config.block_configs[layer_idx].attention.n_heads_in_group
    n_kv_heads = parent_config.num_attention_heads // n_heads_in_group

    wv = parent_state_dict[v_key]
    wv = wv.view(n_kv_heads, head_size, n_embd)
    wv_expanded = torch.repeat_interleave(wv, n_heads_in_group, dim=0).reshape(n_embd, n_embd)

    wo = parent_state_dict[o_key]

    w_linear = wo @ wv_expanded
    return w_linear


def _init_linear_mlp(teacher_mlp_state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """A linear layer that does (W_down @ W_up) @ x, ignoring W_gate."""
    if "linear_mlp.weight" in teacher_mlp_state_dict:  # if the teacher itself is a linear layer
        return teacher_mlp_state_dict["linear_mlp.weight"]

    w_up = teacher_mlp_state_dict["up_proj.weight"]
    w_down = teacher_mlp_state_dict["down_proj.weight"]
    w_linear = w_down @ w_up
    return w_linear


def update_model_config(
    model_config: DeciLMConfig,
    model_config_overrides: None | list[dict[str, Any]] | str | dict | Path = None,
) -> DeciLMConfig:
    new_model_config = deepcopy(model_config)
    if model_config_overrides is None:
        return new_model_config

    model_config_overrides = _parse_model_config_overrides(
        model_config_overrides, model_config.num_hidden_layers
    )

    def override(item, item_overrides):
        if item_overrides is None:
            return item_overrides
        if dataclasses.is_dataclass(item):
            assert isinstance(item_overrides, dict)
            return dataclass_override(item, item_overrides)
        if isinstance(item, list):
            assert isinstance(item_overrides, list)
            return list_override(item, item_overrides)
        return item_overrides

    def list_override(ls, ls_overrides: list):
        assert len(ls) == len(ls_overrides)
        return [override(item, item_overrides) for item, item_overrides in zip(ls, ls_overrides)]

    def dataclass_override(dc, dc_overrides: dict):
        if not set(dc_overrides.keys()).issubset(dataclasses.asdict(dc).keys()):
            raise ValueError(
                f"Uknown overrides for dataclass {type(dc)}: {', '.join(set(dc_overrides.keys()) - dataclasses.asdict(dc).keys())}"
            )
        field_types = {field.name: field.type for field in dataclasses.fields(dc)}
        dc_changes = {}
        for key, item_overrides in dc_overrides.items():
            previous_value, item_type = getattr(dc, key), field_types[key]
            # if original block was no_op, we should not override it
            if getattr(dc, "no_op", False):
                return dc

            if previous_value is None and _is_dataclass_type(item_type):
                new_value = _get_dataclass_type(item_type)(**item_overrides)
            else:
                new_value = override(previous_value, item_overrides)
            check_type(new_value, item_type)
            dc_changes[key] = new_value
        return dataclasses.replace(dc, **dc_changes)

    new_model_config.block_configs = list_override(
        new_model_config.block_configs, model_config_overrides
    )

    return new_model_config


def _parse_model_config_overrides(
    model_config_overrides_json: str | dict | Path | list[dict],
    n_layer: int,
) -> list[dict[str, Any]]:
    """Example model_config_overrides_json:
    {
        "attention": [{"n_heads_in_group": 2}],
        "ffn": [{"intermediate_size": 14336}]
    }
    """
    if isinstance(model_config_overrides_json, list) and isinstance(
        model_config_overrides_json[0], dict
    ):
        return model_config_overrides_json

    if isinstance(model_config_overrides_json, dict):
        model_config_overrides_dict = model_config_overrides_json
    else:
        if os.path.exists(
            model_config_overrides_json
        ):  # using os.path.exists, because Path.exists throws an exception on long strings
            model_config_overrides_json = Path(model_config_overrides_json).read_text()
        print(f"I'm json loadsing over here. {model_config_overrides_json=}")
        model_config_overrides_dict = json.loads(model_config_overrides_json)

    # Sanity checks and conversion to list of dictionaries
    layer_wise_overrides = [{} for _ in range(n_layer)]
    for config_key, config_value in model_config_overrides_dict.items():
        assert config_key in SUBBLOCK_CLS_DICT, f"Unknown config key: {config_key}"
        assert isinstance(config_value, list), (
            f"Expected a list for {config_key}, got {config_value}"
        )
        assert len(config_value) == n_layer or len(config_value) == 1, (
            f"Number of elements in {config_key} must be 1 or equal to the number of layers in the model"
        )

        if len(config_value) == 1:
            model_config_overrides_dict[config_key] = config_value * n_layer

        for layer_idx in range(n_layer):
            layer_wise_overrides[layer_idx][config_key] = model_config_overrides_dict[config_key][
                layer_idx
            ]

    return layer_wise_overrides


def _apply_hidden_size_pruning(
    out_state_dict: dict[str, torch.Tensor],
    original_state_dict: dict[str, torch.Tensor],
    new_config: DeciLMConfig,
    original_config: DeciLMConfig,
    hidden_size_init_mode: HiddenSizeInitMode,
    channel_importance_path: str | None = None,
    owned_block_indexes: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Apply hidden size pruning to all layers that depend on hidden_size.
    This includes embeddings, layer norms, and any linear layers that haven't been handled yet.
    """
    if isinstance(hidden_size_init_mode, str):
        hidden_size_init_mode = HiddenSizeInitMode(hidden_size_init_mode)

    original_hidden_size = original_config.hidden_size
    new_hidden_size = new_config.hidden_size

    if hidden_size_init_mode == HiddenSizeInitMode.CopyAsIs:
        return out_state_dict

    # Load channel ranking if needed
    if hidden_size_init_mode == HiddenSizeInitMode.PruneByChannelRanking:
        if channel_importance_path is not None:
            with open(channel_importance_path) as f:
                channel_ranking = json.load(f)["channel_importance_ranking"]
        else:
            raise ValueError(
                "channel_ranking_path must be provided in hidden_size_init_config for PruneByChannelRanking mode"
            )

    # Handle embedding layer
    embed_key = "model.embed_tokens.weight"
    if embed_key in out_state_dict and embed_key in original_state_dict:
        out_state_dict[embed_key] = _prune_hidden_size_dimension(
            original_state_dict[embed_key],
            new_hidden_size,
            hidden_size_init_mode,
            channel_ranking,
            dim=1,
        )
    else:
        raise ValueError(
            f"Embed key {embed_key} not found in out_state_dict or original_state_dict"
        )

    # Handle final layer norm
    norm_key = "model.norm.weight"
    if norm_key in out_state_dict and norm_key in original_state_dict:
        out_state_dict[norm_key] = _prune_hidden_size_dimension(
            original_state_dict[norm_key],
            new_hidden_size,
            hidden_size_init_mode,
            channel_ranking,
            dim=0,
        )

    # Handle LM head
    lm_head_key = "lm_head.weight"
    if lm_head_key in out_state_dict and lm_head_key in original_state_dict:
        if out_state_dict[lm_head_key].shape[1] != new_hidden_size:
            out_state_dict[lm_head_key] = _prune_hidden_size_dimension(
                original_state_dict[lm_head_key],
                new_hidden_size,
                hidden_size_init_mode,
                channel_ranking,
                dim=1,
            )

    for block_idx in owned_block_indexes:
        if new_config.block_configs[block_idx].parallel_blocks is None:
            key_prefix = f"model.layers.{block_idx}"
            out_state_dict = _prune_hidden_size_dimension_block(
                out_state_dict,
                new_hidden_size,
                hidden_size_init_mode,
                channel_ranking,
                new_config.block_configs[block_idx],
                key_prefix,
            )
        else:
            for internal_block_idx in range(
                len(new_config.block_configs[block_idx].parallel_blocks)
            ):
                block_config = new_config.block_configs[block_idx].parallel_blocks[
                    internal_block_idx
                ]
                key_prefix = f"model.layers.{block_idx}.parallel_blocks.{internal_block_idx}"
                out_state_dict = _prune_hidden_size_dimension_block(
                    out_state_dict,
                    new_hidden_size,
                    hidden_size_init_mode,
                    channel_ranking,
                    block_config,
                    key_prefix,
                )
    return out_state_dict


def _prune_hidden_size_dimension_block(
    out_state_dict,
    new_hidden_size,
    hidden_size_init_mode,
    channel_ranking,
    block_config,
    key_prefix,
):
    for layer_norm in ["input_layernorm", "post_attention_layernorm"]:
        for part in ["weight", "bias"]:
            key = f"{key_prefix}.{layer_norm}.{part}"
            if key in out_state_dict:
                out_state_dict[key] = _prune_hidden_size_dimension(
                    out_state_dict[key],
                    new_hidden_size,
                    hidden_size_init_mode,
                    channel_ranking,
                    dim=0,
                )
    attn_prefix = f"{key_prefix}.self_attn"
    if block_config.attention.replace_with_linear:
        linear_attn_key = f"{attn_prefix}.linear_attn.weight"
        for dim in [0, 1]:
            out_state_dict[linear_attn_key] = _prune_hidden_size_dimension(
                out_state_dict[linear_attn_key],
                new_hidden_size,
                hidden_size_init_mode,
                channel_ranking,
                dim=dim,
            )
    elif block_config.attention.is_mamba:
        for proj in ["in", "out"]:
            mamba_key = f"{attn_prefix}.mamba_mixer.{proj}_proj.weight"
            out_state_dict[mamba_key] = _prune_hidden_size_dimension(
                out_state_dict[mamba_key],
                new_hidden_size,
                hidden_size_init_mode,
                channel_ranking,
                dim=1 if proj == "in" else 0,
            )
    else:
        for k in "qkvo":
            for part in ["weight", "bias"]:
                if k in "qkv" and part == "bias":
                    continue
                key = f"{attn_prefix}.{k}_proj.{part}"
                if key in out_state_dict:
                    out_state_dict[key] = _prune_hidden_size_dimension(
                        out_state_dict[key],
                        new_hidden_size,
                        hidden_size_init_mode,
                        channel_ranking,
                        dim=1 if part == "weight" and k in "qkv" else 0,
                    )
    ffn_prefix = f"{key_prefix}.mlp"
    if block_config.ffn.replace_with_linear:
        linear_mlp_key = f"{ffn_prefix}.linear_mlp.weight"
        for dim in [0, 1]:
            out_state_dict[linear_mlp_key] = _prune_hidden_size_dimension(
                out_state_dict[linear_mlp_key],
                new_hidden_size,
                hidden_size_init_mode,
                channel_ranking,
                dim=dim,
            )
    elif block_config.ffn.moe is not None:
        router_key = f"{ffn_prefix}.router.weight"
        out_state_dict[router_key] = _prune_hidden_size_dimension(
            out_state_dict[router_key],
            new_hidden_size,
            hidden_size_init_mode,
            channel_ranking,
            dim=1,
        )
        _prune_hidden_size_dimension_mlp(
            f"{ffn_prefix}.shared_expert",
            out_state_dict,
            new_hidden_size,
            hidden_size_init_mode,
            channel_ranking,
        )
        for expert_idx in range(block_config.ffn.moe.num_local_experts):
            _prune_hidden_size_dimension_mlp(
                f"{ffn_prefix}.experts.{expert_idx}",
                out_state_dict,
                new_hidden_size,
                hidden_size_init_mode,
                channel_ranking,
            )
    else:
        _prune_hidden_size_dimension_mlp(
            ffn_prefix, out_state_dict, new_hidden_size, hidden_size_init_mode, channel_ranking
        )
    return out_state_dict


def _prune_hidden_size_dimension_mlp(
    name_prefix, out_state_dict, new_hidden_size, hidden_size_init_mode, channel_ranking
):
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        for part in ["weight", "bias"]:
            if proj != "down_proj" and part == "bias":
                continue
            key = f"{name_prefix}.{proj}.{part}"
            if key in out_state_dict:
                out_state_dict[key] = _prune_hidden_size_dimension(
                    out_state_dict[key],
                    new_hidden_size,
                    hidden_size_init_mode,
                    channel_ranking,
                    dim=1 if part == "weight" and proj != "down_proj" else 0,
                )


def _prune_hidden_size_dimension(
    original_tensor: torch.Tensor,
    new_hidden_size: int,
    hidden_size_init_mode: HiddenSizeInitMode,
    channel_ranking: list[int] | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """Prune a tensor along the specified dimension to match the new hidden size."""
    original_size = original_tensor.shape[dim]

    if hidden_size_init_mode == HiddenSizeInitMode.Random:
        # Initialize with random weights
        new_shape = list(original_tensor.shape)
        new_shape[dim] = new_hidden_size
        return torch.randn(new_shape, dtype=original_tensor.dtype, device=original_tensor.device)

    elif hidden_size_init_mode == HiddenSizeInitMode.Truncate:
        # Simple truncation - take the first new_hidden_size elements
        if dim == -1:
            return original_tensor[..., :new_hidden_size]
        elif dim == 0:
            return original_tensor[:new_hidden_size, ...]
        elif dim == 1:
            return original_tensor[:, :new_hidden_size, ...]
        else:
            # Handle other dimensions
            slices = [slice(None)] * original_tensor.ndim
            slices[dim] = slice(new_hidden_size)
            return original_tensor[tuple(slices)]

    elif hidden_size_init_mode == HiddenSizeInitMode.PruneByChannelRanking:
        if channel_ranking is None:
            raise ValueError("Channel ranking must be provided for PruneByChannelRanking mode")

        # Use channel ranking to select the most important channels
        if len(channel_ranking) < new_hidden_size:
            raise ValueError(
                f"Channel ranking has {len(channel_ranking)} channels but need {new_hidden_size}"
            )

        # Take the top new_hidden_size channels according to ranking
        selected_channels = channel_ranking[:new_hidden_size]

        if dim == -1:
            return original_tensor[..., selected_channels]
        elif dim == 0:
            return original_tensor[selected_channels, ...]
        elif dim == 1:
            return original_tensor[:, selected_channels, ...]
        else:
            # Handle other dimensions
            slices = [slice(None)] * original_tensor.ndim
            slices[dim] = selected_channels
            return original_tensor[tuple(slices)]

    else:
        raise ValueError(f"Unsupported hidden_size_init_mode: {hidden_size_init_mode}")
