# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from collections import defaultdict
from typing import Any

import torch
from vllm.distributed.parallel_state import get_tp_group


def _values_equal(v1: Any, v2: Any) -> bool:
    """Compare values, handling dicts with tensors."""
    if isinstance(v1, dict) and isinstance(v2, dict):
        if v1.keys() != v2.keys():
            return False
        return all(
            torch.equal(v1[k], v2[k]) if isinstance(v1[k], torch.Tensor) else v1[k] == v2[k]
            for k in v1
        )
    elif isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
        return torch.equal(v1, v2)
    return v1 == v2


def _convert_key_for_vllm(key: str, value: Any) -> tuple[str, str | None, Any]:
    """
    Transform a single key from HuggingFace format to vLLM format.

    Returns:
        Tuple of (action, new_key_or_group, value) where action is one of:
        - "copy": Copy value to new_key directly
        - "group": Add to merge group identified by new_key
        - "skip": Skip this key entirely
    """
    if "quantizer" not in key:
        return ("copy", key, value)

    # Skip softmax_quantizer and lm_head quantizers(not needed in vLLM)
    if "softmax_quantizer" in key or (key.startswith("lm_head.") and "quantizer" in key):
        return ("skip", None, None)

    # Check if this is a q/k/v projection that needs merging
    qkv_match = re.search(r"(.*\.)([qkv])_proj\.([^.]+_quantizer)(\..+)?$", key)
    if qkv_match:
        suffix = qkv_match.group(4) or ""
        group_key = qkv_match.group(1) + "qkv_proj." + qkv_match.group(3) + suffix
        return ("group", group_key, value)

    # Check if this is an expert gate/up projection
    if "mixer" not in key:
        expert_gate_up_match = re.search(
            r"(.*\.experts)\.\d+\.(gate|up)_proj\.([^.]+_quantizer)(\..+)?$", key
        )
        if expert_gate_up_match:
            suffix = expert_gate_up_match.group(4) or ""
            group_key = (
                expert_gate_up_match.group(1) + ".w13_" + expert_gate_up_match.group(3) + suffix
            )
            return ("group", group_key, value)

    # Check if this is a non-expert gate/up projection that needs merging
    if "mixer" not in key and "experts" not in key:
        gate_up_match = re.search(r"(.*\.)(gate|up)_proj\.([^.]+_quantizer)(\..+)?$", key)
        if gate_up_match:
            suffix = gate_up_match.group(4) or ""
            group_key = gate_up_match.group(1) + "gate_up_proj." + gate_up_match.group(3) + suffix
            return ("group", group_key, value)

    # Check if this is an expert down_proj
    if "mixer" not in key:
        expert_down_match = re.search(
            r"(.*\.experts)\.\d+\.down_proj\.([^.]+_quantizer)(\..+)?$", key
        )
        if expert_down_match:
            suffix = expert_down_match.group(3) or ""
            group_key = expert_down_match.group(1) + ".w2_" + expert_down_match.group(2) + suffix
            return ("group", group_key, value)

    # Transform bmm_quantizer keys: self_attn.q/k/v_bmm_quantizer -> self_attn.attn.q/k/v_bmm_quantizer
    bmm_match = re.search(r"(.*\.self_attn)\.([qkv]_bmm_quantizer.*)$", key)
    if bmm_match:
        new_key = bmm_match.group(1) + ".attn." + bmm_match.group(2)
        return ("copy", new_key, value)

    # Copy other quantizer keys as-is (like o_proj, down_proj)
    return ("copy", key, value)


def _group_keys_for_vllm(
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], defaultdict[str, list[tuple[str, Any]]]]:
    """
    Process state dict and group keys that need merging.

    Returns:
        Tuple of (direct_copy_dict, merge_groups)
    """
    vllm_state_dict = {}
    merge_groups = defaultdict(list)

    for key, value in state_dict.items():
        action, new_key, new_value = _convert_key_for_vllm(key, value)
        if new_key is None or new_value is None:
            assert action == "skip", (
                f"Expected action to be 'skip' for key {key}, value {value}, got {action}"
            )
            continue
        if action == "copy":
            vllm_state_dict[new_key] = new_value
        elif action == "group":
            merge_groups[new_key].append((key, new_value))
        # action == "skip" does nothing

    return vllm_state_dict, merge_groups


def _merge_values_by_max_or_concat(merged_key: str, key_value_pairs: list[tuple[str, Any]]) -> Any:
    """
    Merge values by taking max for amax, concatenating for others.
    Used for quantizer state weights (tensor values).
    """
    values = [value for _, value in key_value_pairs]

    # Check if values are dicts (OrderedDict) containing tensors
    if isinstance(values[0], dict):
        merged_value = {}
        for dict_key in values[0]:
            tensors = [v[dict_key] for v in values]
            if "_amax" in dict_key:
                merged_value[dict_key] = torch.stack(tensors).max(dim=0)[0]
            else:
                merged_value[dict_key] = torch.cat(tensors, dim=0)
        return merged_value
    else:
        # Values are tensors directly
        if "_amax" in merged_key:
            merged_value = torch.stack(values).max(dim=0)[0]
        else:
            merged_value = torch.cat(values, dim=0)
        return merged_value


def _merge_values_require_identical(merged_key: str, key_value_pairs: list[tuple[str, Any]]) -> Any:
    """
    Merge values by requiring all values to be identical.
    Used for quantizer state (config/metadata).
    """
    keys = [k for k, _ in key_value_pairs]
    values = [v for _, v in key_value_pairs]
    first_value = values[0]

    for i, val in enumerate(values[1:], start=1):
        if not _values_equal(val, first_value):
            raise ValueError(
                f"Cannot merge keys into '{merged_key}': values differ.\n"
                f"  '{keys[0]}' has value: {first_value}\n"
                f"  '{keys[i]}' has value: {val}"
            )
    return first_value


def convert_dict_to_vllm(
    state_dict: dict[str, Any], merge_mode: str = "max_or_concat"
) -> dict[str, Any]:
    """
    Common implementation for converting quantizer state from HF to vLLM format.

    Args:
        state_dict: Input state dict
        fuse_experts: Whether to fuse expert projections
        merge_mode: Mode to merge grouped values, "max_or_concat" or "require_identical"
    """
    vllm_state_dict, merge_groups = _group_keys_for_vllm(state_dict)

    merge_fn = (
        _merge_values_require_identical
        if merge_mode == "require_identical"
        else _merge_values_by_max_or_concat
    )

    # Merge grouped values
    for merged_key, key_value_pairs in merge_groups.items():
        if len(key_value_pairs) > 1:
            merged_value = merge_fn(merged_key, key_value_pairs)
            vllm_state_dict[merged_key] = merged_value
        else:
            # Single key, just rename it
            _, value = key_value_pairs[0]
            vllm_state_dict[merged_key] = value

    return vllm_state_dict


def convert_modelopt_state_to_vllm(modelopt_state: dict[str, Any]) -> dict[str, Any]:
    """
    Convert modelopt state from HuggingFace format to vLLM compatible format.

    This function converts the quantizer state from HuggingFace format to vLLM compatible format.

    Args:
        modelopt_state: HuggingFace modelopt state dict

    Returns:
        vLLM compatible modelopt state dict
    """
    modelopt_state_dict = modelopt_state.pop("modelopt_state_dict", [])
    for idx, current_mode in enumerate(modelopt_state_dict):
        current_mode_metadata = current_mode[1].pop("metadata", {})
        current_mode_quant_state = current_mode_metadata.pop("quantizer_state", {})
        if current_mode_quant_state:
            current_mode_metadata["quantizer_state"] = convert_dict_to_vllm(
                current_mode_quant_state, merge_mode="require_identical"
            )
        else:
            current_mode_metadata.pop("quantizer_state", None)
        current_mode[1]["metadata"] = current_mode_metadata
        modelopt_state_dict[idx] = (current_mode[0], current_mode[1])
    modelopt_state["modelopt_state_dict"] = modelopt_state_dict
    return modelopt_state


def process_state_dict_for_tp(saved_qstate_dict, current_state_dict):
    """Shard quantizer tensors for tensor parallelism by matching expected shapes."""
    tp_group = get_tp_group()
    tp_rank = tp_group.rank_in_group
    tp_world_size = tp_group.world_size

    result = {}
    for key, value in saved_qstate_dict.items():
        if key in current_state_dict:
            expected_shape = current_state_dict[key].shape
            if value.shape != expected_shape:
                # Find the dimension that was tensor-parallel sharded.
                # We expect exactly one dimension to satisfy:
                #   checkpoint_dim == expected_dim * tp_world_size
                shard_dims = [
                    d
                    for d in range(len(expected_shape))
                    if value.shape[d] == expected_shape[d] * tp_world_size
                ]
                if len(shard_dims) != 1:
                    raise ValueError(
                        f"Cannot infer TP shard dim for {key}: "
                        f"expected_shape={tuple(expected_shape)}, checkpoint_shape={tuple(value.shape)}, "
                    )

                shard_dim = shard_dims[0]
                shard_size = expected_shape[shard_dim]
                start = tp_rank * shard_size
                end = start + shard_size
                if end > value.shape[shard_dim]:
                    raise ValueError(
                        f"TP shard out of bounds for {key}: "
                        f"expected_shape={tuple(expected_shape)}, checkpoint_shape={tuple(value.shape)})"
                    )
                value = value.narrow(shard_dim, start, shard_size).contiguous()
        result[key] = value

    return result
