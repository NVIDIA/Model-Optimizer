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
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from vllm.distributed.parallel_state import get_tp_group

from modelopt.torch.export.plugins.vllm_fakequant_hf import (
    infer_quantizer_prefix_remap,
    is_weight_quantizer_state_key,
    merge_amax_tensors_for_group,
)
from modelopt.torch.opt.conversion import (
    ModelLikeModule,
    ModeloptStateManager,
    _check_init_modellike,
)
from modelopt.torch.quantization.conversion import (
    convert_to_quantized_model,
    restore_quantizer_state,
)
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils import is_quantized


def _union_quantizer_keys_across_ranks(local_quantizer_keys: list[str]) -> set[str]:
    """Union of quantizer key strings from every rank (same file on all ranks → identical to local)."""
    local = set(local_quantizer_keys)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local
    if torch.distributed.get_world_size() <= 1:
        return local
    try:
        world_size = torch.distributed.get_world_size()
        gathered: list[list[str]] = [[] for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, list(local_quantizer_keys))
        out: set[str] = set()
        for g in gathered:
            out.update(g)
        return out
    except Exception as e:
        warnings.warn(
            f"Could not all_gather quantizer key lists across ranks ({e}); using this rank's keys only."
        )
        return local


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

    # Skip softmax_quantizer and lm_head quantizers (not needed in vLLM).
    if "softmax_quantizer" in key or (key.startswith("lm_head.") and "quantizer" in key):
        return ("skip", None, None)

    # Check if this is a q/k/v projection that needs merging
    qkv_match = re.search(r"(.*\.)([qkv])_proj\.([^.]+_quantizer)(\..+)?$", key)
    if qkv_match:
        suffix = qkv_match.group(4) or ""
        group_key = qkv_match.group(1) + "qkv_proj." + qkv_match.group(3) + suffix
        return ("group", group_key, value)

    # Expert gate/up (per-expert) → w13 merge
    expert_gate_up_match = re.search(
        r"(.*\.experts)\.\d+\.(gate|up)_proj\.([^.]+_quantizer)(\..+)?$", key
    )
    if expert_gate_up_match:
        suffix = expert_gate_up_match.group(4) or ""
        group_key = expert_gate_up_match.group(1) + ".w13_" + expert_gate_up_match.group(3) + suffix
        return ("group", group_key, value)

    # Check if this is a non-expert gate/up projection that needs merging
    if "mixer" not in key and "experts" not in key:
        gate_up_match = re.search(r"(.*\.)(gate|up)_proj\.([^.]+_quantizer)(\..+)?$", key)
        if gate_up_match:
            suffix = gate_up_match.group(4) or ""
            group_key = gate_up_match.group(1) + "gate_up_proj." + gate_up_match.group(3) + suffix
            return ("group", group_key, value)

    expert_down_match = re.search(r"(.*\.experts)\.\d+\.down_proj\.([^.]+_quantizer)(\..+)?$", key)
    if expert_down_match:
        suffix = expert_down_match.group(3) or ""
        group_key = expert_down_match.group(1) + ".w2_" + expert_down_match.group(2) + suffix
        return ("group", group_key, value)

    # Transform bmm_quantizer keys: self_attn.q/k/v_bmm_quantizer -> self_attn.attn.q/k/v_bmm_quantizer
    bmm_match = re.search(r"(.*\.self_attn)\.([qkv]_bmm_quantizer.*)$", key) or re.search(
        r"(.*\.mixer)\.([qkv]_bmm_quantizer.*)$", key
    )
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
            if action != "skip":
                raise RuntimeError(
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
    if not key_value_pairs:
        raise ValueError(f"Cannot merge '{merged_key}': key_value_pairs is empty")
    values = [value for _, value in key_value_pairs]

    # Check if values are dicts (OrderedDict) containing tensors
    if isinstance(values[0], dict):
        merged_value = {}
        for dict_key in values[0]:
            tensors = [v[dict_key] for v in values]
            if "_amax" in dict_key:
                merged_value[dict_key] = merge_amax_tensors_for_group(tensors)
            elif "_pre_quant_scale" in dict_key:
                # _pre_quant_scale is per-input-channel: identical across q/k/v projections
                # since they share the same input. Do not concatenate; take the first value.
                merged_value[dict_key] = tensors[0]
            else:
                merged_value[dict_key] = torch.cat(tensors, dim=0)
        return merged_value
    else:
        # Values are tensors directly
        if "_amax" in merged_key:
            merged_value = merge_amax_tensors_for_group(values)
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

    # If all quantizers are disabled, their shape-specific fields (e.g. _amax_shape_for_export)
    # will differ across q/k/v projections even though the config is logically the same.
    # Since disabled quantizers are not used, skip the equality check.
    if all(isinstance(v, dict) and v.get("_disabled") for v in values):
        return first_value

    for i, val in enumerate(values[1:], start=1):
        if not _values_equal(val, first_value):
            raise ValueError(
                f"Cannot merge keys into '{merged_key}': values differ.\n"
                f"  '{keys[0]}' has value: {first_value}\n"
                f"  '{keys[i]}' has value: {val}"
            )
    return first_value


def convert_dict_to_vllm(
    state_dict: dict[str, Any],
    max_or_concat: bool = True,
    map_fun: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Common implementation for converting quantizer state from HF to vLLM format.

    Args:
        state_dict: Input state dict
        max_or_concat: Whether to merge grouped values by taking max/concatenate or require identical
        map_fun: Function to map the state dict to vLLM format
    """
    # If map_fun is provided, pre-transform quantizer key module-path prefixes so that
    # HF→vLLM model renames (e.g. backbone.layers → model.layers) are applied before
    # key grouping (q/k/v → qkv, experts.N.up_proj → experts.w13, etc.).
    # This is necessary for models where the HF root module differs from vLLM's (e.g.
    # NemotronH uses backbone.layers in HF but model.layers in vLLM), and for
    # modelopt_state_weights where ALL keys are quantizer keys so map_fun is never
    # invoked on non-quantizer keys.
    if map_fun is not None:
        q_only = {k: v for k, v in state_dict.items() if "_quantizer" in k}
        prefix_remap = infer_quantizer_prefix_remap(q_only, map_fun)
        if prefix_remap:
            renamed = {}
            for k, v in state_dict.items():
                if "_quantizer" in k:
                    first = k.split(".")[0]
                    k = prefix_remap.get(first, first) + k[len(first) :]
                renamed[k] = v
            state_dict = renamed

    vllm_state_dict, merge_groups = _group_keys_for_vllm(state_dict)

    merge_fn = _merge_values_by_max_or_concat if max_or_concat else _merge_values_require_identical

    # Merge grouped values
    for merged_key, key_value_pairs in merge_groups.items():
        if len(key_value_pairs) > 1:
            merged_value = merge_fn(merged_key, key_value_pairs)
            vllm_state_dict[merged_key] = merged_value
        else:
            # Single key, just rename it
            _, value = key_value_pairs[0]
            vllm_state_dict[merged_key] = value
    if map_fun is None:
        return vllm_state_dict
    # Quantizer module-path keys (e.g. "layers.0.mlp.gate_proj.input_quantizer") must NOT
    # go through map_fun (hf_to_vllm_mapper.apply_dict), which maps weight tensor paths and
    # drops any key it doesn't recognise — including all quantizer keys. Split them out,
    # apply map_fun only to non-quantizer keys, then merge back.
    quantizer_keys = {k: v for k, v in vllm_state_dict.items() if "_quantizer" in k}
    non_quantizer_keys = {k: v for k, v in vllm_state_dict.items() if "_quantizer" not in k}
    mapped = map_fun(non_quantizer_keys) if non_quantizer_keys else {}
    return {**mapped, **quantizer_keys}


def convert_modelopt_state_to_vllm(
    modelopt_state: dict[str, Any],
    map_fun: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Convert modelopt state from HuggingFace format to vLLM compatible format.

    This function converts the quantizer state from HuggingFace format to vLLM compatible format.

    Note: modifies modelopt_state in place (pops keys). Callers that need the
    original dict should pass a copy.

    Args:
        modelopt_state: HuggingFace modelopt state dict (modified in place)
        map_fun: Optional function to remap non-quantizer keys to vLLM names

    Returns:
        vLLM compatible modelopt state dict
    """
    modelopt_state_dict = modelopt_state.pop("modelopt_state_dict", [])
    for idx, current_mode in enumerate(modelopt_state_dict):
        current_mode_metadata = current_mode[1].pop("metadata", {})
        current_mode_quant_state = current_mode_metadata.pop("quantizer_state", {})
        if current_mode_quant_state:
            current_mode_metadata["quantizer_state"] = convert_dict_to_vllm(
                current_mode_quant_state, max_or_concat=False, map_fun=map_fun
            )
        else:
            current_mode_metadata.pop("quantizer_state", None)
        current_mode[1]["metadata"] = current_mode_metadata
        modelopt_state_dict[idx] = (current_mode[0], current_mode[1])
    modelopt_state["modelopt_state_dict"] = modelopt_state_dict
    return modelopt_state


def filter_modelopt_state_quantizer_state_for_model(
    modelopt_state: dict[str, Any], model: torch.nn.Module
) -> None:
    """
    Align quantizer_state in modelopt_state metadata with the model.

    - Removes keys not in the model (handles TP sharding - each rank has a subset).
    - Removes keys only when the quantizer is disabled (in the model).
    - Adds keys for quantizers in the model but not in metadata (e.g. disabled/excluded).
    Modifies modelopt_state in place. Call after convert_to_quantized_model so the model has
    quantizers.

    Args:
        modelopt_state: Modelopt state dict (modified in place)
        model: Model with quantizers (must already be converted)
    """
    from modelopt.torch.quantization.conversion import quantizer_state
    from modelopt.torch.quantization.nn import TensorQuantizer
    from modelopt.torch.utils import get_unwrapped_name

    model_qstate = quantizer_state(model)
    model_keys = set(model_qstate.keys())
    # Build name -> is_enabled for quantizers in the model
    disabled_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, (TensorQuantizer, SequentialQuantizer)):
            unwrapped_name = get_unwrapped_name(name, model)
            if not getattr(module, "is_enabled", True):
                disabled_keys.add(unwrapped_name)

    for mode_entry in modelopt_state.get("modelopt_state_dict", []):
        metadata = mode_entry[1].get("metadata", {})
        if "quantizer_state" in metadata:
            saved = metadata["quantizer_state"]

            # Keep keys that exist in the model. Remove disabled quantizers UNLESS they
            # have registered buffers (e.g. _pre_quant_scale from AWQ/smoothquant on a
            # disabled input_quantizer). Those buffers must reach _reset_pytorch_state_from_metadata
            # so they get registered before set_quantizer_state_dict loads the values.
            def _has_buffers(state: dict) -> bool:
                return bool(state.get("_pytorch_state_metadata", {}).get("buffers"))

            filtered = {
                k: v
                for k, v in saved.items()
                if k in model_keys and (k not in disabled_keys or _has_buffers(v))
            }
            # Add state for quantizers in model but not in metadata (e.g. disabled/excluded)
            for k in model_keys - filtered.keys():
                state = model_qstate[k]
                # Weight quantizers absent from exported metadata were disabled during export
                # (weights are already fake-quantized and pre_quant_scale is folded in).
                # Keep them disabled on reload so fold_weight does not re-quantize the
                # already-folded weights (re-quantizing distorts the pqs-scaled values).
                if is_weight_quantizer_state_key(k) and not state.get("_disabled"):
                    state = {**state, "_disabled": True}
                filtered[k] = state

            # Invariant: weight quantizers absent from export must be _disabled.
            for wq_k in model_keys:
                if not is_weight_quantizer_state_key(wq_k):
                    continue
                wq_state = filtered[wq_k]
                if wq_k not in saved and not wq_state.get("_disabled"):
                    raise RuntimeError(
                        f"Weight quantizer {wq_k!r} is missing from saved quantizer_state but "
                        f"is not marked _disabled (got _disabled={wq_state.get('_disabled')!r}). "
                        f"vLLM fakequant export omits weight quantizer keys when weights are folded."
                    )
            metadata["quantizer_state"] = filtered


def restore_from_modelopt_state_vllm(
    model: torch.nn.Module, modelopt_state: dict[str, Any]
) -> torch.nn.Module:
    """
    vLLM-specific restore that filters quantizer_state to match the model before restore.

    Handles TP sharding (each rank has a subset of quantizers) and excluded disabled quantizers
    by running convert first, filtering metadata to model keys, then restoring. Uses the same
    restore logic as restore_from_modelopt_state but with filtering for quantize modes.
    """
    model = model if isinstance(model, torch.nn.Module) else ModelLikeModule(model)
    manager = ModeloptStateManager(model=model, init_state=True)
    manager.load_state_dict(
        modelopt_state["modelopt_state_dict"], modelopt_state["modelopt_version"]
    )

    for i, (m, config, metadata) in enumerate(manager.modes_with_states()):
        if i == 0:
            model = _check_init_modellike(model, m)
        # For quantize modes: convert first (if not already), filter metadata to model keys, then restore state.
        # This handles TP (model has subset of quantizers) and excluded disabled quantizers.
        if "quantizer_state" in metadata:
            if not is_quantized(model):
                convert_to_quantized_model(model, config)
            filter_modelopt_state_quantizer_state_for_model(
                {"modelopt_state_dict": manager._state}, model
            )
            # Re-fetch metadata after filtering (manager._state was modified in place)
            metadata = manager._state[i][1]["metadata"]
            model = restore_quantizer_state(model, config, metadata)
        else:
            model = m.restore(model, config, metadata)

    if not manager.has_state and isinstance(model, ModelLikeModule):
        model = model.init_modellike()
    if isinstance(model, ModelLikeModule):
        raise RuntimeError("Model must be a regular Module after restore, got ModelLikeModule")
    return model


def _tp_concat_shard_dims(
    value_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    tp_world_size: int,
) -> list[int]:
    """Dims ``d`` where checkpoint looks like TP concat: ``value[d] == expected[d] * tp_world_size``."""
    return [
        d for d in range(len(expected_shape)) if value_shape[d] == expected_shape[d] * tp_world_size
    ]


def _narrow_tensor_to_tp_local_shard(
    value: torch.Tensor,
    expected_shape: tuple[int, ...] | torch.Size,
    tp_rank: int,
    tp_world_size: int,
    *,
    context: str,
) -> torch.Tensor:
    """Slice ``value`` to this TP rank when it is the concat of per-rank shards along one dim."""
    value_shape = value.shape
    expected_shape = tuple(expected_shape)
    if value_shape == expected_shape:
        return value
    if len(value_shape) != len(expected_shape):
        raise ValueError(
            f"{context}: rank mismatch (checkpoint={tuple(value_shape)}, expected={tuple(expected_shape)})"
        )
    shard_dims = _tp_concat_shard_dims(value_shape, expected_shape, tp_world_size)
    if len(shard_dims) != 1:
        raise ValueError(
            f"{context}: cannot infer TP shard dim "
            f"(expected={tuple(expected_shape)}, checkpoint={tuple(value_shape)}, tp={tp_world_size})"
        )
    d = shard_dims[0]
    shard_size = expected_shape[d]
    start = tp_rank * shard_size
    if start + shard_size > value_shape[d]:
        raise ValueError(
            f"{context}: TP shard out of bounds "
            f"(expected={tuple(expected_shape)}, checkpoint={tuple(value_shape)})"
        )
    return value.narrow(d, start, shard_size).contiguous()


def _pqs_local_expected_shape(pqs: torch.Tensor, expected_in: int) -> tuple[int, ...] | None:
    """Local per-rank shape for ``_pre_quant_scale`` (1-D ``[H]`` or broadcast 2-D ``[1, H]``)."""
    if pqs.ndim == 1:
        return (expected_in,)
    if pqs.ndim == 2 and pqs.shape[0] == 1:
        return (1, expected_in)
    return None


def _expected_in_features_for_input_quantizer(parent: Any, input_quantizer_attr: str) -> int | None:
    """Input feature count for the weight paired with ``*_input_quantizer`` (Linear or FusedMoE)."""
    stem = input_quantizer_attr[: -len("_input_quantizer")]
    w = getattr(parent, (stem + "_weight") if stem else "weight", None)
    if w is None or not isinstance(w, torch.Tensor) or w.is_meta:
        return None
    return int(w.shape[-1] if w.ndim == 3 else w.shape[1])


def shard_pre_quant_scale_for_tp(model: Any) -> None:
    """Shard ``_pre_quant_scale`` in-place for the local TP rank (row-parallel inputs).

    HF exports often store full (unsharded) scales; after load, row-parallel layers need
    ``pqs`` narrowed to ``H_in / tp`` when ``len(pqs) == H_in * tp_world_size``.

    Call after parallel linear modules expose TP-sharded weight shapes (e.g.
    ``post_restore_vllm_parallel_linears``). If run earlier, ``expected_in`` inferred from
    weights can match an unsharded checkpoint and a second call becomes a no-op even when
    pqs should still be narrowed.

    Args:
        model: vLLM model with ``TensorQuantizer`` submodules.
    """
    from modelopt.torch.quantization.nn import TensorQuantizer

    tp_group = get_tp_group()
    tp_rank, tp_world_size = tp_group.rank_in_group, tp_group.world_size
    if tp_world_size == 1:
        return

    for qname, quantizer in model.named_modules():
        if not isinstance(quantizer, TensorQuantizer):
            continue
        pqs = getattr(quantizer, "_pre_quant_scale", None)
        if pqs is None:
            continue
        last = qname.rfind(".")
        if last == -1 or not qname[last + 1 :].endswith("input_quantizer"):
            continue
        try:
            parent = model.get_submodule(qname[:last])
        except (AttributeError, LookupError):
            continue
        expected_in = _expected_in_features_for_input_quantizer(parent, qname[last + 1 :])
        if expected_in is None:
            continue
        expected_shape = _pqs_local_expected_shape(pqs, expected_in)
        if expected_shape is None:
            continue
        quantizer._pre_quant_scale = _narrow_tensor_to_tp_local_shard(
            pqs,
            expected_shape,
            tp_rank,
            tp_world_size,
            context=f"{qname}._pre_quant_scale",
        )


def process_state_dict_for_tp(saved_qstate_dict, current_state_dict):
    """Shard quantizer tensors for tensor parallelism by matching expected shapes."""
    tp_group = get_tp_group()
    tp_rank = tp_group.rank_in_group
    tp_world_size = tp_group.world_size

    result = {}
    for key, value in saved_qstate_dict.items():
        if key in current_state_dict:
            expected = current_state_dict[key]
            if hasattr(value, "shape") and hasattr(expected, "shape"):
                value = _narrow_tensor_to_tp_local_shard(
                    value,
                    expected.shape,
                    tp_rank,
                    tp_world_size,
                    context=f"Key {key!r}",
                )
        result[key] = value

    return result


def load_state_dict_from_path(
    fakequant_runner: Any, quantizer_file_path: str, model: Any
) -> dict[str, Any]:
    # Load on CPU to avoid failures when the checkpoint was saved from a different GPU mapping.
    saved_quant_dict = torch.load(quantizer_file_path, weights_only=True, map_location="cpu")
    if hasattr(fakequant_runner.model_runner.model, "hf_to_vllm_mapper"):
        saved_quant_dict = fakequant_runner.model_runner.model.hf_to_vllm_mapper.apply_dict(
            saved_quant_dict
        )
        saved_quant_dict = {
            key.replace("quantizer_", "quantizer._"): value
            for key, value in saved_quant_dict.items()
            if "quantizer" in key
        }
    saved_quant_dict = convert_dict_to_vllm(saved_quant_dict)

    current_state_dict = model.state_dict()
    checkpoint_quant_keys = [key for key in saved_quant_dict if "quantizer" in key]
    model_quant_keys = [key for key in current_state_dict if "quantizer" in key]
    ckpt_key_set = set(checkpoint_quant_keys)
    global_ckpt_key_set = _union_quantizer_keys_across_ranks(checkpoint_quant_keys)
    # For weight quantizers absent from the checkpoint the weights were already fake-quantized
    # at export time (amax folded into weights). Disable those quantizers so that fold_weight
    # is a no-op for them. Non-weight keys missing on this rank but present on another rank's
    # shard are omitted from global_missing (all_gather union of key strings).
    missing_wq_module_paths: set[str] = set()
    global_missing_non_wq: list[str] = []
    for key in model_quant_keys:
        if key in ckpt_key_set:
            continue
        if "weight_quantizer" in key:
            # Per-rank shard: only disable using this rank's checkpoint contents.
            parts = key.split(".")
            weight_quantizer_index = next(
                (i for i, p in enumerate(parts) if p.endswith("weight_quantizer")),
                None,
            )
            if weight_quantizer_index is not None:
                missing_wq_module_paths.add(".".join(parts[: weight_quantizer_index + 1]))
            else:
                raise ValueError(
                    f"Missing checkpoint key {key!r} looks like a weight quantizer, but no path "
                    "component ends with 'weight_quantizer'; cannot map to a module to disable."
                )
        elif key not in global_ckpt_key_set:
            global_missing_non_wq.append(key)

    if global_missing_non_wq:
        keys = sorted(global_missing_non_wq)
        n = len(keys)
        sample, rest = keys[:8], n - 8
        warnings.warn(
            f"{n} quantizer key(s) missing from every rank's checkpoint (after all_gather):"
            f"{sample}{' ... (+{rest} more)' if rest > 0 else ''}"
        )

    for name, module in model.named_modules():
        if (
            name in missing_wq_module_paths
            and isinstance(module, TensorQuantizer)
            and hasattr(module, "disable")
        ):
            module.disable()

    # Update quant values
    saved_quant_dict = process_state_dict_for_tp(saved_quant_dict, current_state_dict)
    for key, value in saved_quant_dict.items():
        if key in current_state_dict:
            current_state_dict[key] = value.to(current_state_dict[key].device)
    return current_state_dict
