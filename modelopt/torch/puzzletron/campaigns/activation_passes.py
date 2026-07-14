"""Compile descriptor capabilities into executable AutoModel activation passes."""

from __future__ import annotations

import dataclasses
import re
from collections import OrderedDict
from typing import Any

from ..pruning.ffn_intermediate_pruning_mixin import (
    FFNIntermediateLayerDescriptor,
    FFNIntermediatePruningMixIn,
)
from ..pruning.gated_delta_net_pruning_mixin import GatedDeltaNetPruningMixIn
from ..pruning.kv_heads_pruning_mixin import KVHeadsLayerDescriptor, KVHeadsPruningMixIn
from ..pruning.moe_mamba_pruning_mixin import (
    MambaLayerDescriptor,
    MambaPruningMixIn,
    MoELayerDescriptor,
    MoEPruningMixIn,
)

__all__ = ["compile_activation_passes"]


_GROUP_ORDER = (
    "hidden_width",
    "ple_width",
    "ffn_intermediate",
    "attention_grouped",
    "mla_heads",
    "gdn_activation",
    "moe_experts",
    "moe_expert_intermediate",
    "moe_shared_expert_intermediate",
    "moe_latent_dim",
    "mamba_head_and_dim",
    "magnitude_fallback",
)


def _qualname(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _serialize_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _serialize_dataclass(value)
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _serialize_dataclass(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"_target_": _qualname(value)}
    for field in dataclasses.fields(value):
        if field.init:
            result[field.name] = _serialize_value(getattr(value, field.name))
    return result


def _serialize_mixin(mixin: Any) -> dict[str, Any]:
    descriptor = getattr(mixin, "layer_descriptor", None)
    if descriptor is None or not dataclasses.is_dataclass(descriptor):
        raise TypeError(
            f"Activation target mixin {_qualname(mixin)} must own a dataclass layer descriptor"
        )
    return {
        "_target_": _qualname(mixin),
        "layer_descriptor": _serialize_dataclass(descriptor),
    }


def _generic_mixins(descriptor: Any, config: Any) -> dict[str, Any]:
    contract_factory = getattr(descriptor, "generic_decoder_contract", None)
    contract = contract_factory(config) if callable(contract_factory) else None
    if contract is None:
        return {}

    layout = contract.layout
    layer = layout.layer_template
    mixins: dict[str, Any] = {}
    if contract.dense_ffn is not None:
        ffn = contract.dense_ffn
        prefix = f"{layer}.{ffn.module_name}" if ffn.module_name else layer
        target = (
            f"{ffn.module_name}.{ffn.down_proj_name}"
            if ffn.module_name
            else ffn.down_proj_name
        )
        mixins["ffn_intermediate"] = FFNIntermediatePruningMixIn(
            FFNIntermediateLayerDescriptor(
                down_proj_name=target,
                ffn_prefix_name=prefix,
                linear_weight_names=[
                    ffn.down_proj_name,
                    ffn.gate_proj_name,
                    ffn.up_proj_name,
                ],
            )
        )
    if contract.attention is not None:
        attention = contract.attention
        prefix = f"{layer}.{attention.module_name}"
        mixins["kv_heads"] = KVHeadsPruningMixIn(
            KVHeadsLayerDescriptor(
                o_proj_name=f"{attention.module_name}.{attention.o_proj_name}",
                attn_prefix_name=prefix,
                qkvo_weight_names=[
                    attention.q_proj_name,
                    attention.k_proj_name,
                    attention.v_proj_name,
                    attention.o_proj_name,
                ],
            )
        )
    if contract.ple is not None:
        ple = contract.ple
        mixins["ple_width"] = FFNIntermediatePruningMixIn(
            FFNIntermediateLayerDescriptor(
                down_proj_name=ple.layer_projection_name,
                ffn_prefix_name=layer,
                linear_weight_names=[
                    ple.layer_gate_name,
                    ple.layer_projection_name,
                ],
            )
        )
    if contract.latent_attention is not None:
        attention = contract.latent_attention
        prefix = f"{layer}.{attention.module_name}"
        mixins["mla_heads"] = KVHeadsPruningMixIn(
            KVHeadsLayerDescriptor(
                o_proj_name=f"{attention.module_name}.o_proj",
                attn_prefix_name=prefix,
                qkvo_weight_names=["q_b_proj", "kv_b_proj", "o_proj"],
            )
        )
    if contract.routed_moe is not None:
        moe = contract.routed_moe
        prefix = f"{layer}.{moe.module_name}" if moe.module_name else layer
        target_module = moe.module_name or "moe"
        experts_target = f"{target_module}.experts"
        shared_name = moe.shared_expert_name or "shared_expert"
        shared_target = (
            "regex:(?:^|\\.)layers\\.\\d+\\.(?:mlp|moe)\\."
            f"(?:{re.escape(shared_name)}|shared_experts)\\.{re.escape(moe.down_proj_name)}$"
        )
        common = {
            "moe_prefix_name": prefix,
            "gate_name": moe.router_name,
            "experts_name": moe.experts_name,
            "shared_experts_name": shared_name,
        }
        mixins["moe_experts"] = MoEPruningMixIn(
            MoELayerDescriptor(
                target_name=(
                    "regex:(?:^|\\.)layers\\.\\d+\\.(?:mlp|moe)$"
                ),
                require_attrs=("gate", "experts"),
                **common,
            )
        )
        mixins["moe_expert_intermediate"] = MoEPruningMixIn(
            MoELayerDescriptor(target_name=experts_target, **common)
        )
        mixins["moe_shared_expert_intermediate"] = MoEPruningMixIn(
            MoELayerDescriptor(target_name=shared_target, **common)
        )
        mixins["moe_latent_dim"] = mixins["moe_experts"]
    return mixins


def _available_mixins(descriptor: Any, config: Any) -> dict[str, Any]:
    generic = _generic_mixins(descriptor, config)
    specialized = dict(descriptor.pruning_mixins())
    generic.update(specialized)
    return generic


def _group(axis_id: str, method: str) -> str:
    if method == "minitron_hidden_width":
        return "hidden_width"
    if axis_id == "ple_width":
        return "ple_width"
    if axis_id == "ffn_intermediate":
        return "ffn_intermediate"
    if axis_id in {"kv_groups", "q_heads_per_group"}:
        return "attention_grouped"
    if axis_id == "mla_heads":
        return "mla_heads"
    if axis_id.startswith("gdn_"):
        return "gdn_activation"
    if axis_id in {"mamba_heads", "mamba_head_dim"}:
        return "mamba_head_and_dim"
    if method == "magnitude_fallback":
        return "magnitude_fallback"
    return axis_id


def _mixin_key(group: str) -> str | None:
    return {
        "ffn_intermediate": "ffn_intermediate",
        "ple_width": "ple_width",
        "attention_grouped": "kv_heads",
        "mla_heads": "mla_heads",
        "gdn_activation": "gated_delta_net",
        "moe_experts": "moe_experts",
        "moe_expert_intermediate": "moe_expert_intermediate",
        "moe_shared_expert_intermediate": "moe_shared_expert_intermediate",
        "moe_latent_dim": "moe_latent_dim",
        "mamba_head_and_dim": "mamba_heads",
    }.get(group)


def _runtime_method(group: str, declared: str) -> str:
    return {
        "attention_grouped": "grouped_attention_contribution",
        "ple_width": "ple_channel_contribution",
        "mla_heads": "mla_head_contribution",
        "gdn_activation": "gdn_activation_contribution",
        "moe_experts": "removed_expert_diff",
        "moe_expert_intermediate": "expert_intermediate_contribution",
        "moe_shared_expert_intermediate": "shared_expert_intermediate_contribution",
        "moe_latent_dim": "moe_latent",
        "mamba_head_and_dim": "mamba_head_and_dim",
    }.get(group, declared)


def compile_activation_passes(
    descriptor: Any,
    config: Any,
    activation_axes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn axis declarations into deduplicated, Hydra-serializable scorer passes."""

    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for axis in activation_axes:
        group = _group(str(axis["axis_id"]), str(axis["method"]))
        grouped.setdefault(group, []).append(axis)

    mixins = _available_mixins(descriptor, config)
    compiled: dict[str, dict[str, Any]] = {}
    for group, axes in grouped.items():
        axis_ids = [str(axis["axis_id"]) for axis in axes]
        declared = str(axes[0]["method"])
        method = _runtime_method(group, declared)
        hook_kwargs: dict[str, Any] = {"method": method}
        if group == "attention_grouped":
            hook_kwargs.update(optimize_for="memory", scored_axes=axis_ids)
        elif group == "mla_heads":
            hook_kwargs["optimize_for"] = "memory"
        elif group == "gdn_activation":
            hook_kwargs["token_chunk_size"] = 64
        elif group == "magnitude_fallback":
            hook_kwargs["targets"] = [
                axis["magnitude_fallback"] for axis in axes
            ]

        item: dict[str, Any] = {
            "name": group,
            "axis_ids": axis_ids,
            "activation_hooks_kwargs": hook_kwargs,
        }
        key = _mixin_key(group)
        if key is not None:
            aliases = {
                "moe_experts": ("moe_expert_removal", "experts_removal"),
                "mamba_heads": ("mamba_head_dim",),
            }
            mixin = mixins.get(key)
            if mixin is None:
                mixin = next(
                    (mixins[name] for name in aliases.get(key, ()) if name in mixins),
                    None,
                )
            if mixin is None:
                raise ValueError(
                    f"{descriptor.__name__} axis group {group!r} has no executable target mixin"
                )
            item["pruning_mixin"] = _serialize_mixin(mixin)
        compiled[group] = item

    return [compiled[name] for name in _GROUP_ORDER if name in compiled]
