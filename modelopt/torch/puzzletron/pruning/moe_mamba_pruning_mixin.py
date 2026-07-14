# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Tuple, Type

from modelopt.torch.prune.importance_hooks.base_hooks import ForwardHook

from .pruning_mixin import LayerDescriptor, PruningMixIn

__all__ = [
    "MoELayerDescriptor",
    "MoEPruningMixIn",
    "MambaLayerDescriptor",
    "MambaPruningMixIn",
]


@dataclass
class MoELayerDescriptor(LayerDescriptor):
    """Descriptor for MoE activation scoring targets.

    ``target_name`` can be a concrete suffix or a ``regex:`` pattern. The canonical
    key helpers always return checkpoint/AnyModel names under ``moe_prefix_name``,
    while the hook target can be a native AutoModel module path.
    """

    target_name: str = "mixer"
    moe_prefix_name: str = "backbone.layers.{layer_idx}.mixer"
    gate_name: str = "gate"
    experts_name: str = "experts"
    shared_experts_name: str = "shared_experts"
    latent_fc1_name: str = "fc1_latent_proj"
    latent_fc2_name: str = "fc2_latent_proj"
    require_attrs: tuple[str, ...] = ()
    expert_down_proj_regex: str = r"\.experts\.(\d+)\.down_proj$"

    def module_name_regex(self) -> str:
        return self.target_name

    def moe_prefix(self, layer_idx: int) -> str:
        return self.moe_prefix_name.format(layer_idx=layer_idx)

    def gate_prefix(self, layer_idx: int) -> str:
        return f"{self.moe_prefix(layer_idx)}.{self.gate_name}"

    def experts_prefix(self, layer_idx: int) -> str:
        return f"{self.moe_prefix(layer_idx)}.{self.experts_name}"

    def shared_experts_prefix(self, layer_idx: int) -> str:
        return f"{self.moe_prefix(layer_idx)}.{self.shared_experts_name}"

    def expert_idx_from_module_name(self, module_name: str) -> int | None:
        match = re.search(self.expert_down_proj_regex, module_name)
        return int(match.group(1)) if match else None

    def canonical_score_key(self, method: str, block_idx: int | None, fallback: str) -> str:
        if block_idx is None:
            return fallback
        expert_idx = self.expert_idx_from_module_name(fallback)
        if expert_idx is not None:
            return f"{self.experts_prefix(block_idx)}.{expert_idx}.down_proj"
        if method in ("ranked_choice_voting", "router_frequency"):
            return self.gate_prefix(block_idx)
        if method in ("moe_channel", "moe_cett", "expert_intermediate_contribution"):
            return self.experts_prefix(block_idx)
        if method in ("shared_expert_intermediate_contribution", "moe_shared_channel"):
            return f"{self.shared_experts_prefix(block_idx)}.down_proj"
        if method in ("removed_expert_diff", "moe_latent"):
            return self.moe_prefix(block_idx)
        return fallback

    def get_modules_names_to_hook(self, model) -> List[Tuple[int, str]]:
        candidates = super().get_modules_names_to_hook(model)
        if not self.require_attrs:
            return candidates
        filtered: list[tuple[int, str]] = []
        for block_idx, module_name in candidates:
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                continue
            if all(hasattr(module, attr) for attr in self.require_attrs):
                filtered.append((block_idx, module_name))
        return filtered


class MoEPruningMixIn(PruningMixIn):
    def __init__(self, layer_descriptor: MoELayerDescriptor):
        if not isinstance(layer_descriptor, MoELayerDescriptor):
            raise TypeError(f"Expected MoELayerDescriptor, got {type(layer_descriptor).__name__}")
        super().__init__(layer_descriptor)

    def supported_hooks(self) -> List[Type[ForwardHook]]:
        return []


@dataclass
class MambaLayerDescriptor(LayerDescriptor):
    target_name: str = "mixer.in_proj"
    mamba_prefix_name: str = "backbone.layers.{layer_idx}.mixer"
    in_proj_name: str = "in_proj"
    out_proj_name: str = "out_proj"

    def module_name_regex(self) -> str:
        return self.target_name

    def mamba_prefix(self, layer_idx: int) -> str:
        return self.mamba_prefix_name.format(layer_idx=layer_idx)

    def canonical_score_key(self, method: str, block_idx: int | None, fallback: str) -> str:
        if block_idx is None:
            return fallback
        if method in ("mamba_head_and_dim", "mamba_ssm_channel", "channel_contrib"):
            return f"{self.mamba_prefix(block_idx)}.{self.in_proj_name}"
        if method in ("independent_head", "iterative_head"):
            return f"{self.mamba_prefix(block_idx)}.{self.out_proj_name}"
        return fallback


class MambaPruningMixIn(PruningMixIn):
    def __init__(self, layer_descriptor: MambaLayerDescriptor):
        if not isinstance(layer_descriptor, MambaLayerDescriptor):
            raise TypeError(f"Expected MambaLayerDescriptor, got {type(layer_descriptor).__name__}")
        super().__init__(layer_descriptor)

    def supported_hooks(self) -> List[Type[ForwardHook]]:
        return []
