# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from modelopt.torch.prune.importance_hooks.base_hooks import ForwardHook

from .pruning_mixin import LayerDescriptor, PruningMixIn


@dataclass
class GatedDeltaNetLayerDescriptor(LayerDescriptor):
    target_name: str = "linear_attn"
    gdn_prefix_name: str = "model.language_model.layers.{layer_idx}.linear_attn"

    def module_name_regex(self) -> str:
        return self.target_name

    def gdn_prefix(self, layer_idx: int) -> str:
        return self.gdn_prefix_name.format(layer_idx=layer_idx)

    def canonical_score_key(self, method: str, block_idx: int | None, fallback: str) -> str:
        if block_idx is None:
            return fallback
        return self.gdn_prefix(block_idx)


class GatedDeltaNetPruningMixIn(PruningMixIn):
    def __init__(self, layer_descriptor: GatedDeltaNetLayerDescriptor):
        if not isinstance(layer_descriptor, GatedDeltaNetLayerDescriptor):
            raise TypeError(
                f"Expected GatedDeltaNetLayerDescriptor, got {type(layer_descriptor).__name__}"
            )
        super().__init__(layer_descriptor)

    def supported_hooks(self) -> list[type[ForwardHook]]:
        return []
