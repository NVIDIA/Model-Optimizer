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

from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.pruning.pruning_mixin import LayerDescriptor, PruningMixIn
from modelopt.torch.puzzletron.pruning.pruning_utils import resolve_pruning_mixin
from modelopt.torch.puzzletron.tools.bypassed_training.child_init import (
    _process_single_layer,
    update_model_config,
)


class _AddOneMixin:
    def prune_single_layer(self, parent_state_dict, keys_to_remove, **kwargs):
        keys_to_remove["w"] = "w"
        return {"w": parent_state_dict["w"] + 1}


class _TimesTwoMixin:
    def prune_single_layer(self, parent_state_dict, keys_to_remove, **kwargs):
        keys_to_remove["w"] = "w"
        return {"w": parent_state_dict["w"] * 2}


class _PopKeyMixin:
    def prune_single_layer(self, parent_state_dict, keys, **kwargs):
        keys.pop("w")
        return {"w": parent_state_dict["w"]}


class _ShrinkSourceMixin:
    def prune_single_layer(self, parent_state_dict, **kwargs):
        return {"w": parent_state_dict["w"][:2]}


class _UseDestinationShapeMixin:
    def prune_single_layer(self, parent_state_dict, new_state_dict, **kwargs):
        return {"w": torch.zeros_like(new_state_dict["w"]) + parent_state_dict["w"].sum()}


class _ConcretePruningMixIn(PruningMixIn):
    def supported_hooks(self):
        return []


_MAPPED_MIXIN = _ConcretePruningMixIn(LayerDescriptor())


class _DescriptorWithPruningMixins:
    @staticmethod
    def pruning_mixins():
        return {"mapped": _MAPPED_MIXIN}


def _process_with_mixins(
    mixins,
    keys,
    parent_state_dict=None,
    new_state_dict=None,
):
    return _process_single_layer(
        layer_idx=0,
        pruning_mixin=mixins,
        descriptor=None,
        parent_state_dict=parent_state_dict or {"w": torch.tensor([1.0])},
        new_state_dict=new_state_dict or {"w": torch.tensor([0.0])},
        original_config=SimpleNamespace(),
        new_config=SimpleNamespace(),
        gqa_init_mode=None,
        mlp_init_mode=None,
        mlp_init_config=None,
        linear_init_mode=None,
        ignored_keys=set(),
        keys=keys,
        is_original_mha=False,
        head_size=1,
        hidden_size=1,
    )


def test_pruning_mixins_compose_overlapping_outputs_sequentially():
    layer_state_dict, keys_to_remove = _process_with_mixins(
        [_AddOneMixin(), _TimesTwoMixin()], {"w": "w"}
    )

    assert torch.equal(layer_state_dict["w"], torch.tensor([4.0]))
    assert keys_to_remove == {"w": "w"}


def test_pruning_mixins_keep_final_destination_shape_for_later_mixins():
    layer_state_dict, _ = _process_with_mixins(
        [_ShrinkSourceMixin(), _UseDestinationShapeMixin()],
        {"w": "w"},
        parent_state_dict={"w": torch.ones(4)},
        new_state_dict={"w": torch.zeros(3)},
    )

    assert layer_state_dict["w"].shape == torch.Size([3])
    assert torch.equal(layer_state_dict["w"], torch.full((3,), 2.0))


def test_pruning_mixin_key_mutation_is_tracked_without_mutating_shared_keys():
    shared_keys = {"w": "w"}

    _, keys_to_remove = _process_with_mixins([_PopKeyMixin()], shared_keys)

    assert keys_to_remove == {"w": "w"}
    assert shared_keys == {"w": "w"}


def test_resolve_pruning_mixin_accepts_names_instances_and_lists():
    existing = _ConcretePruningMixIn(LayerDescriptor())

    assert resolve_pruning_mixin("mapped", _DescriptorWithPruningMixins) is _MAPPED_MIXIN
    assert resolve_pruning_mixin(existing, _DescriptorWithPruningMixins) is existing
    assert resolve_pruning_mixin(["mapped", existing], _DescriptorWithPruningMixins) == [
        _MAPPED_MIXIN,
        existing,
    ]


def test_resolve_pruning_mixin_reports_available_methods():
    with pytest.raises(ValueError, match="Available methods: \\['mapped'\\]"):
        resolve_pruning_mixin("missing", _DescriptorWithPruningMixins)


def test_update_model_config_treats_null_overrides_as_leave_unchanged():
    config = SimpleNamespace(
        num_hidden_layers=1,
        block_configs=[
            BlockConfig(
                attention=AttentionConfig(num_key_value_heads=8),
                ffn=FFNConfig(intermediate_size=32),
            )
        ],
    )

    updated = update_model_config(
        config,
        [
            {
                "attention": {"num_key_value_heads": 4},
                "ffn": None,
            }
        ],
    )

    assert updated is not config
    assert updated.block_configs[0].attention.num_key_value_heads == 4
    assert updated.block_configs[0].ffn == config.block_configs[0].ffn
    assert config.block_configs[0].attention.num_key_value_heads == 8
