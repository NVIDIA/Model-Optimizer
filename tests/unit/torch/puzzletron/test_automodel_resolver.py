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

"""Tests for the AutoModel target resolver (scorer dispatch + key construction).

Uses fake model/mixin objects to validate the method->scorer dispatch, the
per-method kwarg extraction, dummy/missing-module skipping, and canonical keying.
The resolver's correctness on a real model tree is covered by the in-container
integration check (plan P3).
"""

from types import SimpleNamespace

import pytest
from torch import nn

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, MLAConfig
from modelopt.torch.puzzletron.plugins.automodel.hooks import (
    FFNIndependentScorer,
    FFNIterativeScorer,
    GroupedAttentionScorer,
)
from modelopt.torch.puzzletron.plugins.automodel.reduction import MeshGroups
from modelopt.torch.puzzletron.plugins.automodel.target_resolver import build_scorers


class _FakeMixin:
    def __init__(self, names):
        self._names = names

    def get_module_names_to_hook(self, model):
        return self._names


class _FakeModel:
    def __init__(self, config, submodules):
        self.config = config
        self._submodules = submodules

    def get_submodule(self, name):
        if name not in self._submodules:
            raise AttributeError(name)
        return self._submodules[name]


def test_build_scorers_ffn_independent():
    down_proj = nn.Linear(8, 4, bias=False)
    model = _FakeModel(
        SimpleNamespace(block_configs=[None, None]),
        {"model.layers.0.mlp.down_proj": down_proj},
    )
    mixin = _FakeMixin([(0, "model.layers.0.mlp.down_proj")])

    scorers = build_scorers(model, MeshGroups(), mixin, method="independent", register=False)

    assert len(scorers) == 1
    assert isinstance(scorers[0], FFNIndependentScorer)
    assert scorers[0].name == "model.layers.0.mlp.down_proj"
    assert scorers[0].block_idx == 0


def test_build_scorers_grouped_attention_uses_explicit_block_configs():
    # Heterogeneous teacher: per-layer KV heads come from the explicit block_configs
    # (read from the checkpoint), not model.config.
    o_proj = nn.Linear(16, 6, bias=False)
    config = SimpleNamespace(num_attention_heads=4, hidden_size=6, head_dim=4)
    model = _FakeModel(config, {"model.layers.0.self_attn.o_proj": o_proj})
    mixin = _FakeMixin([(0, "model.layers.0.self_attn.o_proj")])
    block_configs = [BlockConfig(subblock_configs=(AttentionConfig(num_kv_heads=2),))]

    scorers = build_scorers(
        model,
        MeshGroups(),
        mixin,
        method="grouped_attention_contribution",
        block_configs=block_configs,
        validation_full_iters=4,
        register=False,
    )

    assert len(scorers) == 1
    scorer = scorers[0]
    assert isinstance(scorer, GroupedAttentionScorer)
    assert (scorer.num_q_heads, scorer.num_kv_heads, scorer.head_dim) == (4, 2, 4)
    assert scorer.name == "model.layers.0.self_attn.o_proj"


def test_build_scorers_grouped_attention_uses_per_layer_head_dimension():
    o_proj = nn.Linear(4 * 8, 6, bias=False)
    config = SimpleNamespace(num_attention_heads=4, hidden_size=6, head_dim=4)
    model = _FakeModel(config, {"model.layers.0.self_attn.o_proj": o_proj})
    mixin = _FakeMixin([(0, "model.layers.0.self_attn.o_proj")])
    block_configs = [
        BlockConfig(
            subblock_configs=(
                AttentionConfig(num_query_heads=4, num_kv_heads=2, qk_head_dim=8),
            )
        )
    ]

    scorer = build_scorers(
        model,
        MeshGroups(),
        mixin,
        method="grouped_attention_contribution",
        block_configs=block_configs,
        validation_full_iters=4,
        register=False,
    )[0]

    assert scorer.head_dim == 8


def test_build_scorers_grouped_attention_homogeneous_fallback():
    # No block_configs (homogeneous teacher): KV heads fall back to the model config.
    o_proj = nn.Linear(16, 6, bias=False)
    config = SimpleNamespace(
        num_attention_heads=4, num_key_value_heads=2, hidden_size=6, head_dim=4
    )
    model = _FakeModel(config, {"model.layers.0.self_attn.o_proj": o_proj})
    mixin = _FakeMixin([(0, "model.layers.0.self_attn.o_proj")])

    scorers = build_scorers(
        model,
        MeshGroups(),
        mixin,
        method="grouped_attention_contribution",
        validation_full_iters=4,
        register=False,
    )
    assert scorers[0].num_kv_heads == 2


def test_build_scorers_mla_heads_uses_coupled_decoded_head_geometry():
    o_proj = nn.Linear(4 * 6, 8, bias=False)
    config = SimpleNamespace(num_attention_heads=4, hidden_size=8, v_head_dim=6)
    model = _FakeModel(config, {"model.layers.0.self_attn.o_proj": o_proj})
    mixin = _FakeMixin([(0, "model.layers.0.self_attn.o_proj")])
    block_configs = [
        BlockConfig(
            subblock_configs=(
                MLAConfig(num_heads=4, q_lora_rank=8, kv_lora_rank=6),
            )
        )
    ]

    scorers = build_scorers(
        model,
        MeshGroups(),
        mixin,
        method="mla_head_contribution",
        block_configs=block_configs,
        validation_full_iters=4,
        register=False,
    )

    assert len(scorers) == 1
    scorer = scorers[0]
    assert isinstance(scorer, GroupedAttentionScorer)
    assert (scorer.num_q_heads, scorer.num_kv_heads, scorer.head_dim) == (4, 4, 6)
    assert scorer.scored_axes == {"kv_groups"}


class _FakeDescriptor:
    """Descriptor whose canonical names differ from the loaded module path (VL vs text)."""

    def __init__(self):
        self.ffn_prefix_name = "model.language_model.layers.{layer_idx}.mlp"
        self.attn_prefix_name = "model.language_model.layers.{layer_idx}.self_attn"

    def ffn_prefix(self, layer_idx):
        return self.ffn_prefix_name.format(layer_idx=layer_idx)

    def attn_prefix(self, layer_idx):
        return self.attn_prefix_name.format(layer_idx=layer_idx)


class _FakeMixinWithDescriptor(_FakeMixin):
    def __init__(self, names, descriptor):
        super().__init__(names)
        self.layer_descriptor = descriptor


def test_build_scorers_keys_ffn_by_descriptor_not_module_path():
    # The loaded (text-only) model exposes model.layers.7.mlp.down_proj, but the descriptor /
    # pruning step key by model.language_model.layers.7.mlp.down_proj. The scorer must hook the
    # real module yet name its output by the descriptor.
    down_proj = nn.Linear(8, 4, bias=False)
    model = _FakeModel(
        SimpleNamespace(block_configs=[None] * 8),
        {"model.layers.7.mlp.down_proj": down_proj},
    )
    mixin = _FakeMixinWithDescriptor([(7, "model.layers.7.mlp.down_proj")], _FakeDescriptor())

    scorers = build_scorers(model, MeshGroups(), mixin, method="independent", register=False)

    assert len(scorers) == 1
    assert scorers[0].name == "model.language_model.layers.7.mlp.down_proj"
    assert scorers[0].module is down_proj  # still hooks the actual loaded module


def test_build_scorers_keys_grouped_attention_by_descriptor():
    o_proj = nn.Linear(16, 6, bias=False)
    config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, hidden_size=6, head_dim=4)
    model = _FakeModel(config, {"model.layers.3.self_attn.o_proj": o_proj})
    mixin = _FakeMixinWithDescriptor([(3, "model.layers.3.self_attn.o_proj")], _FakeDescriptor())

    scorers = build_scorers(
        model,
        MeshGroups(),
        mixin,
        method="grouped_attention_contribution",
        validation_full_iters=4,
        register=False,
    )
    assert scorers[0].name == "model.language_model.layers.3.self_attn.o_proj"


def test_build_scorers_iterative():
    down_proj = nn.Linear(8, 4, bias=False)
    model = _FakeModel(
        SimpleNamespace(block_configs=[None]), {"model.layers.0.mlp.down_proj": down_proj}
    )
    mixin = _FakeMixin([(0, "model.layers.0.mlp.down_proj")])

    scorers = build_scorers(
        model, MeshGroups(), mixin, method="iterative", validation_full_iters=4, register=False
    )
    assert len(scorers) == 1
    assert isinstance(scorers[0], FFNIterativeScorer)
    assert scorers[0].pruning_iters == 4


def test_build_scorers_iterative_requires_iters():
    down_proj = nn.Linear(8, 4, bias=False)
    model = _FakeModel(SimpleNamespace(block_configs=[None]), {"m.down_proj": down_proj})
    mixin = _FakeMixin([(0, "m.down_proj")])
    with pytest.raises(ValueError, match="validation_full_iters"):
        build_scorers(model, MeshGroups(), mixin, method="iterative", register=False)


def test_build_scorers_skips_missing_module():
    model = _FakeModel(SimpleNamespace(block_configs=[None]), {})  # get_submodule raises
    mixin = _FakeMixin([(0, "model.layers.0.mlp.down_proj")])
    assert build_scorers(model, MeshGroups(), mixin, method="independent", register=False) == []


def test_build_scorers_rejects_unimplemented_method():
    down_proj = nn.Linear(8, 4, bias=False)
    model = _FakeModel(SimpleNamespace(block_configs=[None]), {"m.down_proj": down_proj})
    mixin = _FakeMixin([(0, "m.down_proj")])
    with pytest.raises(NotImplementedError):
        build_scorers(model, MeshGroups(), mixin, method="ranked_choice_voting", register=False)
