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

import json
from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.pruning.pruning_utils import MlpInitMode
from modelopt.torch.puzzletron.tools.bypassed_training.child_init import (
    _concatenate_experts_into_dense_ffn,
    _generate_moe_keys,
    _process_single_layer,
)


def _moe_block(num_experts):
    return BlockConfig(
        attention=AttentionConfig(no_op=True),
        ffn=FFNConfig(
            moe=MoEConfig(
                num_local_experts=num_experts,
                expert_intermediate_dim=2,
                shared_expert_intermediate_dim=2,
            )
        ),
    )


def _moe_state(num_experts):
    router_key, expert_keys = _generate_moe_keys(0, num_experts)
    state = {router_key: torch.arange(num_experts * 2).reshape(num_experts, 2).float()}
    for keys in expert_keys.values():
        state.update({key: torch.full((2, 2), float(i + 1)) for i, key in enumerate(keys)})
    return state


@pytest.mark.parametrize(("higher_is_better", "selected"), [(True, [3, 1]), (False, [0, 2])])
def test_legacy_expert_removal_uses_selected_parent_weights(tmp_path, higher_is_better, selected):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps([0.1, 0.7, 0.2, 0.9]))
    parent = _moe_state(4)
    child = _moe_state(2)
    keys = dict.fromkeys(parent, "parent")
    result, removed = _process_single_layer(
        layer_idx=0,
        pruning_mixin=None,
        descriptor=None,
        parent_state_dict=parent,
        new_state_dict=child,
        original_config=SimpleNamespace(block_configs=[_moe_block(4)]),
        new_config=SimpleNamespace(block_configs=[_moe_block(2)]),
        gqa_init_mode=None,
        mlp_init_mode=MlpInitMode.ExpertRemoval,
        mlp_init_config={
            "expert_scores_file": str(scores_path),
            "higher_is_better": higher_is_better,
        },
        linear_init_mode=None,
        ignored_keys=set(),
        keys=keys,
        is_original_mha=False,
        head_size=1,
        hidden_size=2,
    )
    router_key, child_experts = _generate_moe_keys(0, 2)
    _, parent_experts = _generate_moe_keys(0, 4)
    assert torch.equal(result[router_key], parent[router_key][selected])
    for name, expert_keys in child_experts.items():
        for child_key, expert_idx in zip(expert_keys, selected):
            assert torch.equal(result[child_key], parent[parent_experts[name][expert_idx]])
    assert set(removed) == set(parent)
    assert {key: value.shape for key, value in result.items()} == {
        key: value.shape for key, value in child.items()
    }


def test_concatenate_experts_uses_scores_without_extra_layer_index(tmp_path):
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps([0.1, 0.7, 0.2, 0.9]))
    parent = _moe_state(4)
    for name in ("gate_proj", "up_proj", "down_proj"):
        parent[f"model.layers.0.mlp.shared_expert.{name}.weight"] = torch.zeros(2, 2)
    result = _concatenate_experts_into_dense_ffn(
        parent,
        {"expert_scores_file": str(scores_path)},
        hidden_size=2,
        layer_idx=0,
        child_block_config=BlockConfig(ffn=FFNConfig(intermediate_size=4)),
        parent_block_config=_moe_block(4),
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        expected = torch.cat(
            [torch.zeros(2, 2), torch.full((2, 2), 4.0)], dim=1 if name == "down_proj" else 0
        )
        assert torch.equal(result[f"model.layers.0.mlp.{name}.weight"], expected)
