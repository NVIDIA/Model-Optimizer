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

import torch

from modelopt.torch.puzzletron.pruning.kv_heads_pruning_mixin import (
    KVHeadsLayerDescriptor,
    KVHeadsPruningMixIn,
)
from modelopt.torch.puzzletron.pruning.pruning_utils import (
    GQAInitMode,
    LinearInitMode,
    MlpInitMode,
)
from modelopt.torch.puzzletron.tools.bypassed_training.child_init import _process_single_layer

ATTN_PREFIX = "model.layers.0.self_attn"
QKVO_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class DecoderConfigDescriptor:
    @staticmethod
    def get_language_model_config(config):
        return config.decoder_config


def _make_config():
    return SimpleNamespace(
        decoder_config=SimpleNamespace(head_dim=2, hidden_size=4, num_attention_heads=2),
        block_configs=[
            SimpleNamespace(
                attention=SimpleNamespace(num_key_value_heads=2),
                ffn=SimpleNamespace(is_moe=False),
            )
        ],
        attention_bias=True,
        o_proj_bias=True,
    )


def _make_attention_state_dict(fill_value: float):
    state_dict = {}
    for proj_idx, name in enumerate(QKVO_NAMES):
        weight_key = f"{ATTN_PREFIX}.{name}.weight"
        bias_key = f"{ATTN_PREFIX}.{name}.bias"
        state_dict[weight_key] = torch.full((4, 4), fill_value + proj_idx)
        state_dict[bias_key] = torch.full((4,), fill_value + proj_idx)
    return state_dict


def _assert_attention_state_dict_matches(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_kv_heads_pruning_mixin_uses_descriptor_selected_config_for_attention_init():
    original_config = _make_config()
    new_config = _make_config()
    original_state_dict = _make_attention_state_dict(fill_value=1.0)
    new_state_dict = _make_attention_state_dict(fill_value=10.0)
    keys_to_remove = {}

    mixin = KVHeadsPruningMixIn(
        KVHeadsLayerDescriptor(
            o_proj_name="o_proj",
            attn_prefix_name="model.layers.{layer_idx}.self_attn",
            qkvo_weight_names=QKVO_NAMES,
        )
    )

    layer_state_dict = mixin.prune_single_layer(
        layer_idx=0,
        parent_state_dict=original_state_dict,
        new_state_dict=new_state_dict,
        original_config=original_config,
        new_config=new_config,
        descriptor=DecoderConfigDescriptor,
        gqa_init_mode=GQAInitMode.CopyAsIs,
        mlp_init_config=None,
        is_original_mha=True,
        keys={key: key for key in original_state_dict},
        keys_to_remove=keys_to_remove,
    )

    _assert_attention_state_dict_matches(layer_state_dict, original_state_dict)
    assert keys_to_remove == {key: key for key in original_state_dict}


def test_legacy_process_single_layer_uses_descriptor_selected_config_for_attention_init():
    original_config = _make_config()
    new_config = _make_config()
    original_state_dict = _make_attention_state_dict(fill_value=1.0)
    new_state_dict = _make_attention_state_dict(fill_value=10.0)

    layer_state_dict, keys_to_remove = _process_single_layer(
        layer_idx=0,
        pruning_mixin=None,
        descriptor=DecoderConfigDescriptor,
        parent_state_dict=original_state_dict,
        new_state_dict=new_state_dict,
        original_config=original_config,
        new_config=new_config,
        gqa_init_mode=GQAInitMode.CopyAsIs,
        mlp_init_mode=MlpInitMode.CopyAsIs,
        mlp_init_config=None,
        linear_init_mode=LinearInitMode.Random,
        ignored_keys=set(),
        keys={key: key for key in original_state_dict},
        is_original_mha=True,
        head_size=2,
        hidden_size=4,
    )

    _assert_attention_state_dict_matches(layer_state_dict, original_state_dict)
    assert keys_to_remove == {key: key for key in original_state_dict}
