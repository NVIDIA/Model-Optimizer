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

"""CPU contracts for the Qwen 3.5 0.8B model example."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from modelopt.torch.puzzletron.anymodel.models.qwen3_5.qwen3_5_model_descriptor import (
    Qwen3P5VLModelDescriptor,
)
from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
)
from modelopt.torch.puzzletron.candidates import build_candidate_library

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/model.yaml"
)
CAMPAIGN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/vlm_campaign.yaml"
)


def test_qwen3p5_0p8b_model_and_default_search_are_pinned() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())

    assert model["input_hf_model_path"] == "Qwen/Qwen3.5-0.8B"
    assert model["model_info"]["hf_revision"] == ("2fc06364715b967f1860aea9cf38778875588b17")
    assert model["model_info"]["model_type"] == "qwen3_5"
    assert model["model_info"]["layer_counts"] == {
        "linear_attention": 18,
        "full_attention": 6,
    }
    assert model["pruning"] == {"intermediate_size_list": [3072, 2048]}
    assert model["search_space"]["axes"] == {
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 3584,
            "values": [3072, 2048],
        }
    }


def test_qwen3p5_0p8b_vlm_campaign_pins_width_and_depth_bounds() -> None:
    campaign = yaml.safe_load(CAMPAIGN_PATH.read_text())

    assert campaign["embedding_pruning"]["widths"] == [1024, 960, 896]
    spec = Qwen3P5VLModelDescriptor.embedding_pruning_spec(
        SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=1024, tie_word_embeddings=True),
        ),
        widths=campaign["embedding_pruning"]["widths"],
        alignment=campaign["embedding_pruning"]["alignment"],
    )
    assert [spec.validate_width(width) for width in spec.legal_widths] == [1024, 960, 896]
    assert campaign["depth_importance"]["enabled"] is True
    assert campaign["depth_importance"]["max_subblocks_to_remove"] == 2
    assert campaign["mip"]["runs"]["params-90"]["search_space"] == {
        "depth": [0, 1, 2],
        "embedding": [1024, 960, 896],
        "axes_default": "all",
        "axes": {"ffn.intermediate_size": "all"},
    }


def test_qwen3p5_0p8b_vlm_campaign_generates_axis_values_for_each_block_type() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())
    campaign = yaml.safe_load(CAMPAIGN_PATH.read_text())
    linear_block = BlockConfig(
        subblock_configs=(
            MambaConfig(
                name="linear_attn",
                num_heads=16,
                num_groups=16,
                state_dim=128,
                head_dim=128,
            ),
            FFNConfig(name="mlp", intermediate_size=3584),
        )
    )
    full_attention_block = BlockConfig(
        subblock_configs=(
            AttentionConfig(name="self_attn", num_query_heads=8, num_kv_heads=2),
            FFNConfig(name="mlp", intermediate_size=3584),
        )
    )
    full_attention_layers = set(model["model_info"]["full_attention_layer_indices"])
    block_configs = tuple(
        full_attention_block if layer_idx in full_attention_layers else linear_block
        for layer_idx in range(model["model_info"]["num_hidden_layers"])
    )
    expected_ffn_sizes = {3584, 3328, 3072}
    expected_attention_shapes = {(8, 2), (6, 2), (4, 1), (3, 1)}
    expected_gdn_shapes = {
        (groups, groups, key_dim, value_dim)
        for groups in (16, 14)
        for key_dim in (128, 112)
        for value_dim in (128, 112)
    }

    candidates = build_candidate_library(
        block_configs,
        search_space={"axes": campaign["search_space"]["axes"]},
        parent_checkpoint_identity="qwen3p5-0p8b-teacher",
        include_self=True,
        include_noops=False,
        hidden_width=960,
    )

    assert {candidate.hidden_width for candidate in candidates} == {960}
    assert {candidate.layer_idx for candidate in candidates} == set(
        range(model["model_info"]["num_hidden_layers"])
    )
    for layer_idx in (0, 3):
        layer_candidates = [
            candidate for candidate in candidates if candidate.layer_idx == layer_idx
        ]
        assert {
            candidate.block_config.require_subblock("ffn").intermediate_size
            for candidate in layer_candidates
        } == expected_ffn_sizes

        if layer_idx == 0:
            assert {
                (
                    candidate.block_config.require_subblock("mamba").num_heads,
                    candidate.block_config.require_subblock("mamba").num_groups,
                    candidate.block_config.require_subblock("mamba").state_dim,
                    candidate.block_config.require_subblock("mamba").head_dim,
                )
                for candidate in layer_candidates
            } == expected_gdn_shapes
        else:
            assert {
                (
                    candidate.block_config.require_subblock("attention").num_query_heads,
                    candidate.block_config.require_subblock("attention").num_kv_heads,
                )
                for candidate in layer_candidates
            } == expected_attention_shapes
