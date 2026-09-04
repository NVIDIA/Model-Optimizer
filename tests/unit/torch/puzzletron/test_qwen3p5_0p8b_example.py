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
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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
CONFIG_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs"
MODEL_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/model.yaml"
)
ADVANCED_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/advanced.yaml"
)


def test_qwen3p5_0p8b_model_identity_and_geometry_are_pinned() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())

    assert model["input_hf_model_path"] == "Qwen/Qwen3.5-0.8B"
    assert model["model_info"] == {
        "hf_repo": model["input_hf_model_path"],
        "hf_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "num_hidden_layers": 24,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "tie_word_embeddings": True,
        "max_position_embeddings": 262144,
        "mtp_num_hidden_layers": 1,
        "layer_counts": {"linear_attention": 18, "full_attention": 6},
        "full_attention_layer_indices": [3, 7, 11, 15, 19, 23],
        "mamba": {
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        },
    }


def test_qwen3p5_0p8b_default_search_matches_tracked_runtime_campaign() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())

    assert model["pruning"] == {"intermediate_size_list": [3072, 2048]}
    assert model["search_space"]["axes"] == {
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 3584,
            "values": [3072, 2048],
        }
    }


def test_qwen3p5_0p8b_advanced_search_keeps_mild_domains_explicit() -> None:
    advanced = yaml.safe_load(ADVANCED_PATH.read_text())
    axes = advanced["search_space"]["axes"]

    expected_enabled_domains = {
        "hidden_width": (1024, [960, 896]),
        "kv_groups": (2, [1]),
        "q_heads_per_group": (4, [3]),
        "ffn_intermediate": (3584, [3328, 3072]),
        "gdn_key_groups": (16, [14]),
        "gdn_key_head_dim": (128, [112]),
        "gdn_value_head_dim": (128, [112]),
    }
    enabled_domains = {
        axis_id: (axis["teacher_value"], axis["values"])
        for axis_id, axis in axes.items()
        if axis["enabled"]
    }

    assert advanced["embedding_pruning"]["widths"] == [1024, 960, 896]
    spec = Qwen3P5VLModelDescriptor.embedding_pruning_spec(
        SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=1024, tie_word_embeddings=True),
        ),
        widths=advanced["embedding_pruning"]["widths"],
        alignment=advanced["embedding_pruning"]["alignment"],
    )
    assert [spec.validate_width(width) for width in spec.legal_widths] == [1024, 960, 896]
    assert advanced["pruning"]["intermediate_size_list"] == [3328, 3072]
    assert advanced["pruning"]["attn_heads_list"] == [[8, 2], [6, 2], [4, 1], [3, 1]]
    assert advanced["pruning"]["attention_scored_axes"] == [
        "kv_groups",
        "q_heads_per_group",
    ]
    assert advanced["pruning"]["gdn_scored_axes"] == [
        "gdn_key_groups",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
    ]
    assert advanced["depth_importance"] == {
        "enabled": True,
        "max_removals": 2,
        "max_subblocks_to_remove": 2,
    }
    assert advanced["mip"]["runs"]["params-90"]["search_space"] == {
        "depth": [0, 1, 2],
        "embedding": [1024, 960, 896],
        "axes_default": "all",
        "axes": {"ffn.intermediate_size": "all"},
    }
    assert enabled_domains == expected_enabled_domains
    assert {axis_id: axes[axis_id] for axis_id in ("gdn_value_heads_per_group",)} == {
        "gdn_value_heads_per_group": {
            "enabled": False,
            "teacher_value": 1,
            "values": [],
        },
    }
    assert set(axes) == {
        *expected_enabled_domains,
        "kv_groups",
        "q_heads_per_group",
        "gdn_key_groups",
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
    }


def test_qwen3p5_0p8b_advanced_search_composes_the_pinned_model() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        config = compose(config_name="families/qwen3_5/qwen3p5_0p8b/advanced")
    config = OmegaConf.to_container(config, resolve=True)

    assert config["input_hf_model_path"] == "Qwen/Qwen3.5-0.8B"
    assert config["model_info"]["hf_revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert config["model"]["revision"] == config["model_info"]["hf_revision"]
    assert config["search_space"]["axes"]["ffn_intermediate"]["values"] == [3328, 3072]


def test_qwen3p5_0p8b_advanced_search_generates_every_eligible_axis_value() -> None:
    advanced = yaml.safe_load(ADVANCED_PATH.read_text())
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
    block_configs = (linear_block,) * 18 + (full_attention_block,) * 6
    expected_ffn_sizes = {3584, 3328, 3072}
    expected_attention_shapes = {(8, 2), (6, 2), (4, 1), (3, 1)}
    expected_gdn_shapes = {
        (groups, groups, key_dim, value_dim)
        for groups in (16, 14)
        for key_dim in (128, 112)
        for value_dim in (128, 112)
    }

    for hidden_width in (1024, 960, 896):
        candidates = build_candidate_library(
            block_configs,
            search_space={"axes": advanced["search_space"]["axes"]},
            parent_checkpoint_identity="qwen3p5-0p8b-teacher",
            include_self=True,
            include_noops=False,
            hidden_width=hidden_width,
        )

        assert {candidate.hidden_width for candidate in candidates} == {hidden_width}
        for layer_idx in range(24):
            layer_candidates = [
                candidate for candidate in candidates if candidate.layer_idx == layer_idx
            ]
            assert {
                candidate.block_config.require_subblock("ffn").intermediate_size
                for candidate in layer_candidates
            } == expected_ffn_sizes

            if layer_idx < 18:
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
