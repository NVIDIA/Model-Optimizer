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

"""Tests for candidate counts shown by the Puzzletron setup wizard."""

from puzzletron_setup.profiles import ModelInventory, count_candidate_options


def _axis(teacher, *values):
    return {
        "enabled": True,
        "teacher_value": teacher,
        "values": list(values),
    }


def test_counts_dense_qwen_hybrid_candidates_exactly():
    inventory = ModelInventory(
        family="qwen3_5",
        descriptor="qwen3_5",
        family_config="family.yaml",
        model_type="qwen3_5",
        architectures=(),
        multimodal=False,
        moe=False,
        num_layers=24,
        num_sublayers=48,
        layer_counts={"linear_attention": 18, "full_attention": 6},
        facts={},
        axes=(),
    )
    config = {
        "text_config": {
            "num_hidden_layers": 24,
            "layer_types": ["linear_attention"] * 18 + ["full_attention"] * 6,
        }
    }
    axes = {
        "hidden_width": _axis(1024, 1024, 768),
        "kv_groups": _axis(2, 2, 1),
        "q_heads_per_group": _axis(4, 4, 2),
        "ffn_intermediate": _axis(3584, 3584, 3072),
        "gdn_key_groups": _axis(16, 16, 8),
        "gdn_value_heads_per_group": _axis(1, 1),
        "gdn_key_head_dim": _axis(128, 128, 96),
        "gdn_value_head_dim": _axis(128, 128, 96),
    }

    counts = count_candidate_options(config, inventory, axes)

    assert counts.vllm_subblock == 14
    assert counts.vllm_block == 24
    assert counts.replacement_subblock_per_width == 168
    assert counts.replacement_block_per_width == 312
    assert counts.width_count == 2
    assert counts.replacement_subblock_total == 336
    assert counts.replacement_block_total == 624


def test_deduplicates_teacher_only_axis_selections():
    inventory = ModelInventory(
        family="qwen3_5",
        descriptor="qwen3_5",
        family_config="family.yaml",
        model_type="qwen3_5",
        architectures=(),
        multimodal=False,
        moe=False,
        num_layers=1,
        num_sublayers=2,
        layer_counts={"full_attention": 1},
        facts={},
        axes=(),
    )
    config = {
        "text_config": {
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
        }
    }
    axes = {
        "hidden_width": _axis(1024, 1024, 1024),
        "kv_groups": _axis(2, 2, 2),
        "q_heads_per_group": _axis(4, 4),
        "ffn_intermediate": _axis(3584, 3584, 3584),
    }

    counts = count_candidate_options(config, inventory, axes)

    assert counts.vllm_subblock == 2
    assert counts.vllm_block == 1
    assert counts.replacement_subblock_per_width == 0
    assert counts.replacement_block_per_width == 0
    assert counts.width_count == 1


def test_counts_qwen_moe_domain_once_for_vllm_and_per_layer_for_scoring():
    inventory = ModelInventory(
        family="qwen3_5",
        descriptor="qwen3_5_moe",
        family_config="family.yaml",
        model_type="qwen3_5_moe",
        architectures=(),
        multimodal=False,
        moe=True,
        num_layers=3,
        num_sublayers=6,
        layer_counts={"full_attention": 2, "linear_attention": 1},
        facts={},
        axes=(),
    )
    config = {
        "text_config": {
            "num_hidden_layers": 3,
            "layer_types": [
                "full_attention",
                "full_attention",
                "linear_attention",
            ],
        }
    }
    axes = {
        "hidden_width": _axis(2048, 2048),
        "kv_groups": _axis(2, 2, 1),
        "q_heads_per_group": _axis(8, 8, 4),
        "gdn_key_groups": _axis(16, 16, 8),
        "gdn_value_heads_per_group": _axis(2, 2),
        "gdn_key_head_dim": _axis(128, 128),
        "gdn_value_head_dim": _axis(128, 128),
        "moe_experts": _axis(256, 256, 128),
        "moe_expert_intermediate": _axis(512, 512, 256),
        "moe_shared_expert_intermediate": _axis(512, 512),
        "moe_top_k": _axis(8, 8),
    }

    counts = count_candidate_options(config, inventory, axes)

    assert counts.vllm_subblock == 10
    assert counts.vllm_block == 24
    assert counts.replacement_subblock_per_width == 16
    assert counts.replacement_block_per_width == 37


def test_counts_nemotron_mutually_exclusive_hybrid_pattern():
    inventory = ModelInventory(
        family="nemotron3",
        descriptor="nemotron_h",
        family_config="family.yaml",
        model_type="nemotron_h",
        architectures=(),
        multimodal=False,
        moe=True,
        num_layers=4,
        num_sublayers=4,
        layer_counts={"attention": 1, "mamba": 1, "moe": 1, "ffn": 1},
        facts={},
        axes=(),
    )
    config = {
        "num_hidden_layers": 4,
        "hybrid_override_pattern": "*ME-",
    }
    axes = {
        "hidden_width": _axis(2688, 2688),
        "kv_groups": _axis(2, 2, 1),
        "q_heads_per_group": _axis(16, 16, 8),
        "mamba_heads": _axis(64, 64, 48),
        "mamba_head_dim": _axis(64, 64, 48),
        "moe_experts": _axis(128, 128, 96),
        "moe_expert_intermediate": _axis(1856, 1856, 1600),
        "moe_shared_expert_intermediate": _axis(3712, 3712),
        "moe_top_k": _axis(6, 6),
        "ffn_intermediate": _axis(1856, 1856, 1600),
    }

    counts = count_candidate_options(config, inventory, axes)

    assert counts.vllm_subblock == 14
    assert counts.vllm_block == 14
    assert counts.replacement_subblock_per_width == 10
    assert counts.replacement_block_per_width == 10
