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

"""Tests for the Puzzletron-to-Megatron heterogeneous config contract."""

import pytest

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MoEConfig,
)
from modelopt.torch.puzzletron.plugins.mbridge_schema import build_mcore_heterogeneous_config


def test_build_mcore_config_from_typed_block_config() -> None:
    block = BlockConfig(
        subblock_configs=(
            AttentionConfig(num_query_heads=8, num_kv_heads=2),
            FFNConfig(intermediate_size=32),
        )
    )

    config = build_mcore_heterogeneous_config([block.to_dict()], num_attention_heads=8)

    assert config == {
        "block_configs": [
            {
                "attention": {"no_op": False, "num_query_groups": 2},
                "ffn": {"no_op": False, "ffn_hidden_size": 32},
            }
        ]
    }


def test_build_mcore_config_preserves_no_op_subblocks() -> None:
    block = BlockConfig(subblock_configs=(AttentionConfig(no_op=True), FFNConfig(no_op=True)))

    config = build_mcore_heterogeneous_config([block], num_attention_heads=8)

    assert config["block_configs"][0] == {
        "attention": {"no_op": True, "num_query_groups": None},
        "ffn": {"no_op": True, "ffn_hidden_size": None},
    }


@pytest.mark.parametrize(
    ("block", "match"),
    [
        (
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(num_query_heads=4, num_kv_heads=2),
                    FFNConfig(intermediate_size=32),
                )
            ),
            "cannot vary num_query_heads by layer",
        ),
        (
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(num_query_heads=8, num_kv_heads=2),
                    MoEConfig(num_experts=4, expert_intermediate_size=16, top_k=2),
                )
            ),
            "support only attention and ffn subblocks",
        ),
    ],
)
def test_build_mcore_config_rejects_unrepresentable_blocks(block: BlockConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_mcore_heterogeneous_config([block], num_attention_heads=8)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qk_head_dim", 4),
        ("v_head_dim", 4),
        ("sliding_window_size", 16),
        ("k_eq_v", True),
        ("k_eq_v", False),
        ("kv_source_layer", 0),
        ("llama4", {"attention_chunk_size": 16}),
    ],
)
def test_build_mcore_config_rejects_unrepresentable_attention_fields(
    field: str, value: object
) -> None:
    attention = AttentionConfig(
        num_query_heads=8,
        num_kv_heads=2,
        **{field: value},
    )
    block = BlockConfig(subblock_configs=(attention, FFNConfig(intermediate_size=32)))

    with pytest.raises(ValueError, match="cannot represent attention fields"):
        build_mcore_heterogeneous_config([block], num_attention_heads=8)
