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

"""Megatron-Bridge integration tests for Puzzletron heterogeneous configs."""

import json

from megatron.bridge.models.transformer_config import HeterogeneousTransformerConfig

from modelopt.torch.puzzletron.block_config import AttentionConfig, BlockConfig, FFNConfig
from modelopt.torch.puzzletron.plugins.mbridge.base import HeterogeneousBridgeMixin


class _HfConfig:
    num_attention_heads = 8

    def __init__(self, block_configs: list[BlockConfig]) -> None:
        self.block_configs = block_configs

    def to_json_string(self) -> str:
        return json.dumps(
            {
                "num_attention_heads": self.num_attention_heads,
                "block_configs": [block.to_dict() for block in self.block_configs],
            }
        )


def test_puzzletron_block_config_round_trips_through_mbridge() -> None:
    hf_config = _HfConfig(
        [
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(num_query_heads=8, num_kv_heads=2),
                    FFNConfig(intermediate_size=32),
                )
            ),
            BlockConfig(
                subblock_configs=(
                    AttentionConfig(no_op=True),
                    FFNConfig(intermediate_size=16),
                )
            ),
        ]
    )
    encoded = HeterogeneousBridgeMixin()._build_heterogeneous_config_json(hf_config)

    config = HeterogeneousTransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=8,
        num_query_groups=8,
        ffn_hidden_size=64,
        heterogeneous_layers_config_encoded_json=encoded,
    )
    config.finalize()

    first, second = config.per_block_parameters
    assert first.attention.num_query_groups == 2
    assert first.mlp.ffn_hidden_size == 32
    assert not first.attention.no_op
    assert not first.mlp.no_op
    assert second.attention.no_op
    assert second.mlp.ffn_hidden_size == 16
