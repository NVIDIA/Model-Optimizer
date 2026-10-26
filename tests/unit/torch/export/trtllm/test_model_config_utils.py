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

import pytest
import torch

from modelopt.torch.export.trtllm.model_config import (
    AttentionConfig,
    DecoderLayerConfig,
    EmbeddingConfig,
    LayernormConfig,
    LinearConfig,
    MedusaHeadConfig,
    ModelConfig,
    QKVConfig,
)
from modelopt.torch.export.trtllm.model_config_utils import (
    model_config_from_dict,
    model_config_to_dict,
)


@pytest.mark.parametrize("split_qkv", [False, True])
def test_optional_nested_configs_round_trip(split_qkv):
    linear = LinearConfig(weight=torch.ones(2, 2))
    qkv = QKVConfig(q=linear, k=linear, v=linear) if split_qkv else linear
    config = ModelConfig(
        vocab_embedding=EmbeddingConfig(weight=torch.ones(3, 2)),
        ln_f=LayernormConfig(weight=torch.ones(2)),
        layers=[
            DecoderLayerConfig(
                attention=AttentionConfig(qkv=qkv, dense=linear, clip_qkv=3.0),
                rope_scaling={"type": "linear", "factor": 2.0},
                longrope_scaling_short_factors=[1.0, 2.0],
            )
        ],
        medusa_heads=[MedusaHeadConfig(lm_head=linear)],
    )
    restored = model_config_from_dict(model_config_to_dict(config))
    assert isinstance(restored.vocab_embedding, EmbeddingConfig)
    assert isinstance(restored.ln_f, LayernormConfig)
    attention = restored.layers[0].attention
    assert isinstance(attention, AttentionConfig)
    assert isinstance(attention.qkv, QKVConfig if split_qkv else LinearConfig)
    assert isinstance(attention.dense, LinearConfig)
    assert attention.clip_qkv == 3.0
    assert restored.layers[0].rope_scaling == {"type": "linear", "factor": 2.0}
    assert restored.layers[0].longrope_scaling_short_factors == [1.0, 2.0]
    assert isinstance(restored.medusa_heads[0], MedusaHeadConfig)
    assert isinstance(restored.medusa_heads[0].lm_head, LinearConfig)
    assert restored.lm_head is None
    torch.testing.assert_close(restored.vocab_embedding.weight, config.vocab_embedding.weight)
