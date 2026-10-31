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

from modelopt.torch.export.trtllm.model_config import EmbeddingConfig, LinearConfig, ModelConfig
from modelopt.torch.export.trtllm.postprocess import pad_embedding_lm_head


@pytest.mark.parametrize("has_embedding", [True, False])
@pytest.mark.parametrize("has_lm_head", [True, False])
def test_pad_embedding_lm_head_skips_absent_fields(has_embedding, has_lm_head):
    config = ModelConfig(vocab_size=100)
    if has_embedding:
        config.vocab_embedding = EmbeddingConfig(weight=torch.ones(100, 8))
    if has_lm_head:
        config.lm_head = LinearConfig(weight=torch.ones(100, 8))

    pad_embedding_lm_head(config)

    assert config.vocab_size == 128
    for field in (config.vocab_embedding, config.lm_head):
        if field is not None:
            assert field.weight.shape == (128, 8)
            assert torch.equal(field.weight[:100], torch.ones(100, 8))
            assert not field.weight[100:].any()
