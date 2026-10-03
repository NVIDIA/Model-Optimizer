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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from modelopt.torch.quantization.plugins.huggingface import _QuantQwen3VLMoeTextExperts

try:
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts
except ImportError:
    Qwen3VLMoeTextExperts = None


class TinyQwen3VLMoeTextConfig:
    num_experts = 2
    hidden_size = 8
    moe_intermediate_size = 4
    hidden_act = "silu"
    _experts_implementation = "eager"


def _convert_to_quant_wrapper(module):
    module.__class__ = type(
        "QuantQwen3VLMoeTextExpertsForTest",
        (_QuantQwen3VLMoeTextExperts, Qwen3VLMoeTextExperts),
        {},
    )
    module._setup()
    return module


@pytest.mark.skipif(
    Qwen3VLMoeTextExperts is None, reason="Qwen3-VL-MoE is not available in transformers"
)
def test_qwen3_vl_moe_text_experts_quant_wrapper_matches_hf_forward():
    torch.manual_seed(0)

    original = Qwen3VLMoeTextExperts(TinyQwen3VLMoeTextConfig())
    torch.nn.init.normal_(original.gate_up_proj)
    torch.nn.init.normal_(original.down_proj)

    wrapped = _convert_to_quant_wrapper(copy.deepcopy(original))

    hidden_states = torch.randn(5, TinyQwen3VLMoeTextConfig.hidden_size)
    top_k_index = torch.tensor([[0], [1], [0], [1], [0]])
    top_k_weights = torch.ones(5, 1)

    expected = original(hidden_states, top_k_index, top_k_weights)
    actual = wrapped(hidden_states, top_k_weights, top_k_index)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == hidden_states.shape
