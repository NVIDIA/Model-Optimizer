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

"""Tests for the quantized diffusers Attention in the diffusers plugin."""

import copy

import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("diffusers")

from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.transformers.transformer_flux import FluxAttention

import modelopt.torch.quantization as mtq

# Mirrors ``--quantize-mha`` in examples/diffusers/quantization/quantize.py.
_FP8_MHA_CFG = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
_FP8_MHA_CFG["quant_cfg"].append(
    {"quantizer_name": "*[qkv]_bmm_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}}
)


@pytest.mark.parametrize(
    "make_attention",
    [
        # Calls F.scaled_dot_product_attention positionally.
        pytest.param(
            lambda: Attention(query_dim=32, heads=2, dim_head=16, processor=AttnProcessor2_0()),
            id="Attention",
        ),
        # Calls it by keyword through the Diffusers native attention backend.
        pytest.param(lambda: FluxAttention(query_dim=32, heads=2, dim_head=16), id="FluxAttention"),
    ],
)
@pytest.mark.parametrize(
    ("config", "mha_enabled"),
    [
        pytest.param(mtq.INT8_DEFAULT_CFG, False, id="int8"),
        pytest.param(_FP8_MHA_CFG, True, id="fp8-mha"),
    ],
)
def test_quantized_sdpa_backward(make_attention, config, mha_enabled):
    attention = make_attention()
    inputs = torch.randn(2, 4, 32)
    mtq.quantize(attention, config, lambda model: model(inputs))
    assert attention.q_bmm_quantizer.is_enabled is mha_enabled

    attention(inputs).sum().backward()

    for name in ("to_q", "to_k", "to_v"):
        assert torch.isfinite(getattr(attention, name).weight.grad).all()
