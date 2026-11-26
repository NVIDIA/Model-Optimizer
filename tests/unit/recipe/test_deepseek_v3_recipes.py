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

"""DeepSeek-V3 MLA recipes: the projections they name exist in transformers' DeepSeek-V3."""

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_deepseek_v3

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe

NVFP4 = (2, 1)
FP8 = (4, 3)


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        (
            "nvfp4_omlp-fp8_mla-kv_fp8_cast",
            {"q_a_proj": FP8, "q_b_proj": FP8, "kv_a_proj_with_mqa": FP8, "kv_b_proj": FP8},
        ),
        (
            "nvfp4_omlp_mla-kv_fp8_cast",
            {"q_a_proj": NVFP4, "q_b_proj": NVFP4, "kv_a_proj_with_mqa": NVFP4, "kv_b_proj": None},
        ),
    ],
)
def test_mla_projection_formats(recipe, expected):
    model = get_tiny_deepseek_v3(dtype=torch.float32)
    config = load_recipe(f"model_type/deepseek_v3/ptq/{recipe}").quantize.model_dump()
    config["algorithm"] = None
    mtq.quantize(model, config)

    def fmt(quantizer):
        return quantizer.num_bits if quantizer.is_enabled else None

    layer = model.model.layers[0]
    for name, expected_fmt in {**expected, "o_proj": NVFP4}.items():
        proj = getattr(layer.self_attn, name)
        assert fmt(proj.weight_quantizer) == fmt(proj.input_quantizer) == expected_fmt, name
    assert fmt(layer.mlp.experts.gate_up_proj_input_quantizer) == NVFP4
    assert fmt(layer.self_attn.k_bmm_quantizer) == FP8
