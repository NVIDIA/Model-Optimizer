# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from _test_utils.torch.export.utils import (
    ToyModel,
    partial_fp8_config,
    partial_nvfp4_config,
    partial_w4a8_config,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.layer_utils import get_quantization_format
from modelopt.torch.export.model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_NVFP4,
    QUANTIZATION_W4A8_AWQ,
)
from modelopt.torch.export.quant_utils import (
    get_quant_config,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
)
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer
from modelopt.torch.quantization.utils import reduce_block_amax


@pytest.mark.parametrize(
    ("config", "expected"),
    [(partial_fp8_config, QUANTIZATION_FP8), (partial_w4a8_config, QUANTIZATION_W4A8_AWQ)],
)
def test_get_quantization_format(config, expected):
    model = ToyModel()
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, 10)))
    assert get_quantization_format(model) == expected


def test_nvfp4_static_quantizer_export():
    """NVFP4StaticQuantizer: get_quantization_format returns NVFP4 and get_quant_config returns export config."""
    model = ToyModel()
    mtq.quantize(model, partial_nvfp4_config, lambda x: x(torch.randn(1, 4, 10)))

    # Convert all weight quantizers to NVFP4StaticQuantizer
    for module in model.modules():
        tq = getattr(module, "weight_quantizer", None)
        if tq is not None and hasattr(tq, "_amax") and not isinstance(tq, NVFP4StaticQuantizer):
            global_amax = tq._amax.max() if tq._amax.dim() > 0 else tq._amax
            NVFP4StaticQuantizer.from_tensor_quantizer(tq, global_amax=global_amax)

    assert get_quantization_format(model) == QUANTIZATION_NVFP4

    quant_config = get_quant_config(model)
    assert quant_config["quantization"]["quant_algo"] == "NVFP4"
    assert quant_config["quantization"]["group_size"] == 16


def test_unpromoted_nvfp4_static_quantizer_exports_scalar_global_scale():
    """Export should promote static NVFP4 quantizers that layerwise calibration did not touch."""
    model = ToyModel(dims=[32, 8], bias=False)
    static_nvfp4_config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*.weight_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                    "axis": None,
                },
                "enable": True,
            },
        ],
        "algorithm": None,
    }

    mtq.quantize(model, static_nvfp4_config)
    linear = model.linears
    weight_quantizer = linear.weight_quantizer
    per_block_amax = reduce_block_amax(linear.weight, block_sizes={-1: 16}).flatten()
    weight_quantizer.register_buffer("_amax", per_block_amax)

    assert weight_quantizer.is_nvfp4_static
    assert not isinstance(weight_quantizer, NVFP4StaticQuantizer)

    weight_scale_2 = get_weight_scaling_factor_2(linear)
    weight_scale = get_weight_scaling_factor(linear)

    assert isinstance(linear.weight_quantizer, NVFP4StaticQuantizer)
    assert weight_scale_2.shape == torch.Size([])
    assert torch.allclose(weight_scale_2, per_block_amax.max() / (6.0 * 448.0))
    assert weight_scale.shape == (8, 2)
