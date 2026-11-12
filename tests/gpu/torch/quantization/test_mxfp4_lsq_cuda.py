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

"""GPU end-to-end tests for deployable MXFP4 tied-LSQ."""

import copy

import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import get_weight_scaling_factor, to_quantized_weight
from modelopt.torch.quantization.qtensor import MXFP4QTensor

MXFP4_LSQ_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 32, "type": "static", "scale_bits": (8, 0)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)},
            "axis": None,
            "enable": True,
        },
    },
    "algorithm": {
        "method": "lsq",
        "learnable_amax": ["pre", "post"],
        "tied_amax": True,
        "scale_algorithm": {"method": "max"},
    },
}


def _quantized_linear():
    model = nn.Sequential(nn.Linear(64, 32, bias=False)).cuda()
    calibration_input = torch.randn(2, 64, device="cuda")
    return mtq.quantize(model, MXFP4_LSQ_CFG, lambda module: module(calibration_input))


def test_mxfp4_lsq_gradient_checkpoint_and_export_round_trip():
    model = _quantized_linear()
    linear = model[0]
    quantizer = linear.weight_quantizer

    assert quantizer.is_mxfp4_static
    assert quantizer._tied_amax
    assert isinstance(quantizer._amax_post, nn.Parameter)
    assert linear.input_quantizer.is_mx_format

    initial_scale = quantizer._amax_post.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(torch.randn(2, 64, device="cuda"))
    output.square().mean().backward()
    assert quantizer._amax_post.grad is not None
    assert torch.count_nonzero(quantizer._amax_post.grad) > 0
    optimizer.step()
    assert not torch.equal(quantizer._amax_post.detach(), initial_scale)

    restored_model = copy.deepcopy(model)
    restored_model.load_state_dict(model.state_dict())
    assert torch.equal(
        restored_model[0].weight_quantizer._amax_post,
        quantizer._amax_post,
    )

    e8m0_scale = get_weight_scaling_factor(linear)
    packed = to_quantized_weight(
        linear.weight,
        e8m0_scale,
        quantization="mxfp4",
        block_size=32,
    )
    exported = MXFP4QTensor(
        linear.weight.shape,
        linear.weight.dtype,
        packed,
    ).dequantize(scale=e8m0_scale, block_sizes={-1: 32})
    fake_quantized = quantizer(linear.weight)

    assert torch.equal(exported, fake_quantized)
