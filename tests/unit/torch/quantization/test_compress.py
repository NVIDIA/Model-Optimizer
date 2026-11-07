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

"""CPU tests of mtq.compress."""

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.compress import is_real_quantized
from modelopt.torch.quantization.nn import GroupedQuantizer


@pytest.mark.parametrize("grouped", [False, True], ids=["linear", "grouped"])
def test_compress_refuses_w4a8_weights_before_changing_the_model(grouped):
    """A compressed W4A8 (SequentialQuantizer) weight can neither run forward nor be restored."""
    model = nn.Sequential(nn.Linear(128, 128, bias=False))  # 128 wide for W4A8's INT4 blocks
    mtq.quantize(model, mtq.W4A8_AWQ_BETA_CFG, lambda m: m(torch.randn(2, 128)))
    if grouped:  # one per expert, as Transformer Engine's grouped linear holds them
        model[0].weight_quantizer = GroupedQuantizer(model[0].weight_quantizer)

    with pytest.raises(NotImplementedError, match="W4A8"):
        mtq.compress(model)

    assert not is_real_quantized(model)


@pytest.mark.parametrize(
    "weight_cfg",
    [
        {"num_bits": 4, "block_sizes": {-1: 64}},
        {"num_bits": 8, "axis": 0},
        {"num_bits": 8, "block_sizes": {-1: 64}},
    ],
    ids=["int4_block", "int8_channel", "int8_block"],
)
def test_compress_keeps_calibrated_weight_amax(weight_cfg):
    """Compressed INT4 / INT8 weights use the calibrated amax, not the weight's own max."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 32, bias=False))
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": weight_cfg},
        ],
        "algorithm": "max",
    }
    x = torch.randn(8, 128)
    mtq.quantize(model, config, lambda m: m(x))
    # Stand in for a clip search (AWQ clip, MSE): a calibrated range below the weight's max.
    weight_quantizer = model[0].weight_quantizer
    weight_quantizer.amax = weight_quantizer.amax * 0.8

    expected = model(x)
    mtq.compress(model)

    assert is_real_quantized(model)
    torch.testing.assert_close(model(x), expected)


def test_compress_int4_casts_calibrated_scales_to_weight_dtype():
    """Scales derived from a float32 amax match the weight dtype, as the CUDA kernel requires."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(128, 32, bias=False)).to(torch.bfloat16)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": 4, "block_sizes": {-1: 64}},
            },
        ],
        "algorithm": "max",
    }
    x = torch.randn(8, 128, dtype=torch.bfloat16)
    mtq.quantize(model, config, lambda m: m(x))
    weight_quantizer = model[0].weight_quantizer
    weight_quantizer.amax = weight_quantizer.amax.float()

    mtq.compress(model)

    assert weight_quantizer._scale.dtype == torch.bfloat16
