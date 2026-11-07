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
from modelopt.torch.quantization.qtensor import NVFP4QTensor


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


def test_compress_nvfp4_static_uses_calibrated_per_block_amax():
    """Compressing a static NVFP4 weight packs it with the calibrated per-block amax."""
    model = nn.Sequential(nn.Linear(64, 32, bias=False))
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {
                    "num_bits": (2, 1),
                    "block_sizes": {-1: 16, "type": "static", "scale_bits": (4, 3)},
                    "axis": None,
                },
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, lambda m: m(torch.randn(2, 64)))
    quantizer = model[0].weight_quantizer
    # Clip every block below its max, as MSE or AWQ-clip calibration does.
    quantizer.amax = quantizer.amax * 0.7
    weight = model[0].weight.detach().clone()
    scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(quantizer)
    scale, _ = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(quantizer, weight, scale_2)
    expected, _, _ = NVFP4QTensor.quantize(weight, 16, scale, scale_2)

    mtq.compress(model)

    assert torch.equal(quantizer._scale.float(), scale.float())
    torch.testing.assert_close(
        quantizer(model[0].weight),
        expected.dequantize(scale=scale, double_scale=scale_2, block_sizes={-1: 16}),
        rtol=0,
        atol=0,
    )
