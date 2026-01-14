# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch.nn as nn
from _test_utils.torch.misc import set_seed
from _test_utils.torch.quantization.quantize_common import quantize_model_and_forward

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.extensions import get_cuda_ext_mx
from modelopt.torch.quantization.nn import QuantModule

te = pytest.importorskip("transformer_engine")


class TELinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            te.pytorch.Linear(16, 32), te.pytorch.LayerNormLinear(32, 16, normalization="RMSNorm")
        )

    def forward(self, x):
        return self.net(x)

    def get_input(self):
        return torch.randn(2, 16)


@pytest.mark.parametrize("model_cls", [TELinear])
@pytest.mark.parametrize(
    "config",
    [
        mtq.INT8_DEFAULT_CFG,
        mtq.FP8_DEFAULT_CFG,
        mtq.W4A8_AWQ_BETA_CFG,
        mtq.INT8_SMOOTHQUANT_CFG,
        mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
        mtq.INT4_AWQ_CFG,
        mtq.NVFP4_DEFAULT_CFG,
        mtq.NVFP4_AWQ_LITE_CFG,
        mtq.MXFP8_DEFAULT_CFG,
        mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    ],
)
def test_quantize(model_cls, config):
    """Test quantize function can run without problems."""
    if (
        config
        in [
            mtq.NVFP4_DEFAULT_CFG,
            mtq.NVFP4_AWQ_LITE_CFG,
            mtq.MXFP8_DEFAULT_CFG,
            mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
        ]
        and get_cuda_ext_mx() is None
    ):
        pytest.skip("cuda_ext_mx is not available")

    if config == mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG:
        # reduce block sizes for simple testing models
        config["quant_cfg"]["*weight_quantizer"]["block_sizes"] = {-1: 8, -2: 8}
    model = model_cls().cuda()
    calib_data = [model.get_input().cuda() for _ in range(1)]
    quantize_model_and_forward(model, config, calib_data)


def test_quantize_forward_backward():
    set_seed()
    model = TELinear().cuda()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.copy_(param.data.abs() + 0.1)  # hack to get non-zero gradient

    # hack to get non-zero gradient
    calib_data = [model.get_input().cuda().abs() + 0.1 for _ in range(1)]

    quantize_model_and_forward(model, mtq.INT8_DEFAULT_CFG, calib_data)

    model.train()
    for name, param in model.named_parameters():
        param.grad = None

    loss = model(calib_data[0]).sum()
    loss.backward()

    for i, linear in enumerate(model.net):
        assert isinstance(linear, QuantModule)
        # In-directly tests that data was passed to the quantizers
        assert linear.input_quantizer.amax is not None
        assert linear.weight_quantizer.amax is not None

        # In-directly tests that gradients were computed correctly
        assert linear.weight.grad is not None and linear.weight.grad.abs().sum() > 0.0


@pytest.mark.parametrize(
    ("model_cls", "quant_config"),
    [
        (TELinear, mtq.INT8_SMOOTHQUANT_CFG),
    ],
)
def test_save_restore(model_cls, quant_config):
    # test restoring to an unquantized model
    # Transformer Engine does not support cpu device
    device = "cuda"
    model_quant = model_cls().to(device)
    model_ref = model_cls().to(device)

    calib_data = [model_quant.get_input().to(device) for _ in range(2)]
    quantize_model_and_forward(model_quant.to(device), quant_config, calib_data)

    state_dict = mto.modelopt_state(model_quant)

    mto.restore_from_modelopt_state(model_ref, state_dict)
    model_ref.load_state_dict(model_quant.state_dict())
    assert torch.allclose(model_quant(calib_data[0]), model_ref(calib_data[0]))
