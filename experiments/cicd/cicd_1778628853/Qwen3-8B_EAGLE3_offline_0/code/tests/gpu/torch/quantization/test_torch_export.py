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

"""Unittests for torch export of quantized model."""

import pytest
import torch
import torch.nn.functional as F
from _test_utils.torch.misc import set_seed
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils import export_torch_mode


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=10, out_features=5)
        self.linear2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


RANDOM_SEED = 42

configs = [
    {"quant_cfg": mtq.FP8_DEFAULT_CFG, "rtol": 1e-3, "atol": 1e-2},
    {"quant_cfg": mtq.NVFP4_DEFAULT_CFG, "rtol": 1e-3, "atol": 5e-1},
]


@pytest.mark.parametrize("config", configs)
def test_export_torch_mode(config):
    set_seed(RANDOM_SEED)

    quant_cfg = config["quant_cfg"]
    rtol = config["rtol"]
    atol = config["atol"]

    input_tensor = torch.randn(1, 10).cuda()

    model = SimpleNetwork().eval().cuda()
    with torch.inference_mode():

        def calibrate_loop(model):
            """Simple calibration function for testing."""
            model(input_tensor)

        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

        output_pyt = model(input_tensor)
        with export_torch_mode():
            export_program = torch.export.export(model, (input_tensor,), strict=False)
            outputs_ep = export_program.module()(input_tensor)

            print(f"output_pyt: {output_pyt}, outputs_ep: {outputs_ep}")
            assert torch.allclose(
                output_pyt,
                outputs_ep,
                rtol=rtol,
                atol=atol,
            ), "Output from torch export is not close to that from the original model."


@pytest.mark.parametrize("config", configs)
def test_dynamic_batch_size(config):
    quant_cfg = config["quant_cfg"]

    input_tensor = torch.randn(2, 10).cuda()

    model = SimpleNetwork().eval().cuda()
    with torch.inference_mode():

        def calibrate_loop(model):
            """Simple calibration function for testing."""
            model(input_tensor)

        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

        dynamic_shapes = (
            {
                0: torch.export.Dim("batch_size", min=1, max=8),
            },
        )
        with export_torch_mode():
            exported_program = torch.export.export(
                model, (input_tensor,), dynamic_shapes=dynamic_shapes, strict=False
            )
    assert exported_program
