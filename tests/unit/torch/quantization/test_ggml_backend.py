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

from types import SimpleNamespace

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml.backend import ggml_fake_quant


@pytest.mark.parametrize("num_bits", ["iq1_s", "iq2_xs"])
def test_ggml_backend_via_quantize(num_bits):
    torch.manual_seed(1234)
    model = torch.nn.Linear(256, 2, bias=False)
    inputs = torch.randn(2, 256)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": num_bits, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }

    mtq.quantize(model, config, forward_loop=lambda module: module(inputs))
    output = model(inputs)

    assert model.weight_quantizer.backend == "ggml"
    assert model.weight_quantizer.num_bits == num_bits
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()


def test_ggml_backend_rejects_unknown_format():
    with pytest.raises(ValueError, match="requires num_bits"):
        ggml_fake_quant(torch.ones(1, 256), SimpleNamespace(num_bits="unknown"))
