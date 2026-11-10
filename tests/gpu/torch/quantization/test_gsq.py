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

"""GSQ captures dense updates across every GPU used by one process."""

import pytest
import torch

from modelopt.recipe import load_recipe
from modelopt.torch.quantization.extensions import get_cuda_ext_ggml
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.gsq import quantize_iq_gsq


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_iq_gsq_captures_every_dense_layer_across_gpus():
    assert get_cuda_ext_ggml(raise_if_failed=True) is not None
    model = torch.nn.Module()
    model.mlp = torch.nn.ModuleList(
        torch.nn.Linear(256, 256, bias=False).to(f"cuda:{i % 2}") for i in range(4)
    )
    inputs = torch.randn(32, 256, device="cuda:0")

    def calibrate(model):
        value = inputs
        for layer in model.mlp:
            value = layer(value.to(layer.weight.device))
        return value

    config = load_recipe("general/ptq/iq2_s").quantize.model_dump(exclude_unset=True)
    config["algorithm"] = {"method": "gptq", "block_size": 256, "perc_damp": 0.3}
    result = quantize_iq_gsq(model, config, calibrate, num_steps=(2,))

    assert set(result.payloads) == {f"mlp.{i}" for i in range(4)}
    for i, layer in enumerate(model.mlp):
        assert torch.equal(
            result.payloads[f"mlp.{i}"][0],
            pinned_packed_weight(layer.weight_quantizer, "iq2_s").cpu(),
        )
    result.activate(model, 2)
    assert torch.isfinite(calibrate(model)).all()
