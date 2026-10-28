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

"""GPU tests for imatrix calibration of the GGML formats."""

import copy
import time

import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml.imatrix import GGMLImatrixEncoder

IQ2_XXS_IMATRIX = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {"num_bits": "iq2_xxs", "block_sizes": {-1: 256}, "backend": "ggml"},
        },
    ],
    "algorithm": {"method": "imatrix"},
}


class _Layers(nn.Module):
    """Linear layers fed on whichever GPU each lives on, as a ``device_map`` places them."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(512, 512, bias=False) for _ in range(4))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x.to(layer.weight.device))
        return x


def test_imatrix_encodes_each_gpus_weights_at_once(need_2_gpus, monkeypatch):
    """Weights on two GPUs are encoded concurrently, to the same codes as on one GPU."""
    spans = []
    encode = GGMLImatrixEncoder.encode

    def timed_encode(self, weight, quantizer, second_moments):
        start = time.perf_counter()
        decoded = encode(self, weight, quantizer, second_moments)
        torch.cuda.synchronize(weight.device)
        time.sleep(0.2)  # long enough that encodes on different GPUs must overlap to save time
        spans.append((weight.device.index, start, time.perf_counter()))
        return decoded

    monkeypatch.setattr(GGMLImatrixEncoder, "encode", timed_encode)
    torch.manual_seed(0)
    reference = _Layers().cuda()
    model = copy.deepcopy(reference)
    model.layers[2:].to("cuda:1")
    inputs = torch.randn(8, 128, 512, device="cuda")

    mtq.quantize(reference, IQ2_XXS_IMATRIX, forward_loop=lambda m: m(inputs))
    spans.clear()
    mtq.quantize(model, IQ2_XXS_IMATRIX, forward_loop=lambda m: m(inputs))

    for expected, layer in zip(reference.layers, model.layers):
        assert torch.equal(layer.weight.cuda(0), expected.weight)
    assert any(
        a_start < b_end and b_start < a_end
        for gpu_a, a_start, a_end in spans
        for gpu_b, b_start, b_end in spans
        if gpu_a != gpu_b
    )
