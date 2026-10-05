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
"""GPTQ on a GGML format through ``mtq.quantize``, with the CUDA encoders."""

import copy

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY


def _iq1_s_config(algorithm):
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": "iq1_s", "block_sizes": {-1: 256}, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": algorithm,
    }


def _correlated_inputs(cols=512, seed=0):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(256, cols, generator=generator) @ torch.randn(
        cols, cols, generator=generator
    )
    return inputs.cuda()


def _weighted_error(quantized, weight, inputs):
    return ((quantized - weight) @ inputs.T).square().sum()


def test_gptq_on_a_ggml_format_keeps_the_payload_it_chose():
    torch.manual_seed(0)
    model = torch.nn.Linear(512, 8, bias=False).cuda()
    original = model.weight.detach().clone()
    inputs = _correlated_inputs()
    plain = copy.deepcopy(model)
    mtq.quantize(plain, _iq1_s_config("max"), forward_loop=lambda m: m(inputs))

    mtq.quantize(
        model,
        _iq1_s_config({"method": "gptq", "block_size": 256}),
        forward_loop=lambda m: m(inputs),
    )

    iq1_s = GGML_FORMAT_REGISTRY["iq1_s"]
    packed = model.weight_quantizer._quantizer_cache.packed_weights
    decoded = iq1_s.dequantize(packed, torch.tensor(model.weight.shape), dtype=torch.float32)
    torch.testing.assert_close(model.weight.detach(), decoded)
    torch.testing.assert_close(model(inputs), inputs @ decoded.T)
    assert torch.equal(iq1_s.pack(model.weight, model.weight_quantizer).flatten(), packed.flatten())
    plain_weight = plain.weight_quantizer(plain.weight).detach()
    assert _weighted_error(decoded, original, inputs) < _weighted_error(
        plain_weight, original, inputs
    )


def test_gptq_on_a_ggml_format_rejects_blocks_smaller_than_a_ggml_block():
    model = torch.nn.Linear(512, 8, bias=False).cuda()
    inputs = _correlated_inputs()

    with pytest.raises(ValueError, match="multiples of the quantization group size"):
        mtq.quantize(
            model,
            _iq1_s_config({"method": "gptq", "block_size": 128}),
            forward_loop=lambda m: m(inputs),
        )
