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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.qtensor import MXFP4QTensor


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("prescale", [False, True])
def test_mxfp4_weight_blocks_do_not_cross_rows(materialized, prescale):
    torch.manual_seed(23)
    inputs = torch.randn(2, 64)
    cfg = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "residue_max"}
    model = mtq.quantize(nn.Linear(64, 3), cfg, lambda m: m(inputs))
    original_weight = model.weight.detach().clone()
    if prescale:
        model.input_quantizer.pre_quant_scale = torch.linspace(0.5, 2, 64)
        model.weight_quantizer.pre_quant_scale = model.input_quantizer.pre_quant_scale.reciprocal()
        original_weight *= model.weight_quantizer.pre_quant_scale
        inputs_for_reference = inputs * model.input_quantizer.pre_quant_scale
    else:
        inputs_for_reference = inputs
    if materialized:
        model.materialize_residue_weight()
    # Activation QDQ requires CUDA; this test exercises the real CPU weight packer.
    model.input_quantizer.disable()
    packed, scale = MXFP4QTensor.quantize(original_weight.contiguous(), 32)
    weight_q = packed.dequantize(scale=scale, block_sizes={-1: 32})
    expected = F.linear(inputs_for_reference, weight_q, model.bias)
    torch.testing.assert_close(model(inputs), expected)


def test_materialization_failure_preserves_parameter():
    cfg = deepcopy(mtq.NVFP4_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "residue_max"}
    model = mtq.quantize(nn.Linear(64, 3), cfg, lambda m: m(torch.ones(2, 64)))
    parameter = model.weight
    before = parameter.detach().clone()
    model.input_quantizer.salient_indices.fill_(64)
    with pytest.raises((RuntimeError, IndexError)):
        model.materialize_residue_weight()
    assert model.weight is parameter
    assert torch.equal(model.weight, before)
    assert not hasattr(model, "_original_in_features")


def test_standalone_residue_quantizer_rejected():
    quantizer = mtq.nn.ResidueTensorQuantizer()
    with pytest.raises(RuntimeError, match="requires ResidueQuantLinear"):
        quantizer(torch.ones(2, 8))


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("scale_shape", [(), (1,), (3, 1)])
def test_broadcast_weight_pre_quant_scale(materialized, scale_shape):
    inputs = torch.randn(2, 64)
    cfg = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "residue_max"}
    model = mtq.quantize(nn.Linear(64, 3), cfg, lambda m: m(inputs))
    model.input_quantizer.pre_quant_scale = torch.tensor(2.0)
    model.weight_quantizer.pre_quant_scale = torch.full(scale_shape, 0.5)
    expected = F.linear(inputs, model.weight, model.bias)
    if materialized:
        model.materialize_residue_weight()
    model.input_quantizer.disable()
    model.weight_quantizer.disable()
    torch.testing.assert_close(model(inputs), expected)


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mxfp4_weight_packing_is_bounded(monkeypatch, materialized, dtype):
    inputs = torch.randn(2, 1025, dtype=dtype)
    cfg = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "residue_max"}
    model = mtq.quantize(nn.Linear(1025, 257, dtype=dtype), cfg, lambda m: m(inputs))
    indices = model.input_quantizer.salient_indices
    weight = torch.cat([model.weight, model.weight[:, indices]], dim=-1)
    packed, scale = MXFP4QTensor.quantize(F.pad(weight, (0, -weight.shape[-1] % 32)), 32)
    expected_weight = packed.dequantize(scale=scale, block_sizes={-1: 32})
    expected = F.linear(inputs, expected_weight[:, :1025], model.bias)
    if materialized:
        model.materialize_residue_weight()
    model.input_quantizer.disable()
    quantize = MXFP4QTensor.quantize
    quantized_chunks = []

    def bounded_quantize(weight, block_size):
        assert weight.numel() <= 2**18
        packed, scale = quantize(weight, block_size)
        quantized_chunks.append(packed.dequantize(scale=scale, block_sizes={-1: block_size}))
        return packed, scale

    monkeypatch.setattr(MXFP4QTensor, "quantize", bounded_quantize)
    torch.testing.assert_close(model(inputs), expected)
    torch.testing.assert_close(torch.cat(quantized_chunks), expected_weight, rtol=0, atol=0)
