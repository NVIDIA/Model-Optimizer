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

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.qtensor import MXFP4QTensor
from modelopt.torch.quantization.tensor_quant import dynamic_block_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for FP4 QDQ"
)


def _reference(model, inputs):
    iq, wq = model.input_quantizer, model.weight_quantizer
    indices = iq.salient_indices
    is_mx = iq.block_sizes["scale_bits"] == (8, 0)
    x = inputs if iq.pre_quant_scale is None else inputs * iq.pre_quant_scale
    if is_mx and indices.numel():
        x = x.float()
    block = iq.block_sizes[-1]
    xq = dynamic_block_quant(x, block, iq.amax, None, (2, 1), iq.block_sizes["scale_bits"])
    if indices.numel():
        residual = (x - xq).index_select(-1, indices)
        rq = dynamic_block_quant(
            residual, block, iq.amax, None, (2, 1), iq.block_sizes["scale_bits"]
        )
    weight = model.weight[..., : inputs.shape[-1]]
    if wq.pre_quant_scale is not None:
        weight = weight * wq.pre_quant_scale
    weight_ext = torch.cat([weight, weight.index_select(-1, indices)], dim=-1)
    if is_mx and indices.numel():
        packed, scale = MXFP4QTensor.quantize(
            F.pad(weight_ext, (0, -weight_ext.shape[-1] % 32)), 32
        )
        wq_value = packed.dequantize(scale=scale, block_sizes={-1: 32})[..., : weight_ext.shape[-1]]
    else:
        wq_value = dynamic_block_quant(
            weight_ext, block, wq.amax, None, (2, 1), wq.block_sizes["scale_bits"]
        )
    # Separate GEMMs check the paired residual formula independently of concatenated Linear.
    result = F.linear(xq.to(inputs.dtype), wq_value[..., : inputs.shape[-1]], model.bias)
    if indices.numel():
        result += F.linear(rq.to(inputs.dtype), wq_value[..., inputs.shape[-1] :])
    return result


@pytest.mark.parametrize("fmt", ["NVFP4", "MXFP4"])
@pytest.mark.parametrize("width", [64, 65])
@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.25, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.no_grad()
def test_residue_qdq_and_checkpoint(tmp_path, fmt, width, ratio, dtype):
    torch.manual_seed(31)
    cfg = deepcopy(getattr(mtq, f"{fmt}_DEFAULT_CFG"))
    cfg["algorithm"] = {"method": "residue_max", "residue_ratio": ratio}
    inputs = torch.randn(2, 3, width, device="cuda", dtype=dtype)
    model = mtq.quantize(nn.Linear(width, 17, device="cuda", dtype=dtype), cfg, lambda m: m(inputs))
    model.input_quantizer.pre_quant_scale = torch.linspace(
        0.5, 2, width, device="cuda", dtype=dtype
    )
    model.weight_quantizer.pre_quant_scale = model.input_quantizer.pre_quant_scale.reciprocal()
    reference = _reference(model, inputs)
    atol = rtol = 0.03 if dtype == torch.bfloat16 else 0.004 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(model(inputs), reference, atol=atol, rtol=rtol)
    for materialized in (False, True):
        if materialized:
            model.materialize_residue_weight()
        actual = model(inputs)
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        checkpoint = tmp_path / f"residue-{materialized}.pt"
        mto.save(model, checkpoint)
        restored = mto.restore(nn.Linear(width, 17, device="cuda", dtype=dtype), checkpoint)
        torch.testing.assert_close(restored(inputs), actual, rtol=0, atol=0)


@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.no_grad()
def test_mxfp4_weight_packing_across_chunks(monkeypatch, materialized, dtype):
    torch.manual_seed(31)
    cfg = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "residue_max", "residue_ratio": 0.25}
    inputs = torch.randn(2, 1025, device="cuda", dtype=dtype)
    model = mtq.quantize(
        nn.Linear(1025, 4097, device="cuda", dtype=dtype), cfg, lambda m: m(inputs)
    )
    indices = model.input_quantizer.salient_indices
    weight = torch.cat([model.weight, model.weight[:, indices]], dim=-1)
    packed, scale = MXFP4QTensor.quantize(F.pad(weight, (0, -weight.shape[-1] % 32)), 32)
    expected_weight = packed.dequantize(scale=scale, block_sizes={-1: 32})
    if materialized:
        model.materialize_residue_weight()
    input_q = model._make_residue_input_qdq(inputs, indices)
    expected = F.linear(input_q, expected_weight[..., : input_q.shape[-1]], model.bias)
    quantize = MXFP4QTensor.quantize
    quantized_chunks = []

    def bounded_quantize(weight, block_size):
        assert weight.numel() <= 2**22
        packed, scale = quantize(weight, block_size)
        quantized_chunks.append(packed.dequantize(scale=scale, block_sizes={-1: block_size}))
        return packed, scale

    monkeypatch.setattr(MXFP4QTensor, "quantize", bounded_quantize)
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    assert len(quantized_chunks) > 1
    torch.testing.assert_close(torch.cat(quantized_chunks), expected_weight, rtol=0, atol=0)
