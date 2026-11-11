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
from modelopt.torch.export.residue_export import collect_residue_export_metadata
from modelopt.torch.quantization.residue import get_salient_indices
from modelopt.torch.quantization.tensor_quant import dynamic_block_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for FP4 QDQ"
)


def _reference(model, inputs):
    iq, wq = model.input_quantizer, model.weight_quantizer
    indices = iq._salient_indices
    is_mx = iq.block_sizes["scale_bits"] == (8, 0)
    x = inputs if iq.pre_quant_scale is None else inputs * iq.pre_quant_scale
    if is_mx and indices.numel():
        x = x.float()
    # TensorQuantizer makes inputs contiguous before calling the native FP4 kernel.
    x = x.contiguous()
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
    wq_value = dynamic_block_quant(
        weight_ext, block, wq.amax, None, (2, 1), wq.block_sizes["scale_bits"]
    )
    # Separate GEMMs check the paired residual formula independently of concatenated Linear.
    result = F.linear(xq.to(inputs.dtype), wq_value[..., : inputs.shape[-1]], model.bias)
    if indices.numel():
        result += F.linear(rq.to(inputs.dtype), wq_value[..., inputs.shape[-1] :])
    return result


@pytest.mark.parametrize("fmt", ["NVFP4", "MXFP4"])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 1.0])
def test_residue_qdq_backward_matches_reference(fmt, ratio):
    torch.manual_seed(31)
    cfg = deepcopy(getattr(mtq, f"{fmt}_DEFAULT_CFG"))
    cfg["algorithm"] = {"method": "residue_max", "residue_ratio": ratio}
    inputs = torch.randn(2, 64, 3, device="cuda").transpose(1, 2).requires_grad_()
    model = mtq.quantize(nn.Linear(64, 17, device="cuda"), cfg, lambda m: m(inputs))
    reference = deepcopy(model)
    reference_inputs = inputs.detach().clone().requires_grad_()
    parameter = model.weight
    before = parameter.detach().clone()
    indices = model.input_quantizer._salient_indices.clone()
    output = model(inputs)
    expected = _reference(reference, reference_inputs)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    expected.backward(grad_output)
    for actual, target in (
        (inputs.grad, reference_inputs.grad),
        (model.weight.grad, reference.weight.grad),
        (model.bias.grad, reference.bias.grad),
    ):
        assert actual is not None and torch.isfinite(actual).all()
        assert actual.count_nonzero() > 0
        torch.testing.assert_close(actual, target, atol=1e-5, rtol=1e-5)
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert model.weight is parameter
    assert not torch.equal(model.weight, before)
    assert torch.equal(model.input_quantizer._salient_indices, indices)


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
    checkpoint = tmp_path / "residue.pt"
    mto.save(model, checkpoint)
    restored = mto.restore(nn.Linear(width, 17, device="cuda", dtype=dtype), checkpoint)
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.no_grad()
def test_mxfp4_native_weight_qdq(dtype):
    torch.manual_seed(31)
    cfg = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    cfg["algorithm"] = {"method": "residue_max", "residue_ratio": 0.25}
    inputs = torch.randn(2, 1025, device="cuda", dtype=dtype)
    model = mtq.quantize(
        nn.Linear(1025, 4097, device="cuda", dtype=dtype), cfg, lambda m: m(inputs)
    )
    indices = model.input_quantizer._salient_indices
    weight = torch.cat([model.weight, model.weight[:, indices]], dim=-1)
    expected_weight = dynamic_block_quant(weight, 32, None, None, (2, 1), (8, 0))
    input_q = model._make_residue_input_qdq(inputs, indices)
    expected = F.linear(input_q, expected_weight[..., : input_q.shape[-1]], model.bias)
    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("fmt", "metric"),
    [("NVFP4", "activation_max"), ("MXFP4", "activation_max"), ("NVFP4", "nvfp4_error_weighted")],
)
@pytest.mark.parametrize("patterns", [None, [r"(?:q_proj|k_proj|v_proj)"], []])
@torch.no_grad()
def test_residue_shared_channels_restore_and_export(tmp_path, fmt, metric, patterns):
    torch.manual_seed(71)

    def make_model():
        return nn.ModuleDict(
            {name: nn.Linear(128, 32, device="cuda") for name in ("q_proj", "k_proj", "v_proj")}
        )

    model = make_model()
    inputs = torch.randn(3, 128, device="cuda")
    if metric == "activation_max":
        scores = [inputs.abs().mean(0)] * 3
    else:
        q = dynamic_block_quant(inputs, 16, inputs.abs().amax(), None, (2, 1), (4, 3))
        error_sq = (inputs - q).square().sum(0)
        scores = [(error_sq * layer.weight.square().sum(0)).sqrt() for layer in model.values()]
    expected = (
        get_salient_indices(torch.stack(scores).square().sum(0).sqrt(), 8, 0.25)
        if metric == "nvfp4_error_weighted"
        else get_salient_indices(scores[0], 8, 0.25)
    )
    cfg = deepcopy(getattr(mtq, f"{fmt}_DEFAULT_CFG"))
    cfg["algorithm"] = {
        "method": "residue_max",
        "salience_metric": metric,
        "shared_input_patterns": patterns,
    }
    mtq.quantize(model, cfg, lambda m: [layer(inputs) for layer in m.values()])
    for layer, score in zip(model.values(), scores):
        indices = expected if patterns != [] else get_salient_indices(score, 8, 0.25)
        assert torch.equal(layer.input_quantizer._salient_indices, indices)
    before = [layer(inputs) for layer in model.values()]
    mto.save(model, tmp_path / "model.pt")
    restored = mto.restore(make_model(), tmp_path / "model.pt")
    for actual, expected_output in zip(restored.values(), before):
        torch.testing.assert_close(actual(inputs), expected_output, atol=0, rtol=0)
        assert bool(actual._residue_group) == (patterns != [])
    collect_residue_export_metadata(restored)
    indices = restored["v_proj"].input_quantizer._salient_indices
    restored["v_proj"].input_quantizer._salient_indices = (
        (indices // 8 * 8 + (indices % 8 + 1) % 8).sort().values
    )
    if patterns != []:
        with pytest.raises(ValueError, match="declared residue input group"):
            collect_residue_export_metadata(restored)
    else:
        collect_residue_export_metadata(restored)
