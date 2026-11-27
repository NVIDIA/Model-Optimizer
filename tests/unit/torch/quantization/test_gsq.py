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

"""The GPTQ-initialized IQ refinement entrypoint and native checkpoints."""

import copy

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.gsq import IQGSQResult, quantize_iq_gsq
from modelopt.torch.quantization.ggml.registry import IQ_FORMAT_REGISTRY


@pytest.fixture
def iq_gsq_config():
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": "iq2_s", "block_sizes": {-1: 256}, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": {
            "method": "gptq",
            "block_size": 256,
            "perc_damp": 0.3,
            "importance_weighted": True,
        },
    }


def test_iq_gsq_preserves_the_paired_gptq_baseline_and_selects_a_legal_checkpoint(
    tmp_path, iq_gsq_config
):
    generator = torch.Generator().manual_seed(17)
    model = torch.nn.Linear(256, 1, bias=False)
    model.add_module("vision", torch.nn.Conv3d(1, 1, 2))
    iq_gsq_config["quant_cfg"].append({"quantizer_name": "*vision*", "enable": False})
    model.weight.data.copy_(torch.randn(1, 256, generator=generator) / 16)
    original = copy.deepcopy(model)
    baseline = copy.deepcopy(model)
    inputs = torch.randn(64, 256, generator=generator)
    inputs += 0.5 * torch.randn(64, 1, generator=generator)
    config = iq_gsq_config
    mtq.quantize(baseline, config, forward_loop=lambda m: m(inputs))
    expected_payload = pinned_packed_weight(baseline.weight_quantizer, "iq2_s").clone()
    expected_output = baseline(inputs).detach()

    result = quantize_iq_gsq(model, config, lambda m: m(inputs), num_steps=(200,), seed=17)

    assert torch.equal(pinned_packed_weight(model.weight_quantizer, "iq2_s"), expected_payload)
    torch.testing.assert_close(model(inputs), expected_output)
    result.activate(model, 200)
    baseline_error = (expected_output - original(inputs)).square().sum()
    refined_error = (model(inputs) - original(inputs)).square().sum()
    torch.testing.assert_close(
        torch.tensor(result.metrics[""]["reconstruction_error"][0]),
        baseline_error / expected_output.numel(),
        rtol=1e-5,
        atol=1e-7,
    )
    assert refined_error < baseline_error
    refined_payload = pinned_packed_weight(model.weight_quantizer, "iq2_s")
    decoded = IQ_FORMAT_REGISTRY["iq2_s"].dequantize(
        refined_payload, torch.tensor(model.weight.shape), dtype=model.weight.dtype
    )
    assert torch.equal(model.weight_quantizer(model.weight), decoded)
    # The first two bytes hold only d: a change after them proves categorical refinement.
    assert not torch.equal(refined_payload[..., 2:], expected_payload[..., 2:])
    assert torch.isfinite(model(inputs)).all()

    checkpoint = tmp_path / "iq_gsq.pt"
    result.save(checkpoint)
    restored = IQGSQResult.load(checkpoint)
    refined_output = model(inputs).detach()
    restored.activate(model, 0)
    assert torch.equal(pinned_packed_weight(model.weight_quantizer, "iq2_s"), expected_payload)
    torch.testing.assert_close(model(inputs), expected_output)
    restored.activate(model, 200)
    assert torch.equal(
        IQ_FORMAT_REGISTRY["iq2_s"].pack(model.weight, model.weight_quantizer), refined_payload
    )
    torch.testing.assert_close(model(inputs), refined_output)


@pytest.mark.parametrize("layout", ["weight", "projections"])
def test_iq_gsq_rejects_expert_weights_before_conversion_or_calibration(layout, iq_gsq_config):
    model = torch.nn.Sequential(torch.nn.Linear(256, 1, bias=False), torch.nn.Module())
    experts = model[1]
    if layout == "weight":
        experts.weight = torch.nn.Parameter(torch.randn(2, 1, 256))
    else:
        experts.num_experts = 2
        experts.gate_up_proj = torch.nn.Parameter(torch.randn(2, 2, 256))
        experts.down_proj = torch.nn.Parameter(torch.randn(2, 256, 1))
    original = copy.deepcopy(model.state_dict())
    classes = [type(module) for module in model.modules()]
    calibrated = False

    def calibrate(_model):
        nonlocal calibrated
        calibrated = True

    with pytest.raises(ValueError, match="dense 2-D"):
        quantize_iq_gsq(model, iq_gsq_config, calibrate, num_steps=(2,))

    assert not calibrated
    assert [type(module) for module in model.modules()] == classes
    assert model.state_dict().keys() == original.keys()
    for name, weight in model.state_dict().items():
        assert torch.equal(weight, original[name])
