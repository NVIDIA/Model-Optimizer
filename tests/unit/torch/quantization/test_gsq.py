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
import hashlib

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import gsq
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.gptq import GGMLGPTQHelper
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
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("untouched")
    checkpoint.with_suffix(".pt.tmp").symlink_to(sentinel)
    result.save(checkpoint)
    assert sentinel.read_text() == "untouched"
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


@pytest.mark.parametrize("algorithm_list", [False, True])
def test_iq_gsq_refuses_layerwise_resume_before_mutating_model(
    tmp_path, iq_gsq_config, algorithm_list
):
    model = torch.nn.Linear(256, 1, bias=False)
    original = copy.deepcopy(model.state_dict())
    iq_gsq_config["algorithm"]["layerwise"] = {"enable": True, "checkpoint_dir": str(tmp_path)}
    if algorithm_list:
        iq_gsq_config["algorithm"] = [iq_gsq_config["algorithm"], {"method": "imatrix"}]
    with pytest.raises(ValueError, match=r"layerwise\.checkpoint_dir"):
        quantize_iq_gsq(
            model, iq_gsq_config, lambda _: pytest.fail("calibration ran"), num_steps=(2,)
        )
    assert type(model) is torch.nn.Linear
    for name, weight in model.state_dict().items():
        assert torch.equal(weight, original[name])


def test_iq_gsq_resume_reuses_cache_and_frees_helpers_on_input_mismatch(
    tmp_path, iq_gsq_config, monkeypatch
):
    original = torch.nn.Linear(256, 1, bias=False)
    inputs = torch.randn(8, 256)
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("untouched")
    (tmp_path / f"{hashlib.sha256(b'').hexdigest()}.tmp").symlink_to(sentinel)

    def run(model):
        return quantize_iq_gsq(
            model, iq_gsq_config, lambda m: m(inputs), num_steps=(2,), checkpoint_dir=tmp_path
        )

    first = run(copy.deepcopy(original))
    assert sentinel.read_text() == "untouched"

    def unexpected_refinement(*args):
        pytest.fail("a matching checkpoint should skip refinement")

    monkeypatch.setattr(gsq, "_refine_linear", unexpected_refinement)
    repeated = run(copy.deepcopy(original))
    assert repeated.metrics == first.metrics
    for budget, payload in first.payloads[""].items():
        assert torch.equal(repeated.payloads[""][budget], payload)

    freed = []
    free = GGMLGPTQHelper.free

    def record_free(helper):
        free(helper)
        freed.append(helper)

    monkeypatch.setattr(GGMLGPTQHelper, "free", record_free)
    with pytest.raises(ValueError, match="resume inputs changed"):
        run(torch.nn.Linear(256, 1, bias=False))
    assert len(freed) == 1
    helper = freed[0]
    assert all(
        getattr(helper, name) is None
        for name in (
            "hessian",
            "weight",
            "h_inv",
            "_packed",
            "_unrounded",
            "_importance",
            "_searched",
            "_search_scales",
        )
    )


@pytest.mark.parametrize(
    "case", ["width0", "width255", "width257", "alias_conflict", "alias_match"]
)
def test_iq_gsq_activation_preflights_all_targets(case, iq_gsq_config):
    model = torch.nn.ModuleList(torch.nn.Linear(256, 1, bias=False) for _ in range(3))
    inputs = torch.randn(8, 256)
    result = quantize_iq_gsq(
        model, iq_gsq_config, lambda m: [layer(inputs) for layer in m], num_steps=(2,)
    )
    result.payloads["0"][2] = torch.zeros_like(result.payloads["0"][2])
    if case.startswith("width"):
        model[2].weight = torch.nn.Parameter(torch.randn(1, int(case[5:])))
    else:
        model[2].weight = model[1].weight
        if case == "alias_match":
            result.payloads["2"][2] = result.payloads["1"][2].clone()
    before = copy.deepcopy(model.state_dict())
    if case != "alias_match":
        with pytest.raises(ValueError):
            result.activate(model, 2)
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, before[name])
    else:
        result.activate(model, 2)
        for i, layer in enumerate(model):
            payload = result.payloads[str(i)][2]
            assert torch.equal(pinned_packed_weight(layer.weight_quantizer, "iq2_s"), payload)
            decoded = IQ_FORMAT_REGISTRY["iq2_s"].dequantize(
                payload, torch.tensor(layer.weight.shape), dtype=layer.weight.dtype
            )
            assert torch.equal(layer.weight_quantizer(layer.weight), decoded)
