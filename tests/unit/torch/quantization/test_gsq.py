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

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 4, bias=False), nn.SiLU(), nn.Linear(4, 4, bias=False)
        )

    def forward(self, inputs, *, mask):
        return self.mlp(inputs * mask)


@pytest.mark.parametrize("initializer", ["max", "gptq"])
def test_gsq_reconstructs_region_and_restores_checkpoint(tmp_path, monkeypatch, initializer):
    torch.manual_seed(3)
    model = Decoder().eval()
    pristine = copy.deepcopy(model)
    inputs = torch.randn(2, 4)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

    def forward(m):
        return m(inputs, mask=mask)

    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
        ],
        "algorithm": initializer,
    }
    baseline = mtq.quantize(copy.deepcopy(model), cfg, forward)
    with torch.no_grad():
        initial_error = (forward(baseline) - forward(pristine)).square().mean()
    cfg["algorithm"] = {
        "method": "gsq",
        "initializer": initializer,
        "steps": 2,
        "layerwise": {"enable": False},
        "optimizer_checkpoint_dir": str(tmp_path / "optimizer"),
        "data_fingerprint": "two-masked-rows",
    }
    mtq.quantize(model, cfg, forward)
    result = forward(model).detach()
    assert (result - forward(pristine)).square().mean() <= initial_error * 1.001 + 1e-9
    assert not model.mlp[0].input_quantizer.is_enabled
    mto.save(model, tmp_path / "model.pt")
    restored = mto.restore(copy.deepcopy(pristine), tmp_path / "model.pt")
    torch.testing.assert_close(forward(restored), result, atol=0, rtol=0)
    cfg["algorithm"]["resume"] = True
    resumed = mtq.quantize(copy.deepcopy(pristine), cfg, forward)
    torch.testing.assert_close(forward(resumed), result, atol=0, rtol=0)
    # A changed deployment grouping policy cannot reuse old independently learned scales.
    from modelopt.torch.quantization import gsq

    with monkeypatch.context() as patch:
        patch.setattr(gsq, "SHARED_PATTERNS", ())
        rejected = copy.deepcopy(pristine)
        with pytest.raises(ValueError):
            mtq.quantize(rejected, cfg, forward)
        for candidate, original in zip(rejected.parameters(), pristine.parameters()):
            assert torch.equal(candidate, original)
    cfg["algorithm"]["data_fingerprint"] = "different-data"
    rejected = copy.deepcopy(pristine)
    with pytest.raises(ValueError):
        mtq.quantize(rejected, cfg, forward)
    for candidate, original in zip(rejected.parameters(), pristine.parameters()):
        assert torch.equal(candidate, original)


def test_gsq_interrupted_optimizer_resumes_exactly(tmp_path, monkeypatch):
    torch.manual_seed(11)
    pristine = Decoder().eval()
    inputs = torch.randn(2, 4)
    mask = torch.ones(4)

    def forward(m):
        return m(inputs, mask=mask)

    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
        ],
        "algorithm": {
            "method": "gsq",
            "steps": 4,
            "eval_every": 1,
            "layerwise": {"enable": False},
            "data_fingerprint": "fixed-two-rows",
        },
    }
    expected = mtq.quantize(copy.deepcopy(pristine), cfg, forward)
    cfg["algorithm"]["optimizer_checkpoint_dir"] = str(tmp_path)
    real_replace = Path.replace

    def interrupt_after_save(path, target):
        result = real_replace(path, target)
        if result.name == "decoder.pt":
            raise OSError("simulated interruption after durable optimizer checkpoint")
        return result

    failed = copy.deepcopy(pristine)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupt_after_save)
        with pytest.raises(OSError):
            mtq.quantize(failed, cfg, forward)
    for parameter, original in zip(failed.parameters(), pristine.parameters()):
        assert torch.equal(parameter, original)
    cfg["algorithm"]["resume"] = True
    resumed = mtq.quantize(copy.deepcopy(pristine), cfg, forward)
    torch.testing.assert_close(forward(resumed), forward(expected), rtol=0, atol=0)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("zero", [False, True])
def test_gsq_hard_fp8_values_include_boundaries_and_zero(zero):
    model = Decoder().eval()
    values = torch.tensor(
        [
            [0, 448, -448, 1.0625],
            [-1.0625, 1.1875, -1.1875, 2],
            [0.0001, 0.0029296875, -0.0029296875, 0.5],
            [1, -1, 4, -4],
        ]
    )
    expected = torch.tensor(
        [
            [0, 448, -448, 1],
            [-1, 1.25, -1.25, 2],
            [0, 0.00390625, -0.00390625, 0.5],
            [1, -1, 4, -4],
        ]
    )
    with torch.no_grad():
        model.mlp[0].weight.copy_(torch.zeros_like(values) if zero else values)
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
        ],
        "algorithm": {"method": "gsq", "steps": 0, "layerwise": {"enable": False}},
    }
    mtq.quantize(model, cfg, lambda m: m(torch.ones(1, 4), mask=torch.ones(4)))
    torch.testing.assert_close(
        model.mlp[0].weight, torch.zeros_like(expected) if zero else expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("failure", ["soft", "initial_hard", "final_hard"])
def test_gsq_later_layer_failure_recovers_earlier_weights(failure):
    class FailingMLP(nn.Sequential):
        saw_soft_forward = False

        def forward(self, inputs):
            output = super().forward(inputs)
            soft = torch.is_grad_enabled() and output.requires_grad
            self.saw_soft_forward |= soft
            fail = (
                failure == "initial_hard"
                or (failure == "soft" and soft)
                or (failure == "final_hard" and self.saw_soft_forward and not soft)
            )
            return output * float("nan") if fail else output

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Decoder(), Decoder()])
            self.layers[1].mlp = FailingMLP(*self.layers[1].mlp)

        def forward(self, inputs, mask):
            for layer in self.layers:
                inputs = layer(inputs, mask=mask)
            return inputs

    LayerActivationCollector.register_decoder_layer_support(
        lambda m: isinstance(m, Model), lambda m: m.layers
    )
    model = Model().eval()
    model.layers[0].mlp[0].train()
    model.layers[0].mlp[0].weight.requires_grad_(False)
    training_modes = {name: module.training for name, module in model.named_modules()}
    original = copy.deepcopy(model)
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
        ],
        "algorithm": {"method": "gsq", "steps": 0 if failure == "initial_hard" else 1},
    }
    with pytest.raises(FloatingPointError):
        mtq.quantize(model, cfg, lambda m: m(torch.ones(2, 4), mask=torch.ones(4)))
    for parameter, reference in zip(model.parameters(), original.parameters()):
        assert torch.equal(parameter, reference)
        assert parameter.requires_grad is reference.requires_grad
    assert not hasattr(model, "_original_forward")
    for layer in model.layers:
        assert not hasattr(layer, "_original_forward")
        assert layer.mlp[0].weight_quantizer.amax is None
    for name, training in training_modes.items():
        assert model.get_submodule(name).training is training


@pytest.mark.parametrize("fingerprint", ["", "   "])
def test_gsq_optimizer_checkpoint_requires_initial_data_identity(tmp_path, fingerprint):
    model = Decoder().eval()
    original = copy.deepcopy(model)
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
        ],
        "algorithm": {
            "method": "gsq",
            "steps": 0,
            "layerwise": {"enable": False},
            "optimizer_checkpoint_dir": str(tmp_path),
            "data_fingerprint": fingerprint,
        },
    }
    with pytest.raises(ValueError):
        mtq.quantize(model, cfg, lambda m: m(torch.ones(1, 4), mask=torch.ones(4)))
    for parameter, reference in zip(model.parameters(), original.parameters()):
        assert torch.equal(parameter, reference)
    assert not (tmp_path / "manifest.pt").exists()


def test_gsq_refuses_quantized_embedding_before_layerwise_replay():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.layers = nn.ModuleList([Decoder()])

        def forward(self, inputs):
            return self.layers[0](self.embedding(inputs), mask=torch.ones(4))

    LayerActivationCollector.register_decoder_layer_support(
        lambda m: isinstance(m, Model), lambda m: m.layers
    )
    model = Model().train()
    model.layers[0].mlp[0].eval()
    training_modes = {name: module.training for name, module in model.named_modules()}
    original = copy.deepcopy(model)
    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
        ],
        "algorithm": {"method": "gsq", "steps": 0},
    }
    with pytest.raises(ValueError):
        mtq.quantize(model, config, lambda m: m(torch.tensor([[1, 2]])))
    for parameter, reference in zip(model.parameters(), original.parameters()):
        assert torch.equal(parameter, reference)
    assert not hasattr(model, "_original_forward")
    assert not hasattr(model.layers[0], "_original_forward")
    for name, expected in training_modes.items():
        assert model.get_submodule(name).training is expected
