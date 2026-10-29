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

"""Frozen-mixture calibration; the native LP solver spawns a subprocess."""

import copy

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.recipe import (
    ModelOptAutoQuantizeRecipe,
    freeze_auto_quantize_assignment,
    plan_auto_quantize,
)
from modelopt.torch.quantization._calibration_workers import _digest
from modelopt.torch.quantization._selected_calibration import _SelectedLayerCalibrator
from modelopt.torch.quantization._selected_model import _selected_model
from modelopt.torch.quantization._selected_search import (
    _load_selected_search,
    _selected_search_binding,
)
from modelopt.torch.quantization.mode import MaxCalibrateModeDescriptor, MseCalibrateModeDescriptor
from modelopt.torch.quantization.model_calib import layerwise_calibrate
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils import get_quantizer_state_dict
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16)) for _ in range(3)]
        )

    def forward(self, inputs):
        if hasattr(self, "pool"):
            inputs = self.pool(inputs)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


@pytest.mark.parametrize(
    "invalid",
    [None, "source", "identity", "buffers", "mapping", "outside", "outside_fixed", "empty"],
)
def test_frozen_mixture_preserves_fixed_states_and_recalibrates_selected(monkeypatch, invalid):
    monkeypatch.setattr(LayerActivationCollector, "_decoder_layer_support", [])
    LayerActivationCollector.register_decoder_layer_support(
        lambda model: isinstance(model, _Model), lambda model: model.layers
    )
    # The internal constructor is tested before the public automatic capability is enabled.
    for mode in (MaxCalibrateModeDescriptor, MseCalibrateModeDescriptor):
        monkeypatch.setattr(mode, "_auto_quantize_layerwise_estimator", "source_activation_v1")
    torch.manual_seed(42)
    source = _Model().eval()
    if invalid == "outside":
        source.pool = nn.AvgPool1d(1).eval()
    elif invalid == "outside_fixed":
        source.pool = nn.Linear(16, 16).eval()
    data = [torch.randn(2, 16) for _ in range(2)]
    fixed = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    selected = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    selected["quant_cfg"] = [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 3, "axis": 0}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 4, "axis": None}},
    ]
    selected["algorithm"] = [
        "max",
        {
            "method": "mse",
            "step_size": 0.25,
            "start_multiplier": 0.5,
            "stop_multiplier": 1.0,
            "layerwise": {"enable": True},
        },
    ]
    if invalid == "empty":
        for entry in selected["quant_cfg"]:
            if "cfg" in entry:
                entry["parent_class"] = "nn.Conv1d"
    no_quant = {"quant_cfg": [{"quantizer_name": "*", "enable": False}]}
    spaces = [
        {
            "module_name_patterns": ["layers.1.*"],
            "candidate_formats": [selected],
            "allow_no_quant": False,
        },
        {"module_name_patterns": ["layers.2.0"], "candidate_formats": [no_quant]},
    ]
    recipe = ModelOptAutoQuantizeRecipe(
        quantize=fixed,
        auto_quantize={"constraints": {"effective_bits": 8}, "module_search_spaces": spaces},
    )
    _, state = mtq.auto_quantize(
        copy.deepcopy(source),
        constraints={"effective_bits": 8},
        fixed_quantization_config=fixed,
        module_search_spaces=[
            {
                **{k: v for k, v in space.items() if k != "candidate_formats"},
                "quantization_formats": space["candidate_formats"],
            }
            for space in spaces
        ],
        data_loader=data,
        forward_step=lambda model, batch: model(batch),
        loss_func=lambda output, batch: output.square().mean(),
        num_calib_steps=2,
        num_score_steps=1,
    )
    identity = {
        "source": _digest(dict(source.named_parameters())),
        "inputs": _digest(data),
        "code": "fixture-code",
    }
    plan = plan_auto_quantize(recipe)
    assignment = freeze_auto_quantize_assignment(state, plan, identity)
    actual = copy.deepcopy(source)
    if invalid == "source":
        with torch.no_grad():
            actual.layers[0][0].weight.add_(1)
    elif invalid == "identity":
        identity = dict(identity, inputs="different-inputs")
    elif invalid == "buffers":
        fixed_recipe = state["best"]["recipe"]["layers.0.0.quant_recipe"]
        state["quantizer_states"][fixed_recipe]["state_dict"]["layers.0.0.weight_quantizer"].clear()
    elif invalid == "mapping":
        fixed_recipe = state["best"]["recipe"]["layers.0.0.quant_recipe"]
        for field in ("metadata", "state_dict"):
            payload = state["quantizer_states"][fixed_recipe][field]
            payload["layers.0.0.weight_quantizer.999"] = payload.pop("layers.0.0.weight_quantizer")
    if invalid:
        before = copy.deepcopy(actual.state_dict())
        with pytest.raises(ValueError):
            _selected_model(actual, assignment, plan, state, identity)
        assert not any(isinstance(module, TensorQuantizer) for module in actual.modules())
        torch.testing.assert_close(actual.state_dict(), before, rtol=0, atol=0)
        return

    dispatch = _selected_model(actual, assignment, plan, state, identity)
    assert not actual.layers[2][0].weight_quantizer.is_enabled
    for index in (0, 2):
        for leaf in (0, 2):
            if (index, leaf) == (2, 0):
                continue
            module = actual.layers[index][leaf]
            assert module.weight_quantizer.is_enabled
            assert hasattr(module.weight_quantizer, "_amax")
    for leaf in (0, 2):
        quantizer = actual.layers[1][leaf].weight_quantizer
        assert quantizer.is_enabled and quantizer.num_bits == 3 and not hasattr(quantizer, "_amax")

    config = copy.deepcopy(fixed)
    for name in ("layers.1.0", "layers.1.2"):
        config["quant_cfg"].extend(
            {**entry, "quantizer_name": name + "." + entry["quantizer_name"].removeprefix("*")}
            for entry in selected["quant_cfg"]
            if "cfg" in entry
        )
    config["quant_cfg"].append({"quantizer_name": "layers.2.0.*", "enable": False})
    config["algorithm"] = None
    reference = mtq.quantize(copy.deepcopy(source), config)

    def forward(model):
        for batch in data:
            model(batch)

    with mtq.set_quantizer_by_cfg_context(
        reference, [{"quantizer_name": "layers.1.*", "enable": False}]
    ):
        mtq.calibrate(reference, "max", forward)
    groups = [(["layers.1.0"], selected["algorithm"]), (["layers.1.2"], selected["algorithm"])]
    expected = _SelectedLayerCalibrator(reference, groups)
    layerwise_calibrate(reference, forward, expected, **expected.layerwise_config)
    layerwise_calibrate(actual, forward, dispatch, **dispatch.layerwise_config)
    torch.testing.assert_close(actual.state_dict(), reference.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(actual(data[0]), reference(data[0]), rtol=0, atol=0)
    assert assignment.digest == freeze_auto_quantize_assignment(state, plan, identity).digest


@pytest.mark.usefixtures("skip_on_windows")
def test_frozen_mixture_from_source_estimation_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(LayerActivationCollector, "_decoder_layer_support", [])
    LayerActivationCollector.register_decoder_layer_support(
        lambda model: isinstance(model, _Model), lambda model: model.layers
    )
    monkeypatch.setattr(
        MaxCalibrateModeDescriptor, "_auto_quantize_layerwise_estimator", "source_activation_v1"
    )
    torch.manual_seed(42)
    source = _Model().eval()
    data = [torch.randn(2, 16) for _ in range(2)]
    formats = [
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": bits, "axis": 0}},
                {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": bits, "axis": None}},
            ],
            "algorithm": {
                "method": "max",
                "layerwise": {"enable": True, "get_qdq_activations_from_prev_layer": True},
            },
        }
        for bits in (3, 4)
    ]
    spaces = [
        {
            "module_name_patterns": ["layers.*"],
            "candidate_formats": formats,
            "allow_no_quant": False,
        }
    ]
    no_quant = {"quant_cfg": [{"quantizer_name": "*", "enable": False}]}
    options = {
        "constraints": {"effective_bits": 4},
        "quantization_formats": [no_quant],
        "module_search_spaces": [
            {
                "module_name_patterns": ["layers.*"],
                "quantization_formats": formats,
                "allow_no_quant": False,
            }
        ],
        "data_loader": data,
        "forward_step": lambda model, batch: model(batch),
        "loss_func": lambda output, batch: output.square().mean(),
        "num_calib_steps": 2,
        "num_score_steps": 1,
    }
    limits = {
        "directory": tmp_path / "candidates",
        "source_activation_bytes": 2**20,
        "collect_scores": True,
    }
    unconverted = copy.deepcopy(source).train()
    with pytest.raises(ValueError):
        mtq.auto_quantize(unconverted, **options, calibration=dict(limits, candidate=0))
    assert not any(isinstance(m, TensorQuantizer) for m in unconverted.modules())
    with pytest.raises(ValueError):
        mtq.auto_quantize(
            copy.deepcopy(source),
            **options,
            calibration=dict(limits, candidate=0, source_activation_bytes=1),
        )
    assert not list(tmp_path.rglob("*.pth"))

    for index in (0, 1, 0):
        _, state = mtq.auto_quantize(
            copy.deepcopy(source), **options, calibration=dict(limits, candidate=index)
        )
        ((candidate, entry),) = state["quantizer_states"].items()
        reference = mtq.quantize(
            copy.deepcopy(source), dict(candidate.config.model_dump(), algorithm=None)
        )
        inputs = data
        for original_layer, layer in zip(source.layers, reference.layers):
            with torch.no_grad():
                following = [original_layer(batch) for batch in inputs]
            mtq.calibrate(layer, "max", lambda m: [m(batch) for batch in inputs])
            inputs = following
        torch.testing.assert_close(
            get_quantizer_state_dict(reference), entry["state_dict"], rtol=0, atol=0
        )
    artifacts = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob("rank0.pth")}
    checkpoint = tmp_path / "search"
    _, state = mtq.auto_quantize(
        copy.deepcopy(source), **options, checkpoint=str(checkpoint), calibration=limits
    )
    assert state["candidate_stats"]
    identity = state["calibration_provenance"]["contract"]["identity"]
    assert (
        state["calibration_provenance"]["contract"]["layerwise_estimator"] == "source_activation_v1"
    )
    plan = plan_auto_quantize(
        ModelOptAutoQuantizeRecipe(
            auto_quantize={
                "constraints": {"effective_bits": 4},
                "candidate_formats": [no_quant],
                "module_search_spaces": spaces,
            }
        )
    )
    assignment = freeze_auto_quantize_assignment(state, plan, identity)
    actual = copy.deepcopy(source)
    dispatch = _selected_model(actual, assignment, plan, state, identity)
    binding = _selected_search_binding(checkpoint, assignment, 2**24)
    checkpoint_bytes = (checkpoint / "rank0.pth").read_bytes()
    recovered = _load_selected_search(
        checkpoint, binding, assignment, plan, identity, (2, 1), 2**24
    )
    for changed_binding, changed_identity, steps, limit in [
        (dict(binding, assignment="different"), identity, (2, 1), 2**24),
        (binding, dict(identity, inputs="different"), (2, 1), 2**24),
        (binding, identity, (1, 1), 2**24),
        (binding, identity, (2, 1), 1),
    ]:
        with pytest.raises(ValueError):
            _load_selected_search(
                checkpoint, changed_binding, assignment, plan, changed_identity, steps, limit
            )
    assert (checkpoint / "rank0.pth").read_bytes() == checkpoint_bytes
    restored = copy.deepcopy(source)
    recovered_dispatch = _selected_model(restored, assignment, plan, recovered, identity)
    layerwise_calibrate(
        restored,
        lambda m: [m(batch) for batch in data],
        recovered_dispatch,
        **recovered_dispatch.layerwise_config,
    )
    layerwise_calibrate(
        actual, lambda m: [m(batch) for batch in data], dispatch, **dispatch.layerwise_config
    )
    torch.testing.assert_close(restored.state_dict(), actual.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored(data[0]), actual(data[0]), rtol=0, atol=0)
    assert torch.isfinite(actual(data[0])).all()
    generation = checkpoint.resolve()
    with pytest.raises(ValueError):
        mtq.auto_quantize(
            copy.deepcopy(source),
            **options,
            checkpoint=str(checkpoint),
            calibration={k: v for k, v in limits.items() if k != "source_activation_bytes"},
        )
    assert checkpoint.resolve() == generation
    assert all((p.stat().st_mtime_ns, p.read_bytes()) == value for p, value in artifacts.items())
