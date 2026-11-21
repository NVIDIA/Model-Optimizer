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
from pydantic import ValidationError

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import (
    _export_transformers_checkpoint,
    export_hf_checkpoint,
)
from modelopt.torch.quantization.conversion import restore_residue_model
from modelopt.torch.quantization.nn import ResidueQuantLinear, TensorQuantizer


def _config(fmt="NVFP4", **algorithm):
    cfg = deepcopy(getattr(mtq, f"{fmt}_DEFAULT_CFG"))
    cfg["algorithm"] = {"method": "residue_max", **algorithm}
    return cfg


@pytest.mark.parametrize("fmt", ["NVFP4", "MXFP4"])
def test_equal_batch_weighting(fmt):
    first = torch.tensor([[10.0, 0, 0, 0, 0, 0, 0, 0]]).repeat(1, 8)
    second = torch.tensor([[0.0, 8, 0, 0, 0, 0, 0, 0]]).repeat(9, 8)
    split = mtq.quantize(
        nn.Linear(64, 8),
        _config(fmt, residue_ratio=0.125),
        lambda m: [m(first), m(second)],
    )
    joined = mtq.quantize(
        nn.Linear(64, 8),
        _config(fmt, residue_ratio=0.125),
        lambda m: m(torch.cat([first, second])),
    )
    assert isinstance(split, ResidueQuantLinear)
    assert torch.equal(split.input_quantizer._salient_indices, torch.arange(0, 64, 8))
    assert torch.equal(joined.input_quantizer._salient_indices, torch.arange(1, 64, 8))
    assert split.weight.shape == (8, 64)


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.25, 1.0])
def test_partial_selection_block(ratio):
    model = mtq.quantize(
        nn.Linear(5, 3),
        _config(residue_ratio=ratio),
        lambda m: m(torch.tensor([[1.0, 2, 3, 4, 5]])),
    )
    expected = {0.0: [], 0.01: [], 0.25: [3, 4], 1.0: [0, 1, 2, 3, 4]}[ratio]
    assert model.input_quantizer._salient_indices.tolist() == expected


@pytest.mark.parametrize("fmt", ["NVFP4", "MXFP4"])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 1.0])
def test_standard_checkpoint_roundtrip(tmp_path, fmt, ratio):
    x = torch.randn(2, 64)
    model = mtq.quantize(nn.Linear(64, 3), _config(fmt, residue_ratio=ratio), lambda m: m(x))
    checkpoint = tmp_path / "residue.pt"
    mto.save(model, checkpoint)
    restored = mto.restore(nn.Linear(64, 3), checkpoint)
    assert isinstance(restored, ResidueQuantLinear)
    assert type(restored.input_quantizer) is TensorQuantizer
    assert type(restored.weight_quantizer) is TensorQuantizer
    assert restored.in_features == model.in_features
    assert restored.weight.shape == model.weight.shape
    assert restored.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value), key
    assert restored.input_quantizer.is_enabled
    assert restored.weight_quantizer.is_enabled


@pytest.mark.parametrize("failure", ["callback", "empty", "nonfinite"])
def test_failed_calibration_restores_state(failure):
    model = nn.Sequential(nn.Linear(64, 8), nn.Linear(8, 4))
    cfg = _config()
    cfg["algorithm"] = None
    mtq.quantize(model, cfg)
    quantizers = [q for q in model.modules() if isinstance(q, TensorQuantizer)]
    for q in quantizers:
        q._calibrator.collect(torch.ones(2, 64))
    observer_amax = [q._calibrator._calib_amax.clone() for q in quantizers]
    model[0].output_quantizer.register_buffer("_bias_value", torch.tensor(2.0))
    original_classes = [type(m) for m in model]
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [
        (q._disabled, q._if_quant, q._if_calib, q.axis)
        for q in model.modules()
        if isinstance(q, mtq.nn.TensorQuantizer)
    ]

    def fail(m):
        model[0].output_quantizer._bias_value.fill_(9.0)
        if failure == "empty":
            return
        m(torch.full((2, 64), float("nan")) if failure == "nonfinite" else torch.ones(2, 64))
        if failure == "callback":
            raise RuntimeError("calibration interrupted")

    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        mtq.calibrate(model, {"method": "residue_max"}, fail)
    assert [type(m) for m in model] == original_classes
    for q, expected in zip(quantizers, observer_amax):
        assert torch.equal(q._calibrator._calib_amax, expected)
    assert all(not m._forward_pre_hooks for m in model)
    for key, value in original_state.items():
        assert torch.equal(model.state_dict()[key], value)
    assert model.state_dict().keys() == original_state.keys()
    assert flags == [
        (q._disabled, q._if_quant, q._if_calib, q.axis)
        for q in model.modules()
        if isinstance(q, mtq.nn.TensorQuantizer)
    ]
    assert all(
        not hasattr(m, "residue_max") and not hasattr(m, "_forward_no_residuemax") for m in model
    )
    mtq.calibrate(model, {"method": "residue_max"}, lambda m: m(torch.ones(2, 64)))
    assert all(isinstance(m, ResidueQuantLinear) for m in model)


def test_materialized_checkpoint_rejected_before_conversion():
    model = nn.Linear(64, 8)
    parameter = model.weight
    metadata = {"residue_modules": {"": {"original_in_features": 64, "in_features": 80}}}
    with pytest.raises(NotImplementedError, match="Materialized"):
        restore_residue_model(model, {}, metadata)
    assert type(model) is nn.Linear
    assert model.weight is parameter
    assert model.in_features == 64


@pytest.mark.parametrize("keyword_input", [False, True])
def test_collection_preserves_forward(keyword_input):
    model = nn.Linear(64, 8)
    cfg = _config(residue_ratio=0.125)
    cfg["algorithm"] = None
    mtq.quantize(model, cfg)
    original_forward = model.forward
    inputs = torch.arange(1, 65).float().repeat(3, 1).t()
    inputs = inputs.contiguous().t()
    assert not inputs.is_contiguous()
    expected = torch.nn.functional.linear(inputs, model.weight, model.bias)

    def collect(m):
        assert m.forward == original_forward
        output = m(input=inputs) if keyword_input else m(inputs)
        torch.testing.assert_close(output, expected)

    mtq.calibrate(model, {"method": "residue_max", "residue_ratio": 0.125}, collect)
    assert model.input_quantizer._salient_indices.tolist() == list(range(7, 64, 8))
    assert not model._forward_pre_hooks
    assert "forward" not in model.__dict__


def test_recalibration_preserves_awq_state():
    model = mtq.quantize(nn.Linear(64, 8), _config(), lambda m: m(torch.ones(2, 64)))
    model.input_quantizer.pre_quant_scale = torch.linspace(0.5, 2, 64)
    model.weight_quantizer.pre_quant_scale = model.input_quantizer.pre_quant_scale.reciprocal()
    before = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if "salient" not in k and k != "weight_quantizer._amax"
    }
    inputs = torch.arange(1, 65).float().unsqueeze(0)
    mtq.calibrate(model, {"method": "residue_max", "residue_ratio": 0.125}, lambda m: m(inputs))
    assert model.input_quantizer._salient_indices.tolist() == list(range(7, 64, 8))
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key


def test_recalibration_updates_input_amax():
    inputs = torch.randn(4, 64)
    model = mtq.quantize(nn.Linear(64, 8), _config(), lambda m: m(inputs))
    mtq.calibrate(model, {"method": "residue_max"}, lambda m: m(10 * inputs))
    torch.testing.assert_close(model.input_quantizer.amax, 10 * inputs.abs().max())


def test_ratio_selecting_no_channels_warns():
    with pytest.warns(UserWarning, match="selects no channels"):
        mtq.quantize(nn.Linear(64, 8), _config(residue_ratio=0.1), lambda m: m(torch.ones(2, 64)))


def test_fused_moe_experts_keep_max_calibration(moe):
    inputs = torch.randn(12, moe.mlp.shared_expert.in_features)
    with pytest.warns(UserWarning, match="residue_max does not apply to fused MoE experts"):
        model = mtq.quantize(moe, _config(), lambda m: m(inputs))
    shared, experts = model.mlp.shared_expert, model.mlp.experts
    assert isinstance(shared, ResidueQuantLinear)
    assert shared.input_quantizer._salient_indices.numel() == shared.in_features // 4
    for quantizer in (
        experts.gate_up_proj_input_quantizer,
        experts.down_proj_input_quantizer,
        *experts.gate_up_proj_weight_quantizers,
        *experts.down_proj_weight_quantizers,
    ):
        assert quantizer.amax is not None
        assert not hasattr(quantizer, "_salient_indices")


@pytest.mark.parametrize("ratio", [-0.1, 1.01, float("nan"), float("inf")])
def test_invalid_ratio(ratio):
    with pytest.raises(ValidationError):
        mtq.ResidueMaxCalibConfig(residue_ratio=ratio)


@pytest.mark.parametrize(
    ("algorithm", "match"),
    [
        ({"layerwise": {"enable": True}}, "does not support layerwise"),
        ({"salience_metric": "nvfp4_error_weighted"}, "requires NVFP4"),
    ],
)
def test_unsupported_configuration(algorithm, match):
    with pytest.raises(ValueError, match=match):
        mtq.quantize(
            nn.Linear(64, 8), _config("MXFP4", **algorithm), lambda m: m(torch.ones(2, 64))
        )


def test_fold_and_compress_rejected_without_mutation():
    model = mtq.quantize(nn.Linear(64, 8), _config(), lambda m: m(torch.ones(2, 64)))
    before = deepcopy(model.state_dict())
    with pytest.raises(NotImplementedError, match="folding"):
        mtq.fold_weight(model)
    with pytest.raises(AssertionError, match="Cannot add"):
        mtq.compress(model)
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key


@pytest.mark.parametrize("fmt", ["NVFP4", "MXFP4"])
@pytest.mark.parametrize("exporter", [export_hf_checkpoint, _export_transformers_checkpoint])
def test_unaligned_deployment_export_rejected_without_mutation(tmp_path, fmt, exporter):
    model = mtq.quantize(
        nn.Sequential(nn.Linear(64, 8)), _config(fmt), lambda m: m(torch.ones(2, 64))
    )
    before = deepcopy(model.state_dict())
    attributes = set(model.__dict__)
    parameter = model[0].weight
    export_dir = tmp_path / "export"
    with pytest.raises(ValueError, match="input width divisible by 128"):
        exporter(model, dtype=torch.float32, export_dir=export_dir)
    assert not export_dir.exists()
    assert set(model.__dict__) == attributes
    assert model[0].weight is parameter
    assert model.state_dict().keys() == before.keys()
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key


@pytest.mark.parametrize("standalone", [False, True])
def test_diffusion_pipeline_export_rejected_before_writing(tmp_path, standalone):
    """Reject residue pipelines and aligned components without changing their state."""
    diffusers = pytest.importorskip("diffusers")

    class ResiduePipeline(diffusers.DiffusionPipeline):
        def __init__(self, transformer):
            super().__init__()
            self.register_modules(transformer=transformer)

    class ResidueComponent(diffusers.ModelMixin):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(128, 32)

        def forward(self, inputs):
            return self.proj(inputs)

    model = mtq.quantize(ResidueComponent(), _config(), lambda m: m(torch.ones(2, 128)))
    pipeline = model if standalone else ResiduePipeline(model)
    before = deepcopy(model.state_dict())
    attributes = set(pipeline.__dict__)
    export_dir = tmp_path / "export"
    with pytest.raises(NotImplementedError, match=r"Residue.*export"):
        export_hf_checkpoint(pipeline, export_dir=export_dir)
    assert not export_dir.exists()
    assert set(pipeline.__dict__) == attributes
    assert model.state_dict().keys() == before.keys()
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key


def test_registration_is_opt_in_and_excluded_layers_survive_restore(tmp_path):
    cfg = _config()
    cfg["quant_cfg"].append({"quantizer_name": "1.*", "enable": False})
    model = mtq.quantize(
        nn.Sequential(nn.Linear(64, 8), nn.Linear(8, 4)),
        cfg,
        lambda m: m(torch.ones(2, 64)),
    )
    checkpoint = tmp_path / "mixed.pt"
    mto.save(model, checkpoint)
    restored = mto.restore(nn.Sequential(nn.Linear(64, 8), nn.Linear(8, 4)), checkpoint)
    assert isinstance(restored[0], ResidueQuantLinear)
    assert not isinstance(restored[1], ResidueQuantLinear)
    assert not restored[1].weight_quantizer.is_enabled
    assert not restored[1].input_quantizer.is_enabled
    ordinary = mtq.quantize(nn.Linear(64, 8), mtq.INT8_DEFAULT_CFG, lambda m: m(torch.ones(2, 64)))
    assert not isinstance(ordinary, ResidueQuantLinear)
    assert type(ordinary.input_quantizer) is mtq.nn.TensorQuantizer
