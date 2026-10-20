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
from modelopt.torch.quantization.nn import ResidueQuantLinear


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
    assert torch.equal(split.input_quantizer.salient_indices, torch.arange(0, 64, 8))
    assert torch.equal(joined.input_quantizer.salient_indices, torch.arange(1, 64, 8))
    assert split.weight.shape == (8, 64)


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.25, 1.0])
def test_partial_selection_block(ratio):
    model = mtq.quantize(
        nn.Linear(5, 3),
        _config(residue_ratio=ratio),
        lambda m: m(torch.tensor([[1.0, 2, 3, 4, 5]])),
    )
    expected = {0.0: [], 0.01: [], 0.25: [3, 4], 1.0: [0, 1, 2, 3, 4]}[ratio]
    assert model.input_quantizer.salient_indices.tolist() == expected


@pytest.mark.parametrize("fmt", ["NVFP4", "MXFP4"])
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("ratio", [0.0, 0.25, 1.0])
def test_standard_checkpoint_roundtrip(tmp_path, fmt, materialized, ratio):
    x = torch.randn(2, 64)
    model = mtq.quantize(nn.Linear(64, 3), _config(fmt, residue_ratio=ratio), lambda m: m(x))
    if materialized:
        model.materialize_residue_weight()
    checkpoint = tmp_path / "residue.pt"
    mto.save(model, checkpoint)
    restored = mto.restore(nn.Linear(64, 3), checkpoint)
    assert isinstance(restored, ResidueQuantLinear)
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
    original_classes = [type(m) for m in model]
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [
        (q._disabled, q._if_quant, q._if_calib, q.axis)
        for q in model.modules()
        if isinstance(q, mtq.nn.TensorQuantizer)
    ]

    def fail(m):
        if failure == "empty":
            return
        m(torch.full((2, 64), float("nan")) if failure == "nonfinite" else torch.ones(2, 64))
        if failure == "callback":
            raise RuntimeError("calibration interrupted")

    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        mtq.calibrate(model, {"method": "residue_max"}, fail)
    assert [type(m) for m in model] == original_classes
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
    assert model.input_quantizer.salient_indices.tolist() == list(range(7, 64, 8))
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key


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
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("exporter", [export_hf_checkpoint, _export_transformers_checkpoint])
def test_deployment_export_rejected_without_mutation(tmp_path, fmt, materialized, exporter):
    model = mtq.quantize(
        nn.Sequential(nn.Linear(64, 8)), _config(fmt), lambda m: m(torch.ones(2, 64))
    )
    if materialized:
        model[0].materialize_residue_weight()
    before = deepcopy(model.state_dict())
    attributes = set(model.__dict__)
    parameter = model[0].weight
    export_dir = tmp_path / "export"
    with pytest.raises(NotImplementedError, match=r"Residue.*export"):
        exporter(model, dtype=torch.float32, export_dir=export_dir)
    assert not export_dir.exists()
    assert set(model.__dict__) == attributes
    assert model[0].weight is parameter
    assert model.state_dict().keys() == before.keys()
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key


def test_diffusion_pipeline_export_rejected_before_writing(tmp_path):
    diffusers = pytest.importorskip("diffusers")

    class ResiduePipeline(diffusers.DiffusionPipeline):
        def __init__(self, transformer):
            super().__init__()
            self.register_modules(transformer=transformer)

    model = mtq.quantize(nn.Linear(64, 8), _config(), lambda m: m(torch.ones(2, 64)))
    pipeline = ResiduePipeline(model)
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
