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

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.export.registry import ExportContext
from modelopt.torch.export.residue_export import (
    collect_residue_export_metadata,
    export_residue_weight,
)
from modelopt.torch.export.unified_export_hf import (
    _dispatch_export_handler,
    _export_quantized_weight,
    _process_quantized_modules,
)


@pytest.mark.parametrize("invalid", ["indices", "pre_scale", "weight_shape"])
def test_residue_export_rejects_invalid_state_before_writing(tmp_path, invalid):
    config = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    config["algorithm"] = {"method": "residue_max", "residue_ratio": 1.0}
    model = mtq.quantize(nn.Linear(128, 32), config, lambda m: m(torch.ones(2, 128)))
    if invalid == "indices":
        model.input_quantizer._salient_indices = torch.arange(127, -1, -1)
    elif invalid == "pre_scale":
        model.input_quantizer.pre_quant_scale = torch.ones(128)
    else:
        model.weight = nn.Parameter(torch.cat([model.weight, model.weight], dim=-1))
    before = deepcopy(model.state_dict())
    parameter = model.weight
    target = tmp_path / "checkpoint"
    with pytest.raises((ValueError, NotImplementedError)):
        export_hf_checkpoint(model, export_dir=target)
    assert not target.exists()
    assert model.weight is parameter
    assert model.state_dict().keys() == before.keys()
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("ordinary_sibling", [False, True])
def test_residue_export_preserves_independent_projection_channels(ordinary_sibling):
    config = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    config["algorithm"] = {"method": "residue_max", "residue_ratio": 0.25}
    linear = mtq.quantize(nn.Linear(128, 32), config, lambda m: m(torch.ones(2, 128)))
    model = nn.ModuleDict({name: deepcopy(linear) for name in ("q_proj", "k_proj", "v_proj")})
    indices = model["v_proj"].input_quantizer._salient_indices
    model["v_proj"].input_quantizer._salient_indices = (
        (indices // 8 * 8 + (indices % 8 + 1) % 8).sort().values
    )
    if ordinary_sibling:
        model["k_proj"] = nn.Linear(128, 32)
    names = ["q_proj", "v_proj"] if ordinary_sibling else list(model)
    selected = {name: model[name].input_quantizer._salient_indices.tolist() for name in names}
    expected = {name: export_residue_weight(model[name], torch.bfloat16) for name in names}
    metadata = collect_residue_export_metadata(model)
    assert set(metadata["layers"]) == set(names)
    _process_quantized_modules(model, torch.bfloat16)
    tensors = model.state_dict()
    for name in names:
        assert metadata["layers"][name]["salient_indices"] == selected[name]
        for key, value in expected[name].items():
            torch.testing.assert_close(tensors[f"{name}.{key}"], value, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [0.25, 1.0])
def test_residue_handler_preserves_packed_tensors(ratio):
    config = deepcopy(mtq.MXFP4_DEFAULT_CFG)
    config["algorithm"] = {"method": "residue_max", "residue_ratio": ratio}
    model = mtq.quantize(nn.Linear(128, 32), config, lambda m: m(torch.ones(2, 128)))
    expected = export_residue_weight(model, torch.bfloat16)
    parameter = model.weight
    with pytest.raises(NotImplementedError):
        _export_quantized_weight(model, torch.bfloat16)
    assert model.weight is parameter

    _dispatch_export_handler("linear", model, ExportContext(model, torch.bfloat16))
    for name, tensor in expected.items():
        torch.testing.assert_close(getattr(model, name), tensor, rtol=0, atol=0)
    assert model.weight.dtype == torch.uint8
    assert not model.weight.requires_grad
    assert not any("salient_indices" in name for name in model.state_dict())
