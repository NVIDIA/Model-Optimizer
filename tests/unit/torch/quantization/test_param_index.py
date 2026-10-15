# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the parameter-name index behind FSDP2 export's param mapping."""

from types import SimpleNamespace

import torch.nn as nn

from modelopt.torch.quantization.utils import core_utils
from modelopt.torch.quantization.utils.core_utils import build_param_index, get_prefixed_param_names


def _linear_scan(parent_model, target_module):
    """The pre-index implementation, kept as the correctness oracle."""
    target_ids = {id(p) for p in target_module.parameters()}
    return next(
        (
            name.rsplit(".", 1)[0]
            for name, param in parent_model.named_parameters()
            if id(param) in target_ids
        ),
        None,
    )


def _moe_ish_model(n_layers=3, n_experts=8):
    """Many small sibling modules, i.e. the shape that made the old scan quadratic."""
    return nn.Sequential(
        *[
            nn.ModuleDict(
                {
                    "attn": nn.Linear(8, 8),
                    "experts": nn.ModuleList(
                        [nn.Linear(8, 8, bias=False) for _ in range(n_experts)]
                    ),
                }
            )
            for _ in range(n_layers)
        ]
    )


def test_matches_linear_scan_for_every_module():
    model = _moe_ish_model()
    index = build_param_index(model)
    for _, module in model.named_modules():
        if not any(True for _ in module.parameters()):
            continue
        assert get_prefixed_param_names(model, module, index) == _linear_scan(model, module)


def test_index_is_optional():
    """Callers that pass no index still get the same answer."""
    model = _moe_ish_model()
    target = model[1]["experts"][3]
    assert get_prefixed_param_names(model, target) == _linear_scan(model, target)


def test_returns_none_for_foreign_module():
    model = _moe_ish_model()
    assert get_prefixed_param_names(model, nn.Linear(8, 8), build_param_index(model)) is None


def test_parameterless_module_returns_none():
    model = _moe_ish_model()
    assert get_prefixed_param_names(model, nn.Identity(), build_param_index(model)) is None


def test_index_maps_every_parameter():
    model = _moe_ish_model()
    index = build_param_index(model)
    assert len(index) == len(list(model.parameters()))
    for pos, (name, param) in enumerate(model.named_parameters()):
        assert index[id(param)] == (pos, name)


def test_shared_parameter_resolves_to_first_occurrence():
    """A tied weight must resolve the same way the linear scan did: earliest name wins."""
    model = _moe_ish_model(n_layers=2, n_experts=2)
    model[1]["experts"][0].weight = model[0]["experts"][0].weight
    target = model[1]["experts"][0]
    assert get_prefixed_param_names(model, target, build_param_index(model)) == _linear_scan(
        model, target
    )


def test_mapping_walks_the_parameters_once(monkeypatch):
    """The regression guard: one parameter walk per call, not one per FSDPParam.

    Walking per FSDPParam is what made MoE export quadratic and stalled it for hours.
    """
    model = _moe_ish_model(n_layers=2, n_experts=8)
    experts = [model[0]["experts"][i] for i in range(8)]

    calls = []
    real_build = core_utils.build_param_index
    monkeypatch.setattr(
        core_utils, "build_param_index", lambda m: (calls.append(m), real_build(m))[1]
    )

    fsdp_params = [
        SimpleNamespace(_module_info=SimpleNamespace(module=e, param_name="weight"))
        for e in experts
    ]
    mapping = core_utils.create_fsdp_param_mapping(fsdp_params, model)

    assert len(calls) == 1, f"expected 1 parameter walk, got {len(calls)} (one per FSDPParam?)"
    assert set(mapping) == {f"0.experts.{i}.weight" for i in range(8)}
