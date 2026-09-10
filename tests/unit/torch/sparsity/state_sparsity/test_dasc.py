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

"""CPU tests for DASC state-sparsity policy calibration."""

import copy
import json

import pytest
import torch
from pydantic import ValidationError
from torch import nn

import modelopt.torch.opt as mto
import modelopt.torch.sparsity.state_sparsity as mtss
from modelopt.torch.opt.conversion import ApplyModeError


class TinyGatedDeltaNet(nn.Module):
    """Minimal GDN-shaped module for framework-independent tests."""

    def __init__(self, num_heads: int = 2):
        super().__init__()
        a_log = torch.zeros(num_heads)
        dt_bias = torch.tensor([-2.0, 2.0]) if num_heads == 2 else torch.zeros(num_heads)
        self.A_log = nn.Parameter(a_log)
        self.dt_bias = nn.Parameter(dt_bias)

    def forward(self, inputs):
        return inputs


class TinyGatedDeltaNetForCausalLM(nn.Module):
    """Parent model whose name must not be mistaken for a GDN layer."""

    def __init__(self, num_heads: int = 2):
        super().__init__()
        self.linear_attn = TinyGatedDeltaNet(num_heads)

    def forward(self, inputs):
        return self.linear_attn(inputs)


def _config(**overrides):
    config = {
        "variant": "dasc_wr",
        "epsilon": 1e-3,
        "static_gate_input": 0.0,
        "wmax_candidates": [7, 11],
        "min_perplexity_retention": 0.995,
        "min_top1_agreement": 0.98,
        "min_checkpoint_savings": 0.2,
        "model_id": "tiny-gdn",
        "model_revision": "revision-1",
        "model_config_id": "sha256:config",
        "calibration_data_id": "sha256:calibration",
    }
    config.update(overrides)
    return config


def _candidate(wmax, *, variant="dasc_wr", top1=0.99, convolution_state_exact=True):
    return {
        "variant": variant,
        "wmax": wmax,
        "retained_heads": 1,
        "total_heads": 2,
        "checkpoint_savings": 0.3,
        "quality": [
            {
                "slice_id": "validation-0",
                "perplexity_retention": 0.999,
                "top1_agreement": top1,
                "finite_continuation_logits": True,
                "retained_state_exact": True,
                "omitted_state_matches_recovery": True,
                "convolution_state_exact": convolution_state_exact,
            }
        ],
    }


def test_calibrate_selects_largest_passing_candidate_and_round_trips():
    model = TinyGatedDeltaNetForCausalLM()
    original_state = {name: value.clone() for name, value in model.state_dict().items()}

    model = mtss.calibrate(model, _config(), [_candidate(11, top1=0.9), _candidate(7)])
    policy = mtss.export_policy(model)

    assert policy["selected_wmax"] == 7
    assert policy["variant"] == "dasc_wr"
    assert policy["recovery"] == "suffix_replay"
    assert policy["granularity"] == "gdn_head"
    assert policy["preserve_convolution_state"] is True
    assert policy["active_runtime_state"] == "dense"
    assert policy["layers"]["linear_attn"]["retained_heads"] == [0]
    assert policy["layers"]["linear_attn"]["omitted_heads"] == [1]
    assert [measurement["wmax"] for measurement in policy["measurements"]] == [7, 11]
    assert all(
        torch.equal(original_state[name], value) for name, value in model.state_dict().items()
    )
    json.dumps(policy)

    restored = mto.restore_from_modelopt_state(
        TinyGatedDeltaNetForCausalLM(), mto.modelopt_state(model)
    )
    assert mtss.export_policy(restored) == policy

    policy["selected_wmax"] = 999
    assert mtss.export_policy(model)["selected_wmax"] == 7


@pytest.mark.parametrize(
    ("variant", "recovery"), [("dasc_nr", "zero"), ("dasc_wr", "suffix_replay")]
)
def test_variant_is_explicit_in_exported_policy(variant, recovery):
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(),
        _config(variant=variant, wmax_candidates=[7]),
        [_candidate(7, variant=variant)],
    )
    policy = mtss.export_policy(model)
    assert policy["variant"] == variant
    assert policy["recovery"] == recovery


@pytest.mark.parametrize(
    "override",
    [
        {"wmax_candidates": []},
        {"wmax_candidates": [0]},
        {"wmax_candidates": [7, 7]},
        {"model_revision": ""},
        {"preserve_convolution_state": False},
    ],
)
def test_config_fails_closed(override):
    with pytest.raises(ValidationError):
        mtss.DASCConfig(**_config(**override))

    assert mtss.DASCConfig(**_config(wmax_candidates=[7])).wmax_candidates == [7]


def test_calibration_fails_closed_on_measurements_and_model_mismatch():
    with pytest.raises(ApplyModeError, match="exactly one result"):
        mtss.calibrate(TinyGatedDeltaNetForCausalLM(), _config(), [_candidate(7)])

    with pytest.raises(ApplyModeError, match="No DASC Wmax candidate"):
        mtss.calibrate(
            TinyGatedDeltaNetForCausalLM(),
            _config(wmax_candidates=[7]),
            [_candidate(7, convolution_state_exact=False)],
        )

    mismatched_geometry = _candidate(7)
    mismatched_geometry["retained_heads"] = 2
    with pytest.raises(ApplyModeError, match="geometry"):
        mtss.calibrate(
            TinyGatedDeltaNetForCausalLM(),
            _config(wmax_candidates=[7]),
            [mismatched_geometry],
        )

    class NotGDN(nn.Module):
        pass

    with pytest.raises(ApplyModeError, match="no GatedDeltaNet modules"):
        mtss.calibrate(NotGDN(), _config(wmax_candidates=[7]), [_candidate(7)])


def test_export_rejects_changed_decay_parameters_and_restore_rejects_structure():
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )
    state = mto.modelopt_state(model)

    with torch.no_grad():
        model.linear_attn.A_log.add_(1.0)
    with pytest.raises(ApplyModeError, match="decay parameters"):
        mtss.export_policy(model)
    with pytest.raises(ApplyModeError, match="decay parameters"):
        mto.modelopt_state(model)

    with pytest.raises(ApplyModeError, match="module structure"):
        mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(num_heads=3), state)

    tampered_state = copy.deepcopy(state)
    tampered_state["modelopt_state_dict"][0][1]["metadata"]["policy"]["quality_gates"][
        "min_top1_agreement"
    ] = 0.5
    with pytest.raises(ApplyModeError, match="does not match its mode config"):
        mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), tampered_state)
