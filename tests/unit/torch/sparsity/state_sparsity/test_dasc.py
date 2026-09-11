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
import io
import json

import pytest
import torch
from pydantic import ValidationError
from torch import nn

import modelopt.torch.opt as mto
import modelopt.torch.sparsity.state_sparsity as mtss
from modelopt.torch.opt.conversion import ApplyModeError


class GatedDeltaNet(nn.Module):
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
        self.linear_attn = GatedDeltaNet(num_heads)

    def forward(self, inputs):
        return self.linear_attn(inputs)


def _config(**overrides):
    """Return a complete test configuration with selected overrides."""
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
    """Return passing caller-supplied evidence for one recovery window."""
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
    """Select the largest passing window and preserve weights and ModelOpt state."""
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
    """Keep zero and suffix-replay recovery as explicit deployment contracts."""
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
    """Reject incomplete provenance and invalid window or lifecycle settings."""
    with pytest.raises(ValidationError):
        mtss.DASCConfig(**_config(**override))

    assert mtss.DASCConfig(**_config(wmax_candidates=[7])).wmax_candidates == [7]


def test_perplexity_retention_accepts_parity_improvements():
    """Allow an observed DASC perplexity improvement instead of requiring clamping."""
    measurement = mtss.DASCCalibrationMeasurement(**_candidate(7))
    measurement.quality[0].perplexity_retention = 1.0004

    assert measurement.quality[0].perplexity_retention == 1.0004


def test_calibration_fails_closed_on_measurements_and_model_mismatch():
    """Reject incomplete evidence, failing gates, wrong geometry, and unsupported layers."""
    with pytest.raises(ApplyModeError, match="exactly one result"):
        mtss.calibrate(TinyGatedDeltaNetForCausalLM(), _config(), [_candidate(7)])

    with pytest.raises(ApplyModeError, match="No DASC Wmax candidate"):
        mtss.calibrate(
            TinyGatedDeltaNetForCausalLM(),
            _config(wmax_candidates=[7]),
            [_candidate(7, convolution_state_exact=False)],
        )

    with pytest.raises(ApplyModeError, match="configured variant"):
        mtss.calibrate(
            TinyGatedDeltaNetForCausalLM(),
            _config(wmax_candidates=[7]),
            [_candidate(7, variant="dasc_nr")],
        )

    invalid_measurement = _candidate(7)
    del invalid_measurement["quality"]
    with pytest.raises(ApplyModeError, match="Invalid DASC calibration measurements"):
        mtss.calibrate(
            TinyGatedDeltaNetForCausalLM(),
            _config(wmax_candidates=[7]),
            [invalid_measurement],
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

    with pytest.raises(ApplyModeError, match="no supported GDN modules"):
        mtss.calibrate(NotGDN(), _config(wmax_candidates=[7]), [_candidate(7)])

    class UnsupportedGatedDeltaNet(GatedDeltaNet):
        pass

    with pytest.raises(ApplyModeError, match="no supported GDN modules"):
        mtss.calibrate(UnsupportedGatedDeltaNet(), _config(wmax_candidates=[7]), [_candidate(7)])

    invalid_decay = TinyGatedDeltaNetForCausalLM()
    invalid_decay.linear_attn.dt_bias = nn.Parameter(torch.zeros(3))
    with pytest.raises(ApplyModeError, match="Invalid GDN decay parameters"):
        mtss.analyze_gdn_decay(invalid_decay)


def test_generic_mode_application_reports_missing_measurements():
    """Give generic apply_mode callers an actionable calibration-evidence error."""
    with pytest.raises(ApplyModeError, match="requires calibration measurements"):
        mto.apply_mode(
            TinyGatedDeltaNetForCausalLM(),
            mode=[("dasc", _config(wmax_candidates=[7]))],
        )


def test_public_exports_and_wrapped_model_export():
    """Expose only supported symbols and unwrap recognized parallel wrappers."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )

    assert "DASCLayerPolicy" in mtss.__all__
    assert "mode" not in mtss.__all__
    assert mtss.export_policy(nn.DataParallel(model)) == mtss.export_policy(model)
    with pytest.raises(ApplyModeError, match="no valid attached DASC policy"):
        mtss.export_policy(TinyGatedDeltaNetForCausalLM())


def test_dtype_cast_preserves_policy_when_the_selected_mask_is_unchanged():
    """Treat an ordinary BF16 cast as equivalent when it preserves the policy mask."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )
    policy = mtss.export_policy(model)

    model.to(torch.bfloat16)

    assert mtss.export_policy(model) == policy


def test_export_rejects_changed_decay_parameters_and_restore_rejects_structure():
    """Keep saving recoverable while rejecting stale or tampered deployment policies."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )
    state = mto.modelopt_state(model)

    with torch.no_grad():
        model.linear_attn.A_log.add_(1.0)
    with pytest.raises(ApplyModeError, match="decay parameters"):
        mtss.export_policy(model)
    with pytest.warns(UserWarning, match="saved DASC policy is stale"):
        mto.modelopt_state(model)
    checkpoint = io.BytesIO()
    with pytest.warns(UserWarning, match="saved DASC policy is stale"):
        mto.save(model, checkpoint)
    assert checkpoint.tell() > 0

    with pytest.warns(UserWarning, match="saved DASC policy is stale"):
        model = mtss.calibrate(model, _config(wmax_candidates=[7]), [_candidate(7)])
    policy = mtss.export_policy(model)
    model_state = copy.deepcopy(model.state_dict())
    recalibrated_state = mto.modelopt_state(model)
    restored = mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), recalibrated_state)
    restored.load_state_dict(model_state)
    assert mtss.export_policy(restored) == policy

    with pytest.raises(ApplyModeError, match="module structure"):
        mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(num_heads=3), state)

    tampered_state = copy.deepcopy(state)
    tampered_state["modelopt_state_dict"][0][1]["metadata"]["policy"]["quality_gates"][
        "min_top1_agreement"
    ] = 0.5
    with pytest.raises(ApplyModeError, match="does not match its mode config"):
        mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), tampered_state)

    tampered_state = copy.deepcopy(state)
    tampered_state["modelopt_state_dict"][0][1]["metadata"]["unexpected"] = True
    with pytest.raises(ApplyModeError, match="only the policy field"):
        mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), tampered_state)

    tampered_state = copy.deepcopy(state)
    del tampered_state["modelopt_state_dict"][0][1]["metadata"]["policy"]["variant"]
    with pytest.raises(ApplyModeError, match="Invalid DASC policy metadata"):
        mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), tampered_state)

    tampered_state = copy.deepcopy(state)
    layer = tampered_state["modelopt_state_dict"][0][1]["metadata"]["policy"]["layers"][
        "linear_attn"
    ]
    layer["static_horizons"][0] = 1.0
    layer["retained_heads"] = []
    layer["omitted_heads"] = [0, 1]
    restored = mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), tampered_state)
    with pytest.raises(ApplyModeError, match="horizons do not match"):
        mtss.export_policy(restored)


def test_export_rederives_the_selected_head_mask():
    """Reject a self-consistent stored mask that current decay parameters do not derive."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[54]), [_candidate(54)]
    )
    state = mto.modelopt_state(model)
    layer = state["modelopt_state_dict"][0][1]["metadata"]["policy"]["layers"]["linear_attn"]
    layer["static_horizons"][0] = 53.0
    layer["retained_heads"] = []
    layer["omitted_heads"] = [0, 1]

    restored = mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), state)

    with pytest.raises(ApplyModeError, match="head mask does not match"):
        mtss.export_policy(restored)
