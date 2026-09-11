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
import modelopt.torch.sparsity.state_sparsity.policy as dasc_policy
from modelopt.torch.opt.conversion import ApplyModeError, ModeloptStateManager
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.sparsity.state_sparsity.conversion import replace_dasc_mode
from modelopt.torch.sparsity.state_sparsity.mode import DASCModeRegistry

_resolve_supported_gdn_classes = dasc_policy._supported_gdn_classes.__wrapped__


class GatedDeltaNet(nn.Module):
    """Minimal GDN-shaped module for framework-independent tests."""

    def __init__(self, num_heads: int = 2):
        super().__init__()
        a_log = torch.tensor([0.1, 0.7]) if num_heads == 2 else torch.zeros(num_heads)
        dt_bias = torch.tensor([-2.3, 1.7]) if num_heads == 2 else torch.zeros(num_heads)
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


@pytest.fixture(autouse=True)
def _register_test_gdn_class(monkeypatch):
    """Use the exact toy GDN identity without weakening production class checks."""
    monkeypatch.setattr(dasc_policy, "_supported_gdn_classes", lambda: (GatedDeltaNet,))


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
    assert policy["decay_parameter_storage_dtype"] == "float32"
    assert policy["layers"]["linear_attn"]["retained_heads"] == [0]
    assert policy["layers"]["linear_attn"]["omitted_heads"] == [1]
    assert [measurement["wmax"] for measurement in policy["measurements"]] == [7, 11]
    assert all(
        torch.equal(original_state[name], value) for name, value in model.state_dict().items()
    )
    json.dumps(policy)

    state = mto.modelopt_state(model)
    restored = mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), state)
    assert mtss.export_policy(restored) == policy

    legacy_state = copy.deepcopy(state)
    del legacy_state["modelopt_state_dict"][0][1]["config"]["decay_parameter_storage_dtype"]
    del legacy_state["modelopt_state_dict"][0][1]["metadata"]["policy"][
        "decay_parameter_storage_dtype"
    ]
    legacy_restored = mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), legacy_state)
    assert mtss.export_policy(legacy_restored)["decay_parameter_storage_dtype"] == "float32"

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
        {"decay_parameter_storage_dtype": "float8"},
    ],
)
def test_config_fails_closed(override):
    """Reject incomplete provenance and invalid window or lifecycle settings."""
    with pytest.raises(ValidationError):
        mtss.DASCConfig(**_config(**override))

    assert mtss.DASCConfig(**_config(wmax_candidates=[7])).wmax_candidates == [7]


def test_perplexity_retention_accepts_parity_improvements():
    """Allow improvement measurements and thresholds instead of requiring clamping."""
    candidate = _candidate(7)
    candidate["quality"][0]["perplexity_retention"] = 1.0004
    measurement = mtss.DASCCalibrationMeasurement(**candidate)

    assert measurement.quality[0].perplexity_retention == 1.0004

    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(),
        _config(wmax_candidates=[7], min_perplexity_retention=1.0002),
        [candidate],
    )
    assert mtss.export_policy(model)["selected_wmax"] == 7


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

    class UnsupportedGatedDeltaNet(nn.Module):
        """Unrelated lookalike must not satisfy the supported-base-class contract."""

        def __init__(self):
            super().__init__()
            self.A_log = nn.Parameter(torch.zeros(2))
            self.dt_bias = nn.Parameter(torch.zeros(2))

    with pytest.raises(ApplyModeError, match="no supported GDN modules"):
        mtss.calibrate(UnsupportedGatedDeltaNet(), _config(wmax_candidates=[7]), [_candidate(7)])

    class UnsupportedSubclass(GatedDeltaNet):
        pass

    with pytest.raises(ApplyModeError, match="subclasses that are not ModelOpt dynamic modules"):
        mtss.calibrate(UnsupportedSubclass(), _config(wmax_candidates=[7]), [_candidate(7)])

    same_name_lookalike = type("GatedDeltaNet", (UnsupportedGatedDeltaNet,), {})
    with pytest.raises(ApplyModeError, match="no supported GDN modules"):
        mtss.calibrate(same_name_lookalike(), _config(wmax_candidates=[7]), [_candidate(7)])

    missing_decay = TinyGatedDeltaNetForCausalLM()
    del missing_decay.linear_attn.A_log
    with pytest.raises(ApplyModeError, match="without A_log and dt_bias tensors"):
        mtss.calibrate(missing_decay, _config(wmax_candidates=[7]), [_candidate(7)])

    partially_valid = nn.Module()
    partially_valid.good = GatedDeltaNet()
    partially_valid.bad = GatedDeltaNet()
    del partially_valid.bad.dt_bias
    with pytest.raises(ApplyModeError, match=r"without A_log and dt_bias tensors at: bad$"):
        mtss.analyze_gdn_decay(partially_valid)

    mixed_subclass = nn.Module()
    mixed_subclass.good = GatedDeltaNet()
    mixed_subclass.stale = UnsupportedSubclass()
    with pytest.raises(
        ApplyModeError,
        match=(
            r"not ModelOpt dynamic modules at: stale; convert the module with ModelOpt or use a "
            r"supported class directly$"
        ),
    ):
        mtss.analyze_gdn_decay(mixed_subclass)

    invalid_decay = TinyGatedDeltaNetForCausalLM()
    invalid_decay.linear_attn.dt_bias = nn.Parameter(torch.zeros(3))
    with pytest.raises(ApplyModeError, match="Invalid GDN decay parameters"):
        mtss.analyze_gdn_decay(invalid_decay)


def test_supported_class_resolution_uses_imported_module_identities(monkeypatch):
    """Ignore absent and non-module symbols while retaining exact supported identities."""
    module_paths = (
        ("valid", "GatedDeltaNet"),
        ("invalid", "NotAModule"),
        ("missing", "Missing"),
        ("installed", "Missing"),
        ("broken", "Broken"),
    )
    modules = {
        "valid": type("ValidModule", (), {"GatedDeltaNet": GatedDeltaNet}),
        "invalid": type("InvalidModule", (), {"NotAModule": object()}),
    }

    def import_module(name):
        if name in {"missing", "installed"}:
            raise ModuleNotFoundError(name)
        if name == "broken":
            raise RuntimeError(name)
        return modules[name]

    monkeypatch.setattr(dasc_policy, "_SUPPORTED_GDN_CLASS_PATHS", module_paths)
    monkeypatch.setattr(dasc_policy.importlib, "import_module", import_module)
    monkeypatch.setattr(
        dasc_policy.importlib.util,
        "find_spec",
        lambda name: object() if name == "installed" else None,
    )

    with pytest.warns(UserWarning) as caught:
        assert _resolve_supported_gdn_classes() == (GatedDeltaNet,)
    assert len(caught) == 3


@pytest.mark.parametrize(("module_name", "class_name"), dasc_policy._SUPPORTED_GDN_CLASS_PATHS)
def test_declared_gdn_paths_resolve_when_framework_is_installed(module_name, class_name):
    """Guard supported identities against upstream dependency path drift."""
    root_module = module_name.partition(".")[0]
    pytest.importorskip(root_module)
    module = dasc_policy.importlib.import_module(module_name)
    assert issubclass(getattr(module, class_name), nn.Module)


def test_generic_mode_application_reports_missing_measurements(monkeypatch):
    """Give generic apply_mode callers an actionable calibration-evidence error."""
    assert DASCModeRegistry["dasc"].next_prohibited_modes == {"dasc"}
    assert DASCModeRegistry["dasc"].update_for_new_mode is not None
    with pytest.raises(ApplyModeError, match="requires calibration measurements"):
        mto.apply_mode(
            TinyGatedDeltaNetForCausalLM(),
            mode=[("dasc", _config(wmax_candidates=[7]))],
        )

    model = TinyGatedDeltaNetForCausalLM()
    ModeloptStateManager(model, init_state=True)
    with pytest.raises(ApplyModeError, match="model has no DASC state"):
        replace_dasc_mode(
            model,
            mtss.DASCConfig(**_config(wmax_candidates=[7])),
            [_candidate(7)],
        )

    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )
    ModeloptStateManager(model).state_dict().append(("trailing-mode", {}))
    refreshed = []
    monkeypatch.setattr(
        ModeloptStateManager,
        "update_last_state_before_new_mode",
        lambda _manager, current_model: refreshed.append(current_model),
    )
    replace_dasc_mode(
        model,
        mtss.DASCConfig(**_config(wmax_candidates=[7])),
        [_candidate(7)],
    )
    assert refreshed == [model]


def test_public_exports_and_wrapped_model_export():
    """Expose only supported symbols and accept wrappers and ModelOpt subclasses."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )

    assert "DASCLayerPolicy" in mtss.__all__
    assert "mode" not in mtss.__all__
    assert mtss.export_policy(nn.DataParallel(model)) == mtss.export_policy(model)

    dynamic_class = type("_DynamicGatedDeltaNet", (DynamicModule, GatedDeltaNet), {})
    dynamic_module = GatedDeltaNet()
    dynamic_module.__class__ = dynamic_class
    model.linear_attn = dynamic_module
    assert mtss.export_policy(model)["layers"]["linear_attn"]["num_heads"] == 2

    with pytest.raises(ApplyModeError, match="no valid attached DASC policy"):
        mtss.export_policy(TinyGatedDeltaNetForCausalLM())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dtype_cast_preserves_policy_when_the_selected_mask_is_unchanged(dtype):
    """Treat ordinary low-precision casts as equivalent when they preserve the policy mask."""
    storage_dtype = "bfloat16" if dtype == torch.bfloat16 else "float16"
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(),
        _config(wmax_candidates=[7], decay_parameter_storage_dtype=storage_dtype),
        [_candidate(7)],
    )
    policy = mtss.export_policy(model)
    original_decay = torch.cat(
        [model.linear_attn.A_log.detach(), model.linear_attn.dt_bias.detach()]
    )

    model.to(dtype)

    cast_decay = torch.cat(
        [model.linear_attn.A_log.detach().float(), model.linear_attn.dt_bias.detach().float()]
    )
    assert not torch.equal(original_decay, cast_decay)
    assert mtss.export_policy(model) == policy


def test_calibration_uses_storage_canonical_mask_at_wmax_boundary():
    """Derive the mask from stored decay values and accept their explained boundary flip."""
    model = TinyGatedDeltaNetForCausalLM()
    with torch.no_grad():
        model.linear_attn.A_log[0] = 0.0
        model.linear_attn.dt_bias[0] = 0.520263671875
    live_horizon = mtss.compute_gdn_decay_horizons(
        model.linear_attn.A_log, model.linear_attn.dt_bias, static_gate_input=0.0
    )[0]
    stored_horizon = mtss.analyze_gdn_decay(
        model,
        static_gate_input=0.0,
        decay_parameter_storage_dtype="float16",
    )["linear_attn"][0]
    assert live_horizon > 7
    assert stored_horizon < 7

    measurement = _candidate(7)
    measurement["retained_heads"] = 0
    model = mtss.calibrate(
        model,
        _config(wmax_candidates=[7], decay_parameter_storage_dtype="float16"),
        [measurement],
    )
    policy = mtss.export_policy(model)

    assert policy["layers"]["linear_attn"]["retained_heads"] == []
    assert policy["layers"]["linear_attn"]["static_horizons"][0] < 7


@pytest.mark.parametrize("invalid_storage_dtype", ["float8", []])
def test_analysis_arguments_fail_at_the_public_boundary(invalid_storage_dtype):
    """Report invalid analysis arguments uniformly without blaming a GDN module."""
    model = TinyGatedDeltaNetForCausalLM()
    with pytest.raises(ValueError, match="decay_parameter_storage_dtype must be one of"):
        mtss.analyze_gdn_decay(
            model,
            decay_parameter_storage_dtype=invalid_storage_dtype,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("epsilon", [1.0, [], 10**1000])
def test_analysis_rejects_invalid_epsilon_at_the_public_boundary(epsilon):
    """Normalize invalid epsilon values to the public ValueError contract."""
    with pytest.raises(ValueError, match=r"epsilon must be finite and in \(0, 1\)"):
        mtss.analyze_gdn_decay(TinyGatedDeltaNetForCausalLM(), epsilon=epsilon)  # type: ignore[arg-type]


@pytest.mark.parametrize("static_gate_input", [torch.nan, [], 10**1000])
def test_analysis_rejects_invalid_static_gate_input_at_the_public_boundary(static_gate_input):
    """Normalize invalid static gate values to the public ValueError contract."""
    with pytest.raises(ValueError, match="static_gate_input must be finite"):
        mtss.analyze_gdn_decay(
            TinyGatedDeltaNetForCausalLM(),
            static_gate_input=static_gate_input,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("epsilon", [], r"epsilon must be finite and in \(0, 1\)"),
        ("epsilon", torch.nan, r"epsilon must be finite and in \(0, 1\)"),
        ("epsilon", 10**1000, r"epsilon must be finite and in \(0, 1\)"),
        ("static_gate_input", [], "static_gate_input must be finite"),
        ("static_gate_input", torch.nan, "static_gate_input must be finite"),
        ("static_gate_input", 10**1000, "static_gate_input must be finite"),
    ],
)
def test_horizon_computation_rejects_invalid_public_arguments(argument, value, message):
    """Use the same public argument contract for direct horizon computation."""
    kwargs = {argument: value}
    with pytest.raises(ValueError, match=message):
        mtss.compute_gdn_decay_horizons(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            **kwargs,  # type: ignore[arg-type]
        )


def test_analysis_accepts_tensor_scalar_arguments():
    """Preserve support for real-like scalar values accepted by the numeric operations."""
    horizons = mtss.compute_gdn_decay_horizons(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        epsilon=torch.tensor(1e-3),  # type: ignore[arg-type]
        static_gate_input=torch.tensor(-0.3),  # type: ignore[arg-type]
    )
    assert horizons.shape == (1,)
    assert torch.isfinite(horizons).all()


def test_bf16_storage_round_trip_loaded_in_fp32_preserves_policy():
    """Accept BF16-rounded values after a checkpoint loader materializes FP32 tensors."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(),
        _config(wmax_candidates=[7], decay_parameter_storage_dtype="bfloat16"),
        [_candidate(7)],
    )
    policy = mtss.export_policy(model)
    original_decay = torch.cat(
        [model.linear_attn.A_log.detach(), model.linear_attn.dt_bias.detach()]
    )

    with torch.no_grad():
        model.linear_attn.A_log.copy_(model.linear_attn.A_log.to(torch.bfloat16).float())
        model.linear_attn.dt_bias.copy_(model.linear_attn.dt_bias.to(torch.bfloat16).float())

    reloaded_decay = torch.cat(
        [model.linear_attn.A_log.detach(), model.linear_attn.dt_bias.detach()]
    )
    assert reloaded_decay.dtype == torch.float32
    assert not torch.equal(original_decay, reloaded_decay)
    assert mtss.export_policy(model) == policy

    model.linear_attn.A_log = nn.Parameter(
        model.linear_attn.A_log.detach().to(torch.int64), requires_grad=False
    )
    with pytest.raises(ApplyModeError, match="floating-point dtype"):
        mtss.export_policy(model)


@pytest.mark.parametrize(
    ("storage_name", "storage_dtype", "live_dtype"),
    [
        ("float16", torch.float16, torch.bfloat16),
        ("bfloat16", torch.bfloat16, torch.float16),
    ],
)
def test_cross_dtype_reload_accumulates_both_rounding_bounds(
    storage_name, storage_dtype, live_dtype
):
    """Accept two distinct declared-storage and live-materialization rounding steps."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(),
        _config(wmax_candidates=[7], decay_parameter_storage_dtype=storage_name),
        [_candidate(7)],
    )
    policy = mtss.export_policy(model)

    model.to(storage_dtype).to(live_dtype)

    assert mtss.export_policy(model) == policy


@pytest.mark.parametrize("live_dtype", [torch.float16, torch.bfloat16])
def test_storage_rounding_excludes_exact_fp32_widening(live_dtype):
    """Do not add FP32 slack when only the low-precision cast can round values."""
    tensor = torch.tensor([1.25], dtype=live_dtype)

    assert torch.equal(
        dasc_policy._storage_rounding_radius(tensor, torch.float32),
        dasc_policy._storage_rounding_radius(tensor, live_dtype),
    )


def test_non_finite_decay_parameters_are_rejected_on_export():
    """Reject NaNs before interval comparisons can silently accept them."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )
    with torch.no_grad():
        model.linear_attn.dt_bias[0] = torch.nan

    with pytest.raises(ApplyModeError, match="GDN decay parameters must be finite"):
        mtss.export_policy(model)


def test_export_rejects_changed_decay_parameters_and_restore_rejects_structure():
    """Keep saving recoverable while rejecting stale or tampered deployment policies."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )
    state = mto.modelopt_state(model)

    with torch.no_grad():
        model.linear_attn.A_log.add_(1.0)
    with pytest.raises(ApplyModeError, match="horizons"):
        mtss.export_policy(model)
    with pytest.warns(UserWarning, match="saved DASC policy is stale"):
        mto.modelopt_state(model)
    checkpoint = io.BytesIO()
    with pytest.warns(UserWarning, match="saved DASC policy is stale"):
        mto.save(model, checkpoint)
    assert checkpoint.tell() > 0

    manager_state = ModeloptStateManager(model).state_dict()
    manager_state.append(copy.deepcopy(manager_state[0]))
    delattr(model, "_modelopt_dasc_policy")
    model = mtss.calibrate(
        model,
        _config(wmax_candidates=[7], model_revision="revision-2"),
        [_candidate(7)],
    )
    policy = mtss.export_policy(model)
    assert policy["model_revision"] == "revision-2"
    model_state = copy.deepcopy(model.state_dict())
    recalibrated_state = mto.modelopt_state(model)
    assert [mode for mode, _ in recalibrated_state["modelopt_state_dict"]] == ["dasc"]
    restored = mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), recalibrated_state)
    restored.load_state_dict(model_state)
    assert mtss.export_policy(restored) == policy

    with pytest.warns(UserWarning, match="restored DASC policy is stale"):
        mismatched = mto.restore_from_modelopt_state(
            TinyGatedDeltaNetForCausalLM(num_heads=3), state
        )
    with pytest.raises(ApplyModeError, match="module structure"):
        mtss.export_policy(mismatched)

    with pytest.raises(ApplyModeError, match="no supported GDN modules"):
        mto.restore_from_modelopt_state(nn.Linear(2, 2), state)

    unsupported = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )
    unsupported.linear_attn = nn.Linear(2, 2)
    with pytest.raises(ApplyModeError, match="no supported GDN modules"):
        mto.modelopt_state(unsupported)

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
    layer["static_horizons"][0] *= 1.0001
    restored = mto.restore_from_modelopt_state(TinyGatedDeltaNetForCausalLM(), tampered_state)
    with pytest.raises(ApplyModeError, match="horizons do not match"):
        mtss.export_policy(restored)


def test_structure_staleness_does_not_block_checkpoint_save():
    """Keep checkpoint save and restore available after a calibrated GDN structure changes."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )
    model.linear_attn = GatedDeltaNet(num_heads=3)

    checkpoint = io.BytesIO()
    with pytest.warns(UserWarning, match="saved DASC policy is stale"):
        mto.save(model, checkpoint)

    assert checkpoint.tell() > 0
    with pytest.raises(ApplyModeError, match="module structure"):
        mtss.export_policy(model)

    with pytest.warns(UserWarning, match="saved DASC policy is stale"):
        stale_state = mto.modelopt_state(model)
    with pytest.warns(UserWarning, match="restored DASC policy is stale"):
        restored = mto.restore_from_modelopt_state(
            TinyGatedDeltaNetForCausalLM(num_heads=3), stale_state
        )
    restored.load_state_dict(model.state_dict())
    with pytest.raises(ApplyModeError, match="module structure"):
        mtss.export_policy(restored)


def test_temporarily_unavailable_decay_tensors_are_recoverable_staleness():
    """Keep save and restore symmetric when a supported GDN is temporarily flattened."""
    model = mtss.calibrate(
        TinyGatedDeltaNetForCausalLM(), _config(wmax_candidates=[7]), [_candidate(7)]
    )
    state = mto.modelopt_state(model)
    model.linear_attn.A_log = None

    with pytest.warns(UserWarning, match="saved DASC policy is stale"):
        mto.modelopt_state(model)

    target = TinyGatedDeltaNetForCausalLM()
    target.linear_attn.A_log = None
    with pytest.warns(UserWarning, match="restored DASC policy is stale"):
        restored = mto.restore_from_modelopt_state(target, state)
    with pytest.raises(ApplyModeError, match="without A_log and dt_bias tensors"):
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
