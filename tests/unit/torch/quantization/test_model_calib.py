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

"""Unit tests for calibration orchestration in modelopt.torch.quantization.model_calib.

Covers the CPU-testable orchestration paths not exercised by the sibling suites:
``max_calibrate`` amax contracts, ``enable_stats_collection`` / ``finish_stats_collection``
choreography, ``mse_calibrate`` model-level orchestration, entry-point error paths
(``smoothquant`` / ``awq_lite`` / ``awq_clip``), pre_quant_scale helpers, and the pure
helpers (``svd``, ``_iter_leaf_quantizers``, EP-sync predicates, AWQ block-size lookup).

AWQ search math, layerwise calibration, local-Hessian, GPTQ, NVFP4 shared-state sync and
the MSE calibrator internals are covered by test_calib.py, test_layerwise_calibrate.py,
test_local_hessian.py, test_gptq.py, test_shared_input.py and test_mse_calibrator.py.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.quantization.models import SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import (
    _get_awq_quantizer_block_size,
    _is_routed_expert,
    _iter_leaf_quantizers,
    _mse_quant_func,
    _should_sync_amax_across_ep,
    apply_pre_quant_scale_and_smooth,
    awq,
    awq_clip,
    awq_lite,
    disable_pre_quant_scale_and_resmooth,
    enable_stats_collection,
    finish_stats_collection,
    max_calibrate,
    mse_calibrate,
    smoothquant,
    svd,
    weight_only_quantize,
)
from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.utils.shared_input import iter_shared_quant_states

_INT8_QUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
]

_FP8_QUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
    {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": (4, 3), "axis": None}},
]


class _TinyMLP(nn.Module):
    """Two-linear MLP used for direct max_calibrate / mse_calibrate calls."""

    def __init__(self, fi=16, fh=32, fo=16):
        super().__init__()
        self.fc1 = nn.Linear(fi, fh)
        self.fc2 = nn.Linear(fh, fo)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _TwoBranch(nn.Module):
    """Two parallel linears; the default forward exercises only branch ``a``."""

    def __init__(self, dim=16):
        super().__init__()
        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)

    def forward(self, x, branch="a"):
        return self.a(x) if branch == "a" else self.b(x)


class _QKVAttention(nn.Module):
    """q/k/v projection triplet matching the default shared-state name patterns."""

    def __init__(self, dim=16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)


def _quantize_model(model, quant_cfg=_INT8_QUANT_CFG):
    """Convert to quant modules and apply a quantizer config without calibrating."""
    mtq.replace_quant_module(model)
    mtq.set_quantizer_by_cfg(model, quant_cfg)
    return model


def _quantized_linears(model):
    return [
        m for m in model.modules() if isinstance(m, nn.Linear) and hasattr(m, "weight_quantizer")
    ]


def _forward_loop(model, batches):
    for batch in batches:
        model(batch)


# ---------------------------------------------------------------------------
# max_calibrate
# ---------------------------------------------------------------------------


def test_max_calibrate_weight_only_without_forward_loop():
    """forward_loop=None calibrates weights only; input quantizers stay uncalibrated."""
    torch.manual_seed(0)
    model = _quantize_model(_TinyMLP())

    max_calibrate(model, forward_loop=None)

    for linear in _quantized_linears(model):
        expected = linear.weight.abs().amax(dim=1, keepdim=True)
        assert torch.allclose(linear.weight_quantizer.amax, expected)
        # Quantization re-enabled, calibration off — the model is ready for inference.
        assert linear.weight_quantizer._if_quant
        assert not linear.weight_quantizer._if_calib
        # No data was forwarded, so the (enabled) input quantizers have no amax.
        assert getattr(linear.input_quantizer, "_amax", None) is None


def test_max_calibrate_amax_matches_data_max():
    """Input amax equals max |activation| over all batches; weight amax is per-channel."""
    torch.manual_seed(1)
    model = _TinyMLP()
    ref = copy.deepcopy(model)
    _quantize_model(model)
    batches = [torch.randn(4, 16) for _ in range(3)]

    max_calibrate(model, lambda m: _forward_loop(m, batches))

    expected_fc1_in = max(b.abs().max() for b in batches)
    with torch.no_grad():
        expected_fc2_in = max(torch.relu(ref.fc1(b)).abs().max() for b in batches)

    assert torch.isclose(model.fc1.input_quantizer.amax, expected_fc1_in)
    assert torch.isclose(model.fc2.input_quantizer.amax, expected_fc2_in)
    assert model.fc1.input_quantizer.amax.numel() == 1  # per-tensor (axis=None)

    for linear, ref_linear in ((model.fc1, ref.fc1), (model.fc2, ref.fc2)):
        expected_w = ref_linear.weight.abs().amax(dim=1, keepdim=True)
        assert torch.allclose(linear.weight_quantizer.amax, expected_w)
        assert linear.weight_quantizer.amax.shape == (linear.weight.shape[0], 1)


def test_max_calibrate_distributed_sync_flag_is_noop_single_process():
    """distributed_sync=True vs False must be equivalent without an initialized process group."""
    torch.manual_seed(2)
    model_a = _TinyMLP()
    model_b = copy.deepcopy(model_a)
    _quantize_model(model_a)
    _quantize_model(model_b)
    batches = [torch.randn(4, 16) for _ in range(2)]

    max_calibrate(model_a, lambda m: _forward_loop(m, batches), distributed_sync=True)
    max_calibrate(model_b, lambda m: _forward_loop(m, batches), distributed_sync=False)

    quantizers_a = {n: m for n, m in model_a.named_modules() if isinstance(m, TensorQuantizer)}
    quantizers_b = {n: m for n, m in model_b.named_modules() if isinstance(m, TensorQuantizer)}
    assert set(quantizers_a) == set(quantizers_b)
    amax_seen = 0
    for name, qa in quantizers_a.items():
        qb = quantizers_b[name]
        amax_a, amax_b = getattr(qa, "_amax", None), getattr(qb, "_amax", None)
        assert (amax_a is None) == (amax_b is None), name
        if amax_a is not None:
            assert torch.equal(amax_a, amax_b), name
            amax_seen += 1
    assert amax_seen > 0


def test_max_calibrate_accumulates_running_max_across_calls():
    """Repeated max_calibrate on the same model keeps the running max (amax never shrinks)."""
    torch.manual_seed(3)
    model = _quantize_model(_TinyMLP())
    x = torch.randn(4, 16)
    x[0, 0] = 5.0  # pin the max

    max_calibrate(model, lambda m: m(x))
    amax_first = model.fc1.input_quantizer.amax.clone()
    assert torch.isclose(amax_first, torch.tensor(5.0))

    # Smaller data: the calibrator's running max keeps the old value.
    max_calibrate(model, lambda m: m(x * 0.5))
    assert torch.equal(model.fc1.input_quantizer.amax, amax_first)

    # Larger data: the running max grows.
    max_calibrate(model, lambda m: m(x * 3))
    assert torch.isclose(model.fc1.input_quantizer.amax, torch.tensor(15.0))


def test_max_calibrate_shared_states_pattern_plumbing():
    """The shared_states kwarg controls shared weight-global-amax group discovery.

    Scenarios 1-2 overlap test_shared_input.py (NVFP4/promotion angle there,
    INT8/iter_shared_quant_states here); scenario 3 (custom patterns through
    the max_calibrate kwarg) is only covered here."""
    # Default patterns: q/k/v form one group whose global_amax is the member max.
    torch.manual_seed(4)
    model = _quantize_model(_QKVAttention())
    max_calibrate(model, forward_loop=None)
    states = list(iter_shared_quant_states(model))
    assert len(states) == 1
    assert len(states[0].members) == 3
    expected = max(p.weight.abs().max() for p in (model.q_proj, model.k_proj, model.v_proj)).float()
    assert torch.isclose(states[0].global_amax, expected)

    # Empty pattern list disables the shared state entirely.
    torch.manual_seed(4)
    model = _quantize_model(_QKVAttention())
    max_calibrate(model, forward_loop=None, shared_states={"weight_global_amax": {"patterns": []}})
    assert list(iter_shared_quant_states(model)) == []

    # Custom patterns override the defaults (group q/k only).
    torch.manual_seed(4)
    model = _quantize_model(_QKVAttention())
    max_calibrate(
        model,
        forward_loop=None,
        shared_states={"weight_global_amax": {"patterns": [r"(?:(.*)\.)?(?:q_proj|k_proj)"]}},
    )
    states = list(iter_shared_quant_states(model))
    assert len(states) == 1
    assert len(states[0].members) == 2


# ---------------------------------------------------------------------------
# enable_stats_collection / finish_stats_collection choreography
# ---------------------------------------------------------------------------


def test_enable_stats_collection_disables_quant_and_enables_calib():
    torch.manual_seed(5)
    model = _quantize_model(_TinyMLP())

    enable_stats_collection(model)

    checked = 0
    for module in model.modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            assert module._if_calib
            assert not module._if_quant
            checked += 1
    assert checked > 0


def test_finish_stats_collection_loads_amax_and_restores_quant():
    torch.manual_seed(6)
    model = _quantize_model(_TinyMLP())

    enable_stats_collection(model)
    model(torch.randn(4, 16))
    finish_stats_collection(model)

    for linear in _quantized_linears(model):
        for quantizer in (linear.input_quantizer, linear.weight_quantizer):
            assert quantizer.amax is not None
            assert quantizer._if_quant
            assert not quantizer._if_calib


def test_finish_stats_collection_without_data_leaves_amax_unset():
    """A calibrator that saw no data yields no amax, but quant mode is still restored."""
    torch.manual_seed(7)
    model = _quantize_model(_TinyMLP())

    enable_stats_collection(model)
    finish_stats_collection(model)

    for linear in _quantized_linears(model):
        assert getattr(linear.input_quantizer, "_amax", None) is None
        assert linear.input_quantizer._if_quant
        assert not linear.input_quantizer._if_calib


def test_quantizer_without_calibrator_stays_disabled_after_finish():
    """enable_stats_collection fully disables calibrator-less quantizers and
    finish_stats_collection never re-enables them.

    # NOTE: documents current behavior; arguably a bug because a quantizer that was
    # enabled before calibration silently remains disabled afterwards —
    # finish_stats_collection skips `module._disabled` quantizers, so the disable
    # applied by enable_stats_collection is never undone.
    """
    model = nn.Module()
    model.q = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=None))
    model.q._calibrator = None
    assert model.q.is_enabled

    enable_stats_collection(model)
    assert not model.q.is_enabled

    finish_stats_collection(model)
    assert not model.q.is_enabled  # never restored


def test_constant_amax_quantizer_quant_toggled_but_never_calibrated():
    """use_constant_amax quantizers skip calibration; quant is disabled then re-enabled."""
    model = nn.Module()
    model.q = TensorQuantizer(
        QuantizerAttributeConfig(num_bits=(4, 3), axis=None, use_constant_amax=True)
    )

    enable_stats_collection(model)
    assert not model.q._if_quant
    assert not model.q._if_calib  # calibration never turned on

    finish_stats_collection(model)
    assert model.q._if_quant
    assert getattr(model.q, "_amax", None) is None  # amax is constant, not a buffer


def test_weight_only_quantize_visits_aliased_module_once():
    """A module reachable under two names must be calibrated exactly once.

    The dedup is provided by named_modules(remove_duplicate=True); this pins
    the behavioral contract regardless of which layer implements it."""

    class _Aliased(nn.Module):
        def __init__(self):
            super().__init__()
            shared = nn.Linear(8, 8)
            self.a = shared
            self.b = shared

        def forward(self, x):
            return self.b(self.a(x))

    torch.manual_seed(8)
    model = _quantize_model(_Aliased())
    assert model.a is model.b  # conversion preserved aliasing

    calls = []
    calibrator = model.a.weight_quantizer._calibrator
    original_collect = calibrator.collect
    calibrator.collect = lambda x: (calls.append(1), original_collect(x))[1]

    enable_stats_collection(model)
    weight_only_quantize(model)
    finish_stats_collection(model)

    assert len(calls) == 1
    assert model.a.weight_quantizer.amax is not None


# ---------------------------------------------------------------------------
# mse_calibrate orchestration
# ---------------------------------------------------------------------------


def test_mse_calibrate_refines_only_weight_amax():
    """MSE refines weight amax within the multiplier search range; input amax is
    identical to plain max calibration."""
    torch.manual_seed(9)
    model_max = _TinyMLP()
    model_mse = copy.deepcopy(model_max)
    _quantize_model(model_max)
    _quantize_model(model_mse)
    batches = [torch.randn(4, 16) for _ in range(2)]

    max_calibrate(model_max, lambda m: _forward_loop(m, batches))
    mse_calibrate(model_mse, lambda m: _forward_loop(m, batches))

    for name in ("fc1", "fc2"):
        linear_max = getattr(model_max, name)
        linear_mse = getattr(model_mse, name)
        # Inputs: MSE calibration must not touch activation amax.
        assert torch.equal(linear_mse.input_quantizer.amax, linear_max.input_quantizer.amax)
        # Weights: refined per-channel amax stays within [start, stop] x max-amax.
        base = linear_max.weight_quantizer.amax
        refined = linear_mse.weight_quantizer.amax
        assert refined.shape == base.shape
        assert torch.all(refined >= 0.25 * base - 1e-7)
        assert torch.all(refined <= 4.0 * base + 1e-7)
        assert torch.all(refined > 0)
        # MSE must genuinely refine, not silently fall back to the max amax
        # (deterministic under this seed: every channel moves).
        assert not torch.equal(refined, base)


def test_mse_calibrate_is_deterministic():
    torch.manual_seed(10)
    model_a = _TinyMLP()
    model_b = copy.deepcopy(model_a)
    _quantize_model(model_a)
    _quantize_model(model_b)
    batches = [torch.randn(4, 16) for _ in range(2)]

    mse_calibrate(model_a, lambda m: _forward_loop(m, batches))
    mse_calibrate(model_b, lambda m: _forward_loop(m, batches))

    for name in ("fc1", "fc2"):
        assert torch.equal(
            getattr(model_a, name).weight_quantizer.amax,
            getattr(model_b, name).weight_quantizer.amax,
        )


def test_mse_quant_func_restores_preexisting_amax():
    """_mse_quant_func quantizes with the trial amax but must restore the original."""
    torch.manual_seed(11)
    x = torch.randn(32)
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=None), amax=2.0)
    reference = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=None), amax=1.0)

    out = _mse_quant_func(x, torch.tensor(1.0), quantizer)

    assert torch.allclose(out, reference(x))
    assert torch.isclose(quantizer.amax, torch.tensor(2.0))  # restored


def test_mse_quant_func_removes_temporary_amax():
    """When the quantizer had no amax, the temporary trial amax must be cleaned up."""
    torch.manual_seed(12)
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=None))
    assert not hasattr(quantizer, "_amax")

    _mse_quant_func(torch.randn(32), torch.tensor(1.0), quantizer)

    assert not hasattr(quantizer, "_amax")


# ---------------------------------------------------------------------------
# smoothquant entry point
# ---------------------------------------------------------------------------


def test_smoothquant_requires_forward_loop():
    model = _quantize_model(_TinyMLP())
    with pytest.raises(AssertionError, match="forward_loop must be provided"):
        smoothquant(model, forward_loop=None)


def test_smoothquant_converts_input_quantizers_to_per_tensor():
    """After smoothing, activation quantizers are per-tensor with the per-channel
    history stashed in _amax_for_smoothing, and pre_quant_scale is positive."""
    torch.manual_seed(13)
    model = SimpleLinear()
    calib_data = [SimpleLinear.get_input() for _ in range(2)]
    mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, lambda m: _forward_loop(m, calib_data))

    smoothed = 0
    for linear in _quantized_linears(model):
        input_quantizer = linear.input_quantizer
        if input_quantizer.pre_quant_scale is None:
            continue
        smoothed += 1
        assert input_quantizer.axis is None
        assert input_quantizer.amax.numel() == 1
        assert input_quantizer._amax_for_smoothing.numel() == linear.weight.shape[1]
        assert input_quantizer.pre_quant_scale.numel() == linear.weight.shape[1]
        assert torch.all(input_quantizer.pre_quant_scale > 0)
    assert smoothed > 0


def test_smoothquant_skips_non_int8_linears():
    torch.manual_seed(14)
    model = _quantize_model(_TinyMLP(), _FP8_QUANT_CFG)
    batches = [torch.randn(4, 16)]

    with pytest.warns(UserWarning, match="Only int8 smoothing is supported"):
        smoothquant(model, lambda m: _forward_loop(m, batches))

    for linear in _quantized_linears(model):
        assert linear.input_quantizer.pre_quant_scale is None


def test_smoothquant_skips_uncalibrated_linear():
    """A linear that never saw data is skipped with a warning; the exercised one smooths."""
    torch.manual_seed(15)
    model = _quantize_model(_TwoBranch())
    batches = [torch.randn(4, 16) for _ in range(2)]

    with pytest.warns(UserWarning, match="not calibrated, skip smoothing"):
        smoothquant(model, lambda m: _forward_loop(m, batches))  # only branch "a" runs

    assert model.a.input_quantizer.pre_quant_scale is not None
    assert model.b.input_quantizer.pre_quant_scale is None


# ---------------------------------------------------------------------------
# awq entry points
# ---------------------------------------------------------------------------


def test_awq_lite_without_forward_loop_warns_and_skips():
    torch.manual_seed(16)
    model = _quantize_model(_TinyMLP())

    with pytest.warns(UserWarning, match="forward_loop must be provided for awq_lite"):
        awq_lite(model, forward_loop=None)

    # Nothing was calibrated or mutated.
    for linear in _quantized_linears(model):
        assert getattr(linear.weight_quantizer, "_amax", None) is None
        assert not hasattr(linear, "awq_lite")


def test_awq_clip_without_forward_loop_raises():
    model = _quantize_model(_TinyMLP())
    with pytest.raises(AssertionError, match="forward_loop must be provided for awq_clip"):
        awq_clip(model, forward_loop=None)


def test_awq_unknown_algorithm_is_silent_noop():
    """awq() with an unrecognized algorithm string runs neither awq_lite nor awq_clip.

    # NOTE: documents current behavior; arguably a bug because a typo such as
    # "awq-lite" silently returns an uncalibrated model instead of raising —
    # awq() only checks membership in the two `if algorithm in [...]` lists.
    # (mtq.quantize validates the method name at the config layer, but direct
    # callers of awq() get no such protection.)
    """
    torch.manual_seed(17)
    model = _quantize_model(_TinyMLP())

    awq(model, lambda m: m(torch.randn(4, 16)), algorithm="awq-lite")

    for linear in _quantized_linears(model):
        assert getattr(linear.weight_quantizer, "_amax", None) is None


# ---------------------------------------------------------------------------
# pre_quant_scale helpers
# ---------------------------------------------------------------------------


def _smoothquant_simple_linear():
    torch.manual_seed(18)
    model = SimpleLinear()
    calib_data = [SimpleLinear.get_input() for _ in range(2)]
    mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, lambda m: _forward_loop(m, calib_data))
    return model


def test_disable_pre_quant_scale_with_delete_removes_attribute():
    model = _smoothquant_simple_linear()
    linear = model.net[0]
    assert linear.input_quantizer.pre_quant_scale is not None

    disable_pre_quant_scale_and_resmooth(linear, delete_pre_quant_scale=True)

    assert not hasattr(linear.input_quantizer, "_pre_quant_scale")
    assert linear.input_quantizer.pre_quant_scale is None
    assert not linear.input_quantizer._enable_pre_quant_scale
    # Resmoothing folded the scale back, so the weight amax was recalibrated.
    assert linear.weight_quantizer.amax is not None


def test_apply_pre_quant_scale_asserts_when_already_smoothed():
    model = _smoothquant_simple_linear()
    linear = model.net[2]
    with pytest.raises(AssertionError, match="pre_quant_scale should be None first"):
        apply_pre_quant_scale_and_smooth(linear, torch.ones(linear.weight.shape[1]))


def test_apply_pre_quant_scale_requires_positive_scale():
    model = _smoothquant_simple_linear()
    linear = model.net[4]
    disable_pre_quant_scale_and_resmooth(linear)  # keep _pre_quant_scale, disable it
    with pytest.raises(AssertionError, match="pre_quant_scale should be positive"):
        apply_pre_quant_scale_and_smooth(linear, torch.zeros(linear.weight.shape[1]))


# ---------------------------------------------------------------------------
# svd helper
# ---------------------------------------------------------------------------


def test_svd_full_rank_reconstructs_weight():
    torch.manual_seed(19)
    weight = torch.randn(8, 12)

    us, vt = svd(weight, rank=8)

    assert us.shape == (8, 8)
    assert vt.shape == (8, 12)
    assert us.dtype == weight.dtype
    assert torch.allclose(us @ vt, weight, atol=1e-5)


def test_svd_low_rank_error_decreases_with_rank():
    torch.manual_seed(20)
    weight = torch.randn(8, 12)

    errors = []
    for rank in (1, 4, 8):
        us, vt = svd(weight, rank)
        assert us.shape == (8, rank)
        assert vt.shape == (rank, 12)
        errors.append((weight - us @ vt).norm().item())

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] == pytest.approx(0.0, abs=1e-4)


def test_svd_rank_exceeding_min_dim_pads_and_warns():
    torch.manual_seed(21)
    weight = torch.randn(4, 6)

    with pytest.warns(UserWarning, match="low-rank dimensions do not match"):
        us, vt = svd(weight, rank=10)

    assert us.shape == (4, 10)
    assert vt.shape == (10, 6)
    # The valid singular directions still reconstruct the weight; padding is zero.
    assert torch.allclose(us @ vt, weight, atol=1e-5)
    assert torch.all(us[:, 4:] == 0)
    assert torch.all(vt[4:, :] == 0)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("block_sizes", "expected"),
    [
        ({-1: 8}, 8),
        ({1: 16}, 16),
        (None, None),
    ],
)
def test_get_awq_quantizer_block_size(block_sizes, expected):
    quantizer = SimpleNamespace(block_sizes=block_sizes)
    assert _get_awq_quantizer_block_size(torch.empty(4, 8), quantizer) == expected


def test_get_awq_quantizer_block_size_rejects_other_axes():
    quantizer = SimpleNamespace(block_sizes={0: 8})
    with pytest.raises(ValueError, match="AWQ requires block quantization along -1 axis"):
        _get_awq_quantizer_block_size(torch.empty(4, 8), quantizer)


@pytest.mark.parametrize(
    ("parent_name", "expected"),
    [
        ("model.layers.0.mlp.experts.3.w1", True),
        ("model.layers.0.mlp.shared_experts.w1", False),
        ("model.layers.0.self_attn.q_proj", False),
        ("experts", True),
    ],
)
def test_is_routed_expert(parent_name, expected):
    assert _is_routed_expert(parent_name) is expected


@pytest.mark.parametrize(
    ("parent_name", "child_name", "sync_flag", "expected"),
    [
        # Routed-expert weights sync only when explicitly opted in.
        ("model.experts.3.w1", "weight_quantizer", False, False),
        ("model.experts.3.w1", "weight_quantizer", True, True),
        # Activation quantizers always sync, even for routed experts.
        ("model.experts.3.w1", "input_quantizer", False, True),
        # Shared experts and non-expert modules always sync.
        ("model.shared_experts.w1", "weight_quantizer", False, True),
        ("model.mlp.fc1", "weight_quantizer", False, True),
    ],
)
def test_should_sync_amax_across_ep(parent_name, child_name, sync_flag, expected):
    assert _should_sync_amax_across_ep(parent_name, child_name, sync_flag) is expected


def test_iter_leaf_quantizers_flattens_sequential():
    q1 = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    q2 = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    seq = SequentialQuantizer(q1, q2)

    assert list(_iter_leaf_quantizers(seq)) == [q1, q2]
    assert list(_iter_leaf_quantizers(q1)) == [q1]
