# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for local Hessian-weighted MSE calibration (CPU)."""

import copy
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from _test_utils.torch.distributed.utils import bounded_collectives
from _test_utils.torch.quantization.models import SimpleConv, SimpleLinear

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import calib
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import (
    _FP8_SWEEP_CALIBRATOR_REGISTRY,
    _grouped_linear_slices,
    _LocalHessianAccumulator,
    _make_weight_mse_calibrator,
    _register_fp8_sweep_calibrator,
    _warn_if_block_size_mismatch,
    local_hessian_calibrate,
    lsq,
    mse_calibrate,
)
from modelopt.torch.quantization.nn import (
    GroupedQuantizer,
    QuantModule,
    SequentialQuantizer,
    StaticBlockScaleQuantizer,
    TensorQuantizer,
)
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    _QUANT_FUNCTIONAL_BACKENDS,
    register_quant_backend,
)
from modelopt.torch.quantization.utils import calib_utils
from modelopt.torch.utils.distributed import ParallelState

# Weight-only INT8 per-channel; calibration is re-run explicitly per test.
INT8_WEIGHT_CFG = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
    ],
    "algorithm": "max",
}


def _weight_amaxes(model):
    return {
        n: m.amax
        for n, m in model.named_modules()
        if isinstance(m, TensorQuantizer) and m.is_enabled and m.amax is not None
    }


def _make_forward_loop(seed=0):
    def forward_loop(model):
        torch.manual_seed(seed)
        for _ in range(3):
            x = torch.randn(8, 16)
            x[:, 0] *= 40.0  # skew so the Hessian is non-trivial vs plain weight MSE
            model(x)

    return forward_loop


class TestLocalHessianAccumulator:
    def test_accumulate_shape_samples_fp32_buffer(self):
        torch.manual_seed(0)
        acc = _LocalHessianAccumulator(8, 32, 16)
        assert acc.is_enabled
        acc.accumulate(torch.randn(10, 32, dtype=torch.bfloat16))
        assert acc.hessian_per_block.shape == (2, 16, 16)
        assert acc.hessian_per_block.dtype == torch.float32  # fp32 despite bf16 input
        acc.accumulate(torch.randn(5, 32))
        assert acc.num_samples == 15
        assert acc.build_error_func() is not None
        assert acc.hessian_per_block is None  # raw buffer freed

    def test_error_func_matches_explicit_hessian_weighted_loss(self):
        torch.manual_seed(1)
        cout, cin, bs = 4, 32, 16
        n_blocks = cin // bs
        acc = _LocalHessianAccumulator(cout, cin, bs)
        x = torch.randn(7, cin)
        acc.accumulate(x)
        error_func = acc.build_error_func()

        xb = x.reshape(-1, cin).T.reshape(n_blocks, bs, -1)
        hessian = (xb @ xb.transpose(-1, -2)) / acc.num_samples
        w = torch.randn(cout * n_blocks, bs)
        wq = w + 0.05 * torch.randn_like(w)
        err = error_func(w, wq).view(-1, bs)

        assert err.shape == (cout * n_blocks, bs)
        assert torch.allclose(err, err[:, :1].expand(-1, bs))  # per-block scalar broadcast
        dw = (w - wq).view(cout, n_blocks, bs)
        expected = torch.einsum("cnb,nbd,cnd->cn", dw, hessian, dw).reshape(-1)
        assert torch.allclose(err[:, 0], expected, atol=1e-5)

    def test_returns_none_when_disabled_or_no_samples(self):
        not_divisible = _LocalHessianAccumulator(8, 30, 16)
        assert not not_divisible.is_enabled
        not_divisible.accumulate(torch.randn(4, 30))  # no-op
        assert not_divisible.build_error_func() is None
        assert _LocalHessianAccumulator(8, 32, 16).build_error_func() is None  # no samples


class TestLocalHessianCalibrateDense:
    def test_refines_amax_beyond_max_and_plain_mse(self):
        forward_loop = _make_forward_loop()
        torch.manual_seed(0)
        model_lh = SimpleLinear()
        mtq.quantize(model_lh, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        max_amax = {n: a.clone() for n, a in _weight_amaxes(model_lh).items()}
        local_hessian_calibrate(model_lh, forward_loop, fp8_scale_sweep=False, debug=True)

        torch.manual_seed(0)
        model_mse = SimpleLinear()
        mtq.quantize(model_mse, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        mse_calibrate(model_mse, forward_loop, fp8_scale_sweep=False)

        accs = model_lh._local_hessian_accumulators
        assert accs and all(a.num_samples > 0 for a in accs.values())
        lh, mse = _weight_amaxes(model_lh), _weight_amaxes(model_mse)
        assert all(torch.isfinite(a).all() and (a > 0).all() for a in lh.values())
        assert any(not torch.allclose(lh[n], max_amax[n]) for n in lh)  # refined past max-cal
        assert any(not torch.allclose(lh[n], mse[n]) for n in lh)  # Hessian changed the choice

    def test_bf16_hessian_input_on_cpu_gives_the_float32_result_and_warns_once(self):
        """CPU GEMMs take no ``out_dtype``, so ``hessian_input_dtype="bfloat16"`` falls back to
        the float32 Hessians, warning once however many batches they read."""
        calib_utils._warn_bfloat16_hessian_unavailable.cache_clear()
        amaxes = {}
        for hessian_input_dtype in ("float32", "bfloat16"):
            torch.manual_seed(0)
            model = SimpleLinear()
            mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=_make_forward_loop())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                local_hessian_calibrate(
                    model,
                    _make_forward_loop(),
                    fp8_scale_sweep=False,
                    hessian_input_dtype=hessian_input_dtype,
                )
            fallbacks = [w for w in caught if "hessian_input_dtype" in str(w.message)]
            assert len(fallbacks) == (hessian_input_dtype == "bfloat16")
            amaxes[hessian_input_dtype] = _weight_amaxes(model)
        for name, amax in amaxes["float32"].items():
            assert torch.equal(amax, amaxes["bfloat16"][name])

    def test_warns_with_module_name_when_cin_not_divisible(self):
        class _OddModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.odd = nn.Linear(24, 32)  # 24 not divisible by block_size 16

            def forward(self, x):
                return self.odd(x)

        torch.manual_seed(0)
        model = _OddModel()
        forward_loop = lambda m: m(torch.randn(4, 24))  # noqa: E731
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        with pytest.warns(UserWarning, match=r"odd input features \(24\) not divisible"):
            local_hessian_calibrate(model, forward_loop, fp8_scale_sweep=False)

    def test_no_forward_loop_is_skipped(self):
        torch.manual_seed(0)
        model = SimpleLinear()
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=_make_forward_loop())
        before = {n: a.clone() for n, a in _weight_amaxes(model).items()}
        with pytest.warns(UserWarning, match="forward_loop must be provided"):
            local_hessian_calibrate(model, forward_loop=None)
        assert all(torch.equal(before[n], a) for n, a in _weight_amaxes(model).items())


class TestLocalHessianFallbacks:
    """Weights local-Hessian can't pair with an input fall back to plain MSE (no Hessian)."""

    def test_conv_weight_falls_back_without_crash(self):
        torch.manual_seed(0)
        model = SimpleConv()  # 4-D conv weights — no single 2-D weight to pair
        forward_loop = lambda m: m(SimpleConv.get_input())  # noqa: E731
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=forward_loop)
        local_hessian_calibrate(model, forward_loop, fp8_scale_sweep=False, debug=True)
        conv = model.net[0]
        assert id(conv.weight_quantizer) not in model._local_hessian_accumulators
        assert conv.weight_quantizer.amax is not None  # still calibrated via plain MSE

    def test_sequential_quantizer_weight_falls_back_without_crash(self):
        torch.manual_seed(0)
        model = SimpleLinear()
        mtq.quantize(model, INT8_WEIGHT_CFG, forward_loop=_make_forward_loop())
        linear = model.net[0]
        linear.weight_quantizer = SequentialQuantizer(TensorQuantizer(), TensorQuantizer())
        local_hessian_calibrate(model, _make_forward_loop(), fp8_scale_sweep=False, debug=True)
        assert id(linear.weight_quantizer) not in model._local_hessian_accumulators


class TestBlockSizeMismatchWarning:
    def test_block_size_mismatch_warns_only_on_mismatch(self):
        def q(block):
            return TensorQuantizer(
                QuantizerAttributeConfig(
                    num_bits=(2, 1), block_sizes={-1: block, "type": "static", "scale_bits": (4, 3)}
                )
            )

        with pytest.warns(UserWarning, match="will not align"):
            _warn_if_block_size_mismatch(q(32), 16, "layer")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_block_size_mismatch(q(16), 16, "layer")  # matching block
            per_channel = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=0))
            _warn_if_block_size_mismatch(per_channel, 16, "layer")  # no block_sizes


class TestMakeWeightMseCalibratorErrorFunc:
    def setup_method(self):
        self._orig_fp8_registry = dict(_FP8_SWEEP_CALIBRATOR_REGISTRY)
        self._orig_quant_backends = dict(_QUANT_FUNCTIONAL_BACKENDS)

    def teardown_method(self):
        _FP8_SWEEP_CALIBRATOR_REGISTRY.clear()
        _FP8_SWEEP_CALIBRATOR_REGISTRY.update(self._orig_fp8_registry)
        _QUANT_FUNCTIONAL_BACKENDS.clear()
        _QUANT_FUNCTIONAL_BACKENDS.update(self._orig_quant_backends)

    def _make_quantizer(self, backend=None):
        q = TensorQuantizer(QuantizerAttributeConfig(num_bits=8, axis=None, backend=backend))
        q.amax = torch.tensor(1.0)
        return q

    def test_error_func_threaded_to_mse_calibrator(self):
        marker = lambda x, xq: (x - xq) ** 2  # noqa: E731
        cal = _make_weight_mse_calibrator(
            self._make_quantizer(), 0.1, 0.25, 4.0, fp8_scale_sweep=False, error_func=marker
        )
        assert isinstance(cal, calib.MseCalibrator)
        assert cal._error_func is marker

    def test_registered_backend_with_error_func_is_skipped(self):
        register_quant_backend("_lh_test_backend", lambda x, tq: x)
        _register_fp8_sweep_calibrator(
            "_lh_test_backend",
            lambda amax, axis, qf: calib.MseCalibrator(amax=amax, axis=axis, quant_func=qf),
        )
        q = self._make_quantizer(backend="_lh_test_backend")
        with pytest.warns(UserWarning, match="does not support a custom error"):
            cal = _make_weight_mse_calibrator(
                q, 0.1, 0.25, 4.0, fp8_scale_sweep=True, error_func=lambda x, xq: x
            )
        assert cal is None


class _RawTEGroupedLinear(nn.Module):
    """Unquantized CPU stand-in with ``TEGroupedLinear``'s weight0..N and ``m_splits`` layout."""

    def __init__(self, num_gemms: int, in_features: int, out_features: int):
        super().__init__()
        self.num_gemms = num_gemms
        for i in range(num_gemms):
            self.register_parameter(
                f"weight{i}", nn.Parameter(torch.randn(out_features, in_features))
            )


class _QuantFakeTEGroupedLinear(QuantModule):
    """Quantized CPU stand-in matching ``_QuantTEGroupedLinear``'s attribute and forward contract."""

    def _setup(self):
        self.input_quantizer = TensorQuantizer()
        self.weight_quantizer = GroupedQuantizer(
            *(TensorQuantizer() for _ in range(self.num_gemms))
        )
        self.output_quantizer = TensorQuantizer()
        self.output_quantizer.disable()

    def iter_weights_for_calibration(self):
        for i in range(self.num_gemms):
            yield getattr(self, f"weight{i}"), self.weight_quantizer[i]

    def forward(self, inp: torch.Tensor, m_splits):
        inp = self.input_quantizer(inp)
        splits = m_splits.tolist() if isinstance(m_splits, torch.Tensor) else list(m_splits)
        chunks = torch.split(inp.reshape(-1, inp.shape[-1]), splits)
        outs = [
            nn.functional.linear(chunk, self.weight_quantizer[i](getattr(self, f"weight{i}")))
            for i, chunk in enumerate(chunks)
        ]
        return self.output_quantizer(torch.cat(outs, dim=0))


class _HybridDenseAndGroupedModel(nn.Module):
    """Minimal dense + TEGroupedLinear model for CPU LSQ/local_hessian and DP sync tests."""

    def __init__(self, num_gemms: int = 3, hidden: int = 16, expert_parallel_state=None):
        super().__init__()
        self.dense = mtq.nn.QuantLinear(hidden, hidden, bias=False)
        raw = _RawTEGroupedLinear(num_gemms, hidden, hidden)
        if expert_parallel_state is not None:
            raw._parallel_state = expert_parallel_state
        self.experts = _QuantFakeTEGroupedLinear.convert(raw)

    def forward(self, x: torch.Tensor, m_splits, use_kwargs: bool = False):
        h = self.dense(x)
        if use_kwargs:
            return self.experts(h, m_splits=m_splits)
        return self.experts(h, m_splits)


_STATIC_INT4_QUANT_CFG = [
    {"quantizer_name": "*", "enable": False},
    {
        "quantizer_name": "*weight_quantizer",
        "cfg": {"num_bits": 4, "block_sizes": {-1: 16, "type": "static"}},
    },
]


class TestLocalHessianTEGroupedLinear:
    def test_captures_per_expert_hessians_and_refines_lsq(self):
        """TEGroupedLinear splits inputs by m_splits per expert instead of falling back to MSE."""
        torch.manual_seed(0)
        model_lh = _HybridDenseAndGroupedModel(num_gemms=3, hidden=16)
        model_mse = copy.deepcopy(model_lh)

        # Expert 0 gets skewed rows via list m_splits; expert 1 gets different skew via tensor
        # m_splits (and kwargs); expert 2 receives 0 tokens across all batches.
        x0 = torch.randn(5, 16)
        x0[:3, 0] *= 40.0
        x0[3:, 1] *= 40.0
        x1 = torch.randn(4, 16)
        x1[:, 2] *= 40.0
        x_empty = torch.empty(0, 16)

        def forward_loop(m):
            m(x0, [3, 2, 0])
            m(x1, torch.tensor([0, 4, 0]), use_kwargs=True)
            m(x_empty, [0, 0, 0])

        lh_cfg = {
            "quant_cfg": _STATIC_INT4_QUANT_CFG,
            "algorithm": {
                "method": "lsq",
                "scale_algorithm": {
                    "method": "local_hessian",
                    "fp8_scale_sweep": False,
                    "debug": True,
                },
            },
        }
        mse_cfg = {
            "quant_cfg": _STATIC_INT4_QUANT_CFG,
            "algorithm": {
                "method": "lsq",
                "scale_algorithm": {"method": "mse", "fp8_scale_sweep": False},
            },
        }
        mtq.quantize(model_lh, lh_cfg, forward_loop=forward_loop)
        mtq.quantize(model_mse, mse_cfg, forward_loop=forward_loop)

        accs = model_lh._local_hessian_accumulators
        wqs = model_lh.experts.weight_quantizer
        assert accs[id(wqs[0])].num_samples == 3
        assert accs[id(wqs[1])].num_samples == 6
        assert id(wqs[2]) not in accs  # unrouted expert allocates no Hessian

        # Verify each routed expert's Hessian matches its exact routed hidden activations.
        with torch.no_grad():
            h0 = model_lh.dense.weight_quantizer
            assert isinstance(h0, StaticBlockScaleQuantizer) and h0._lsq
            dense_h0 = nn.functional.linear(x0, model_lh.dense.weight)
            dense_h1 = nn.functional.linear(x1, model_lh.dense.weight)
            e0_in = dense_h0[:3]
            e1_in = torch.cat([dense_h0[3:], dense_h1], dim=0)
            torch.testing.assert_close(
                accs[id(wqs[0])].hessian_per_block, (e0_in.T @ e0_in).unsqueeze(0)
            )
            torch.testing.assert_close(
                accs[id(wqs[1])].hessian_per_block, (e1_in.T @ e1_in).unsqueeze(0)
            )

        # Routed experts refine LSQ _amax_post differently from plain MSE; unrouted matches MSE.
        wqs_mse = model_mse.experts.weight_quantizer
        assert all(q._lsq for q in wqs)
        assert any(not torch.allclose(wqs[i]._amax_post, wqs_mse[i]._amax_post) for i in (0, 1))
        assert torch.equal(wqs[2]._amax_post, wqs_mse[2]._amax_post)

    @pytest.mark.parametrize(
        "bad_splits",
        [[2, 2], [-1, 3, 2], [2, 3, 2], torch.tensor([[2, 2, 2]])],
        ids=["wrong-len", "negative", "exceeds-rows", "not-1d"],
    )
    def test_grouped_linear_slices_rejects_invalid_m_splits(self, bad_splits):
        assert list(_grouped_linear_slices([2, 0, 3], 3, 6)) == [(0, 0, 2), (2, 2, 5)]
        with pytest.raises(RuntimeError):
            list(_grouped_linear_slices(bad_splits, 3, 6))


def _dp_local_hessian_lsq_job(rank, size):
    """Multi-rank DP Hessian sync for dense + TEGroupedLinear under LSQ, plus opt-out & EP cases."""
    assert size == 2
    # Distinct per-rank calibration inputs and asymmetric MoE routing:
    # - Expert 0: routed on both ranks with different token counts (3 vs 5) and opposite skew.
    # - Expert 1: routed ONLY on rank 0 (0 tokens on rank 1) -> must not deadlock and must match.
    # - Expert 2: routed on neither rank -> skipped on both ranks without Hessian allocation.
    torch.manual_seed(123)
    x_r0 = torch.randn(7, 16)
    x_r0[:3, 0] *= 50.0
    x_r0[3:, 4] *= 50.0
    splits_r0 = [3, 4, 0]

    x_r1 = torch.randn(5, 16)
    x_r1[:, 8] *= 50.0
    splits_r1 = [5, 0, 0]

    local_x, local_splits = (x_r0, splits_r0) if rank == 0 else (x_r1, splits_r1)

    def make_model(expert_ps=None):
        torch.manual_seed(0)
        return _HybridDenseAndGroupedModel(num_gemms=3, hidden=16, expert_parallel_state=expert_ps)

    def run_lsq_lh(model, batches, distributed_sync=True):
        mtq.set_quantizer_by_cfg(model, _STATIC_INT4_QUANT_CFG)
        lsq(
            model,
            forward_loop=lambda m: [m(x, sp) for x, sp in batches],
            scale_algorithm={
                "method": "local_hessian",
                "distributed_sync": distributed_sync,
                "fp8_scale_sweep": False,
                "debug": True,
            },
        )
        return model

    # 1. Synced DP run vs pooled single-rank reference (inside bounded_collectives to fail fast).
    with bounded_collectives():
        dp_model = run_lsq_lh(make_model(), [(local_x, local_splits)], distributed_sync=True)

    ref_model = run_lsq_lh(
        make_model(), [(x_r0, splits_r0), (x_r1, splits_r1)], distributed_sync=False
    )

    # Every weight's synced Hessian, sample count, and initial LSQ _amax_post/_amax_pre must match
    # both across DP ranks and with the pooled single-rank reference.
    dp_accs = dp_model._local_hessian_accumulators
    ref_accs = ref_model._local_hessian_accumulators
    pairs = [
        ("dense", dp_model.dense.weight_quantizer, ref_model.dense.weight_quantizer, 12),
        (
            "expert0",
            dp_model.experts.weight_quantizer[0],
            ref_model.experts.weight_quantizer[0],
            8,
        ),
        (
            "expert1",
            dp_model.experts.weight_quantizer[1],
            ref_model.experts.weight_quantizer[1],
            4,
        ),
        (
            "expert2",
            dp_model.experts.weight_quantizer[2],
            ref_model.experts.weight_quantizer[2],
            0,
        ),
    ]
    for label, q_dp, q_ref, expected_samples in pairs:
        if expected_samples > 0:
            assert dp_accs[id(q_dp)].num_samples == expected_samples, label
            torch.testing.assert_close(
                dp_accs[id(q_dp)].normalized_hessian(),
                ref_accs[id(q_ref)].normalized_hessian(),
                msg=label,
            )
        else:
            assert id(q_dp) not in dp_accs, label
        # Peer rank comparison: broadcast rank 0's _amax_post/_amax_pre and require exact equality.
        for attr in ("_amax_post", "_amax_pre"):
            local_t = getattr(q_dp, attr).detach().clone()
            r0_t = local_t.clone()
            dist.broadcast(r0_t, src=0)
            assert torch.equal(local_t, r0_t), f"{label}.{attr} diverged across DP ranks"
            assert torch.equal(local_t, getattr(q_ref, attr)), f"{label}.{attr} != pooled ref"

    # 2. distributed_sync=False keeps rank-local Hessians (so ranks diverge as expected).
    nosync_model = run_lsq_lh(make_model(), [(local_x, local_splits)], distributed_sync=False)
    e1_q = nosync_model.experts.weight_quantizer[1]
    assert (id(e1_q) in nosync_model._local_hessian_accumulators) == (rank == 0)

    # 3. EP-sharded experts (data_parallel_group=-1, expert_model_parallel_group=None) must NOT
    # merge Hessians across EP ranks (different ranks own different experts).
    ep_ps = ParallelState(data_parallel_group=-1, expert_model_parallel_group=None)
    with bounded_collectives():
        ep_model = run_lsq_lh(make_model(ep_ps), [(local_x, local_splits)], distributed_sync=True)
    ep_accs = ep_model._local_hessian_accumulators
    assert ep_accs[id(ep_model.dense.weight_quantizer)].num_samples == 12  # dense still DP-synced
    assert ep_accs[id(ep_model.experts.weight_quantizer[0])].num_samples == (3 if rank == 0 else 5)
    assert (id(ep_model.experts.weight_quantizer[1]) in ep_accs) == (rank == 0)


class TestLocalHessianDistributedSync:
    def test_dp_sync_matches_pooled_hessian_and_agrees_on_lsq_amax(self, cpu_dist_workers):
        """DP sync merges per-block Hessians (incl. sparse TEGroupedLinear experts) and unifies LSQ."""
        cpu_dist_workers.run(_dp_local_hessian_lsq_job)
