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

"""Tests for the P^2 quantile calibrator."""

import json

import numpy as np
import pytest
import torch

from modelopt.torch.quantization.calib.quantile import (
    P2QuantileEstimator,
    QuantileCalibrator,
    save_quantile_data,
)
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer


class TestP2QuantileEstimator:
    def test_known_quantile_uniform(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0.0, 1.0, size=100_000).astype(np.float64)
        est = P2QuantileEstimator(0.5)
        est.update_bulk(data)
        expected = float(np.quantile(data, 0.5))
        assert abs(est.estimate() - expected) < 0.01, (
            f"median estimate {est.estimate():.4f} too far from empirical {expected:.4f}"
        )

    def test_known_quantile_normal_tail(self):
        rng = np.random.default_rng(1)
        data = np.abs(rng.standard_normal(size=200_000)).astype(np.float64)
        est = P2QuantileEstimator(0.999)
        est.update_bulk(data)
        expected = float(np.quantile(data, 0.999))
        # P^2 is approximate by construction; extreme quantiles carry more
        # error than the median. For p=0.999 of |N(0,1)| (true value ~3.29),
        # 15% absolute error is well within published P^2 envelopes and is
        # the right order of magnitude for a streaming O(1)-memory estimator.
        rel_err = abs(est.estimate() - expected) / expected
        assert rel_err < 0.15, (
            f"0.999-quantile estimate {est.estimate():.4f} too far from empirical "
            f"{expected:.4f} (rel_err={rel_err:.3f})"
        )

    def test_known_quantile_extreme_tail(self):
        # Sanity for the extreme p used by the GLM-5.1 recipe.
        rng = np.random.default_rng(2)
        data = np.abs(rng.standard_normal(size=500_000)).astype(np.float64)
        est = P2QuantileEstimator(0.99999)
        est.update_bulk(data)
        expected = float(np.quantile(data, 0.99999))
        # Tail estimate at p=0.99999 with 500k samples is noisy by construction;
        # 0.5 absolute tolerance is enough to catch ordering / monotonicity bugs.
        assert abs(est.estimate() - expected) < 0.5

    def test_streaming_matches_batch(self):
        rng = np.random.default_rng(3)
        data = rng.uniform(-5.0, 5.0, size=50_000).astype(np.float64)
        data = np.abs(data)

        est_batch = P2QuantileEstimator(0.99)
        est_batch.update_bulk(data)

        est_chunked = P2QuantileEstimator(0.99)
        for chunk in np.array_split(data, 50):
            est_chunked.update_bulk(chunk)

        # Chunk-update is approximate to per-value but must be very close.
        assert abs(est_batch.estimate() - est_chunked.estimate()) < 0.05

    def test_state_dict_roundtrip(self):
        rng = np.random.default_rng(4)
        data = rng.uniform(0.0, 1.0, size=2_000).astype(np.float64)
        est = P2QuantileEstimator(0.9)
        est.update_bulk(data)

        state = est.state_dict()
        rehydrated = P2QuantileEstimator(0.9)
        rehydrated.load_state_dict(state)

        assert rehydrated.estimate() == est.estimate()
        assert rehydrated.count == est.count
        assert rehydrated.q == est.q
        assert rehydrated.n == est.n

    def test_handles_small_n(self):
        est = P2QuantileEstimator(0.5)
        # Fewer than 5 samples — graceful "return the max of buffer".
        est.update_bulk(np.array([0.1, 0.4, 0.2], dtype=np.float64))
        assert est.estimate() == pytest.approx(0.4)

        # Exactly 5 samples flushes the buffer and initialises markers.
        est.update_bulk(np.array([0.3, 0.5], dtype=np.float64))
        # With 5 distinct values [0.1,0.2,0.3,0.4,0.5], adaptive init for p=0.5
        # places q[2] at sorted[round(4*0.5)]=sorted[2]=0.3.
        assert est.estimate() == pytest.approx(0.3)

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            P2QuantileEstimator(0.0)
        with pytest.raises(ValueError):
            P2QuantileEstimator(1.0)
        with pytest.raises(ValueError):
            P2QuantileEstimator(1.5)


class TestQuantileCalibrator:
    def test_basic(self):
        torch.manual_seed(0)
        cal = QuantileCalibrator(8, None, False)
        x_1 = torch.randn(4096)
        x_2 = torch.randn(4096)
        cal.collect(x_1)
        cal.collect(x_2)

        amax = cal.compute_amax()
        assert amax is not None
        assert amax.dim() == 0  # per-tensor scalar
        assert torch.isfinite(amax)
        assert amax.item() > 0.0
        all_abs = torch.cat([x_1.abs(), x_2.abs()])
        # Quantile-based amax must not exceed the absolute max of the data.
        assert amax.item() <= all_abs.max().item() + 1e-6

    def test_overrides_emulate_toolkit(self):
        """Emulate quant-toolkit's `_override_quantile_levels`."""
        cal = QuantileCalibrator(8, None, False)
        cal._quantile_probs = [0.5, 0.99]
        cal._estimators = {p: P2QuantileEstimator(p) for p in cal._quantile_probs}

        rng = np.random.default_rng(7)
        data = torch.from_numpy(rng.uniform(0, 1, size=10_000).astype(np.float32))
        cal.collect(data)
        amax = cal.compute_amax()
        assert amax is not None
        # max over {0.5, 0.99} quantiles of |U(0,1)| ≈ the 0.99 estimate ≈ 0.99.
        assert 0.85 < amax.item() < 1.0

    def test_axis_unsupported(self):
        with pytest.raises(NotImplementedError):
            QuantileCalibrator(8, axis=0, unsigned=False)

    def test_reset(self):
        cal = QuantileCalibrator(8, None, False, quantiles=[0.5, 0.99])
        cal.collect(torch.randn(1024))
        first_amax = cal.compute_amax()
        assert first_amax is not None

        cal.reset()
        assert cal._calib_amax is None
        # Reset should reinstall fresh estimators.
        for est in cal._estimators.values():
            assert est.count == 0
            assert not est._initialized

    def test_empty_collect_is_safe(self):
        cal = QuantileCalibrator(8, None, False, quantiles=[0.5])
        cal.collect(torch.empty(0))
        # No data — compute_amax returns None like MaxCalibrator pre-collect.
        assert cal.compute_amax() is None

    def test_calibrator_string_registry(self):
        """Ensure TensorQuantizer dispatches `calibrator='quantile'` to QuantileCalibrator."""
        cfg = QuantizerAttributeConfig(num_bits=8, axis=None, calibrator="quantile")
        tq = TensorQuantizer(cfg)
        assert isinstance(tq._calibrator, QuantileCalibrator)
        # And the default quantile probs come through.
        assert tq._calibrator._quantile_probs == sorted([0.99, 0.999, 0.9999, 0.99999])

    def test_save_quantile_data_roundtrip(self, tmp_path):
        cfg = QuantizerAttributeConfig(num_bits=8, axis=None, calibrator="quantile")
        tq_a = TensorQuantizer(cfg)
        tq_b = TensorQuantizer(cfg)
        model = torch.nn.Sequential()
        model.add_module("a", tq_a)
        model.add_module("b", tq_b)

        # Push some data through both quantizers' calibrators.
        torch.manual_seed(11)
        for _ in range(3):
            tq_a._calibrator.collect(torch.randn(2048))
            tq_b._calibrator.collect(torch.randn(2048))

        out_path = tmp_path / "quantile_dump.json"
        n_saved = save_quantile_data(model, str(out_path))
        assert n_saved == 2

        with open(out_path) as f:
            payload = json.load(f)
        assert set(payload.keys()) == {"a", "b"}
        for entry in payload.values():
            # Default probs: 4 levels.
            assert len(entry) == 4
            for p_key, state in entry.items():
                assert "q" in state and "n" in state and "count" in state
                assert len(state["q"]) == 5 and len(state["n"]) == 5
                assert float(p_key) > 0.0 and float(p_key) < 1.0
