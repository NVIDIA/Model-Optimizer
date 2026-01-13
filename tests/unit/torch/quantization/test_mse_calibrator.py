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

"""Tests for per-tensor and per-channel MseCalibrator (MSE-based amax search)."""

import torch

from modelopt.torch.quantization import calib
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils import enable_fake_quant


# TODO: avoid code duplication in this file
def _mse_at_a(x: torch.Tensor, a: torch.Tensor, num_bits: int = 8, unsigned: bool = False):
    """Compute MSE at a given amax value."""
    qmin = 0 if unsigned else -(1 << (num_bits - 1))
    qmax = (1 << num_bits) - 1 if unsigned else (1 << (num_bits - 1)) - 1

    s = a / max(abs(qmin), abs(qmax))
    s = torch.clamp(s, min=torch.finfo(torch.float32).eps)
    q = torch.clamp(torch.round(x / s), qmin, qmax)
    xq = q * s
    return ((x - xq) ** 2).mean()


class TestMseCalibrator:
    def test_one_tensor_reduces_outlier_signed(self):
        torch.manual_seed(0)
        x = torch.ones(1024, dtype=torch.float32)
        x[0] = 10.0

        # Initial amax is the max of the tensor
        initial_amax = x.abs().max()

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=None, unsigned=False)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(
            amax=initial_amax,
            step_size=0.075,
            start_multiplier=0.1,
            stop_multiplier=1.5,
            quant_func=quant_func,
        )

        cal.collect(x)

        a_best = cal.compute_amax()

        assert torch.isfinite(a_best)
        assert 0 < a_best <= x.abs().max() * 1.5 + 1e-6

        loss_best = _mse_at_a(x, a_best, num_bits=8, unsigned=False)
        loss_bulk = _mse_at_a(x, torch.tensor(1.0, device=x.device), num_bits=8, unsigned=False)
        assert loss_best <= loss_bulk + 1e-6

    def test_one_tensor_reduces_negative_outlier_signed(self):
        torch.manual_seed(0)
        x = torch.rand(4096) * 2.0 - 1.0
        x[0] = -12.0

        initial_amax = x.abs().max()

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=None, unsigned=False)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(
            amax=initial_amax,
            step_size=0.045,
            start_multiplier=0.1,
            stop_multiplier=1.2,
            quant_func=quant_func,
        )

        cal.collect(x)

        a_best = cal.compute_amax()

        assert torch.isfinite(a_best)
        assert 0 < a_best <= x.abs().max() * 1.2 + 1e-6

        loss_best = _mse_at_a(x, a_best, num_bits=8, unsigned=False)
        loss_bulk = _mse_at_a(x, torch.tensor(1.0, device=x.device), num_bits=8, unsigned=False)
        assert loss_best <= loss_bulk + 1e-6

    def test_unsigned_one_tensor(self):
        torch.manual_seed(0)
        x = torch.ones(11, 7, 3, 3, dtype=torch.float32) * 512.0
        x[1, 1, 1, 1] = 513.0

        initial_amax = x.abs().max()

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=None, unsigned=True)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(
            amax=initial_amax,
            step_size=0.008,
            start_multiplier=0.8,
            stop_multiplier=1.2,
            quant_func=quant_func,
        )

        cal.collect(x)

        a_best = cal.compute_amax()

        assert torch.isfinite(a_best)
        assert 0 < a_best <= x.abs().max() * 1.2 + 1e-6

        # The calibrator should find a reasonable amax value
        # It should be better than using the max value directly for most distributions
        loss_best = _mse_at_a(x, a_best, num_bits=8, unsigned=True)

        # The found amax should be close to initial or potentially better
        # For this specific case with mostly 512s and one 513, using initial (513) is reasonable
        assert torch.isfinite(loss_best)
        assert loss_best < 1.0  # Should be much better than no quantization

    def test_multiple_collections_accumulate(self):
        torch.manual_seed(0)
        x1 = torch.ones(2048) * 2.0
        x1[0] = 16.0

        initial_amax = x1.abs().max()

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=None, unsigned=False)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(
            amax=initial_amax,
            step_size=0.075,
            start_multiplier=0.1,
            stop_multiplier=1.5,
            quant_func=quant_func,
        )

        cal.collect(x1)

        x2 = torch.ones(2048) * 2.0
        cal.collect(x2)

        a_best = cal.compute_amax()

        assert torch.isfinite(a_best)
        assert 0 < a_best <= initial_amax * 1.5 + 1e-6

    def test_custom_error_function(self):
        """Test that custom error function is used correctly."""
        torch.manual_seed(0)
        x = torch.ones(512) * 5.0
        x[0] = 10.0

        initial_amax = x.abs().max()

        # Custom error function (L1 loss instead of MSE)
        def l1_loss(x, xq):
            return torch.abs(x - xq).mean()

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=None, unsigned=False)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(
            amax=initial_amax,
            step_size=0.07,
            start_multiplier=0.5,
            stop_multiplier=1.5,
            quant_func=quant_func,
            error_func=l1_loss,
        )

        cal.collect(x)

        a_best = cal.compute_amax()

        assert torch.isfinite(a_best)
        assert 0 < a_best <= initial_amax * 1.5 + 1e-6

    def test_reset_clears_state(self):
        """Test that reset clears the calibrator state."""
        torch.manual_seed(0)
        x = torch.ones(512) * 2.0

        initial_amax = x.abs().max()

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=None, unsigned=False)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(amax=initial_amax, step_size=0.4, quant_func=quant_func)

        cal.collect(x)

        a_before_reset = cal.compute_amax()
        assert a_before_reset is not None

        cal.reset()
        a_after_reset = cal.compute_amax()
        assert a_after_reset is None

    def test_per_channel_basic(self):
        """Test per-channel MSE calibration with axis=0."""
        torch.manual_seed(0)
        # Create a weight tensor with 2 output channels, 3 input channels
        # W = [[1, 2, 3], [4, 5, 6]] (cout = 2, cin = 3)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Initial amax per channel: [[3], [6]]
        initial_amax = torch.tensor([[3.0], [6.0]])

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=0, unsigned=False)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(
            amax=initial_amax,
            axis=0,
            step_size=0.15,
            start_multiplier=0.5,
            stop_multiplier=2.0,
            quant_func=quant_func,
        )

        cal.collect(x)

        a_best = cal.compute_amax()

        # Check that best amax has the correct shape
        assert a_best.shape == initial_amax.shape
        assert a_best.numel() == 2  # Should have 2 channels
        assert torch.all(torch.isfinite(a_best))
        assert torch.all(a_best > 0)

    def test_per_channel_multiple_collections(self):
        """Test per-channel MSE calibration with multiple collections."""
        torch.manual_seed(0)
        # Initial amax per channel
        initial_amax = torch.tensor([[3.0], [6.0]])

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=0, unsigned=False)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(
            amax=initial_amax,
            axis=0,
            step_size=0.1,
            start_multiplier=0.5,
            stop_multiplier=2.0,
            quant_func=quant_func,
        )

        # Collect from multiple batches
        batch1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        batch2 = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        batch3 = torch.tensor([[1.2, 2.2, 3.2], [4.2, 5.2, 6.2]])

        cal.collect(batch1)
        cal.collect(batch2)
        cal.collect(batch3)

        a_best = cal.compute_amax()

        # Check that best amax has the correct shape
        assert a_best.shape == initial_amax.shape
        assert a_best.numel() == 2
        assert torch.all(torch.isfinite(a_best))
        assert torch.all(a_best > 0)

    def test_per_channel_independent_optimization(self):
        """Test that per-channel calibration optimizes each channel independently."""
        torch.manual_seed(0)
        # Create a tensor where channels have very different scales
        # Channel 0: small values (around 1.0)
        # Channel 1: large values (around 100.0)
        x = torch.zeros(2, 1000)
        x[0, :] = torch.randn(1000) * 0.5 + 1.0  # Mean ~1.0, std ~0.5
        x[1, :] = torch.randn(1000) * 5.0 + 100.0  # Mean ~100.0, std ~5.0

        # Initial amax per channel based on actual max
        initial_amax = torch.tensor([[x[0].abs().max()], [x[1].abs().max()]])

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=0, unsigned=False)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(
            amax=initial_amax,
            axis=0,
            step_size=0.05,
            start_multiplier=0.5,
            stop_multiplier=1.5,
            quant_func=quant_func,
        )

        cal.collect(x)

        a_best = cal.compute_amax()

        assert a_best.shape == initial_amax.shape
        assert a_best.numel() == 2
        assert torch.all(torch.isfinite(a_best))
        assert torch.all(a_best > 0)

        # The ratio of amax values should roughly reflect the ratio of scales
        # Channel 1 should have much larger amax than channel 0
        assert a_best[1, 0] > a_best[0, 0] * 10  # At least 10x larger

    def test_per_channel_with_custom_error_func(self):
        """Test per-channel MSE calibration with custom error function."""
        torch.manual_seed(0)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        initial_amax = torch.tensor([[3.0], [6.0]])

        # Custom error function (element-wise L1 loss)
        def l1_loss(x, xq):
            return torch.abs(x - xq)

        quant_cfg = QuantizerAttributeConfig(num_bits=8, axis=0, unsigned=False)
        tq = TensorQuantizer(quant_attribute_cfg=quant_cfg, amax=initial_amax)

        def quant_func(x, amax):
            original_amax = tq._amax.clone() if hasattr(tq, "_amax") else None
            was_quant_enabled = tq._if_quant
            was_calib_enabled = tq._if_calib

            tq._amax = amax
            tq._if_quant = True
            tq._if_calib = False

            with enable_fake_quant(tq):
                xq = tq(x)

            if original_amax is not None:
                tq._amax = original_amax
            tq._if_quant = was_quant_enabled
            tq._if_calib = was_calib_enabled
            return xq

        cal = calib.MseCalibrator(
            amax=initial_amax,
            axis=0,
            step_size=0.15,
            start_multiplier=0.5,
            stop_multiplier=2.0,
            quant_func=quant_func,
            error_func=l1_loss,
        )

        cal.collect(x)

        a_best = cal.compute_amax()

        assert a_best.shape == initial_amax.shape
        assert a_best.numel() == 2
        assert torch.all(torch.isfinite(a_best))
        assert torch.all(a_best > 0)
