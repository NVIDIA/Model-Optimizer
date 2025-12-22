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

"""Tests of calibrators"""

import numpy as np
import pytest
import torch
from _test_utils.torch.quantization.models import QuantConvLinear

from modelopt.torch.quantization import calib
from modelopt.torch.quantization import nn as qnn
from modelopt.torch.quantization import utils as quant_utils


class TestMaxCalibrator:
    def test_simple_run(self):
        max_calibrator = calib.MaxCalibrator(8, None, False)

        x_1 = torch.rand(16)
        x_2 = torch.rand(16)
        max_calibrator.collect(x_1)
        max_calibrator.collect(x_2)

        assert torch.allclose(
            max_calibrator.compute_amax(), torch.max(x_1.max(), x_2.max()), atol=0, rtol=0
        )

        # Nothing to test other than creation
        max_calibrator = calib.MaxCalibrator(8, None, True)

    @pytest.mark.parametrize("axis", [0, -4])
    def test_fine_grain(self, axis):
        reduce_axis = (1, 2, 3)
        max_calibrator = calib.MaxCalibrator(8, axis, False)

        x_1 = torch.rand(3, 4, 2, 2)
        x_2 = torch.rand(3, 4, 2, 2)
        max_calibrator.collect(x_1)
        max_calibrator.collect(x_2)

        assert max_calibrator.compute_amax().shape[0] == 3

        assert torch.allclose(
            max_calibrator.compute_amax(),
            quant_utils.reduce_amax(torch.max(x_1, x_2), axis=reduce_axis),
            atol=0,
            rtol=0,
        )

        max_calibrator.reset()
        assert max_calibrator.compute_amax() is None

    def test_track_amax(self):
        max_calibrator = calib.MaxCalibrator(8, None, False, track_amax=True)

        x_1 = torch.rand(16)
        x_2 = torch.rand(16)
        max_calibrator.collect(x_1)
        max_calibrator.collect(x_2)

        assert torch.allclose(
            max_calibrator.compute_amax(), torch.max(x_1.max(), x_2.max()), atol=0, rtol=0
        )
        np.testing.assert_array_equal(max_calibrator.amaxs[0], x_1.max().cpu().numpy())
        np.testing.assert_array_equal(max_calibrator.amaxs[1], x_2.max().cpu().numpy())

    def test_track_amax_raises(self):
        axis = 0
        max_calibrator = calib.MaxCalibrator(8, axis, False)

        x_2 = torch.rand(3, 4, 2, 2)
        x_3 = torch.rand(4, 4, 2, 2)
        max_calibrator.collect(x_2)
        with pytest.raises(RuntimeError, match="shape changed"):
            max_calibrator.collect(x_3)


@pytest.mark.manual(reason="slow test, run with --run-manual")
class TestHistogramCalibrator:
    def test_grow(self, verbose):
        x_1 = torch.tensor([0, 255, 255, 255, 255, 255])
        x_2 = torch.tensor([0, 255, 255, 255, 255, 256])

        hist_calibrator = calib.HistogramCalibrator(
            8, None, False, num_bins=2048, grow_method="stretch"
        )
        hist_calibrator.collect(x_1)
        hist_calibrator.collect(x_2)

        amax = hist_calibrator.compute_amax(method="entropy", start_bin=128)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        # amax should be closer to 256 because the last bin gets stretched to (~255, 257)
        assert (amax - 255.0).abs() < (amax - 256.0).abs()

        hist_calibrator = calib.HistogramCalibrator(
            8, None, False, num_bins=2048, grow_method="append"
        )
        hist_calibrator.collect(x_1)
        hist_calibrator.collect(x_2)

        amax = hist_calibrator.compute_amax(method="mse", start_bin=128)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        # amax should be closer to 255
        assert (amax - 255.0).abs() < 0.5

    def test_skip_zeros(self, verbose):
        x_1 = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5])
        x_2 = torch.tensor([0, 0, 0, 0, 0, 6, 7, 8, 9, 10])

        calibrator = calib.HistogramCalibrator(8, None, False, num_bins=2048, skip_zeros=True)
        calibrator.collect(x_1)
        calibrator.collect(x_2)

        amax = calibrator.compute_amax("percentile", percentile=50, start_bin=128)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        # amax should be close to 5
        assert (amax - 5.0).abs() < 10 / 2048

    def test_torch_hist(self):
        torch.manual_seed(0)
        x_1 = torch.rand(15)
        x_1[0] = 0
        x_2 = torch.rand(15) + 1  # Make sure histogram bins need to be grown
        x_2[1] = 0

        calibrator_np = calib.HistogramCalibrator(8, None, False, num_bins=19, torch_hist=False)
        calibrator_torch = calib.HistogramCalibrator(8, None, False, num_bins=19, torch_hist=True)

        calibrator_np.collect(x_1)
        calibrator_torch.collect(x_1)
        assert calibrator_torch._calib_hist.numel() == calibrator_torch._calib_bin_edges.numel() - 1
        np.testing.assert_array_equal(
            calibrator_np._calib_hist, calibrator_torch._calib_hist.cpu().numpy()
        )
        np.testing.assert_array_almost_equal(
            calibrator_np._calib_bin_edges, calibrator_torch._calib_bin_edges.cpu().numpy()
        )

        # Test multiple collections with some of them needs to expand range
        for _ in range(3):
            calibrator_np.collect(x_2)
            calibrator_torch.collect(x_2)
            calibrator_np.collect(x_1)
            calibrator_torch.collect(x_1)

            # Test compute_amax function doesn't convert _calib_hist and _calib_bin_edges unnecessarily
            calibrator_np.compute_amax("percentile", percentile=99.99)
            calibrator_torch.compute_amax("percentile", percentile=99.99)

            np.testing.assert_array_equal(
                calibrator_np._calib_hist, calibrator_torch._calib_hist.cpu().numpy()
            )
            np.testing.assert_array_almost_equal(
                calibrator_np._calib_bin_edges, calibrator_torch._calib_bin_edges.cpu().numpy()
            )
            assert (
                calibrator_torch._calib_hist.numel()
                == calibrator_torch._calib_bin_edges.numel() - 1
            )


@pytest.mark.manual(reason="slow test, run with --run-manual")
class TestEntropyCalibrator:
    def test_one_tensor(self, verbose):
        hist_calibrator = calib.HistogramCalibrator(
            8, None, False, num_bins=2048, grow_method="stretch"
        )

        x_2 = torch.rand(11, 7, 3, 3)  # uniform in (0,1)
        x_2[1, 1, 1, 1] = 10.0  # create outlier
        hist_calibrator.collect(x_2)

        # Don't have a better test metric. One outlier 10 should be discarded by KL-divergence
        amax = hist_calibrator.compute_amax("entropy", start_bin=128)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        assert amax < 1.1

    def test_unsigned(self, verbose):
        hist_calibrator = calib.HistogramCalibrator(
            8, None, True, num_bins=2048, grow_method="stretch"
        )

        x_2 = torch.rand(11, 7, 3, 3)  # uniform in (0,1)
        x_2[1, 1, 1, 1] = 10.0  # create outlier
        hist_calibrator.collect(x_2)

        amax = hist_calibrator.compute_amax("entropy", start_bin=128)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        assert amax < 1.1

    @pytest.mark.parametrize("torch_hist", [False, True])
    def test_two_tensor(self, torch_hist, verbose):
        hist_calibrator = calib.HistogramCalibrator(
            8, None, False, num_bins=2048, torch_hist=torch_hist
        )

        x_2 = torch.rand(11, 7, 3, 3)  # uniform in (0,1)
        x_2[1, 1, 1, 1] = 10.0  # create outlier

        x_2 = torch.rand(11, 7, 3, 3)  # uniform in (0,1)
        x_2[1, 1, 1, 1] = 10.0  # create outlier
        hist_calibrator.collect(x_2)
        x_3 = torch.rand(11, 7, 3, 3)
        hist_calibrator.collect(x_3)

        # Don't have a better test metric. One outlier 10 should be discarded by KL-divergence
        amax = hist_calibrator.compute_amax("entropy", start_bin=128)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        assert amax < 1.1

    def test_repr(self):
        hist_calibrator = calib.HistogramCalibrator(8, None, True, num_bins=2048)
        repr(hist_calibrator)


@pytest.mark.manual(reason="slow test, run with --run-manual")
class TestMSECalibrator:
    def test_one_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False, num_bins=32)

        x_1 = torch.ones(4, 4, 4) * 255.0
        x_1[1, 1, 1] = 256.0  # create an outlier
        calibrator.collect(x_1)

        amax = calibrator.compute_amax("mse", start_bin=16)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        # amax should be closer to 255
        assert (amax - 255.0).abs() < (amax - 256.0).abs()

    def test_unsigned_one_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator(8, None, True, num_bins=32)

        x_1 = torch.ones(11, 7, 3, 3) * 512.0
        x_1[1, 1, 1, 1] = 513.0  # create an outlier
        calibrator.collect(x_1)

        amax = calibrator.compute_amax("mse", start_bin=8)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        # amax should be closer to 512
        assert (amax - 512.0).abs() < (amax - 513.0).abs()

    @pytest.mark.parametrize("torch_hist", [False, True])
    def test_two_tensor(self, torch_hist, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False, torch_hist=torch_hist)

        x_1 = torch.ones(11, 7, 3, 3) * 255.0
        x_1[1, 1, 1, 1] = 256.0  # create an outlier
        calibrator.collect(x_1)
        x_2 = torch.ones(11, 7, 3, 3) * 255.0
        calibrator.collect(x_2)

        amax = calibrator.compute_amax("mse")

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        # amax should be closer to 255
        assert (amax - 255.0).abs() < (amax - 256.0).abs()

    def test_repr(self):
        calibrator = calib.HistogramCalibrator(8, None, False)
        repr(calibrator)


@pytest.mark.manual(reason="slow test, run with --run-manual")
class TestPercentileCalibrator:
    def test_one_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False)

        x_1 = torch.arange(100)
        calibrator.collect(x_1)

        amax = calibrator.compute_amax("percentile", percentile=90)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        # amax should be approximately 89
        assert (amax - 89.0).abs() < 100 / 1024

    def test_unsigned_one_tensor(self, verbose):
        calibrator = calib.HistogramCalibrator(8, None, True)

        x_1 = torch.arange(100)
        calibrator.collect(x_1)

        amax = calibrator.compute_amax("percentile", percentile=80)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        # amax should be approximately 79
        assert (amax - 79.0).abs() < 100 / 2048

    @pytest.mark.parametrize("torch_hist", [False, True])
    def test_two_tensor(self, torch_hist, verbose):
        calibrator = calib.HistogramCalibrator(8, None, False, torch_hist=torch_hist)

        x_1 = torch.arange(100)
        calibrator.collect(x_1)
        x_2 = torch.arange(0, 50, 0.5)
        calibrator.collect(x_2)
        amax = calibrator.compute_amax("percentile", percentile=99)

        if verbose:
            print(f"amax={amax.item():.4f}", end=" ")

        # amax should be approximately 97
        assert (amax - 97.0).abs() < 100 / 1024

    def test_repr(self):
        calibrator = calib.HistogramCalibrator(8, None, False)
        repr(calibrator)

    def test_range(self):
        calibrator = calib.HistogramCalibrator(8, None, False)
        x_1 = torch.arange(100)
        calibrator.collect(x_1)
        with pytest.raises(ValueError, match="range"):
            calibrator.compute_amax("percentile", percentile=-10)
        with pytest.raises(ValueError, match="range"):
            calibrator.compute_amax("percentile", percentile=200)


@pytest.mark.manual(reason="slow test, run with --run-manual")
class TestCalibrateWeights:
    def test_max(self):
        ref_lenet = QuantConvLinear()
        test_lenet = QuantConvLinear()
        test_lenet.load_state_dict(ref_lenet.state_dict())

        for module in ref_lenet.modules():
            if isinstance(module, (qnn.QuantConv2d, qnn.QuantLinear)):
                module.weight_quantizer.enable_calib()
                module.weight_quantizer.disable_quant()
                module.weight_quantizer(module.weight)
                module.weight_quantizer.load_calib_amax()

        calib.calibrate_weights(test_lenet, method="max")

        for ref_module, test_module in zip(ref_lenet.modules(), test_lenet.modules()):
            if isinstance(ref_module, (qnn.QuantConv2d, qnn.QuantLinear)):
                assert torch.allclose(
                    ref_module.weight_quantizer.amax,
                    test_module.weight_quantizer.amax,
                    rtol=0,
                    atol=0,
                )
                assert (
                    ref_module.weight_quantizer.amax.shape
                    == test_module.weight_quantizer.amax.shape
                )

    def test_shape_with_axis(self):
        """Check calibrate_weight function returns same shape as TensorQuantizer"""
        ref_lenet = QuantConvLinear()
        test_lenet = QuantConvLinear()
        test_lenet.load_state_dict(ref_lenet.state_dict())

        for module in ref_lenet.modules():
            if isinstance(module, (qnn.QuantConv2d, qnn.QuantLinear)):
                module.weight_quantizer.enable_calib()
                module.weight_quantizer.disable_quant()
                module.weight_quantizer(module.weight)
                module.weight_quantizer.load_calib_amax()

        calib.calibrate_weights(test_lenet, method="percentile")

        for ref_module, test_module in zip(ref_lenet.modules(), test_lenet.modules()):
            if isinstance(ref_module, (qnn.QuantConv2d, qnn.QuantLinear)):
                assert (
                    ref_module.weight_quantizer.amax.shape
                    == test_module.weight_quantizer.amax.shape
                )

    @pytest.mark.parametrize("method", ["mse", "percentile"])
    def test_per_tensor(self, method):
        test_lenet = QuantConvLinear()

        ref_calibrator = calib.HistogramCalibrator(8, None, False)

        calib.calibrate_weights(test_lenet, method=method, perchannel=False)
        ref_calibrator.collect(test_lenet.conv1.weight)
        ref_amax = ref_calibrator.compute_amax(method)
        assert torch.allclose(ref_amax, test_lenet.conv1.weight_quantizer.amax, rtol=0, atol=0)

    @pytest.mark.parametrize("method", ["mse", "percentile"])
    def test_with_axis(self, method):
        test_lenet = QuantConvLinear()

        ref_calibrator = calib.HistogramCalibrator(8, None, False)

        calib.calibrate_weights(test_lenet, method=method, perchannel=True)
        ref_calibrator.collect(test_lenet.conv2.weight[1])
        ref_amax = ref_calibrator.compute_amax(method)
        assert torch.allclose(ref_amax, test_lenet.conv2.weight_quantizer.amax[1], rtol=0, atol=0)
