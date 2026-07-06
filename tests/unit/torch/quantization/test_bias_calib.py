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

"""Targeted unit tests for modelopt.torch.quantization.calib.bias.

Only exercises the paths not covered elsewhere: per-tensor bias math and method
dispatch, bias subtraction/addition, the running-average aggregation of the mean
method, dynamic bias computation, and calibrator reset.
"""

import torch

from modelopt.torch.quantization.calib.bias import (
    BiasCalibrator,
    add_bias,
    compute_bias,
    subtract_bias,
)


class TestBiasFunctions:
    def test_compute_bias_dispatch_per_tensor(self):
        # [0, 1, 8]: mean = 3, (max + min) / 2 = 4 -- distinguishes the two methods.
        x = torch.tensor([0.0, 1.0, 8.0])
        mean_bias = compute_bias(x, None, method="mean")
        maxmin_bias = compute_bias(x, None, method="max_min")
        assert mean_bias.shape == ()
        assert maxmin_bias.shape == ()
        assert mean_bias.item() == 3.0
        assert maxmin_bias.item() == 4.0

    def test_subtract_add_bias_round_trip(self):
        x = torch.tensor([[1.0, 3.0], [5.0, 9.0]])
        bias = torch.tensor([[2.0], [7.0]])
        centered = subtract_bias(x, bias)
        assert torch.equal(centered, torch.tensor([[-1.0, 1.0], [-2.0, 2.0]]))
        restored = add_bias(centered, bias)
        assert restored.shape == x.shape
        assert torch.equal(restored, x)


class TestBiasCalibrator:
    def test_mean_running_average_weights_collects_equally(self):
        # Aggregation is a running average over *collect calls*, not elements:
        #   collect([1, 2, 3]) -> bias = 2
        #   collect([4])       -> bias = (2 * 1 + 4) / 2 = 3
        #   collect([9])       -> bias = (3 * 2 + 9) / 3 = 5
        # An element-weighted mean of all 5 values would be 3.8 instead.
        calibrator = BiasCalibrator(method="mean", axis=None)
        assert calibrator.compute_bias() is None  # nothing collected yet
        calibrator.collect(torch.tensor([1.0, 2.0, 3.0]))
        assert calibrator.compute_bias().item() == 2.0
        calibrator.collect(torch.tensor([4.0]))
        assert calibrator.compute_bias().item() == 3.0
        calibrator.collect(torch.tensor([9.0]))
        assert calibrator.compute_bias().item() == 5.0
        assert calibrator._cnt == 3

    def test_mean_running_average_preserves_dtype(self):
        # The running average is computed in float32 for stability but cast back.
        calibrator = BiasCalibrator(method="mean", axis=None)
        calibrator.collect(torch.tensor([1.0, 3.0], dtype=torch.float16))  # mean 2
        calibrator.collect(torch.tensor([5.0], dtype=torch.float16))  # (2 + 5) / 2 = 3.5
        bias = calibrator.compute_bias()
        assert bias.dtype == torch.float16
        assert bias.item() == 3.5

    def test_compute_dynamic_bias_is_stateless_for_both_methods(self):
        # [0, 1, 8]: mean = 3, (max + min) / 2 = 4
        x = torch.tensor([0.0, 1.0, 8.0])
        mean_calib = BiasCalibrator(method="mean", axis=None)
        mean_calib.collect(torch.tensor([100.0, 200.0]))
        assert mean_calib.compute_dynamic_bias(x).item() == 3.0  # collected history ignored
        assert mean_calib.compute_bias().item() == 150.0  # collected state untouched
        maxmin_calib = BiasCalibrator(method="max_min", axis=None)
        assert maxmin_calib.compute_dynamic_bias(x).item() == 4.0

    def test_reset_clears_state_for_both_methods(self):
        mean_calib = BiasCalibrator(method="mean", axis=None)
        mean_calib.collect(torch.tensor([1.0, 2.0, 3.0]))
        mean_calib.collect(torch.tensor([10.0]))
        maxmin_calib = BiasCalibrator(method="max_min", axis=None)
        maxmin_calib.collect(torch.tensor([-100.0, 100.0]))
        for calibrator in (mean_calib, maxmin_calib):
            calibrator.reset()
            assert calibrator.compute_bias() is None
            assert calibrator._calib_max is None
            assert calibrator._calib_min is None
            assert calibrator._cnt == 0
        # No history left: fresh collects match a brand-new calibrator.
        mean_calib.collect(torch.tensor([10.0, 20.0]))
        assert mean_calib.compute_bias().item() == 15.0
        maxmin_calib.collect(torch.tensor([0.0, 2.0]))
        assert maxmin_calib.compute_bias().item() == 1.0  # old extrema forgotten
