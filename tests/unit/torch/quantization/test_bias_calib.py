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

"""Tests of the bias calibrator (modelopt.torch.quantization.calib.bias)."""

import pytest
import torch

from modelopt.torch.quantization import calib
from modelopt.torch.quantization.calib.bias import (
    add_bias,
    compute_bias,
    compute_maxmin,
    compute_maxmin_bias,
    compute_mean_bias,
    subtract_bias,
)
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.model_calib import enable_stats_collection, finish_stats_collection
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

# A tiny 3x4 tensor with hand-checkable statistics:
#   rows: [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]
X_3X4 = torch.arange(12.0).reshape(3, 4)


class TestComputeMaxmin:
    def test_axis_none_reduces_all(self):
        max_, min_ = compute_maxmin(X_3X4, None)
        assert max_.shape == () and min_.shape == ()
        assert max_.item() == 11.0
        assert min_.item() == 0.0

    def test_axis_dims_are_reduced_not_kept(self):
        # NOTE: the docstring of ``compute_maxmin`` claims ``axis`` lists the dims to
        # *keep* ("(-1,): reduce all except last dim"), but the implementation reduces
        # exactly the dims listed in ``axis``. The behavior below (reduce last dim,
        # keep rows) is the actual, shipped semantic that TensorQuantizer relies on.
        max_, min_ = compute_maxmin(X_3X4, (-1,))
        assert max_.shape == (3, 1)
        assert torch.equal(max_, torch.tensor([[3.0], [7.0], [11.0]]))
        assert torch.equal(min_, torch.tensor([[0.0], [4.0], [8.0]]))

    def test_reduce_first_dim(self):
        max_, min_ = compute_maxmin(X_3X4, (0,))
        assert max_.shape == (1, 4)
        assert torch.equal(max_, torch.tensor([[8.0, 9.0, 10.0, 11.0]]))
        assert torch.equal(min_, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))

    def test_negative_and_positive_axis_equivalent(self):
        max_neg, min_neg = compute_maxmin(X_3X4, (-1,))
        max_pos, min_pos = compute_maxmin(X_3X4, (1,))
        assert torch.equal(max_neg, max_pos)
        assert torch.equal(min_neg, min_pos)

    def test_multi_axis_4d(self):
        x = torch.arange(2.0 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        max_, min_ = compute_maxmin(x, (-1, -3))  # reduce dims 1 and 3
        assert max_.shape == (2, 1, 4, 1)
        assert torch.equal(max_, torch.amax(x, dim=(1, 3), keepdim=True))
        assert torch.equal(min_, torch.amin(x, dim=(1, 3), keepdim=True))

    def test_int_axis_unsupported(self):
        # NOTE: the annotation advertises ``axis: int | tuple[int, ...] | None`` but a
        # plain int axis raises TypeError (``i in axis`` requires an iterable). Latent
        # bug: TensorQuantizer always passes a tuple, so it never trips in-tree.
        with pytest.raises(TypeError, match="not iterable"):
            compute_maxmin(X_3X4, -1)


class TestComputeBiasFunctions:
    def test_maxmin_bias_per_tensor(self):
        # (max + min) / 2 = (5 + (-1)) / 2 = 2
        bias = compute_maxmin_bias(torch.tensor([-1.0, 0.0, 5.0]), None)
        assert bias.item() == 2.0

    def test_maxmin_bias_per_row(self):
        x = torch.tensor([[-2.0, 4.0], [6.0, 10.0]])
        bias = compute_maxmin_bias(x, (-1,))
        assert torch.equal(bias, torch.tensor([[1.0], [8.0]]))

    def test_mean_bias_per_tensor(self):
        # mean([1, 2, 3, 6]) = 3
        bias = compute_mean_bias(torch.tensor([1.0, 2.0, 3.0, 6.0]), None)
        assert bias.shape == ()
        assert bias.item() == 3.0

    def test_mean_bias_per_axis(self):
        x = torch.tensor([[1.0, 3.0], [5.0, 9.0]])
        assert torch.equal(compute_mean_bias(x, (-1,)), torch.tensor([[2.0], [7.0]]))
        assert torch.equal(compute_mean_bias(x, (0,)), torch.tensor([[3.0, 6.0]]))

    def test_dispatch_mean_vs_maxmin(self):
        # [0, 1, 8]: mean = 3, (max + min) / 2 = 4 -- distinguishes the two methods.
        x = torch.tensor([0.0, 1.0, 8.0])
        assert compute_bias(x, None, method="mean").item() == 3.0
        assert compute_bias(x, None, method="max_min").item() == 4.0

    def test_dispatch_unknown_method_falls_back_to_maxmin(self):
        # NOTE: ``compute_bias`` silently routes any method != "mean" to max_min
        # instead of raising, unlike ``BiasCalibrator.collect``/``compute_dynamic_bias``
        # which raise ValueError. Documents current (lenient) behavior.
        x = torch.tensor([0.0, 1.0, 8.0])
        assert compute_bias(x, None, method="bogus").item() == 4.0

    def test_subtract_add_bias_round_trip(self):
        x = torch.tensor([[1.0, 3.0], [5.0, 9.0]])
        bias = torch.tensor([[2.0], [7.0]])
        centered = subtract_bias(x, bias)
        assert torch.equal(centered, torch.tensor([[-1.0, 1.0], [-2.0, 2.0]]))
        restored = add_bias(centered, bias)
        assert restored.shape == x.shape
        assert torch.equal(restored, x)


class TestBiasCalibratorMean:
    def test_single_collect_per_tensor(self):
        calibrator = calib.BiasCalibrator(method="mean", axis=None)
        assert calibrator.compute_bias() is None  # nothing collected yet
        calibrator.collect(torch.tensor([1.0, 2.0, 3.0, 6.0]))
        bias = calibrator.compute_bias()
        assert bias.shape == ()
        assert bias.item() == 3.0

    def test_running_average_weights_batches_equally(self):
        # Aggregation is a running average over *collect calls*, not elements:
        #   collect([1, 2, 3]) -> bias = 2
        #   collect([4])       -> bias = (2 * 1 + 4) / 2 = 3
        #   collect([9])       -> bias = (3 * 2 + 9) / 3 = 5
        # An element-weighted mean of all 5 values would be 3.8 instead.
        calibrator = calib.BiasCalibrator(method="mean", axis=None)
        calibrator.collect(torch.tensor([1.0, 2.0, 3.0]))
        assert calibrator.compute_bias().item() == 2.0
        calibrator.collect(torch.tensor([4.0]))
        assert calibrator.compute_bias().item() == 3.0
        calibrator.collect(torch.tensor([9.0]))
        assert calibrator.compute_bias().item() == 5.0
        assert calibrator._cnt == 3

    def test_per_axis_running_average(self):
        calibrator = calib.BiasCalibrator(method="mean", axis=(-1,))
        calibrator.collect(torch.tensor([[1.0, 3.0], [5.0, 9.0]]))  # row means [2, 7]
        calibrator.collect(torch.tensor([[4.0, 8.0], [1.0, 1.0]]))  # row means [6, 1]
        # running average: [(2 + 6) / 2, (7 + 1) / 2] = [4, 4]
        bias = calibrator.compute_bias()
        assert bias.shape == (2, 1)
        assert torch.equal(bias, torch.tensor([[4.0], [4.0]]))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_running_average_preserves_dtype(self, dtype):
        # The running average is computed in float32 for stability but cast back.
        calibrator = calib.BiasCalibrator(method="mean", axis=None)
        calibrator.collect(torch.tensor([1.0, 3.0], dtype=dtype))  # mean 2
        calibrator.collect(torch.tensor([5.0], dtype=dtype))  # (2 + 5) / 2 = 3.5
        bias = calibrator.compute_bias()
        assert bias.dtype == dtype
        assert bias.item() == 3.5

    def test_negative_values(self):
        calibrator = calib.BiasCalibrator(method="mean", axis=None)
        calibrator.collect(torch.tensor([-3.0, -5.0, -7.0]))
        assert calibrator.compute_bias().item() == -5.0

    def test_single_element(self):
        calibrator = calib.BiasCalibrator(method="mean", axis=None)
        calibrator.collect(torch.tensor([42.0]))
        assert calibrator.compute_bias().item() == 42.0

    def test_empty_tensor_yields_nan(self):
        # NOTE: empty inputs are not guarded; mean over zero elements silently
        # produces NaN which would poison the running average. Documents current
        # behavior rather than endorsing it.
        calibrator = calib.BiasCalibrator(method="mean", axis=None)
        calibrator.collect(torch.empty(0, 4))
        assert torch.isnan(calibrator.compute_bias())


class TestBiasCalibratorMaxMin:
    def test_single_collect(self):
        calibrator = calib.BiasCalibrator(method="max_min", axis=None)
        calibrator.collect(torch.tensor([-1.0, 0.0, 5.0]))
        assert calibrator.compute_bias().item() == 2.0

    def test_running_extrema_across_collects(self):
        # Unlike "mean", aggregation keeps global extrema over all collects:
        #   collect([0, 2])  -> max 2, min 0,  bias 1
        #   collect([-4, 1]) -> max 2, min -4, bias -1
        calibrator = calib.BiasCalibrator(method="max_min", axis=None)
        calibrator.collect(torch.tensor([0.0, 2.0]))
        assert calibrator.compute_bias().item() == 1.0
        calibrator.collect(torch.tensor([-4.0, 1.0]))
        assert calibrator.compute_bias().item() == -1.0
        assert calibrator._calib_max.item() == 2.0
        assert calibrator._calib_min.item() == -4.0

    def test_inner_batch_does_not_move_extrema(self):
        calibrator = calib.BiasCalibrator(method="max_min", axis=None)
        calibrator.collect(torch.tensor([-4.0, 2.0]))
        calibrator.collect(torch.tensor([-1.0, 1.0]))  # strictly inside [-4, 2]
        assert calibrator.compute_bias().item() == -1.0

    def test_per_axis(self):
        calibrator = calib.BiasCalibrator(method="max_min", axis=(-1,))
        calibrator.collect(torch.tensor([[-2.0, 4.0], [6.0, 10.0]]))
        calibrator.collect(torch.tensor([[-6.0, 0.0], [7.0, 8.0]]))
        # row 0: max 4, min -6 -> -1; row 1: max 10, min 6 -> 8
        bias = calibrator.compute_bias()
        assert bias.shape == (2, 1)
        assert torch.equal(bias, torch.tensor([[-1.0], [8.0]]))

    def test_single_element(self):
        calibrator = calib.BiasCalibrator(method="max_min", axis=None)
        calibrator.collect(torch.tensor([-7.0]))
        assert calibrator.compute_bias().item() == -7.0

    def test_empty_tensor_raises(self):
        calibrator = calib.BiasCalibrator(method="max_min", axis=None)
        with pytest.raises(RuntimeError, match="Expected reduction dim to be specified"):
            calibrator.collect(torch.empty(0, 4))


class TestBiasCalibratorContract:
    def test_reset(self):
        calibrator = calib.BiasCalibrator(method="mean", axis=None)
        calibrator.collect(torch.tensor([1.0, 2.0, 3.0]))
        calibrator.reset()
        assert calibrator.compute_bias() is None
        assert calibrator._calib_max is None
        assert calibrator._calib_min is None
        assert calibrator._cnt == 0
        # No history: bias after reset equals a fresh single-collect result.
        calibrator.collect(torch.tensor([10.0, 20.0]))
        assert calibrator.compute_bias().item() == 15.0

    def test_reset_max_min(self):
        calibrator = calib.BiasCalibrator(method="max_min", axis=None)
        calibrator.collect(torch.tensor([-100.0, 100.0]))
        calibrator.reset()
        calibrator.collect(torch.tensor([0.0, 2.0]))
        assert calibrator.compute_bias().item() == 1.0  # old extrema forgotten

    def test_collect_unsupported_method_raises(self):
        calibrator = calib.BiasCalibrator(method="bogus", axis=None)
        with pytest.raises(ValueError, match="Unsupported method: bogus"):
            calibrator.collect(torch.tensor([1.0]))

    def test_compute_dynamic_bias_unknown_method_raises(self):
        calibrator = calib.BiasCalibrator(method="bogus", axis=None)
        with pytest.raises(ValueError, match="Unknown bias method: bogus"):
            calibrator.compute_dynamic_bias(torch.tensor([1.0]))

    def test_compute_dynamic_bias_is_stateless(self):
        # Dynamic bias depends only on the current inputs, not collected history.
        calibrator = calib.BiasCalibrator(method="mean", axis=None)
        calibrator.collect(torch.tensor([100.0, 200.0]))
        assert calibrator.compute_dynamic_bias(torch.tensor([10.0, 20.0])).item() == 15.0
        assert calibrator.compute_bias().item() == 150.0  # collected state untouched

    def test_compute_dynamic_bias_max_min(self):
        calibrator = calib.BiasCalibrator(method="max_min", axis=None)
        assert calibrator.compute_dynamic_bias(torch.tensor([0.0, 1.0, 8.0])).item() == 4.0

    def test_methods_agree_on_symmetric_data(self):
        x = torch.tensor([-3.0, -1.0, 1.0, 3.0])  # mean == (max + min) / 2 == 0
        mean_calib = calib.BiasCalibrator(method="mean", axis=None)
        maxmin_calib = calib.BiasCalibrator(method="max_min", axis=None)
        mean_calib.collect(x)
        maxmin_calib.collect(x)
        assert mean_calib.compute_bias().item() == 0.0
        assert maxmin_calib.compute_bias().item() == 0.0

    def test_repr(self):
        calibrator = calib.BiasCalibrator(method="mean", axis=(-1,))
        assert "axis=(-1,)" in repr(calibrator)


class TestTensorQuantizerBiasIntegration:
    """End-to-end bias calibration through TensorQuantizer on CPU.

    ``bias`` config keys: int keys are the reduction axes (block_sizes-style),
    e.g. ``{-1: None, "type": "static", "method": "mean"}``.
    """

    @staticmethod
    def _make_quantizer(bias_cfg):
        quant_cfg = QuantizerAttributeConfig(num_bits=(4, 3), bias=bias_cfg)
        return TensorQuantizer(quant_attribute_cfg=quant_cfg)

    def test_static_mean_bias_value_and_centered_amax(self):
        quantizer = self._make_quantizer({-1: None, "type": "static", "method": "mean"})
        x = torch.tensor([[1.0, 3.0], [5.0, 9.0]])

        enable_stats_collection(quantizer)
        y = quantizer(x)
        finish_stats_collection(quantizer)

        assert y.shape == x.shape
        expected_bias = torch.tensor([[2.0], [7.0]])  # per-row means
        assert torch.equal(quantizer.bias_value, expected_bias)
        # During calibration the quantizer collects amax on the *centered* tensor:
        # x - bias = [[-1, 1], [-2, 2]] -> per-tensor amax = 2 (not 9).
        assert quantizer.amax.item() == 2.0

    def test_static_bias_running_average_over_forwards(self):
        quantizer = self._make_quantizer({-1: None, "type": "static", "method": "mean"})
        enable_stats_collection(quantizer)
        quantizer(torch.tensor([[1.0, 3.0]]))  # batch mean [2]
        quantizer(torch.tensor([[4.0, 8.0]]))  # batch mean [6]
        finish_stats_collection(quantizer)
        assert torch.equal(quantizer.bias_value, torch.tensor([[4.0]]))

    def test_static_max_min_bias_value(self):
        quantizer = self._make_quantizer({-1: None, "type": "static", "method": "max_min"})
        enable_stats_collection(quantizer)
        quantizer(torch.tensor([[-2.0, 4.0], [6.0, 10.0]]))
        finish_stats_collection(quantizer)
        assert torch.equal(quantizer.bias_value, torch.tensor([[1.0], [8.0]]))

    def test_quantized_forward_restores_bias(self):
        # With per-row bias removed, rows centered around large offsets survive
        # FP8 fake-quant nearly unscathed: quant(x - bias) + bias stays close to x
        # even though x itself spans [99, 101] with amax-scale dominated by 101.
        quantizer = self._make_quantizer({-1: None, "type": "static", "method": "mean"})
        x = torch.tensor([[99.0, 101.0], [-101.0, -99.0]])

        enable_stats_collection(quantizer)
        quantizer(x)
        finish_stats_collection(quantizer)

        y = quantizer(x)  # quantization enabled now
        assert torch.equal(quantizer.bias_value, torch.tensor([[100.0], [-100.0]]))
        assert quantizer.amax.item() == 1.0  # centered residual is exactly +-1
        assert torch.allclose(y, x)  # +-1 residual is exactly representable in FP8

    def test_dynamic_bias_uses_current_input(self):
        quantizer = self._make_quantizer({-1: None, "type": "dynamic", "method": "mean"})
        enable_stats_collection(quantizer)
        quantizer(torch.tensor([[1.0, 3.0]]))
        finish_stats_collection(quantizer)

        # Dynamic bias is never baked into a buffer...
        assert quantizer.bias_value is None
        # ...and is recomputed from whatever tensor comes in.
        x_new = torch.tensor([[10.0, 20.0], [30.0, 50.0]])
        assert torch.equal(quantizer._get_bias(x_new), torch.tensor([[15.0], [40.0]]))

    def test_load_calib_bias_without_data_raises(self):
        quantizer = self._make_quantizer({-1: None, "type": "static", "method": "mean"})
        with pytest.raises(RuntimeError, match="Calibrator returned None"):
            quantizer.load_calib_bias()

    def test_bias_calibrator_constructed_from_config(self):
        quantizer = self._make_quantizer(
            {-1: None, -3: None, "type": "static", "method": "max_min"}
        )
        calibrator = quantizer.bias_calibrator
        assert isinstance(calibrator, calib.BiasCalibrator)
        assert calibrator._method == "max_min"
        assert set(quantizer.bias_axis) == {-1, -3}
