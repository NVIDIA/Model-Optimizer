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

"""Temporary MSE collectors must not survive calibration."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import model_calib
from modelopt.torch.quantization.calib import MseCalibrator
from modelopt.torch.quantization.model_calib import module_name_maps


class TestMseCalibratorLifetime(unittest.TestCase):
    def model(self, layer, numeric=None):
        model = nn.Sequential(layer)
        return mtq.quantize(model, {"quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": numeric or {"num_bits": 8, "axis": 0}},
        ], "algorithm": None})

    def test_repeated_embedding_and_linear_calibration(self):
        for layer in [nn.Embedding(8, 16), nn.Linear(16, 8, bias=False)]:
            with self.subTest(layer=type(layer).__name__):
                model = self.model(layer)
                q = model[0].weight_quantizer
                original = q._calibrator
                original_weight = model[0].weight.detach().clone()
                for _ in range(2):
                    model_calib.mse_calibrate(model, start_multiplier=1., stop_multiplier=1., step_size=1.)
                    self.assertIs(q._calibrator, original)
                    torch.testing.assert_close(q._amax.flatten(), original_weight.abs().amax(dim=1))
                    self.assertTrue(torch.isfinite(q._amax).all())
                    model_calib.max_calibrate(model)
                torch.testing.assert_close(model[0].weight, original_weight)

    def test_fp8_embedding_then_max(self):
        model = self.model(nn.Embedding(8, 16), {"num_bits": (4, 3), "axis": None})
        q = model[0].weight_quantizer
        original = q._calibrator
        model_calib.mse_calibrate(model, start_multiplier=1., stop_multiplier=1.5, step_size=.5)
        self.assertIs(q._calibrator, original)
        self.assertTrue(torch.isfinite(q._amax).all())
        model_calib.max_calibrate(model)
        self.assertTrue(torch.isfinite(q._amax).all())

    def test_refined_amax_survives_restoration(self):
        model = self.model(nn.Embedding(8, 16))
        model_calib.max_calibrate(model)
        q = model[0].weight_quantizer
        original = q._calibrator
        target = q._amax.clone() * .5
        with patch.object(MseCalibrator, "compute_amax", return_value=target):
            model_calib._mse_calibrate_weights(model, module_name_maps(model), 1., 1., 1., False)
        self.assertIs(q._calibrator, original)
        torch.testing.assert_close(q._amax, target)

    def test_error_restores_collector_and_propagates(self):
        model = self.model(nn.Embedding(8, 16))
        model_calib.max_calibrate(model)
        q = model[0].weight_quantizer
        original = q._calibrator
        amax = q._amax.clone()
        failure = RuntimeError("sentinel collection failure")
        with patch.object(model_calib, "_run_and_load_max_stats", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "sentinel collection failure"):
                model_calib._mse_calibrate_weights(model, module_name_maps(model), 1., 1., 1., False)
        self.assertIs(q._calibrator, original)
        torch.testing.assert_close(q._amax, amax)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
