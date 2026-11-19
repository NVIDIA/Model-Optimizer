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

import pytest
import torch

from examples.diffusers.quantization.calib.plugin_calib import PercentileCalibrator


@pytest.mark.parametrize(
    ("observations", "expected_amax"),
    [
        pytest.param([2.0, 4.0, 6.0, 8.0], 2.0, id="one-cycle"),
        pytest.param([2.0, 4.0, 6.0, 8.0] * 3, 2.0, id="repeated-cycles"),
        pytest.param([2.0, 4.0, 6.0, 8.0, 6.0, 8.0, 10.0, 12.0], 4.0, id="varying-cycles"),
        pytest.param([2.0, 8.0, 6.0, 10.0, 6.0], 4.0, id="partial-final-cycle"),
    ],
)
def test_min_mean_averages_observations_per_step(observations, expected_amax):
    calibrator = PercentileCalibrator(
        num_bits=8, axis=None, percentile=1.0, total_step=4, collect_method="min-mean"
    )
    for value in observations:
        calibrator.collect(torch.tensor([value]))

    torch.testing.assert_close(calibrator.compute_amax(), torch.tensor([expected_amax]))
