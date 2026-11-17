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

"""Independent mathematical checks for Bypass updates."""

import pytest
import torch


def test_lion_conflicting_gradients_and_effective_schedule():
    from modelopt.torch.distill import BypassConfig
    from modelopt.torch.distill._bypass_optimizer import Lion, learning_rate

    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = Lion([parameter], lr=0.001)
    parameter.grad = torch.tensor([2.0, -4.0])
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([0.999, -0.999]))
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.tensor([0.02, -0.04]))
    parameter.grad = torch.tensor([-0.1, 0.1])
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([0.998, -0.998]))
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg"], torch.tensor([0.0188, -0.0386])
    )
    config = BypassConfig(num_steps=23, max_lr=0.001, warmup_fraction=0.1)
    for step, expected in {1: 0.001 / 3, 2: 0.002 / 3, 3: 0.001, 13: 0.0005, 23: 0.0}.items():
        assert learning_rate(config, step) == pytest.approx(expected, abs=1e-12)
