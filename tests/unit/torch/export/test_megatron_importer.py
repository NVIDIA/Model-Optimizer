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

import torch

from modelopt.torch.export.plugins.megatron_importer import _get_mamba_conv1d


class _Mixer:
    pass


def test_get_mamba_conv1d_returns_legacy_module():
    mixer = _Mixer()
    mixer.conv1d = torch.nn.Conv1d(4, 4, 3)

    assert _get_mamba_conv1d(mixer) is mixer.conv1d


def test_get_mamba_conv1d_wraps_direct_params():
    mixer = _Mixer()
    mixer.conv1d_weight = torch.nn.Parameter(torch.zeros(4, 1, 3))
    mixer.conv1d_bias = torch.nn.Parameter(torch.zeros(4))

    conv1d = _get_mamba_conv1d(mixer)
    new_weight = torch.ones_like(mixer.conv1d_weight)
    new_bias = torch.ones_like(mixer.conv1d_bias)
    conv1d.load_state_dict({"weight": new_weight, "bias": new_bias})

    assert set(conv1d.state_dict()) == {"weight", "bias"}
    assert conv1d.weight is mixer.conv1d_weight
    assert conv1d.bias is mixer.conv1d_bias
    torch.testing.assert_close(mixer.conv1d_weight, new_weight)
    torch.testing.assert_close(mixer.conv1d_bias, new_bias)
