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

"""CPU unit tests for GPTQ utilities."""

import torch

from modelopt.torch.quantization.utils.calib_utils import update_hessian


def test_update_hessian_zero_token_input_noops():
    features = 4
    hessian = torch.eye(features, dtype=torch.float32)
    expected_hessian = hessian.clone()
    n_samples = 7

    updated_hessian, new_n_samples = update_hessian(
        torch.empty(0, features, dtype=torch.float32), hessian, n_samples
    )

    assert updated_hessian is hessian
    assert new_n_samples == n_samples
    torch.testing.assert_close(hessian, expected_hessian)
