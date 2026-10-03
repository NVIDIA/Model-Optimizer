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

"""CUDA validation with ModelOpt INT4 max and GPTQ calibration."""

import copy

import pytest
import torch
from torch import nn

import modelopt.torch.quantization as mtq
from experimental.softmax_reparameterization import head_kl, search_head


@pytest.mark.parametrize("algorithm", ["max", "gptq"])
def test_cuda_head_search(algorithm):
    torch.manual_seed(7)
    head = nn.Linear(128, 256, bias=False, device="cuda", dtype=torch.bfloat16).eval()
    original = head.weight.detach().clone()
    fit, val, test = torch.randn(3, 64, 128, device="cuda", dtype=torch.bfloat16).unbind()
    config = copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG)
    config["algorithm"] = algorithm
    result = search_head(head, fit, val, config, (0, 1, 4), batch_size=32)
    curve = dict(result.validation_kl)
    assert curve[result.coefficient] <= curve[0]
    assert head_kl(head, result.model, val, 64) == pytest.approx(
        curve[result.coefficient], abs=2e-5
    )
    assert head_kl(head, result.model, test) >= -1e-6
    torch.testing.assert_close(head.weight, original, rtol=0, atol=0)
