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

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torchprofile")

import modelopt.torch.prune as mtp
from modelopt.torch.nas.utils import inference_flops


def test_pruning(use_channel_div_4):
    def get_model():
        return nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv2d(4, 8, 3), nn.Conv2d(8, 8, 3))

    def score_func(x):
        return 0.5

    dummy_input = torch.randn(1, 3, 8, 8)
    constraints = {
        "flops": inference_flops(get_model(), dummy_input, unit=1) * 0.9,
    }

    # Assert mtp.prune can be called and the return is valid
    searched_model, search_history = mtp.prune(
        model=get_model(),
        mode="fastnas",
        constraints=constraints,
        dummy_input=dummy_input,
        config={"num_iters": 5, "score_func": score_func},
    )
    assert isinstance(search_history, dict)
