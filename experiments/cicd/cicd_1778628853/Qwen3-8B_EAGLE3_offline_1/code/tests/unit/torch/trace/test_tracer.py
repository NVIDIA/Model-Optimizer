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
from torch import nn
from torch.fx import Tracer
from torch.fx.proxy import TraceError

from modelopt.torch.trace import RobustTracer


class ModelWithFlag(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x, test_flag=False):
        if test_flag:
            return self.conv(x + 1)
        else:
            return self.conv(x)


class ModelWithError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return iter(x)


def test_flag():
    model = ModelWithFlag()
    tracer = RobustTracer()
    graph = tracer.trace(model, {"x": torch.randn(1, 3, 16, 16)})
    assert "conv" in [node.name for node in graph.nodes]
    assert "add" not in [node.name for node in graph.nodes]


def test_error():
    model = nn.Sequential(ModelWithError())
    tracer = Tracer()
    with pytest.raises(TraceError):
        tracer.trace(model)

    tracer = RobustTracer()
    graph = tracer.trace(model, {"x": torch.randn(1, 3, 16, 16)})
    assert len(graph.nodes) == 3
