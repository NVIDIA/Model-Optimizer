# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch.nn.functional as F
from torch import nn

from modelopt.torch.utils.graph import match


class LinearRelu(nn.Module):
    def __init__(self, features=8):
        super().__init__()
        self.fc = nn.Linear(features, features)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))


class LinearSigmoid(nn.Module):
    def __init__(self, features=8):
        super().__init__()
        self.fc = nn.Linear(features, features)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc(x))


class LinearFuncRelu(nn.Module):
    """Same math as LinearRelu but relu as call_function instead of call_module."""

    def __init__(self, features=8):
        super().__init__()
        self.fc = nn.Linear(features, features)

    def forward(self, x):
        return F.relu(self.fc(x))


class LinearMethodRelu(nn.Module):
    """relu invoked as a tensor method (call_method node)."""

    def __init__(self, features=8):
        super().__init__()
        self.fc = nn.Linear(features, features)

    def forward(self, x):
        return self.fc(x).relu()


class JustLinear(nn.Module):
    def __init__(self, features=8):
        super().__init__()
        self.fc = nn.Linear(features, features)

    def forward(self, x):
        return self.fc(x)


class DiamondShared(nn.Module):
    """The fc output is consumed by two branches (a node is revisited during matching)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        return self.relu(y) + self.sig(y)


class DiamondChained(nn.Module):
    """Same node count and node types as DiamondShared but different connectivity."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.fc(x)
        r = self.relu(y)
        return self.sig(r) + r


class AddTwoInputs(nn.Module):
    def forward(self, x, y):
        return x + y


class ReluUnusedInput(nn.Module):
    """Same node count as AddTwoInputs but the output node has a single input."""

    def forward(self, x, y):
        return F.relu(x)


class TwoOutputs(nn.Module):
    def forward(self, x):
        return x.relu(), x.sigmoid()


class ChainedOutput(nn.Module):
    """Same node count and node types as TwoOutputs but a single chained output."""

    def forward(self, x):
        return x.relu().sigmoid()


class Untraceable(nn.Module):
    """Data-dependent control flow makes this module untraceable by torch.fx."""

    def forward(self, x):
        if x.sum() > 0:
            return x
        return -x


@pytest.mark.parametrize(
    ("make_module", "make_pattern"),
    [
        (LinearRelu, LinearRelu),
        # different layer sizes must still match since submodules are compared by type
        (lambda: LinearRelu(features=4), lambda: LinearRelu(features=16)),
        (LinearFuncRelu, LinearFuncRelu),
        (LinearMethodRelu, LinearMethodRelu),
        # equivalent graph built from different module classes
        (lambda: nn.Sequential(nn.Linear(8, 8), nn.ReLU()), LinearRelu),
        # shared intermediate node is matched consistently on revisit
        (DiamondShared, DiamondShared),
    ],
)
def test_match_equivalent_graphs(make_module, make_pattern):
    assert match(make_module(), [make_pattern()])


@pytest.mark.parametrize(
    ("make_module", "make_pattern"),
    [
        (LinearRelu, LinearSigmoid),  # different call_module target type
        (JustLinear, LinearRelu),  # different node count
        (LinearRelu, JustLinear),
        (LinearRelu, LinearFuncRelu),  # call_module vs call_function op
        (LinearMethodRelu, LinearFuncRelu),  # call_method vs call_function op
        (AddTwoInputs, ReluUnusedInput),  # same node count, different input arity
        (DiamondShared, DiamondChained),  # same node count/types, different connectivity
        (TwoOutputs, ChainedOutput),  # output nodes differ in input degree
    ],
)
def test_no_match(make_module, make_pattern):
    assert not match(make_module(), [make_pattern()])


def test_empty_patterns_never_match():
    assert not match(LinearRelu(), [])


def test_match_any_of_multiple_patterns():
    assert match(LinearRelu(), [LinearSigmoid(), JustLinear(), LinearRelu()])


def test_untraceable_module_returns_false():
    assert not match(Untraceable(), [LinearRelu()])
