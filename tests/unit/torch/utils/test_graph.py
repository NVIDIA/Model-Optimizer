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


class Untraceable(nn.Module):
    """Data-dependent control flow makes this module untraceable by torch.fx."""

    def forward(self, x):
        if x.sum() > 0:
            return x
        return -x


def test_match_same_structure():
    assert match(LinearRelu(), [LinearRelu()])


def test_match_compares_module_types_not_instances():
    # different layer sizes must still match since submodules are compared by type
    assert match(LinearRelu(features=4), [LinearRelu(features=16)])


def test_no_match_different_activation():
    assert not match(LinearRelu(), [LinearSigmoid()])


def test_no_match_different_node_count():
    assert not match(JustLinear(), [LinearRelu()])
    assert not match(LinearRelu(), [JustLinear()])


def test_empty_patterns_never_match():
    assert not match(LinearRelu(), [])


def test_match_any_of_multiple_patterns():
    assert match(LinearRelu(), [LinearSigmoid(), JustLinear(), LinearRelu()])


def test_call_function_nodes_match():
    assert match(LinearFuncRelu(), [LinearFuncRelu()])


def test_call_module_does_not_match_call_function():
    # nn.ReLU (call_module) vs F.relu (call_function) are different graph ops
    assert not match(LinearRelu(), [LinearFuncRelu()])


def test_call_method_nodes_match():
    assert match(LinearMethodRelu(), [LinearMethodRelu()])
    assert not match(LinearMethodRelu(), [LinearFuncRelu()])


def test_untraceable_module_returns_false():
    assert not match(Untraceable(), [LinearRelu()])


def test_sequential_matches_equivalent_custom_module():
    seq = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    assert match(seq, [LinearRelu()])
    assert match(LinearRelu(), [seq])
