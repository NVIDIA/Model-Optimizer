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

from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.utils import make_divisible


@pytest.mark.parametrize(
    ("in_features", "out_features", "bias", "active_in_features", "active_out_features"),
    [
        (12, 6, False, 2, 3),
        (4, 8, True, 1, 7),
    ],
)
def test_dynamic_linear(
    in_features: list[int],
    out_features: list[int],
    bias: bool,
    active_in_features: int,
    active_out_features: int,
) -> None:
    def _get_dyn_model():
        model = nn.Linear(in_features, out_features, bias=bias)
        return DMRegistry.convert(model)

    model = _get_dyn_model()

    # check if for max subnet the weight/bias are the original ones
    model.out_features = model.get_hparam("out_features").max
    model.in_features = model.get_hparam("in_features").max

    assert model.weight is model._parameters["weight"]
    assert model.bias is model._parameters["bias"]

    # reassign active choices
    model.out_features = active_out_features
    model.in_features = active_in_features

    model.train()
    inputs = torch.randn(12, active_in_features)
    targets = model(inputs)
    assert model.in_features == active_in_features
    assert model.out_features == active_out_features

    loss = targets.sum()
    loss.backward()

    assert torch.allclose(
        model._parameters["weight"].grad[active_out_features:, active_in_features:],
        torch.tensor([0.0]),
    )

    if bias:
        assert torch.allclose(
            model._parameters["bias"].grad[active_out_features:], torch.tensor([0.0])
        )

    # store removable attributes before export
    attr_to_be_removed = model._dm_attribute_manager.attr_keys()

    model.eval()
    model = model.export()
    assert type(model) is nn.Linear
    assert model.training is False

    assert model.in_features == active_in_features
    assert model.out_features == active_out_features
    assert (model.bias is not None) == bias

    # check that special attributes are removed
    for attr in attr_to_be_removed:
        assert not hasattr(model, attr)

    outputs = model(inputs)
    assert outputs.shape == targets.shape
    assert torch.allclose(outputs, targets)

    # get new model and check that we can correctly modify choices
    model = _get_dyn_model()

    # now check if we can reduce the choices
    f_ratio = [0.5, 1.0]
    f_divisor = 2
    model.modify(features_ratio=f_ratio, feature_divisor=f_divisor)

    # a few sanity checks
    # NOTE: check is not 100% generic (even # numbers of channels and more constraints).
    # Make sure to check here if you add more test cases
    features = ["out_features", "in_features"]
    for feature in features:
        hp = model.get_hparam(feature)
        assert all(c % f_divisor == 0 for c in hp.choices), (feature, hp)
        assert hp.choices == [make_divisible(r * hp.original, f_divisor) for r in f_ratio]
