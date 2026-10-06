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
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from modelopt.torch.nas.registry import DMRegistry


def _batchnorm(ndim: int) -> type[_BatchNorm]:
    return [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][ndim - 1]


@pytest.mark.parametrize(
    ("ndim", "num_features", "active_num_features"),
    [
        (1, 10, 6),
        (2, 8, 4),
        (3, 6, 2),
    ],
)
def test_dynamic_batchnorm(ndim: int, num_features: int, active_num_features: int) -> None:
    model = _batchnorm(ndim)(num_features)
    model = DMRegistry.convert(model)

    # check if for max subnet the weight/bias are the original ones
    model.num_features = model.get_hparam("num_features").max
    assert model.weight is model._parameters["weight"]
    assert model.bias is model._parameters["bias"]

    # reassign active choice
    model.num_features = active_num_features

    # check that inactive weight and bias parameters have no grad on training
    model.train()
    for _ in range(10):
        inputs = torch.randn(10, active_num_features, *[10] * ndim)
        outputs = model(inputs)

    loss = outputs.sum()
    loss.backward()

    assert torch.allclose(
        model._parameters["weight"].grad[active_num_features:],
        torch.tensor([0.0]),
    )

    assert torch.allclose(model._parameters["bias"].grad[active_num_features:], torch.tensor([0.0]))

    model.eval()
    inputs = torch.randn(1, active_num_features, *[10] * ndim)
    targets = model(inputs)
    assert model.num_features == active_num_features

    # check that you can correctly limit choices
    f_ratio = [active_num_features / num_features, 1.0]
    f_divisor = active_num_features
    model.modify(features_ratio=f_ratio, feature_divisor=f_divisor)
    assert model.get_hparam("num_features").choices == [active_num_features, num_features]

    # store removable attributes before export
    attr_to_be_removed = model._dm_attribute_manager.attr_keys()

    model_exported = model.export()
    assert type(model_exported) is _batchnorm(ndim)
    assert model_exported.training is False
    assert model_exported.num_features == active_num_features

    # check that special attributes are removed
    for attr in attr_to_be_removed:
        assert not hasattr(model, attr)

    # check export
    outputs = model_exported(inputs)
    assert outputs.shape == targets.shape
    assert torch.allclose(outputs, targets)


def _instancenorm(ndim: int) -> type[_InstanceNorm]:
    return [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d][ndim - 1]


@pytest.mark.parametrize(
    ("ndim", "num_features", "active_num_features"),
    [
        (1, 10, 6),
        (1, 10, 8),
        (2, 8, 4),
        (3, 6, 2),
    ],
)
def test_dynamic_instancenorm(ndim: int, num_features: int, active_num_features: int) -> None:
    model = _instancenorm(ndim)(num_features, affine=True)
    model = DMRegistry.convert(model)

    # check if for max subnet the weight/bias are the original ones
    model.num_features = model.get_hparam("num_features").max
    assert model.weight is model._parameters["weight"]
    assert model.bias is model._parameters["bias"]

    # reassign active choice
    model.num_features = active_num_features

    model.train()
    for _ in range(2):
        inputs = torch.randn(2, active_num_features, *[10] * ndim)
        outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()

    assert torch.allclose(
        model._parameters["weight"].grad[active_num_features:],
        torch.tensor([0.0]),
    )

    assert torch.allclose(model._parameters["bias"].grad[active_num_features:], torch.tensor([0.0]))

    model.eval()
    inputs = torch.randn(1, active_num_features, *[10] * ndim)
    targets = model(inputs)
    assert model.num_features == active_num_features

    # check that you can correctly limit choices
    f_ratio = [active_num_features / num_features, 1.0]
    f_divisor = active_num_features
    model.modify(features_ratio=f_ratio, feature_divisor=f_divisor)
    assert model.get_hparam("num_features").choices == [active_num_features, num_features]

    model_exported = model.export()
    assert type(model_exported) is _instancenorm(ndim)
    assert model_exported.training is False
    assert model_exported.num_features == active_num_features

    outputs = model_exported(inputs)
    assert outputs.shape == targets.shape
    assert torch.allclose(outputs, targets)


@pytest.mark.parametrize(
    ("normlayer_shape", "active_num_features"),
    [
        ([10], 8),
        ([32, 6], 4),
        ([32, 28, 7], 7),
        ([6], 3),
    ],
)
def test_dynamic_layernorm(normlayer_shape, active_num_features: int) -> None:
    model = nn.LayerNorm(normlayer_shape)
    model = DMRegistry.convert(model)

    # check if for max subnet the weight/bias are the original ones
    model.num_features = model.get_hparam("num_features").max
    assert torch.allclose(model.weight, model._parameters["weight"])
    assert torch.allclose(model.bias, model._parameters["bias"])

    # reassign active choice
    model.num_features = active_num_features

    active_shape = (*tuple(normlayer_shape[:-1]), active_num_features)
    assert model.normalized_shape == active_shape

    model.train()
    for _ in range(2):
        inputs = torch.randn(1, 4, *active_shape)
        outputs = model(inputs)

    loss = outputs.sum()
    loss.backward()

    assert torch.allclose(
        model._parameters["weight"].grad[..., active_num_features:],
        torch.tensor([0.0]),
    )

    assert torch.allclose(
        model._parameters["bias"].grad[..., active_num_features:], torch.tensor([0.0])
    )

    model.eval()
    inputs = torch.randn(2, *active_shape)
    targets = model(inputs)

    # check that you can correctly limit choices
    f_ratio = [active_num_features / normlayer_shape[-1], 1.0]
    f_divisor = active_num_features
    model.modify(features_ratio=f_ratio, feature_divisor=f_divisor)
    expected_choices = sorted({active_num_features, normlayer_shape[-1]})
    assert model.get_hparam("num_features").choices == expected_choices

    model_exported = model.export()
    assert type(model_exported) is nn.LayerNorm
    assert model_exported.training is False

    outputs = model_exported(inputs)
    assert outputs.shape == targets.shape
    assert torch.allclose(outputs, targets)


@pytest.mark.parametrize(
    ("num_groups", "num_channels", "active_num_channels"),
    [
        (1, 8, 8),
        (4, 8, 4),
        (4, 8, 8),
        (8, 8, 8),
    ],
)
def test_dynamic_groupnorm(num_groups, num_channels, active_num_channels) -> None:
    model = nn.GroupNorm(num_groups, num_channels)
    model = DMRegistry.convert(model)

    # check if for max subnet the weight/bias are the original ones
    model.num_channels = model.get_hparam("num_channels").max
    assert torch.allclose(model.weight, model._parameters["weight"])
    assert torch.allclose(model.bias, model._parameters["bias"])

    # reassign active choice
    model.num_channels = active_num_channels

    model.train()
    for _ in range(2):
        inputs = torch.randn(1, active_num_channels, 8, 8)
        outputs = model(inputs)

    loss = outputs.sum()
    loss.backward()

    assert torch.allclose(
        model._parameters["weight"].grad[active_num_channels:], torch.tensor([0.0])
    )
    assert torch.allclose(model._parameters["bias"].grad[active_num_channels:], torch.tensor([0.0]))

    model.eval()
    inputs = torch.randn(1, active_num_channels, 8, 8)
    targets = model(inputs)

    # check that you can correctly limit choices
    f_ratio = [active_num_channels / num_channels, 1.0]
    f_divisor = active_num_channels
    model.modify(channels_ratio=f_ratio, channel_divisor=f_divisor)
    expected_choices = sorted({active_num_channels, num_channels})
    assert model.get_hparam("num_channels").choices == expected_choices

    model_exported = model.export()
    assert type(model_exported) is nn.GroupNorm
    assert model_exported.training is False

    model_exported.eval()
    outputs = model_exported(inputs)
    assert outputs.shape == targets.shape
    assert torch.allclose(outputs, targets)
