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

from collections import OrderedDict

import pytest
import torch
import torchvision
from _test_utils.torch.vision_models import TinyMobileNetFeatures
from torch import nn
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.models.resnet import BasicBlock, Bottleneck

from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.nas.search_space import SearchSpace, generate_search_space


@pytest.fixture(scope="module")
def ConvNormActivation():  # noqa: N802
    if hasattr(torchvision.ops.misc, "ConvNormActivation"):
        return torchvision.ops.misc.ConvNormActivation

    from torchvision.models.mobilenetv2 import ConvBNActivation

    return ConvBNActivation


@pytest.fixture(scope="module")
def test_rules():
    return {
        "nn.Sequential": {"min_depth": 1},
        "nn.Conv2d": {
            "channels_ratio": (0.4, 0.6, 0.8, 1.0),
            "kernel_size": (3, 5),
            "channel_divisor": 16,
        },
        "nn.BatchNorm2d": {},
        "nn.SyncBatchNorm": {},
    }


def get_data_loader(num_batches):
    """Yield some fake data that's consistent over the test."""
    for _ in range(num_batches):
        yield torch.rand(2, 3, 56, 56), int(torch.randint(1000, (1,)))


@pytest.mark.parametrize(
    ("module", "key_layer", "expected_kernel_size_set", "expected_mid_channels"),
    [
        (Bottleneck(32, 2, base_width=512), "conv2", {(3, 3)}, {16}),
        (BasicBlock(16, 32), "conv1", {(3, 3)}, {16, 32}),
        (InvertedResidual(16, 16, 1, 6), "conv.1.0", {(3, 3)}, {48, 64, 80, 96}),
    ],
)
def test_base_module(
    module: nn.Module, key_layer: str, expected_kernel_size_set, expected_mid_channels, test_rules
):
    search_space = generate_search_space(module, test_rules)
    str(search_space)

    for _, hp in search_space.named_hparams():
        str(hp)

    assert (
        set(search_space.get_hparam(f"{key_layer}.kernel_size").choices) == expected_kernel_size_set
    )
    assert (
        set(search_space.get_hparam(f"{key_layer}.out_channels").choices) == expected_mid_channels
    )


def _change_to_nn_type(rules):
    return {getattr(nn, k.rpartition("nn.")[-1]): v for k, v in rules.items()}


def _change_to_mixed_keys(rules):
    return {nn.Sequential if k == "nn.Sequential" else k: v for k, v in rules.items()}


def _add_wrong_key(rules):
    return {**rules, "wrong_key": {"wrong_value": None}}


def _add_wrong_value(rules):
    return {**rules, "nn.Conv2d": {**rules["nn.Conv2d"], "wrong_value": None}}


@pytest.mark.parametrize(
    ("change_rule", "is_valid_rule"),
    [
        (lambda rules: rules, True),
        (_change_to_nn_type, True),
        (_change_to_mixed_keys, True),
        (_add_wrong_key, False),
        (_add_wrong_value, False),
    ],
    ids=["keep_unchanged", "change_to_nn_type", "change_to_mixed_keys", "wrong_key", "wrong_value"],
)
def test_nest_module_with_different_rules(change_rule, is_valid_rule, test_rules):
    nested_model = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 16, 3),
            InvertedResidual(16, 16, 1, 4),
            Bottleneck(16, 4, base_width=128),
        ),
        nn.ReLU(),
    )
    updated_rules = change_rule(test_rules)
    if not is_valid_rule:
        with pytest.raises(Exception):
            generate_search_space(nested_model, updated_rules)
        return
    search_space = generate_search_space(nested_model, updated_rules)
    assert type(search_space.model) is nn.Sequential
    assert len(search_space.model) == 2

    submodule = search_space.model[0]
    assert type(submodule) is DMRegistry[nn.Sequential]
    assert set(submodule.get_hparam("depth").choices) == {1, 2, 3}


@pytest.fixture
def search_space() -> SearchSpace:
    seq = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 8, 3)),
                ("bn1", nn.BatchNorm2d(8)),
                ("act1", nn.ReLU(True)),
                ("conv2", nn.Conv2d(8, 1, 3)),
            ]
        )
    )
    rules = {
        "nn.Conv2d": {
            "channels_ratio": tuple(0.1 * i for i in range(1, 11)),
            "kernel_size": (),
            "channel_divisor": 4,
        },
        "nn.BatchNorm2d": {},
        "nn.Sequential": None,
    }
    return generate_search_space(seq, rules)


def test_searchspace(search_space: SearchSpace) -> None:
    config = search_space.sample()
    assert set(config.keys()) == {
        "conv1.in_channels",
        "conv1.out_channels",
        "conv1.kernel_size",
        "bn1.num_features",
        "conv2.in_channels",
        "conv2.out_channels",
        "conv2.kernel_size",
    }

    inputs = torch.randn(1, 1, 10, 10)
    targets = search_space.model(inputs)

    while search_space.sample() == config:
        continue
    search_space.select(config)

    outputs = search_space.model(inputs)
    assert outputs.shape == targets.shape
    assert torch.allclose(outputs, targets)

    sub_model = search_space.export()
    assert type(sub_model.conv1) is nn.Conv2d
    assert type(sub_model.bn1) is nn.BatchNorm2d
    assert type(sub_model.conv2) is nn.Conv2d

    outputs = sub_model(inputs)
    assert outputs.shape == targets.shape
    assert torch.allclose(outputs, targets)


def test_config(search_space: SearchSpace) -> None:
    # sample a new configuration
    config_original = search_space.config()
    while search_space.sample() == config_original:
        continue

    # also try creating config
    search_space.select(config_original)
    search_space.config()

    # re-select original config --> exporting should work
    search_space.select(config_original)
    search_space.export()


def test_model_match(test_rules):
    model = TinyMobileNetFeatures()
    dummy_input = torch.randn(1, 3, 128, 128)
    targets_nn = model(dummy_input)

    search_space = generate_search_space(model, test_rules)
    config = search_space.sample(max)
    search_space.select(config)

    targets_dyn = model(dummy_input)
    assert targets_nn.shape == targets_dyn.shape
    assert torch.allclose(targets_nn, targets_dyn)


def test_format_match(test_rules):
    model = TinyMobileNetFeatures()
    model.half()
    generate_search_space(model, test_rules)
    for p in model.parameters():
        assert p.dtype == torch.float16


def test_conv_search_space():
    # start with a simple conv
    conv = nn.Conv2d(3, 16, 7)
    dummy_input = torch.randn(1, 3, 24, 24)

    # conversion rules
    conversion_rules = {"nn.Conv2d": {"channel_divisor": 16, "kernel_size": ()}}

    # generate search space now
    search_space = generate_search_space(conv, conversion_rules)

    # check correctness of search space
    config_space_expected = {}
    config_space_actual = {n: hp.choices for n, hp in search_space.named_hparams(configurable=True)}
    assert config_space_expected == config_space_actual

    # check bias
    assert conv.bias is not None

    # check ability to sample from search space and run forward pass
    search_space.sample()
    conv(dummy_input)

    # Test dilated conv
    conv = nn.Conv2d(3, 32, 7, dilation=2)
    search_space = generate_search_space(conv, conversion_rules)
    assert conv.dilation == (2, 2)

    # Test depthwise conv
    conv = nn.Conv2d(32, 32, 5, groups=32)
    search_space = generate_search_space(conv, conversion_rules)
    assert type(conv) is DMRegistry[nn.Conv2d]

    # Test groupwise conv
    conv = nn.Conv2d(64, 64, 5, groups=4)
    search_space = generate_search_space(conv, conversion_rules)
    assert type(conv) is DMRegistry[nn.Conv2d]

    # Test groupwise conv with unsatisfiable settings
    conv = nn.Conv2d(64, 64, 5, groups=2)
    search_space = generate_search_space(conv, conversion_rules)
    assert search_space.size() == 1
    conv = nn.Conv2d(64, 32, 5, groups=4)
    search_space = generate_search_space(conv, conversion_rules)
    assert search_space.size() == 1

    # try custom conversion rules now
    conv = nn.Conv2d(3, 32, 7, padding=3)
    conversion_rules["nn.Conv2d"]["kernel_size"] = (3, (3, 5), 5)
    conversion_rules["nn.Conv2d"]["channel_divisor"] = 16
    conversion_rules["nn.Conv2d"]["channels_ratio"] = [0.33, 0.67]
    search_space = generate_search_space(conv, conversion_rules)

    # check correctness of search space
    config_space_expected2 = {"kernel_size": [(3, 3), (3, 5), (5, 5), (7, 7)]}
    config_space_actual2 = {n: hp.choices for n, hp in search_space.named_hparams(True)}
    assert config_space_expected2 == config_space_actual2

    # Check correctness of padding
    config = {"in_channels": 3, "kernel_size": (3, 5), "out_channels": 32}
    search_space.select(config)
    assert conv.padding == (1, 2)

    # Check correctness of output shape
    out = conv(dummy_input)
    assert out.shape == (1, 32, 24, 24)

    # Test unsupported conv with non-zero padding mode
    conv = nn.Conv2d(3, 32, 7, padding="same", padding_mode="circular")
    search_space = generate_search_space(conv, conversion_rules)
    assert search_space.size() == 1

    # Test unsupported conv with groups
    conv = nn.Conv2d(4, 32, 7, groups=4)
    search_space = generate_search_space(conv, conversion_rules)
    assert search_space.size() == 1


def test_basic_layerwise_search_space(ConvNormActivation):  # noqa: N803
    # setup model and search space
    model = nn.Sequential(
        ConvNormActivation(3, 16, 3),
        ConvNormActivation(16, 32, 3),
        ConvNormActivation(32, 64, 3),
    )
    conversion_rules = {
        "nn.Sequential": {"min_depth": 1},
        "nn.Conv2d": {
            "channels_ratio": [0.1 * i for i in range(1, 11)],
            "channel_divisor": 16,
            "kernel_size": (),
        },
        "nn.BatchNorm2d": {},
        "nn.SyncBatchNorm": {},
    }
    search_space = generate_search_space(model, conversion_rules)

    # check correctness of configuration space
    config_space_expected = {"1.0.out_channels": [16, 32]}
    config_space_actual = {n: hp.choices for n, hp in search_space.named_hparams(configurable=True)}
    assert config_space_expected == config_space_actual

    # check ability to sample from search space and run forward pass
    search_space.sample()
    dummy_input = torch.randn(1, 3, 128, 128)
    model(dummy_input)
