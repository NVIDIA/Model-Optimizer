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

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from contextlib import nullcontext

import pytest
import torch
from _test_utils.torch.nas_prune.utils import param_num
from torch import nn
from torchvision.models import MobileNetV2

from modelopt.torch.utils.network import (
    compare_dict,
    create_param_grad_clear_hook,
    get_model_attributes,
    get_same_padding,
    make_divisible,
    set_submodule,
    standardize_model_args,
)


@pytest.mark.parametrize(
    ("in_channel", "out_channel", "kernel_size", "groups"),
    [
        (2, 4, 3, 2),
        (3, 5, 1, 1),
        (4, 2, 3, 1),
    ],
)
def test_param_count(in_channel, out_channel, kernel_size, groups) -> None:
    conv_module = nn.Conv2d(
        in_channel, out_channel, kernel_size=kernel_size, groups=groups, bias=False
    )
    assert (
        param_num(conv_module, unit=1)
        == in_channel * out_channel * kernel_size * kernel_size // groups
    )

    linear_module = nn.Linear(in_channel, out_channel, bias=False)
    assert param_num(linear_module, unit=1) == in_channel * out_channel


@pytest.mark.parametrize(
    ("v", "divisor", "min_val"),
    [
        (23, 8, None),
        (23, 10, None),
        (55, 12, None),
    ],
)
def test_make_divisible(v, divisor, min_val):
    new_v = make_divisible(v, divisor, min_val)
    assert new_v % divisor == 0


@pytest.mark.parametrize(
    ("test_input", "expected_output"),
    [
        ((3,), 1),
        ((5,), 2),
        ((7,), 3),
        (((3, 3),), (1, 1)),
        (((5, 5),), (2, 2)),
        (((7, 7),), (3, 3)),
    ],
)
def test_get_same_padding(test_input, expected_output):
    assert get_same_padding(*test_input) == expected_output


def test_compare_dict():
    test_dict1 = {"a": 1, "b": 2, "c": 3}
    test_dict2 = {"b": 2, "c": 1, "d": "4"}
    keys_unmatched = compare_dict(test_dict1, test_dict2)
    assert set(keys_unmatched) == {"a", "c", "d"}


@pytest.mark.parametrize(
    ("model", "target", "submodule"),
    [
        (nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU()), "1", nn.SiLU()),
        (MobileNetV2(width_mult=0.25), "classifier.0", nn.SiLU()),
        (MobileNetV2(width_mult=0.25), "features.17.conv.1.2", nn.SiLU()),
        (MobileNetV2(width_mult=0.25), "features.17.conv", nn.Conv2d(3, 16, 3)),
    ],
)
def test_regular(model, target, submodule):
    parent_module = model.get_submodule(target.rpartition(".")[0])
    num_original_modules = len(list(parent_module.children()))

    set_submodule(model, target, submodule)

    assert model.get_submodule(target) == submodule
    assert len(list(parent_module.children())) == num_original_modules


@pytest.mark.parametrize(
    ("model", "training"),
    [
        (nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU()), True),
        (MobileNetV2(width_mult=0.25), False),
    ],
)
def test_get_model_attributes(model, training):
    model.train(training)
    assert get_model_attributes(model)["training"] == training
    assert get_model_attributes(model)["type(model)"] == type(model).__name__


def _get_forward_dict(wrap_dict):
    a = {"a": "1", "b": "2", "c": "3", "d": "4"}
    if wrap_dict:
        return (a,), {}
    return a, None


def forward0(a, b, *args, c="9", d="8", **kwargs):
    return a + b + c + d + "".join(args) + "".join(kwargs.values())


def forward1(a, b, *args, c="9", d="8"):
    return a + b + c + d + "".join(args)


def forward2(a, b, *, c="9", d="8"):
    return a + b + c + d


def forward3(a, b, c="9", d="8"):
    return a + b + c + d


def forward4(a):
    return a


def forward5(a):
    return a["a"] + a["b"] + a["c"] + a["d"]


def forward6(a, b, *args, c, d="8"):
    return a + b + c + d + "".join(args)


# using strings since order matters with string addition!
@pytest.mark.parametrize(
    ("forward", "args", "kwargs", "expect_assert_without_kwargs", "fw_fails"),
    [
        (forward0, ("0", "1", "2"), {"d": "3", "e": "4"}, True, (False, False)),
        (forward0, ("0", "1", "2", "3", "4"), {}, False, (False, False)),
        (forward0, ("0", "1", "2", "3", "4"), None, False, (False, False)),
        (forward1, ("0", "1", "2", "3", "4"), {"d": "5"}, True, (False, False)),
        (forward1, ("0", "1", "2", "3", "4"), {}, False, (False, False)),
        (forward2, ("0", "1"), {"d": "5"}, True, (False, False)),
        (forward2, ("0", "1"), {}, False, (False, False)),
        (forward3, ("0", "1", "2", "3"), None, False, (False, False)),
        (forward3, ("0", "1"), {"d": "2"}, False, (False, False)),
        (forward3, ("0",), {"d": "2"}, True, (True, True)),
        (forward4, "1", None, False, (False, False)),
        (forward5, *_get_forward_dict(True), False, (False, False)),
        (forward5, *_get_forward_dict(False), True, (False, True)),
        (forward6, ("0", "1", "2", "3"), {"c": "4"}, True, (False, False)),
        (forward6, ("0", "1", "2", "3"), {}, True, (True, True)),
    ],
)
@pytest.mark.parametrize("use_kwargs", [True, False])
def test_standardize_model_args(
    use_kwargs, forward, args, kwargs, expect_assert_without_kwargs, fw_fails
):
    expect_assert = False if use_kwargs else expect_assert_without_kwargs

    # wrap into format expected by standardize_model_args
    if kwargs is not None:
        args = (*args, kwargs)

    # check if we can process as expected
    with pytest.raises(AssertionError) if expect_assert else nullcontext():
        args_standard = standardize_model_args(forward, args, use_kwargs)

    # no need to compare if we expect an assertion error
    if expect_assert:
        return

    # compare outputs of standardized and non-standardized args
    with pytest.raises(TypeError) if fw_fails[0] else nullcontext():
        if kwargs is None:
            inputs = args if isinstance(args, tuple) else (args,)
            output = forward(*inputs)
        else:
            output = forward(*args[:-1], **args[-1])

    with pytest.raises(TypeError) if fw_fails[1] else nullcontext():
        if use_kwargs:
            output_standardized = forward(*args_standard[:-1], **args_standard[-1])
        else:
            output_standardized = forward(*args_standard)

    # can only compare outputs when we don't have failures
    if not any(fw_fails):
        assert output == output_standardized


def test_create_param_post_grad_hook():
    model = nn.Linear(16, 16)
    model.train()
    accum_grad, handle = create_param_grad_clear_hook(model.weight)
    model(torch.randn(16)).sum().backward()
    assert model.weight.grad is None
