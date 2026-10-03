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
from torch.nn.modules.conv import _ConvNd

from modelopt.torch.nas.registry import DMRegistry
from modelopt.torch.utils import make_divisible

# TODO: enable remaining unit tests!!!


def _conv(ndim: int, transposed: bool) -> type[_ConvNd]:
    return getattr(nn, f"Conv{'Transpose' if transposed else ''}{ndim}d")


def _check_inactive_weight_grad_allzero(
    weight: torch.tensor, active_shape: tuple[int, ...], max_weight_matrix_size: tuple[int, ...]
) -> bool:
    for dim, active_dim_size in enumerate(active_shape):
        if dim < 2:
            # dim corresponding to in_channels  or out_channels
            indices = torch.arange(active_dim_size, max_weight_matrix_size[dim])
        else:
            # dim corresponding to kernel,
            max_kernel_size_dim = max_weight_matrix_size[dim]
            indices = list(range(max_kernel_size_dim // 2 - active_dim_size // 2))
            nonzero_active_kernel_start_index = max_kernel_size_dim // 2 - active_dim_size // 2
            indices += range(
                nonzero_active_kernel_start_index + active_dim_size, max_kernel_size_dim
            )
            indices = torch.tensor(indices)
        if indices.shape[0] == 0:
            continue
        if not torch.allclose(torch.index_select(weight, dim, indices), torch.tensor([0.0])):
            return False
    return True


def _get_weight_tensor_shape(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, ...],
    ndim: int,
    groups: int,
    transposed: bool,
):
    ks_dim = (kernel_size,) * ndim if isinstance(kernel_size, int) else kernel_size
    c_dim = [out_channels, in_channels][:: (-1 if transposed else 1)]
    assert c_dim[1] % groups == 0
    c_dim[1] //= groups
    return tuple(c_dim) + ks_dim


@pytest.mark.parametrize(
    (
        "ndim",
        "in_channels",
        "out_channels",
        "kernel_size",
        "active_in_channels",
        "active_out_channels",
        "active_kernel_size",
        "padding",
        "active_padding",
        "groups",
        "transposed",
    ),
    [
        # ConvNd
        (1, 10, 10, 7, 6, 5, (5,), "same", "same", 1, False),
        (2, 8, 10, 5, 4, 6, (3, 3), "same", "same", 1, False),
        (3, 6, 6, 3, 2, 4, (3, 3, 3), "same", "same", 1, False),
        # ConvTransposeNd (does not allow string padding)
        (1, 10, 10, 7, 6, 5, (5,), 3, (2,), 1, True),
        (2, 8, 10, 5, 4, 6, (3, 3), 3, (2, 2), 1, True),
        (3, 6, 6, 3, 2, 4, (3, 3, 3), 3, (3, 3, 3), 1, True),
        # ConvNd - depthwise
        (1, 10, 10, 7, 4, 4, (5,), "same", "same", 10, False),
        (2, 8, 8, 5, 3, 3, (3, 3), "same", "same", 8, False),
        (3, 6, 6, 3, 1, 1, (3, 3, 3), "same", "same", 6, False),
        # ConvTransposeNd - depthwise
        (1, 10, 10, 7, 9, 9, (5,), 3, (2,), 10, True),
        (2, 8, 8, 5, 6, 6, (3, 3), 3, (2, 2), 8, True),
        (3, 6, 6, 3, 4, 4, (3, 3, 3), 3, (3, 3, 3), 6, True),
        # Conv(Tranpose)Nd - grouped, in_c == out_c
        (2, 8, 8, 5, 6, 6, (3, 3), "same", "same", 4, False),
        (3, 12, 12, 3, 6, 6, (3, 3, 3), 3, (3, 3, 3), 2, True),
        # Conv(Transpose)Nd - grouped, no channel support
        (1, 16, 8, 7, 16, 8, (5,), "same", "same", 4, False),
        (1, 6, 12, 7, 6, 12, (5,), 3, (2,), 6, True),
    ],
)
def test_dynamic_conv_and_conv_transpose(
    ndim: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    active_in_channels: int,
    active_out_channels: int,
    active_kernel_size: tuple[int, ...],
    padding: str | int,
    active_padding: str | int,
    groups: int,
    transposed: bool,
) -> None:
    def _get_dyn_model():
        model = _conv(ndim, transposed)(
            in_channels, out_channels, kernel_size, padding=padding, groups=groups
        )
        return DMRegistry.convert(model)

    # get a model
    model = _get_dyn_model()

    # compute channel choices
    oc_per_group = out_channels // groups
    ic_per_group = in_channels // groups
    if groups > 1 and in_channels != out_channels:
        oc_choices = [out_channels]
        ic_choices = [in_channels]
    else:
        oc_choices = [c for c in range(1, out_channels + 1) if groups == 1 or c % oc_per_group == 0]
        ic_choices = [c for c in range(1, in_channels + 1) if groups == 1 or c % ic_per_group == 0]

    # check out_channels
    hparam_out_channels = model.get_hparam("out_channels")
    assert hparam_out_channels.choices == oc_choices

    # check in_channels
    assert model.get_hparam("in_channels").max == in_channels
    assert model.get_hparam("in_channels").choices == ic_choices

    # check groups
    assert model.groups == groups

    # check weight shape
    max_weight_tensor_shape = _get_weight_tensor_shape(
        in_channels, out_channels, kernel_size, ndim, groups, transposed
    )
    assert model.weight.shape == max_weight_tensor_shape

    # check that we cannot assign a wrong value
    with pytest.raises(AssertionError):
        wrong_out_channels = model.get_hparam("out_channels").max * 2
        model.out_channels = wrong_out_channels

    # check that we cannot query arbitrary attribute
    with pytest.raises(KeyError):
        model.get_hparam("weight")

    # check if for max subnet the weight/bias are the original ones
    model.out_channels = model.get_hparam("out_channels").max
    model.kernel_size = model.get_hparam("kernel_size").max
    model.in_channels = model.get_hparam("in_channels").max

    assert model.weight is model._parameters["weight"]
    assert model.bias is model._parameters["bias"]

    # re-assign hyperparameters
    model.out_channels = active_out_channels
    model.kernel_size = active_kernel_size
    model.in_channels = active_in_channels

    # check that we can access dynamic hparams as expected
    assert model.in_channels == active_in_channels
    assert model.out_channels == active_out_channels
    assert model.kernel_size == active_kernel_size

    # check that active slice is as expected
    assert model.get_hparam("in_channels").active_slice == slice(active_in_channels)
    assert model.get_hparam("out_channels").active_slice == slice(active_out_channels)
    with pytest.raises(AssertionError):
        # kernel size is a tuple and doesn't have a slice
        model.get_hparam("kernel_size").active_slice

    # check that dynamic attributes work as expected
    active_groups = 1 if groups == 1 else active_out_channels // oc_per_group
    assert model.groups == active_groups
    active_shape = _get_weight_tensor_shape(
        active_in_channels, active_out_channels, active_kernel_size, ndim, active_groups, transposed
    )
    assert model.weight.shape == active_shape

    assert model.bias.shape[0] == active_out_channels
    assert model.padding == active_padding

    inputs = torch.randn(9, active_in_channels, *[10] * ndim)

    model.train()
    targets = model(inputs)
    outputs = model(torch.ones(1, active_in_channels, *[10] * ndim))
    loss = outputs.sum()
    loss.backward()

    assert _check_inactive_weight_grad_allzero(
        model._parameters["weight"].grad, active_shape, max_weight_tensor_shape
    )

    assert torch.allclose(model._parameters["bias"].grad[active_out_channels:], torch.tensor([0.0]))

    # check export
    model = model.export()
    assert type(model) is _conv(ndim, transposed)
    assert model.in_channels == active_in_channels
    assert model.out_channels == active_out_channels
    assert model.kernel_size == active_kernel_size

    outputs = model(inputs)
    assert outputs.shape == targets.shape
    assert torch.allclose(outputs, targets)

    # get a fresh model
    model = _get_dyn_model()

    # now check if we can reduce the choices
    c_ratio = [0.5, 1.0]
    c_divisor = 2
    model.modify(channels_ratio=c_ratio, channel_divisor=c_divisor)

    # a few sanity checks
    # NOTE: check is not 100% generic (even # numbers of channels and more constraints).
    # Make sure to check here if you add more test cases
    channels = {"out_channels": oc_choices, "in_channels": ic_choices}
    for channel, orig_choices in channels.items():
        hp = model.get_hparam(channel)
        assert all(c % c_divisor == 0 for c in hp.choices), (channel, hp)
        if len(orig_choices) > 1:
            assert hp.choices == [make_divisible(r * hp.original, c_divisor) for r in c_ratio]
        else:
            assert hp.choices == [hp.original]
