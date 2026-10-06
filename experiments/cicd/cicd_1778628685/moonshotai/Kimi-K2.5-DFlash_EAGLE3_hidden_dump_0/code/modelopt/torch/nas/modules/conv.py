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

"""Dynamic Conv implementations based on torch.nn.modules.conv."""

import itertools

import torch
from torch import nn

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils import make_divisible, val2tuple

from ..registry import DMRegistry
from ..traced_hp import TracedHp
from .utils import get_sliced_tensor, get_sliced_tensor_by_slices

__all__ = ["_DynamicConvNd", "_DynamicConvTransposeNd"]


@DMRegistry.register({nn.Conv1d: "nn.Conv1d", nn.Conv2d: "nn.Conv2d", nn.Conv3d: "nn.Conv3d"})
class _DynamicConvNd(DynamicModule):
    @property
    def ndim(self):
        return len(self.kernel_size)

    @staticmethod
    def _assert_input_format(
        mod: "_DynamicConvNd", input: tuple[torch.Tensor] | torch.Tensor
    ) -> None:
        if isinstance(input, tuple):
            assert len(input) == 1, f"Expected single input, but got {input}."
            input = input[0]
        if input.dim() != mod.ndim + 2:
            raise ValueError(f"Expected batched input (e.g. NCHW vs CHW), but got {input.shape}.")

    @staticmethod
    def _get_padding(
        mod: "_DynamicConvNd", padding: str | tuple[int, ...]
    ) -> str | tuple[int, ...]:
        """New padding such that output size stays the same as with max kernel size and padding."""
        if isinstance(padding, str):
            return padding

        active_padding = []
        max_kernel_size = mod.get_hparam("kernel_size").max
        for dim in range(mod.ndim):
            max_ks = max_kernel_size[dim]  # type: ignore[index]
            active_ks = mod.kernel_size[dim]
            assert max_ks % 2 == active_ks % 2, (max_ks, active_ks)

            max_p = padding[dim]
            active_p = max_p - (max_ks - active_ks) * mod.dilation[dim] // 2
            assert active_p >= 0, (max_p, active_p)

            active_padding.append(active_p)
        return tuple(active_padding)

    @staticmethod
    def _get_bias(mod: "_DynamicConvNd", bias: torch.Tensor | None) -> torch.Tensor | None:
        return get_sliced_tensor(mod, bias, "out_channels")

    @staticmethod
    def _get_channel_order() -> list[str]:
        return ["out_channels", "in_channels"]

    @staticmethod
    def _get_weight(mod: "_DynamicConvNd", weight: torch.Tensor) -> torch.Tensor:
        slices = []

        # get channel hparams in correct order
        hps_c = [mod.get_hparam(c_name) for c_name in mod._get_channel_order()]
        c_active = [hp.active for hp in hps_c]

        # add slice for 1st channel dimension
        slices.append(hps_c[0].active_slice)

        # slice for 2nd channel dimension depends on groups
        slice_1 = hps_c[1].active_slice
        groups = mod.groups
        if groups == 1:
            slices.append(slice_1)
        elif all(c % groups == 0 for c in c_active):
            assert slice_1 == slice(c_active[1]) or c_active[0] == c_active[1] == groups
            slices.append(slice(c_active[1] // groups))
        else:
            raise NotImplementedError(f"Group size {groups} not supported!")

        # kernel size slices
        max_kernel_size = mod.get_hparam("kernel_size").max
        active_kernel_size = mod.kernel_size
        assert isinstance(max_kernel_size, tuple)
        for max_ks, active_ks in zip(max_kernel_size, active_kernel_size):
            a = max_ks // 2 - active_ks // 2
            b = max_ks // 2 + active_ks // 2 + max_ks % 2
            slices.append(slice(a, b))

        return get_sliced_tensor_by_slices(weight, slices)

    @staticmethod
    def _get_groups(mod: "_DynamicConvNd", groups: int) -> int:
        # retrieve channel hparams
        hp_oc = mod.get_hparam("out_channels")
        hp_ic = mod.get_hparam("in_channels")

        # vanilla cases
        if groups == 1 or (hp_oc.active == hp_oc.original and hp_ic.active == hp_ic.original):
            return groups

        # only support out_channels == in_channels if grouped
        assert hp_oc.active == hp_ic.active, (hp_oc.active, hp_ic.active)

        # compute current number of groups
        assert hp_oc.original % groups == 0, (hp_oc.original, groups)
        active_groups = (hp_oc.active * groups) // hp_oc.original

        # sanity checks
        assert isinstance(active_groups, int)
        assert hp_oc.active % active_groups == 0, (hp_oc.active, active_groups)
        assert hp_ic.active % active_groups == 0, (hp_ic.active, active_groups)

        return active_groups

    def _estimate_importance(self) -> TracedHp.Importance:
        # for group > 1, we do not know how to handle it yet
        if self.groups > 1:
            return None
        weight = self.weight
        c_in = weight.shape[1]
        return torch.linalg.vector_norm(
            torch.reshape(weight.detach().transpose(0, 1), (c_in, -1)), dim=1
        )

    def _setup(self):
        # only support ungrouped conv or grouped conv with in_channels == out_channels
        if self.groups == 1:
            oc_step = ic_step = 1
        elif self.out_channels == self.in_channels:
            oc_step = ic_step = self.out_channels // self.groups
        else:
            oc_step = self.out_channels
            ic_step = self.in_channels
        oc_choices = [c for c in range(1, self.out_channels + 1) if c % oc_step == 0]
        ic_choices = [c for c in range(1, self.in_channels + 1) if c % ic_step == 0]

        # construct choices for kernel size hyperparameter
        ks_set = {self.kernel_size}
        # TODO: padding mode other than "zeros" is not supported since in other padding mode, conv
        # will use "_reversed_padding_repeated_twice" instead of "padding" in the forward function.
        if self.padding_mode == "zeros":
            for ks in itertools.product(*[range(1, k + 1) for k in self.kernel_size]):
                for d in range(self.ndim):
                    # check kernel size: smaller than original, same parity
                    if ks[d] > self.kernel_size[d] or ks[d] % 2 != self.kernel_size[d] % 2:
                        break

                    # check padding: non-negative
                    if (
                        isinstance(self.padding, tuple)
                        and self.padding[d] < (self.kernel_size[d] - ks[d]) * self.dilation[d] // 2
                    ):
                        break
                else:
                    ks_set.add(ks)
        ks_choices = list(ks_set)

        # register hyperparameters
        self._register_hparam("out_channels", TracedHp(oc_choices, self.out_channels))
        self._register_hparam("in_channels", TracedHp(ic_choices, self.in_channels))
        self._register_hparam("kernel_size", TracedHp(ks_choices, self.kernel_size))

        # We restrict the input to be batched (e.g. NCHW) format so searchable_tensor_dims can be 1.
        # Ideally searchable_tensor_dims for Conv2d would be -3, but its more common to use
        # torch.cat([...], dim=1), and enforcing batched input would allow us to support these.
        hook_handle = self.register_forward_pre_hook(self._assert_input_format)
        self._register_temp_attribute(
            "_hook_handle", hook_handle, del_hook=lambda m, n: m._hook_handle.remove()
        )

        # register dynamic attributes
        self._register_dynamic_attribute("padding", self._get_padding)
        self._register_dynamic_attribute("weight", self._get_weight)
        self._register_dynamic_attribute("bias", self._get_bias)
        self._register_dynamic_attribute("groups", self._get_groups)

        # register importance for in_channels
        self.get_hparam("in_channels").register_importance(self._estimate_importance)

    def modify(
        self,
        *,
        channels_ratio: tuple[float, ...] | None = None,
        channel_divisor: int = 1,
        kernel_size: tuple[int | tuple[int, ...], ...] = (),
    ):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            channels_ratio: The ratios of the desired number of out/in channels over original
                number of out/in channels.
            channel_divisor: The divisor of the out/in channels.
            kernel_sizes: The desired kernel sizes.
        """
        # modify both in_channels and out_channels
        channels = ["in_channels", "out_channels"]
        choices: set[float]
        for channel in channels:
            hp = self.get_hparam(channel)
            if channels_ratio is not None:
                assert isinstance(hp.original, int)
                choices = {r * hp.original for r in channels_ratio}
            else:
                choices = set(hp.choices)  # type: ignore[arg-type]
            choices_rounded: set[int] = {int(make_divisible(c, channel_divisor)) for c in choices}
            hp.choices = list(set(hp.choices) & choices_rounded | {hp.original})

        # modify kernel size choices
        hp_ks = self.get_hparam("kernel_size")
        kernel_size = hp_ks.choices if kernel_size is None else kernel_size
        ks_choices = {val2tuple(ks, self.ndim) for ks in kernel_size}
        hp_ks.choices = list(set(hp_ks.choices) & ks_choices | {hp_ks.original})


@DMRegistry.register(
    {
        nn.ConvTranspose1d: "nn.ConvTranspose1d",
        nn.ConvTranspose2d: "nn.ConvTranspose2d",
        nn.ConvTranspose3d: "nn.ConvTranspose3d",
    }
)
class _DynamicConvTransposeNd(_DynamicConvNd):
    @staticmethod
    def _get_channel_order() -> list[str]:
        return ["in_channels", "out_channels"]

    def _estimate_importance(self) -> TracedHp.Importance:
        # for group > 1, we do not know how to handle it yet
        if self.groups > 1:
            return None
        weight = self.weight
        c_in = weight.shape[0]
        return torch.linalg.vector_norm(torch.reshape(weight.detach(), (c_in, -1)), dim=1)
