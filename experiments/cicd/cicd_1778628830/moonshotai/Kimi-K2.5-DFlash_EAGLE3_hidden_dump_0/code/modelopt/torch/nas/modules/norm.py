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

"""Dynamic norm implementations based on norm modules in torch.nn.modules."""

from collections.abc import Sequence

import torch
from torch import nn

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils import make_divisible

from ..registry import DMRegistry
from ..traced_hp import TracedHp
from .utils import get_sliced_tensor

__all__ = ["_DynamicBatchNorm", "_DynamicGroupNorm", "_DynamicInstanceNorm", "_DynamicLayerNorm"]


class _DynamicBatchInstance(DynamicModule):
    """Dynamic base class for batch norm and instance norm layers.

    NOTE: Don't use this class for instance checks. Use _DynamicBatchNorm or _DynamicInstanceNorm
    instead!
    """

    @staticmethod
    def _cut_to_active_features(mod: "_DynamicBatchInstance", value: torch.Tensor | None):
        return get_sliced_tensor(mod, value, "num_features")

    def _setup(self):
        # register hyperparameters
        self._register_hparam("num_features", TracedHp(list(range(1, self.num_features + 1))))

        # register dynamic attributes
        dyn_attrs = ["running_mean", "running_var", "weight", "bias"]
        for attr in dyn_attrs:
            self._register_dynamic_attribute(attr, self._cut_to_active_features)

    def modify(self, *, features_ratio: tuple[float, ...] | None = None, feature_divisor: int = 1):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            features_ratio: The ratios of the desired number of features over original number of
                features.
            feature_divisor: The divisor of the number of features.
        """
        hp = self.get_hparam("num_features")
        choices = (
            {r * hp.original for r in features_ratio}
            if features_ratio is not None
            else set(hp.choices)
        )
        choices = {int(make_divisible(c, feature_divisor)) for c in choices}
        hp.choices = list(set(hp.choices) & choices | {hp.original})


@DMRegistry.register(
    {
        nn.BatchNorm1d: "nn.BatchNorm1d",
        nn.BatchNorm2d: "nn.BatchNorm2d",
        nn.BatchNorm3d: "nn.BatchNorm3d",
        nn.SyncBatchNorm: "nn.SyncBatchNorm",
    }
)
class _DynamicBatchNorm(_DynamicBatchInstance):
    """Just syntactic sugar so we have a common base class for batch norm only."""


@DMRegistry.register(
    {
        nn.InstanceNorm1d: "nn.InstanceNorm1d",
        nn.InstanceNorm2d: "nn.InstanceNorm2d",
        nn.InstanceNorm3d: "nn.InstanceNorm3d",
    }
)
class _DynamicInstanceNorm(_DynamicBatchInstance):
    """Just syntactic sugar so we have a common base class for instance norm only."""


@DMRegistry.register({nn.LayerNorm: "nn.LayerNorm"})
class _DynamicLayerNorm(DynamicModule):
    """An ``nn.LayerNorm`` layer with dynamic hyperparams."""

    @staticmethod
    def _get_normalized_shape(mod: "_DynamicLayerNorm", value: Sequence[int | TracedHp]) -> tuple:
        return (*tuple(value[:-1]), mod.num_features)

    @staticmethod
    def _cut_to_active_features(
        mod: "_DynamicLayerNorm", value: torch.Tensor | None
    ) -> torch.Tensor | None:
        if value is None:
            return value
        nf_slice = mod.get_hparam("num_features").active_slice
        return value[..., nf_slice].contiguous()

    def _setup(self):
        # construct normalized shape with Hparam as last dimension
        normalized_shape = list(self.normalized_shape)
        normalized_shape[-1] = TracedHp(list(range(1, normalized_shape[-1] + 1)))

        # register the hyperparameter with a new name
        self._register_hparam("num_features", normalized_shape[-1])

        # register dynamic attributes
        dyn_attrs = ["weight", "bias"]
        for attr in dyn_attrs:
            self._register_dynamic_attribute(attr, self._cut_to_active_features)

        self._register_dynamic_attribute("normalized_shape", self._get_normalized_shape)

    def modify(self, *, features_ratio: tuple[float, ...] | None = None, feature_divisor: int = 1):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            features_ratio: The ratios of the desired number of features over original number of
                features.
            feature_divisor: The divisor of the number of features.
        """
        hp = self.get_hparam("num_features")
        choices = (
            {r * hp.original for r in features_ratio}
            if features_ratio is not None
            else set(hp.choices)
        )
        choices = {int(make_divisible(c, feature_divisor)) for c in choices}
        hp.choices = list(set(hp.choices) & choices | {hp.original})


@DMRegistry.register({nn.GroupNorm: "nn.GroupNorm"})
class _DynamicGroupNorm(DynamicModule):
    """An ``nn.GroupNorm`` layer with dynamic hyperparams."""

    _group_size: int

    @staticmethod
    def _get_num_groups(mod: "_DynamicGroupNorm", value: int) -> int:
        return mod.num_channels // mod._group_size

    @staticmethod
    def _cut_to_active_channels(mod: "_DynamicGroupNorm", value: torch.Tensor | None):
        return get_sliced_tensor(mod, value, "num_channels")

    def _setup(self):
        # register num_channels as hyperparameter
        group_size = self.num_channels // self.num_groups
        choices = [
            c
            for c in range(self.num_groups, self.num_channels + 1)
            if c % self.num_groups == 0 and c % group_size == 0
        ]
        self._register_hparam("num_channels", TracedHp(choices, original=self.num_channels))

        # register num_groups as a dynamic attribute so group size is same
        self._register_temp_attribute("_group_size", group_size)
        self._register_dynamic_attribute("num_groups", self._get_num_groups)

        # register dynamic attributes
        dyn_attrs = ["weight", "bias"]
        for attr in dyn_attrs:
            self._register_dynamic_attribute(attr, self._cut_to_active_channels)

    def modify(self, *, channels_ratio: tuple[float, ...] | None = None, channel_divisor: int = 1):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            channels_ratio: The ratios of the desired number of out/in channels over original
                number of out/in channels.
            channel_divisor: The divisor of the out/in channels.
        """
        hp = self.get_hparam("num_channels")
        choices: list[int]
        choices = (
            {r * hp.original for r in channels_ratio}
            if channels_ratio is not None
            else set(hp.choices)
        )
        choices = {int(make_divisible(c, channel_divisor)) for c in choices}
        hp.choices = list(set(hp.choices) & choices | {hp.original})
