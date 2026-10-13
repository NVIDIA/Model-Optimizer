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

"""Dynamic linear implementations based on torch.nn.modules.linear."""

import torch
from torch import nn

from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.utils import make_divisible

from ..registry import DMRegistry
from ..traced_hp import TracedHp
from .utils import get_sliced_tensor

__all__ = ["_DynamicLinear"]


@DMRegistry.register({nn.Linear: "nn.Linear"})
class _DynamicLinear(DynamicModule):
    """An ``nn.Linear`` layer with dynamic hyperparams."""

    @staticmethod
    def _get_weight(mod: "_DynamicLinear", weight: torch.Tensor) -> torch.Tensor:
        return get_sliced_tensor(mod, weight, "out_features", "in_features")

    @staticmethod
    def _get_bias(mod: "_DynamicLinear", bias: torch.Tensor | None) -> torch.Tensor | None:
        return get_sliced_tensor(mod, bias, "out_features")

    def _estimate_importance(self) -> TracedHp.Importance:
        return torch.linalg.vector_norm(self.weight.detach(), dim=0)

    def _setup(self):
        # register hyperparameters
        self._register_hparam("in_features", TracedHp(list(range(1, self.in_features + 1))))
        self._register_hparam("out_features", TracedHp(list(range(1, self.out_features + 1))))

        # register dynamic attributes of the class
        self._register_dynamic_attribute("weight", self._get_weight)
        self._register_dynamic_attribute("bias", self._get_bias)

        # register importance for in_features
        self.get_hparam("in_features").register_importance(self._estimate_importance)

    def modify(self, *, features_ratio: tuple[float, ...] | None = None, feature_divisor: int = 1):
        """Modify the dynamic choices of the module according to provided keyword arguments.

        Args:
            features_ratio: The ratios of the desired number of output/input features over original
                number of output/input features.
            feature_divisor: The divisor of the output/input features.
        """
        # modify both in_features and out_features
        features = ["in_features", "out_features"]
        for feature in features:
            hp = self.get_hparam(feature)
            choices = (
                {r * hp.original for r in features_ratio}
                if features_ratio is not None
                else set(hp.choices)
            )
            choices = {int(make_divisible(c, feature_divisor)) for c in choices}
            hp.choices = list(set(hp.choices) & choices | {hp.original})
