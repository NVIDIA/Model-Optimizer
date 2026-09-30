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

"""Dynamic class for all sparse modules."""

import torch
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor

from modelopt.torch.opt.dynamic import DynamicModule, _DMRegistryCls
from modelopt.torch.opt.hparam import Hparam

__all__ = ["SpDMRegistry", "SparseModule"]


SpDMRegistry = _DMRegistryCls(prefix="Sparse")  # global instance for the sparsity registry


@SpDMRegistry.register({nn.Linear: "nn.Linear", nn.Conv2d: "nn.Conv2d"})
class SparseModule(DynamicModule):
    """Base dynamic class for all sparse modules."""

    @staticmethod
    def _get_weight(mod: "SparseModule", weight: torch.Tensor) -> torch.Tensor:
        if mod.is_sparse and mod._weight_mask is not None:
            mask = mod._weight_mask
            # Under FSDP the weight is sharded into a DTensor while the mask buffer stays a
            # plain tensor; align the mask to the weight's sharding so the elementwise multiply
            # does not mix DTensor and torch.Tensor operands. The distributed mask is cached and
            # only rebuilt when the sharding changes, since re-sharding on every access is costly.
            if isinstance(weight, DTensor) and not isinstance(mask, DTensor):
                cached = mod._weight_mask_dtensor
                if (
                    cached is None
                    or cached.device_mesh != weight.device_mesh
                    or cached.placements != weight.placements
                ):
                    cached = distribute_tensor(mask, weight.device_mesh, weight.placements)
                    mod._weight_mask_dtensor = cached
                mask = cached
            masked_weight = weight * mask
            # Quick workaround for the custom attribute for Megatron.
            # TODO: maybe we need a more general way for customized attributes
            if hasattr(weight, "main_grad"):
                masked_weight.main_grad = weight.main_grad
            return masked_weight
        return weight

    def _setup(self):
        # define hparam to check if sparsity is activated
        hp = Hparam([0, -1], original=0)
        hp.active = 0
        self._register_hparam("is_sparse", hp)

        # define the sparse mask here (don't pre-allocate memory to maximize memory savings)
        self._register_temp_attribute("_weight_mask", None, lambda m, n, v: m.register_buffer(n, v))

        # cache of the FSDP-sharded (DTensor) mask; rebuilt lazily in _get_weight and invalidated
        # whenever _weight_mask changes. Not a buffer, so it is never serialized/restored.
        self._register_temp_attribute("_weight_mask_dtensor", None)

        # register dynamic attributes of the class
        self._register_dynamic_attribute("weight", self._get_weight)

    def modify(self, *args, **kwargs):
        """Initialize the sparsity mask when this is called.

        Note that for any module that is not frozen via ``None`` in the rules, this function will be
        called. Hence, we use this function to initialize the sparsity mask only when necessary.
        """
        hp = self.get_hparam("is_sparse")
        if -1 in hp.choices and self._weight_mask is None:
            hp.active = -1
            self._weight_mask = torch.ones_like(self.weight, dtype=torch.bool)

    def set_mask(self, value: torch.BoolTensor | None):
        """Set the active sparse mask of the module weights."""
        # invalidate the cached DTensor mask since the underlying mask is changing
        self._weight_mask_dtensor = None

        if value is None:
            self._weight_mask = None
            return

        # sanity checks on the mask
        w_shape = self.weight.shape
        assert value is not None, "Mask cannot be None."
        assert value.shape == w_shape, f"Mask must have shape {w_shape}, got {value.shape} instead."
        assert value.dtype == torch.bool, f"Mask must be of type torch.bool, but got {value.dtype}."

        # assign mask
        with torch.no_grad():
            if torch.all(value):
                self._weight_mask = None
            elif self._weight_mask is None:
                self._weight_mask = value.detach().clone().to(self.weight.device)
            else:
                self._weight_mask.copy_(value.to(self._weight_mask.device))
