# Adapted from ResInfer (tensor_quantizer.py / residue_linear.py):
# https://github.com/ShawRong/resinfer/tree/51b906836a69fa43915040314df57bc9f23030ad/quant/modelopt_integration
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tensor quantizer with residue metadata."""

from contextlib import contextmanager

import torch

from .tensor_quantizer import TensorQuantizer

__all__ = ["ResidueTensorQuantizer"]


class ResidueTensorQuantizer(TensorQuantizer):
    """TensorQuantizer carrying salient channels for residue Linear QDQ."""

    def __init__(
        self,
        quant_attribute_cfg=None,
        if_quant: bool = True,
        if_calib: bool = False,
        amax=None,
    ) -> None:
        """Initialize the quantizer and enable salient channel metadata."""
        super().__init__(
            quant_attribute_cfg=quant_attribute_cfg,
            if_quant=if_quant,
            if_calib=if_calib,
            amax=amax,
        )
        self._enable_salient_indices = True

    # mimic pre quant scale in the modelopt, in (Tensorquantizer, svdlinear, awq)
    @property
    def salient_indices(self):
        """Return salient_indices used for residue."""
        if not hasattr(self, "_salient_indices") or not self._enable_salient_indices:
            return None
        return self._salient_indices

    @salient_indices.setter
    def salient_indices(self, value):
        assert value is not None, "salient_indices cannot be set to None."
        assert self._enable_salient_indices, (
            "salient_indices cannot be set when salient indices are disabled."
        )
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if not hasattr(self, "_salient_indices"):
            self.register_buffer("_salient_indices", value.clone().detach())
        else:
            if self._salient_indices.shape != value.shape:
                raise RuntimeError("Changing shape when setting salient_indices is not allowed.")
            self._salient_indices.data.copy_(
                value.clone().detach().to(self._salient_indices.device)
            )

    def _fake_quantize(self, inputs):
        """Require Linear-level QDQ to pair residuals with copied weight channels."""
        if getattr(self, "_allow_residue_linear_fake_quant", False):
            return super()._fake_quantize(inputs)
        raise RuntimeError("Residue fake quantization requires ResidueQuantLinear.")

    @contextmanager
    def allow_residue_linear_fake_quant(self):
        """Temporarily allow ResidueQuantLinear to use TensorQuantizer.forward()."""
        was_allowed = getattr(self, "_allow_residue_linear_fake_quant", False)
        self._allow_residue_linear_fake_quant = True
        try:
            yield
        finally:
            self._allow_residue_linear_fake_quant = was_allowed

    def copy_extension(
        self,
        input: torch.Tensor,
        salient_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append selected channels unchanged. Used for materialized weights."""
        if salient_indices is None:
            raise RuntimeError(
                "Salient indices not loaded. Provide salient_indices to copy_extension()."
            )
        selected = input.index_select(-1, salient_indices.to(input.device))
        return torch.cat([input, selected], dim=-1)
