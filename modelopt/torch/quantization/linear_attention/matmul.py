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

"""GDN prefill operand QDQ at GEMM boundaries that exist in the algorithm."""

import torch
from torch import nn

__all__ = []

_GDN_MATMUL_SITES = (
    "key_interaction",
    "wy_value",
    "wy_key",
    "state_read",
    "output_state",
    "output_score",
    "output_value",
    "state_update",
)


def _apply_operand_quantizer(quantizer, operand):
    """Apply dynamic NVFP4 independently to every sequence/chunk and head matrix."""
    if not quantizer.is_enabled:
        return operand
    if quantizer.is_nvfp4_dynamic:
        if operand.ndim != 4:
            raise ValueError("GDN operand QDQ requires [sequence-or-chunk, head, row, K]")
        shape = operand.shape
        matrices = operand.flatten(0, 1)
        # TensorQuantizer's NVFP4 second-level amax is scalar. Invoke it per matrix
        # so unrelated sequences and heads cannot change one another's numerics.
        return torch.stack([quantizer(matrix) for matrix in matrices]).reshape(shape)
    return quantizer(operand)


class _MatmulSite(nn.Module):
    def __init__(self):
        # QuantizeConfig imports this package before TensorQuantizer is defined.
        from modelopt.torch.quantization.config import QuantizerAttributeConfig
        from modelopt.torch.quantization.nn import TensorQuantizer

        super().__init__()
        self.lhs_quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False))
        self.rhs_quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False))

    def forward(self, lhs, rhs):
        if lhs.ndim != 4 or rhs.ndim != 4 or lhs.shape[:2] != rhs.shape[:2]:
            raise ValueError("GDN operand matmuls require compatible four-dimensional operands")
        if lhs.shape[-1] != rhs.shape[-1]:
            raise ValueError("GDN operand matmuls require a shared reduction dimension")
        with torch.autocast(device_type=lhs.device.type, enabled=False):
            lhs = _apply_operand_quantizer(self.lhs_quantizer, lhs)
            rhs = _apply_operand_quantizer(self.rhs_quantizer, rhs)
            return lhs @ rhs.transpose(-1, -2)


class GDNMatmulSites(nn.ModuleDict):
    """Independent LHS/RHS quantizers for the eight GDN prefill GEMMs.

    KDA is deliberately excluded: its gated pairwise operands are not produced by
    any GEMM kernel, and materializing them would train against the wrong error model.
    """

    def __init__(self):
        super().__init__({name: _MatmulSite() for name in _GDN_MATMUL_SITES})

    @property
    def is_enabled(self):
        return any(quantizer.is_enabled for site in self.values() for quantizer in site.children())

    def matmul(self, name, lhs, rhs):
        return self[name](lhs, rhs)

    def validate(self):
        for site_name, site in self.items():
            for operand_name, quantizer in site.named_children():
                _validate_operand_quantizer(
                    quantizer, f"gdn_matmul_sites.{site_name}.{operand_name}"
                )


def _validate_operand_quantizer(quantizer, name):
    from modelopt.torch.quantization.nn import TensorQuantizer

    if not isinstance(quantizer, TensorQuantizer):
        raise ValueError(f"{name} requires a single TensorQuantizer")
    if not quantizer.is_enabled:
        return
    fp8 = (
        quantizer.num_bits == (4, 3)
        and quantizer.axis == (0, 1, 2)
        and quantizer.block_sizes is None
    )
    nvfp4 = quantizer.is_nvfp4_dynamic and quantizer.block_sizes == {
        -1: 16,
        "type": "dynamic",
        "scale_bits": (4, 3),
    }
    if not (
        (fp8 or nvfp4)
        and quantizer._dynamic
        and quantizer.fake_quant
        and quantizer._pass_through_bwd
        and quantizer.rounding == "nearest"
        and not quantizer.rotate_is_enabled
        and quantizer.pre_quant_scale is None
        and quantizer.backend is None
        and not quantizer._bias
        and not quantizer._use_constant_amax
    ):
        raise ValueError(
            f"{name} requires nearest-rounded dynamic per-row E4M3 or block16 NVFP4 "
            "with identity STE and no extra transformations"
        )
