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

"""Native scalar block codes and fields for formats without a packed GEMM codec.

Codes use their native sign/exponent/mantissa bits in a byte container. Codebooks
come from the existing CUDA converter; this adds no quantization algorithm or
learnable catalog. Deployment still requires a backend that packs that format.
"""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from ..extensions import get_cuda_ext_mx
from .base_qtensor import BaseQuantizedTensor

__all__ = ["MXQTensor"]


@lru_cache(maxsize=16)
def _positive_levels(bits):
    from ..tensor_quant import mx_format_map

    extension = get_cuda_ext_mx(raise_if_failed=True)
    kind = getattr(extension.Types, mx_format_map[bits])
    # Every supported FP4/6/8 codepoint is exactly representable in FP16.
    # Query the native scalar converter rather than duplicate its format tables.
    samples = torch.arange(0x7C00, dtype=torch.int16).view(torch.float16).tolist()
    values = {extension.convert_to_exmy(value, kind) for value in samples}
    return tuple(sorted(value for value in values if math.isfinite(value)))


@lru_cache(maxsize=64)
def _level_tensor(values, device):
    return torch.tensor(values, device=device, dtype=torch.float32)


class MXQTensor(BaseQuantizedTensor):
    """Byte-contained scalar codes with native E8M0 or E4M3 block scales."""

    @classmethod
    def quantize(cls, inputs, bits, block_size, scale_bits=(8, 0), amax=None):
        """Capture exact native block fields and encode their immutable codepoints."""
        from ..tensor_quant import mx_format_map

        if not inputs.is_cuda:
            raise NotImplementedError(
                "Native scalar MX initialization requires the CUDA converter."
            )
        extension = get_cuda_ext_mx(raise_if_failed=True)
        shape = tuple(inputs.shape)
        columns = shape[-1]
        blocks = math.ceil(columns / block_size)
        scales = torch.empty((*shape[:-1], blocks), device=inputs.device, dtype=torch.float32)
        if scale_bits not in ((8, 0), (4, 3)):
            raise NotImplementedError("Scalar block fields require E8M0 or E4M3 scales.")
        if scale_bits == (4, 3):
            if amax is None:
                amax = inputs.detach().float().abs().amax()
            else:
                amax = amax.float().amax()
        values = extension.fused_amax_convert(
            inputs.contiguous(),
            block_size,
            getattr(extension.Types, mx_format_map[bits]),
            getattr(extension.Types, mx_format_map[scale_bits]),
            amax if scale_bits == (4, 3) else None,
            scales,
        )
        expanded = scales.repeat_interleave(block_size, dim=-1)[..., :columns]
        normalized = values.float() / expanded
        positive: tuple[float, ...]
        if bits == 8:
            codes = normalized.round().to(torch.int8).view(torch.uint8)
            positive, sign_bit = (), 0
        else:
            positive = _positive_levels(bits)
            levels = _level_tensor(positive, inputs.device)
            boundaries = (levels[:-1] + levels[1:]) / 2
            indices = torch.bucketize(normalized.abs().contiguous(), boundaries, out_int32=True)
            sign_bit = 1 << sum(bits)
            codes = indices.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) * sign_bit)
        native = cls(shape, inputs.dtype, codes)
        native.metadata.update(
            bits=bits, positive_levels=positive, sign_bit=sign_bit, scale_bits=scale_bits
        )
        if scale_bits == (8, 0):
            _, exponent = torch.frexp(scales)
            return native, (exponent - 1 + 127).clamp(0, 254).to(torch.uint8), None
        maximum = 127 if bits == 8 else positive[-1]
        global_scale = amax / (maximum * torch.finfo(torch.float8_e4m3fn).max)
        global_scale = torch.where((global_scale > 0) & global_scale.isfinite(), global_scale, 1)
        local = (scales / global_scale).clamp(0, torch.finfo(torch.float8_e4m3fn).max)
        return native, local.to(torch.float8_e4m3fn), global_scale

    def dequantize(self, *, scale, block_sizes, double_scale=None, dtype=None, **kwargs):
        """Decode fixed native levels and apply projected native block fields."""
        dtype = dtype or self.metadata["dtype"]
        codes = self._quantized_data
        if self.metadata["bits"] == 8:
            levels = codes.view(torch.int8).float()
        else:
            sign_bit = self.metadata["sign_bit"]
            table = _level_tensor(tuple(self.metadata["positive_levels"]), codes.device)
            indices = (codes & (sign_bit - 1)).to(torch.int32)
            levels = F.embedding(indices, table.unsqueeze(-1)).squeeze(-1)
            levels = torch.where((codes & sign_bit) != 0, -levels, levels)
        columns = self.metadata["shape"][-1]
        fields = (
            torch.exp2(scale.float() - 127)
            if self.metadata.get("scale_bits", (8, 0)) == (8, 0)
            else scale.float() * double_scale
        )
        expanded = fields.repeat_interleave(block_sizes[-1], dim=-1)[..., :columns]
        return (levels.reshape(self.metadata["shape"]) * expanded).to(dtype)
