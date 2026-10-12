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

"""imatrix calibration for GGML formats: one encode per weight, weighted by its inputs' ``E[x_j^2]``."""

import torch

from ..utils.calib_utils import ImatrixEncoder, register_imatrix_encoder
from .common import floored_second_moments, pin_packed_weight, pinned_packed_weight, validate_weight
from .registry import GGML_FORMAT_REGISTRY

__all__ = ["GGMLImatrixEncoder"]


class GGMLImatrixEncoder(ImatrixEncoder):
    """Encodes a ``ggml``-backend weight with llama.cpp's imatrix statistic and pins its codes.

    The search is weighted by the square root of ``E[x_j^2]``, as GPTQ's is. Without error
    compensation the statistic itself is the exact diagonal of the output error, but it is
    heavy-tailed and, for an MoE expert that sees few calibration tokens, noisy; its root did better
    on both dense and MoE models. The pinned payload is what later forwards and export use, and it
    marks the weight as final: a quantizer that already holds one, such as GPTQ's, is left alone.
    """

    def accepts(self, quantizer) -> bool:
        """Enabled formats with a weighted search that do not hold final codes yet."""
        num_bits = getattr(quantizer, "num_bits", None)
        if not isinstance(num_bits, str) or not quantizer.is_enabled:
            return False
        ggml_format = GGML_FORMAT_REGISTRY.get(num_bits)
        return (
            ggml_format is not None
            and ggml_format.weighted
            and pinned_packed_weight(quantizer, num_bits) is None
        )

    def validate(self, weight, quantizer):
        """Refuse a weight the format cannot encode, such as one whose width is not a block multiple."""
        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        validate_weight(weight, ggml_format.name.upper(), block_size=ggml_format.block_size)

    def encode(self, weight, quantizer, second_moments):
        """Encode ``weight`` once, weighted by ``second_moments``, and pin the payload."""
        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        extra_args = quantizer.backend_extra_args or {}
        floored = floored_second_moments(second_moments)
        kwargs = {} if floored is None else {"importance": floored.sqrt()}
        packed, shape = ggml_format.quantize(
            weight.contiguous(),
            block_chunk_size=extra_args.get("block_chunk_size", ggml_format.block_chunk_size),
            **kwargs,
        )
        pin_packed_weight(quantizer, quantizer.num_bits, packed)
        decoded = ggml_format.dequantize(
            packed,
            shape,
            dtype=torch.float32,
            block_chunk_size=extra_args.get("decode_chunk_size", ggml_format.decode_chunk_size),
        )
        return decoded.to(weight.dtype)


register_imatrix_encoder("ggml", GGMLImatrixEncoder())
