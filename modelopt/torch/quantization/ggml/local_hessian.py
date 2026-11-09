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

"""local_hessian for GGML formats: each group's local scale picked by its inputs' local Hessian."""

import torch

from ..utils.calib_utils import LocalHessianEncoder, register_local_hessian_encoder
from .choice_search import search_choices
from .common import (
    GGML_BLOCK_SIZE,
    floored_second_moments,
    pin_packed_weight,
    pinned_packed_weight,
    validate_weight,
)
from .registry import GGML_FORMAT_REGISTRY

__all__ = ["GGMLLocalHessianEncoder"]


class GGMLLocalHessianEncoder(LocalHessianEncoder):
    """Encodes a ``ggml``-backend weight with Hessian-picked local scales and pins its codes.

    The codebook search keeps imatrix's weighting, the square root of the Hessian diagonal; the
    per-group choice that search would take by that diagonal is instead picked by the full
    256-column block of the Hessian (:func:`~.choice_search.search_choices`), damped by
    ``perc_damp`` times its mean diagonal: undamped, the picks overfit the calibration inputs. A
    weight no calibration input reached is encoded plainly. Like imatrix, the pinned payload marks
    the weight as final, so a quantizer that already holds one, such as GPTQ's, is left alone, and
    a format that keeps its pinned scales leaves the weight unchanged.
    """

    block_size = GGML_BLOCK_SIZE

    def accepts(self, quantizer) -> bool:
        """Enabled formats with per-group choices that do not hold final codes yet."""
        num_bits = getattr(quantizer, "num_bits", None)
        if not isinstance(num_bits, str) or not quantizer.is_enabled:
            return False
        ggml_format = GGML_FORMAT_REGISTRY.get(num_bits)
        return (
            ggml_format is not None
            and ggml_format.num_choices > 0
            and pinned_packed_weight(quantizer, num_bits) is None
        )

    def validate(self, weight, quantizer):
        """Refuse a weight the format cannot encode, such as one whose width is not a block multiple."""
        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        validate_weight(weight, ggml_format.name.upper(), block_size=ggml_format.block_size)

    def encode(self, weight, quantizer, hessian, perc_damp=0.0):
        """Encode ``weight`` with the choices the damped block Hessians pick, and pin the payload."""
        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        extra_args = quantizer.backend_extra_args or {}
        block_chunk_size = extra_args.get("block_chunk_size", ggml_format.block_chunk_size)
        decode_chunk_size = extra_args.get("decode_chunk_size", ggml_format.decode_chunk_size)
        kwargs = {}
        if hessian is not None:
            moments = floored_second_moments(hessian.diagonal(dim1=1, dim2=2).reshape(-1))
            if moments is not None:
                if ggml_format.weighted:
                    kwargs["importance"] = moments.sqrt().to(weight.device)
                hessian = hessian.to(device=weight.device, dtype=torch.float32)
                # Symmetrized, as the CUDA search reads each block by column.
                hessian = (hessian + hessian.transpose(1, 2)) / 2
                ridge = perc_damp * hessian.diagonal(dim1=1, dim2=2).mean()
                hessian.diagonal(dim1=1, dim2=2).add_(ridge)
                kwargs["choices"] = search_choices(
                    ggml_format,
                    weight,
                    hessian,
                    kwargs.get("importance"),
                    block_chunk_size,
                    decode_chunk_size,
                )
        packed, shape = ggml_format.quantize(
            weight.contiguous(), block_chunk_size=block_chunk_size, **kwargs
        )
        # Pinned with the weight its choices were searched for, so that weight keeps these codes.
        pin_packed_weight(
            quantizer,
            quantizer.num_bits,
            packed,
            kwargs.get("importance"),
            source=weight if "choices" in kwargs else None,
        )
        if ggml_format.payload_scales is not None:
            return weight
        decoded = ggml_format.dequantize(
            packed, shape, dtype=torch.float32, block_chunk_size=decode_chunk_size
        )
        return decoded.to(weight.dtype)


register_local_hessian_encoder("ggml", GGMLLocalHessianEncoder())
