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

from ..utils.calib_utils import ImatrixEncoder, register_imatrix_encoder
from .common import (
    floored_second_moments,
    pin_energy_matched,
    pinned_packed_weight,
    validate_weight,
)
from .registry import GGML_FORMAT_REGISTRY

__all__ = ["GGMLImatrixEncoder"]


class GGMLImatrixEncoder(ImatrixEncoder):
    """Encodes a ``ggml``-backend weight with llama.cpp's imatrix statistic and pins its codes.

    The encode is :func:`~.common.pin_energy_matched` under ``E[x_j^2]``, the exact diagonal of the
    output error, whose search takes its square root. The pinned payload is what later forwards and
    export use, and it marks the weight as final: a quantizer that already holds one, such as
    GPTQ's, is left alone. The weight itself is left as it is.
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
        if ggml_format.scale_fit is None or ggml_format.payload_scales is None:
            raise ValueError(
                f"imatrix encodes formats that derive and keep a super-block scale, not "
                f"{ggml_format.name.upper()}"
            )

    def encode(self, weight, quantizer, second_moments):
        """Encode ``weight`` once, weighted by ``second_moments``, and pin the payload."""
        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        pin_energy_matched(ggml_format, quantizer, weight, floored_second_moments(second_moments))
        return weight


register_imatrix_encoder("ggml", GGMLImatrixEncoder())
