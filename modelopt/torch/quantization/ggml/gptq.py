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

"""GPTQ for GGML block formats, which encode each block of consecutive weights jointly."""

from collections.abc import Callable

import torch

from ..utils.calib_utils import GPTQHelper, register_gptq_helper
from .common import pin_packed_weight
from .registry import GGML_FORMAT_REGISTRY

__all__ = ["GGMLGPTQHelper", "gptq_group_update"]


def gptq_group_update(
    weight: torch.Tensor,
    h_inv: torch.Tensor,
    block_size: int,
    group_size: int,
    quantize_group: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """GPTQ update for formats that quantize ``group_size`` consecutive columns together.

    Each group is fake-quantized as a whole and its error ``E`` folded into the later columns as
    ``E @ inv(U_gg) @ U_g,rest``, where ``U`` is ``h_inv``. With ``group_size=1`` this is the
    column-wise GPTQ update. Later columns are updated lazily, ``block_size`` columns at a time.

    Args:
        weight: ``[out_features, in_features]`` float weight, replaced in place by its
            fake-quantized values.
        h_inv: Upper-triangular Cholesky factor of the damped inverse Hessian.
        quantize_group: Fake-quantizes one ``[out_features, group_size]`` column group.
    """
    num_cols = weight.shape[1]
    if block_size % group_size or num_cols % group_size:
        raise ValueError(
            f"GPTQ block_size ({block_size}) and in_features ({num_cols}) must be multiples of "
            f"the quantization group size ({group_size})."
        )
    for block_start in range(0, num_cols, block_size):
        block_end = min(block_start + block_size, num_cols)
        errs = torch.empty_like(weight[:, block_start:block_end])
        for start in range(block_start, block_end, group_size):
            end = start + group_size
            qdq = quantize_group(weight[:, start:end])
            err = torch.linalg.solve_triangular(
                h_inv[start:end, start:end], weight[:, start:end] - qdq, upper=True, left=False
            )
            weight[:, start:end] = qdq
            weight[:, end:block_end].addmm_(err, h_inv[start:end, end:block_end], alpha=-1)
            errs[:, start - block_start : end - block_start] = err
        weight[:, block_end:].addmm_(errs, h_inv[block_start:block_end, block_end:], alpha=-1)


class GGMLGPTQHelper(GPTQHelper):
    """GPTQ for ``ggml``-backend weight quantizers, one GGML block of columns at a time.

    The payload GPTQ chose is pinned to the weight quantizer, so later forwards and export use
    those exact codes rather than encoding the GPTQ'd weight again, which would not return them.
    """

    def update_weights(self, block_size, perc_damp):
        """Run the GPTQ update, then pin the chosen payload to the weight quantizer."""
        super().update_weights(block_size, perc_damp)
        quantizer = self.module.weight_quantizer
        pin_packed_weight(quantizer, quantizer.num_bits, self._packed)
        self._packed = None

    def _blockwise_update(self, block_size):
        quantizer = self.module.weight_quantizer
        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        extra_args = quantizer.backend_extra_args or {}
        block_chunk_size = extra_args.get("block_chunk_size", ggml_format.block_chunk_size)
        decode_chunk_size = extra_args.get("decode_chunk_size", ggml_format.decode_chunk_size)
        payloads = []

        def quantize_group(group):
            packed, shape = ggml_format.quantize(
                group.contiguous(), block_chunk_size=block_chunk_size
            )
            payloads.append(packed)
            return ggml_format.dequantize(
                packed, shape, dtype=group.dtype, block_chunk_size=decode_chunk_size
            )

        gptq_group_update(
            self.weight, self.h_inv, block_size, ggml_format.block_size, quantize_group
        )
        self._packed = torch.cat(payloads, dim=-2)


register_gptq_helper("ggml", GGMLGPTQHelper)
