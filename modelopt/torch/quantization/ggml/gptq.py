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

from ..extensions import get_cuda_ext_ggml
from ..utils.calib_utils import GPTQHelper, register_gptq_helper
from .choice_search import search_choices, search_scratch
from .common import (
    checked_encode_inputs,
    derive_scales,
    floored_second_moments,
    pin_packed_weight,
    quantize_energy_matched,
)
from .gsq import _GSQ_CAPTURE
from .registry import GGML_FORMAT_REGISTRY

__all__ = ["GGMLGPTQHelper", "compensated_block_hessian", "gptq_group_update", "row_gains"]

# Rows the row gain solves at a time.
_GAIN_ROWS = 4096


def gptq_group_update(
    weight: torch.Tensor,
    h_inv: torch.Tensor,
    block_size: int,
    group_size: int,
    quantize_group: Callable[[torch.Tensor, slice], torch.Tensor],
) -> None:
    """GPTQ update for formats that quantize ``group_size`` consecutive columns together.

    Each group is fake-quantized as a whole and its error ``E`` folded into the later columns as
    ``E @ inv(U_gg) @ U_g,rest``, where ``U`` is ``h_inv``. With ``group_size=1`` this is the
    column-wise GPTQ update. Later columns are updated lazily, ``block_size`` columns at a time.

    Args:
        weight: ``[out_features, in_features]`` float weight, replaced in place by its
            fake-quantized values.
        h_inv: Upper-triangular Cholesky factor of the damped inverse Hessian.
        quantize_group: Fake-quantizes one ``[out_features, group_size]`` column group, given the
            group and the slice of columns it holds.
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
            qdq = quantize_group(weight[:, start:end], slice(start, end))
            err = torch.linalg.solve_triangular(
                h_inv[start:end, start:end], weight[:, start:end] - qdq, upper=True, left=False
            )
            weight[:, start:end] = qdq
            weight[:, end:block_end].addmm_(err, h_inv[start:end, end:block_end], alpha=-1)
            errs[:, start - block_start : end - block_start] = err
        weight[:, block_end:].addmm_(errs, h_inv[block_start:block_end, block_end:], alpha=-1)


def compensated_block_hessian(h_inv: torch.Tensor, columns: slice) -> torch.Tensor:
    """The Hessian GPTQ's error on one block of columns is measured by, ``[n, n]``.

    Once the later columns compensate, fixing the block to ``q`` costs ``||(w - q) inv(U_gg)||^2``
    (the error :func:`gptq_group_update` folds forward), that is ``inv(U_gg^T U_gg)``: the inverse of
    the block of ``inv(H)`` restricted to the not-yet-quantized columns, where ``U`` is ``h_inv``.
    """
    u = h_inv[columns, columns]
    u_inv = torch.linalg.solve_triangular(
        u, torch.eye(u.shape[0], device=u.device, dtype=u.dtype), upper=True
    )
    return u_inv @ u_inv.T


def row_gains(h_inv: torch.Tensor, original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Per row, the factor that gives ``quantized`` a gain of one on ``original`` under GPTQ's Hessian.

    GPTQ measures a row's error ``w - q`` by ``||(w - q) inv(U)||^2``, ``U`` being ``h_inv``: under
    the damped Hessian ``inv(U^T U)``, or the identity where its Cholesky failed. As a least-squares
    fit, ``q`` comes out shrunk along ``w``; ``<w, w> / <q, w>`` in that metric scales it back so its
    projection on ``w`` is ``w`` itself. Rows ``q`` has no component of ``w`` along keep a factor of 1.

    A gain of one rather than :func:`quantize_energy_matched`'s energy match: under a Hessian it
    measured better on dense MLPs and at IQ1, the energy match only on MoE experts at IQ2.
    """
    factors = []
    for start in range(0, len(original), _GAIN_ROWS):
        rows = slice(start, start + _GAIN_ROWS)
        a = torch.linalg.solve_triangular(h_inv, original[rows].float(), upper=True, left=False)
        b = torch.linalg.solve_triangular(h_inv, quantized[rows], upper=True, left=False)
        num, den = torch.einsum("rc,rc->r", a, a), torch.einsum("rc,rc->r", b, a)
        del a, b  # free before the next chunk's solves
        ratio = num / den
        factors.append(torch.where((den > 0) & ratio.isfinite(), ratio, 1.0))
    return torch.cat(factors)


class GGMLGPTQHelper(GPTQHelper):
    """GPTQ for ``ggml``-backend weight quantizers, one GGML block of columns at a time.

    With ``importance_weighted``, formats whose search takes an importance get the square root of
    the Hessian diagonal ``E[x_j^2]`` (llama.cpp's imatrix statistic), so each block's codes favour
    the columns its outputs depend on most. The raw diagonal is heavy-tailed, so a few outlier columns would
    dominate the search; its root keeps the column ranking at half the log-range.
    The payload GPTQ chose is pinned to the weight quantizer, so later forwards and export use
    those exact codes. For a format that keeps its pinned scales, the weight is left at what GPTQ
    encoded each block from -- the error-compensated weight before rounding -- rather than at its
    decoding: under the pinned scales and importance it encodes to the same codes, and QAD trains
    it from there like an unquantized weight instead of from the grid points.
    With ``local_hessian_scales``, formats with per-group choices pick each block's by
    :func:`~.choice_search.search_choices` under :func:`compensated_block_hessian`, and the weight
    left keeps those codes until QAD changes it.

    A format with a :attr:`~.common.GGMLFormat.scale_fit` searches each group at super-block scales
    derived from the group as GPTQ has compensated it. GPTQ's fit still shrinks each row's output,
    so once every group is encoded, each row's stored scales are raised by one factor that gives
    the row a gain of one under GPTQ's Hessian (:func:`row_gains`); a scale per block would undo the
    compensation the later groups made for it. The pin keeps the derived scales as the ones its
    codes were searched at. A weight no calibration input reached gets the plain encode its
    forwards would give.
    """

    def updates_on_any_device(self):
        """Whether the CUDA encoders are available.

        The update reads only the weight, its Hessian and the format's codec, but the PyTorch
        fallback's scratch grows with ``block_chunk_size``, past the room the update is checked for.
        """
        return get_cuda_ext_ggml() is not None

    def update_scratch(self, weight):
        """The scratch of the row gain, and of the local-Hessian search when it picks the local scales."""
        ggml_format = GGML_FORMAT_REGISTRY[self.weight_quantizer.num_bits]
        rows, columns = weight.numel() // weight.shape[-1], weight.shape[-1]
        scratch = 0
        if ggml_format.scale_fit is not None:
            # The weight before GPTQ in its own dtype, and two float32 solves of a chunk of its rows.
            scratch += (
                weight.element_size() * rows * columns + 2 * 4 * min(rows, _GAIN_ROWS) * columns
            )
        if self._searches(ggml_format):
            scratch += search_scratch(ggml_format, rows)
        return scratch

    def _searches(self, ggml_format):
        """Whether ``local_hessian_scales`` picks this weight's local scales.

        A Hessian no calibration input reached leaves the identity inverse: the plain search.
        """
        return self.local_hessian_scales and ggml_format.num_choices > 0 and self.n_samples > 0

    def update_weights(self, block_size, perc_damp, weight=None, device=None):
        """Run the GPTQ update, then pin the chosen payload to the weight quantizer.

        On a ``device`` other than the weight's, it returns once its results are on the weight's GPU.
        """
        weight = self.module.weight if weight is None else weight
        self._weight_dtype = weight.dtype
        capture = _GSQ_CAPTURE.get()
        teacher = weight.detach().clone() if capture is not None else None
        covariance = self.hessian.detach().clone() if capture is not None else None
        super().update_weights(block_size, perc_damp, weight, device)
        if self._unrounded is not None:
            weight.data.copy_(self._unrounded.reshape(weight.shape))
        quantizer = self.weight_quantizer
        importance = None if self._importance is None else self._importance.to(weight.device)
        # Re-encoding would re-pick searched local scales, so the pin keeps them for this weight.
        search_scales = (
            None if self._search_scales is None else self._search_scales.to(weight.device)
        )
        pin_packed_weight(
            quantizer,
            quantizer.num_bits,
            self._packed.to(weight.device),
            importance,
            source=weight if self._searched else None,
            search_scales=search_scales,
        )
        if capture is not None:
            capture(self, teacher, covariance, self._packed)
        self._packed = self._unrounded = self._importance = self._searched = None
        self._search_scales = None
        target = weight.device if device is None else torch.device(device)
        if target.type == "cuda" and target != weight.device:
            # The weight's GPU's allocator does not see the copies into it running on this
            # device's stream, so emptying its cache could release their targets mid-copy.
            torch.cuda.current_stream(target).synchronize()

    def _blockwise_update(self, block_size):
        quantizer = self.weight_quantizer
        ggml_format = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        extra_args = quantizer.backend_extra_args or {}
        block_chunk_size = extra_args.get("block_chunk_size", ggml_format.block_chunk_size)
        decode_chunk_size = extra_args.get("decode_chunk_size", ggml_format.decode_chunk_size)
        payloads = []
        assert self.weight is not None, "_blockwise_update called before update_weights()"
        diagonal = importance = None
        if self.importance_weighted and ggml_format.weighted:
            diagonal = floored_second_moments(self.hessian.diagonal().to(self.weight.device))
            importance = None if diagonal is None else diagonal.sqrt()
        self._importance = importance
        # A format that keeps its scales re-encodes the weight it is left with, so encode each
        # group as that weight will hold it, rounded to its dtype.
        self._unrounded = (
            torch.empty_like(self.weight, dtype=self._weight_dtype)
            if ggml_format.payload_scales is not None
            else None
        )
        self._search_scales = None
        derive = ggml_format.scale_fit is not None
        if derive and self.n_samples == 0:
            self._encode_plain(ggml_format, self.weight, block_chunk_size, decode_chunk_size)
            return

        search = self._searched = self._searches(ggml_format)
        # The float32 working copy holds the weight's own dtype exactly, so this copy is lossless.
        original = self.weight.to(self._weight_dtype, copy=True) if derive else None

        def quantize_group(group, columns):
            kwargs = {} if importance is None else {"importance": importance[columns]}
            dtype = group.dtype
            if self._unrounded is not None:
                self._unrounded[:, columns] = group
                group = self._unrounded[:, columns]
            scales = None
            if derive:
                moments = None if diagonal is None else diagonal[columns]
                scales = kwargs["scales"] = derive_scales(ggml_format, group, moments)
            if search:
                kwargs["choices"] = search_choices(
                    ggml_format,
                    group,
                    compensated_block_hessian(self.h_inv, columns)[None],
                    kwargs.get("importance"),
                    block_chunk_size,
                    decode_chunk_size,
                    scales,
                )
            packed, shape = ggml_format.quantize(
                group.contiguous(), block_chunk_size=block_chunk_size, **kwargs
            )
            payloads.append(packed)
            return ggml_format.dequantize(
                packed, shape, dtype=dtype, block_chunk_size=decode_chunk_size
            )

        # Each group's derived scale and encode read slices of one importance vector (the root of
        # the diagonal they read), checked here once.
        with checked_encode_inputs(importance):
            gptq_group_update(
                self.weight, self.h_inv, block_size, ggml_format.block_size, quantize_group
            )
        self._packed = torch.cat(payloads, dim=-2)
        if derive:
            gains = row_gains(self.h_inv, original, self.weight)
            del original
            rows = self._packed.reshape(-1, ggml_format.block_bytes)
            payload_scales = ggml_format.payload_scales
            replace_payload_scales = ggml_format.replace_payload_scales
            # Every format with a scale fit keeps its scales in the payload, so has both.
            assert payload_scales is not None and replace_payload_scales is not None
            searched = payload_scales(rows)
            stored = searched.float().reshape(len(gains), -1) * gains[:, None]
            stored = stored.clamp(max=65504.0).to(torch.float16).reshape(-1)
            self._packed = replace_payload_scales(rows, stored).reshape(self._packed.shape)
            self._search_scales = searched
            # The decoding at the raised scales, up to rounding them to FP16; only logged.
            self.weight.mul_(gains[:, None])

    def _encode_plain(self, ggml_format, weight, block_chunk_size, decode_chunk_size):
        """Encode a weight no calibration input reached as its forwards would, with no GPTQ update."""
        self._searched = False
        if self._unrounded is not None:
            self._unrounded.copy_(weight)
        packed, shape, self._search_scales = quantize_energy_matched(
            ggml_format, weight, block_chunk_size=block_chunk_size
        )
        self._packed = packed
        weight.copy_(
            ggml_format.dequantize(
                packed, shape, dtype=torch.float32, block_chunk_size=decode_chunk_size
            )
        )


register_gptq_helper("ggml", GGMLGPTQHelper)
