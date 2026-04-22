# Adapted from https://github.com/IST-DASLab/FP-Quant/blob/d2e3092/src/quantization/gptq.py
# with minor modifications to the original forms to accommodate minor architectural differences
# to be reused in the Model-Optimizer pipeline.
# Copyright (c) Andrei Panferov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""GPTQ helper and Hessian utilities for calibration."""

import math

import torch

from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.network import bind_forward_method, unpatch_forward_method
from modelopt.torch.utils.perf import get_used_gpu_mem_fraction


def update_hessian(input, hessian, n_samples):
    """Update hessian matrix with new input samples using incremental formula.

    Args:
        input: Input tensor (batch_size, ..., features)
        hessian: Current Hessian matrix to update in-place
        n_samples: Number of samples already processed
    Returns:
        Tuple of (updated_hessian, new_sample_count)

    Note: input must be non-empty (batch_size > 0); a zero-sized input causes division by zero.
    """
    # Flatten to 2D (total_tokens, features) first, so batch_size counts tokens
    input_flat = input.reshape(-1, input.shape[-1]).t().float()
    batch_size = input_flat.shape[1]

    # Incremental averaging: scale down old hessian
    hessian *= n_samples / (n_samples + batch_size)
    n_samples += batch_size

    # Compute outer product: H += (2/n_samples) * X @ X^T
    scaled_input = math.sqrt(2 / n_samples) * input_flat
    hessian.add_((scaled_input @ scaled_input.t()).to(hessian.device))

    return hessian, n_samples


def compute_hessian_inverse(hessian, weight, perc_damp):
    """Compute damped upper-Cholesky inverse Hessian.

    Dead-neuron columns (all-zero in ``weight``) are zeroed in the
    Hessian before inversion, matching the FP-Quant reference:
    https://github.com/IST-DASLab/FP-Quant/blob/d2e3092f968262c4de5fb050e1aef568a280dadd/src/quantization/gptq.py#L200

    Args:
        hessian: Hessian matrix ``[in_features, in_features]``.
        weight: Weight matrix ``[out_features, in_features]`` for dead-neuron detection.
        perc_damp: Percentage of average Hessian diagonal for damping.

    Returns:
        Upper-triangular Cholesky factor of the damped inverse Hessian
        ``[in_features, in_features]``.  Falls back to the identity matrix
        when the Hessian is not positive definite.
    """
    h = hessian.clone()
    zero_cols = torch.nonzero(weight.eq(0).all(dim=0)).unsqueeze(-1)

    h[zero_cols, :] = 0
    h[:, zero_cols] = 0
    h[zero_cols, zero_cols] = 1

    damp = perc_damp * torch.mean(torch.diag(h))
    diag_indices = torch.arange(h.shape[0], device=h.device)
    h[diag_indices, diag_indices] += damp

    try:
        h = torch.cholesky_inverse(torch.linalg.cholesky(h))
        return torch.linalg.cholesky(h, upper=True)
    except (RuntimeError, torch.linalg.LinAlgError):
        print_rank_0("Warning: Hessian is not positive definite, using identity matrix")
        return torch.eye(h.shape[0], device=h.device, dtype=h.dtype)


class GPTQHelper:
    """Encapsulates per-module GPTQ state and operations.

    Owns the Hessian, patches the forward during collection, and contains
    the blockwise weight-update logic.

    Instance attributes set during ``__init__``:
        module, name, hessian, n_samples

    Instance attributes set during ``update_weights``:
        weight: float working copy of module weights (mutated in-place by update methods)
        h_inv: upper-triangular Cholesky factor of the damped inverse Hessian
    """

    CACHE_NAME = "_forward_no_gptq_hessian"

    def __init__(self, module, name, offload_to_cpu=False, fused=False):
        """Initialize GPTQHelper with module state and Hessian storage."""
        self.module = module
        self.name = name
        self.fused = fused
        in_features = module.weight.shape[-1]
        device = module.weight.device
        if device.type == "meta" or (offload_to_cpu and get_used_gpu_mem_fraction(device) > 0.65):
            device = "cpu"
        self.hessian = torch.zeros(in_features, in_features, dtype=torch.float32, device=device)
        self.n_samples = 0
        # Set by update_weights(); listed here for documentation.
        self.weight: torch.Tensor | None = None
        self.h_inv: torch.Tensor | None = None

    def setup(self):
        """Patch the module's forward to accumulate Hessian during the collection pass."""
        gptq_helper = self

        def hessian_forward(self, input, *args, **kwargs):
            inp = input.to_local() if hasattr(input, "to_local") else input
            if self.input_quantizer is not None and self.input_quantizer.is_enabled:
                hessian_input = self.input_quantizer(inp)
            else:
                hessian_input = inp
            gptq_helper.hessian, gptq_helper.n_samples = update_hessian(
                hessian_input, gptq_helper.hessian, gptq_helper.n_samples
            )

            out = self._forward_no_gptq_hessian(input, *args, **kwargs)

            return out

        bind_forward_method(self.module, hessian_forward, self.CACHE_NAME)

    def cleanup(self):
        """Unpatch the module's forward method."""
        unpatch_forward_method(self.module, self.CACHE_NAME)

    def free(self):
        """Release Hessian and working tensors to reclaim memory."""
        self.hessian = None
        self.weight = None
        self.h_inv = None

    def update_weights(self, block_size, perc_damp):
        """Run GPTQ blockwise weight update on this module.

        Populates ``self.weight`` and ``self.h_inv``, runs the blockwise update,
        logs MSE, and writes the result back to the module.
        """
        hessian = self.hessian.to(self.module.weight.device)
        self.weight = self.module.weight.data.float().clone()
        self._prepare_hessian_inverse(hessian, perc_damp)
        self._blockwise_update(block_size)
        self._print_mse_error(hessian)
        self.module.weight.data = self.weight.reshape(self.module.weight.shape).to(
            self.module.weight.data.dtype
        )

    # ------------------------------------------------------------------
    # Quantize helpers — all read from self.module, self.weight, self.h_inv
    # ------------------------------------------------------------------

    def _prepare_hessian_inverse(self, hessian, perc_damp):
        """Compute damped inverse Hessian and store as ``self.h_inv``."""
        assert self.weight is not None, "_prepare_hessian_inverse called before update_weights()"
        self.h_inv = compute_hessian_inverse(hessian, self.weight, perc_damp)

    def _blockwise_update(self, block_size):
        """Column-wise GPTQ update.

        When ``self.fused`` is True and the weight quantizer is an
        ``NVFP4StaticQuantizer``, uses :func:`gptq_blockwise_update_fused_scalar`
        (a fused Triton kernel).  Otherwise falls back to
        :func:`gptq_blockwise_update` (unfused column-by-column loop).
        """
        assert self.weight is not None and self.h_inv is not None, (
            "_blockwise_update called before _prepare_hessian_inverse()"
        )
        quantizer = self.module.weight_quantizer

        if self.fused and getattr(quantizer, "_is_nvfp4_static_quantizer", False):
            block_sizes = quantizer.block_sizes
            quant_block_size = block_sizes.get(-1) or block_sizes.get(1)
            if quant_block_size is not None and block_size % quant_block_size != 0:
                raise ValueError(
                    f"GPTQ block_size ({block_size}) must be divisible by the quantizer"
                    f" group_size ({quant_block_size})"
                )
            out_features, num_cols = self.weight.shape
            n_blocks = num_cols // quant_block_size
            block_amax = quantizer.amax.reshape(out_features, n_blocks).float()
            global_scale = quantizer.global_amax.float().item() / (6.0 * 448.0)
            gptq_blockwise_update_fused_scalar(
                self.weight, block_amax, global_scale, self.h_inv, block_size, quant_block_size
            )
        else:
            gptq_blockwise_update(self.weight, self.h_inv, block_size, quantizer)

    def _print_mse_error(self, hessian):
        """Log Hessian-weighted relative MSE between ``self.weight`` and original weights."""
        w_orig = self.module.weight.float()
        delta = self.weight - w_orig
        mse = (delta).mm(hessian).mul(delta).mean() / (w_orig.mm(hessian).mul(w_orig).mean() + 1e-6)
        suffix = f", n_hessian_samples: {self.n_samples}" if self.n_samples else ""
        print_rank_0(f"[{self.name}] Relative MSE error: {mse.item():.2e}{suffix}")


def gptq_blockwise_update(weight, h_inv, block_size, quantize_fn):
    """Column-wise GPTQ update using full-matrix fake quantization.

    For each column, quantizes the full weight matrix via ``quantize_fn`` and
    extracts the quantized column.  Error is propagated to remaining columns
    within the block and then to all subsequent columns via the inverse Hessian.

    Args:
        weight: Weight tensor ``[out_features, in_features]``, modified **in-place**
            with fake-quantized values.
        h_inv: Upper-triangular Cholesky factor of the damped inverse Hessian
            ``[in_features, in_features]``.
        block_size: Number of columns to process per GPTQ block.
        quantize_fn: Callable ``(weight) -> qdq_weight`` that fake-quantizes
            the full weight matrix.
    """
    num_cols = weight.shape[1]

    for block_start in range(0, num_cols, block_size):
        block_end = min(block_start + block_size, num_cols)
        n_cols_blk = block_end - block_start
        h_inv_cho_blk = h_inv[block_start:block_end, block_start:block_end]

        wblk = weight.clone()
        errs = torch.zeros_like(weight[:, block_start:block_end])

        for i in range(n_cols_blk):
            w_ci = wblk[:, block_start + i]
            d = h_inv_cho_blk[i, i]
            qdq = quantize_fn(wblk)
            weight[:, block_start + i] = qdq[:, block_start + i]
            err = (w_ci - qdq[:, block_start + i]) / d
            wblk[:, block_start + i : block_end].addr_(err, h_inv_cho_blk[i, i:], alpha=-1)
            errs[:, i] = err

        weight[:, block_end:].addmm_(errs, h_inv[block_start:block_end, block_end:], alpha=-1)


def gptq_blockwise_update_fused_scalar(
    weight, block_amax, global_scale, h_inv, block_size, quant_block_size
):
    """Fused GPTQ blockwise update for NVFP4 scalar quantization.

    Uses a fused Triton kernel that combines scale computation, quantization,
    and per-column error propagation into one launch per GPTQ block, avoiding
    the Python-level per-column loop in :func:`gptq_blockwise_update`.

    Args:
        weight: Weight tensor ``[out_features, in_features]``, modified **in-place**
            with fake-quantized values.
        block_amax: Per-block amax values ``[out_features, n_amax_blocks]``.
        global_scale: Pre-computed ``global_amax / (6.0 * 448.0)`` (scalar).
        h_inv: Upper-triangular Cholesky factor of the damped inverse Hessian
            ``[in_features, in_features]``.
        block_size: Number of columns to process per GPTQ block.
        quant_block_size: Number of elements sharing one quantization scale factor.
    """
    from modelopt.torch.quantization.triton.gptq_fused_kernel import gptq_fused_block_scalar

    num_cols = weight.shape[1]
    for bs in range(0, num_cols, block_size):
        be = min(bs + block_size, num_cols)
        qw, err = gptq_fused_block_scalar(
            weight[:, bs:be].clone().contiguous(),
            block_amax,
            global_scale,
            h_inv[bs:be, bs:be].contiguous(),
            quant_block_size,
            bs,
        )
        weight[:, bs:be] = qw
        if be < num_cols:
            weight[:, be:].addmm_(err, h_inv[bs:be, be:], alpha=-1)


_GPTQ_HELPER_REGISTRY: dict[str, type[GPTQHelper]] = {}


def register_gptq_helper(backend: str, factory: type[GPTQHelper]) -> None:
    """Register a :class:`GPTQHelper` subclass for a quantizer backend.

    When :func:`modelopt.torch.quantization.model_calib.gptq` encounters a
    module whose ``weight_quantizer.backend`` matches ``backend``, it will
    construct ``factory`` instead of the default ``GPTQHelper``.
    """
    _GPTQ_HELPER_REGISTRY[backend] = factory
