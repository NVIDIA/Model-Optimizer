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

    def __init__(self, module, name, offload_to_cpu=False):
        """Initialize GPTQHelper with module state and Hessian storage."""
        self.module = module
        self.name = name
        in_features = module.weight.shape[-1]
        device = module.weight.device
        if offload_to_cpu and get_used_gpu_mem_fraction(device) > 0.65:
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
        """Compute damped inverse Hessian and store as ``self.h_inv``.

        Dead-neuron columns (all-zero in ``self.weight``) are zeroed in the
        Hessian before inversion, matching the FP-Quant reference:
        https://github.com/IST-DASLab/FP-Quant/blob/d2e3092f968262c4de5fb050e1aef568a280dadd/src/quantization/gptq.py#L200
        """
        assert self.weight is not None, "_prepare_hessian_inverse called before update_weights()"
        h = hessian.clone()
        zero_cols = torch.nonzero(self.weight.eq(0).all(dim=0)).unsqueeze(-1)

        h[zero_cols, :] = 0
        h[:, zero_cols] = 0
        h[zero_cols, zero_cols] = 1

        damp = perc_damp * torch.mean(torch.diag(h))
        diag_indices = torch.arange(h.shape[0], device=h.device)
        h[diag_indices, diag_indices] += damp

        try:
            h = torch.cholesky_inverse(torch.linalg.cholesky(h))
            self.h_inv = torch.linalg.cholesky(h, upper=True)
        except (RuntimeError, torch.linalg.LinAlgError):
            print_rank_0("Warning: Hessian is not positive definite, using identity matrix")
            self.h_inv = torch.eye(h.shape[0], device=h.device, dtype=h.dtype)

    def _blockwise_update(self, block_size):
        """Column-wise GPTQ update using full-matrix QDQ.

        For each column, quantizes the full weight matrix via the quantizer and
        extracts the quantized column. This is the standard GPTQ approach.

        Reads/writes ``self.weight`` and ``self.h_inv`` in-place.
        """
        assert self.weight is not None and self.h_inv is not None, (
            "_blockwise_update called before _prepare_hessian_inverse()"
        )
        quantizer = self.module.weight_quantizer
        block_sizes = getattr(quantizer, "block_sizes", None)
        if block_sizes is not None:
            group_size = block_sizes.get(-1)
            if group_size is not None and block_size % group_size != 0:
                raise ValueError(
                    f"GPTQ block_size ({block_size}) must be divisible by the quantizer"
                    f" group_size ({group_size})"
                )
        num_cols = self.weight.shape[1]

        for block_start in range(0, num_cols, block_size):
            block_end = min(block_start + block_size, num_cols)
            n_cols_blk = block_end - block_start
            h_inv_cho_blk = self.h_inv[block_start:block_end, block_start:block_end]

            wblk = self.weight.clone()
            errs = torch.zeros_like(wblk[:, block_start:block_end])

            for i in range(n_cols_blk):
                w_ci = wblk[:, block_start + i]
                d = h_inv_cho_blk[i, i]
                qdq = quantizer(wblk)
                self.weight[:, block_start + i] = qdq[:, block_start + i]
                err = (w_ci - qdq[:, block_start + i]) / d
                wblk[:, block_start + i : block_end].addr_(err, h_inv_cho_blk[i, i:], alpha=-1)
                errs[:, i] = err

            self.weight[:, block_end:].addmm_(
                errs, self.h_inv[block_start:block_end, block_end:], alpha=-1
            )

    def _print_mse_error(self, hessian):
        """Log Hessian-weighted relative MSE between ``self.weight`` and original weights."""
        w_orig = self.module.weight.float()
        delta = self.weight - w_orig
        mse = (delta).mm(hessian).mul(delta).mean() / (w_orig.mm(hessian).mul(w_orig).mean() + 1e-6)
        suffix = f", n_hessian_samples: {self.n_samples}" if self.n_samples else ""
        print_rank_0(f"[{self.name}] Relative MSE error: {mse.item():.2e}{suffix}")
