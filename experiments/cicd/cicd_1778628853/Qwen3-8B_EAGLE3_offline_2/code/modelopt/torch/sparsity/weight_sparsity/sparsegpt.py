# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Utility functions of SparseGPT."""

import math
import warnings

import torch
import torch.nn as nn

from modelopt.torch.opt.searcher import SearchConfig
from modelopt.torch.utils import print_rank_0

from .magnitude import get_nmprune_info
from .module import SparseModule
from .searcher import BaseSparseSearcher


def invert(hessian: torch.Tensor) -> torch.Tensor:
    """Invert a Hessian matrix."""
    try:
        hessian_inv = torch.linalg.cholesky(hessian)
        hessian_inv = torch.cholesky_inverse(hessian_inv)
        hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
    except RuntimeError:
        cols = hessian.size(0)
        eps = 1e-6 * torch.eye(cols).to(hessian.device)
        hessian_inv = torch.cholesky_inverse(torch.linalg.cholesky(hessian + eps))

    return hessian_inv


def prepare(
    tensor: torch.Tensor, hessian: torch.Tensor, hessian_damp: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare the inverse Hessian matrix."""
    weight = tensor.detach().clone()
    # move the hessian matrix from CPU to GPU for acceleration
    hessian = hessian.to(weight.device)
    if len(weight.size()) == 4:
        weight = weight.flatten(1)

    zero = torch.diag(hessian) == 0
    hessian[zero, zero] = 1
    weight[:, zero] = 0

    damp = hessian_damp * torch.mean(torch.diag(hessian))
    cols = weight.size(1)
    diag = torch.arange(cols)
    hessian[diag, diag] += damp

    hessian_inv = invert(hessian)

    # remove the Hessian matrix to save GPU memory
    del hessian
    torch.cuda.empty_cache()

    return weight, hessian_inv


@torch.no_grad()
def create_sgpt_mask(
    tensor: torch.Tensor, hessian: torch.Tensor, config: SearchConfig
) -> torch.Tensor:
    """Create a sparse mask for the given tensor."""
    shape = tensor.size()
    weight, hessian_inv = prepare(tensor, hessian, config["hessian_damp"])
    rows, cols = weight.size()
    hessian_inv_diag = torch.diagonal(hessian_inv, dim1=0, dim2=1)

    is_nm_prune, n, m = get_nmprune_info(config["pattern"])
    col_bs = config["col_block_size"]
    row_bs = config["row_block_size"]
    # if row_bs is not specified, prune the whole weight block
    if row_bs == -1:
        row_bs = rows

    for r1 in range(0, rows, row_bs):
        r2 = min(r1 + row_bs, rows)
        # the mask of the weights not to be pruned
        w_rows = weight[r1:r2].float()

        # pruning the weight block W[row:row+row_bs, i1:i1+col_bs]
        for i1 in range(0, cols, col_bs):
            i2 = min(i1 + col_bs, cols)
            w_blk = w_rows[:, i1:i2].clone()
            q_blk = torch.zeros_like(w_blk)
            # the error of the weights to be pruned
            delta_blk = torch.zeros_like(w_blk)
            hinv_blk = hessian_inv[i1:i2, i1:i2]
            hinv_diag_blk = hessian_inv_diag[i1:i2]

            errors_blk = (w_blk**2) / (hinv_diag_blk**2 + 1e-9)
            if torch.isnan(errors_blk).any():
                print("nan in errors_blk.")

            mask_blk = torch.zeros_like(w_blk, dtype=torch.bool)

            for j in range(i2 - i1):
                # compute the error of the weights to be pruned
                w = w_blk[:, j]
                d = hinv_diag_blk[j]
                if is_nm_prune and j % m == 0:
                    errors_blk = (w_blk[:, j : j + m] ** 2) / (hinv_diag_blk[j : j + m] ** 2 + 1e-9)
                    mask_blk.scatter_(
                        1, j + torch.topk(errors_blk, n, dim=1, largest=False)[1], True
                    )

                q = w.clone()
                q[mask_blk[:, j]] = 0
                q_blk[:, j] = q

                # update the remaining weights in the col_bs block to compensate the error caused by pruning W[:, j]
                err = (w - q) / d
                w_blk[:, j:] -= err.unsqueeze(1).matmul(hinv_blk[j, j:].unsqueeze(0))
                delta_blk[:, j] = err

            # compensate the error caused by pruning W[:, i: i + col_bs] with the weights update in W[:, i + col_bs:]
            w_rows[:, i1:i2] = q_blk
            w_rows[:, i2:] -= delta_blk.matmul(hessian_inv[i1:i2, i2:])
            if torch.isnan(w_rows[:, i2:]).any():
                print("nan")

        weight[r1:r2] = w_rows

    mask = weight != 0

    return mask.view(shape)


class SparseGPTSearcher(BaseSparseSearcher):
    """SparseGPT-based sparse mask searching algorithm."""

    @property
    def default_search_config(self) -> SearchConfig:
        """Get the default config for the searcher."""
        return {
            **super().default_search_config,
            "col_block_size": 128,  # column block size in sparsegpt
            "row_block_size": -1,  # row block size in sparsegpt
            "hessian_damp": 0.1,  # hessian damp in sparsegpt
            "calib_size": 256,  # calibration size for hessian matrix calculation
            "device": "cuda",  # device of hessian matrix
        }

    def _check_weight_size(self, weight, mod_name) -> bool:
        """Check if the weight size is supported by SparseGPT."""
        _, _, m = get_nmprune_info(self.config["pattern"])

        # the column size must be divisible by m
        if weight.size(0) % m != 0 or weight.size(1) % m != 0:
            warnings.warn(
                f"Skipping pruning {mod_name} of size={weight.size()!s} and"
                f" type={weight.dtype!s} for SparseGPT"
            )
            return False

        return True

    def _compute_mask(self, module: SparseModule) -> torch.BoolTensor:
        """Compute the mask (and weight update) for the given module."""
        return create_sgpt_mask(module.weight, module.hessian, self.config)

    @torch.no_grad()
    def before_search(self):
        """Register the forward hook to collect the hessian matrix."""
        super().before_search()

        handles = []
        for _, module in self._named_sparsifiable_modules():
            # setup and register the forward hook
            self._setup_forward_hook(module)
            handles.append(module.register_forward_hook(self._hook_compute_hessian))

        print_rank_0(f"Collecting Hessian statistics for {len(handles)} modules.")

        # run a forward loop to collect the hessian matrix
        assert self.forward_loop is not None, "Please provide `data_loader` or `forward_loop`!"
        self.forward_loop(self.model)

        # remove the forward hooks
        for handle in handles:
            handle.remove()

    def after_search(self):
        """Remove Hessian artifacts from network."""
        super().after_search()
        for _, module in self._named_sparsifiable_modules():
            del module.hessian
            del module.samples

    @staticmethod
    def _is_memory_sufficient(device_id, threshold):
        """Check if the memory usage on the CUDA device is below the threshold."""
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        free_memory = total_memory - allocated_memory
        return free_memory / total_memory > (1 - threshold)

    @classmethod
    def _setup_forward_hook(cls, mod: SparseModule) -> None:
        """Setup the attributes we need for our forward hook during the SparseGPT search."""
        # initialize the hessian matrix
        if isinstance(mod, nn.Conv2d):
            # For conv2d layers, the hessian matrix is calculated as X * X^T, where X is the
            # flattened weight matrix.
            cols = mod.weight.size(1) * mod.weight.size(2) * mod.weight.size(3)
        else:
            # For linear layers, the hessian matrix is calculated as X * X^T, where X is the
            # weight matrix.
            cols = mod.weight.size(1)

        target_device = mod.weight.device
        # Hessian matrix is stored in the GPU memory by default
        if target_device.type == "cuda" and cls._is_memory_sufficient(target_device.index, 0.8):
            hessian = torch.zeros((cols, cols), dtype=torch.float32).to(target_device)
        else:
            hessian = torch.zeros((cols, cols), dtype=torch.float32).to("cpu")

        # store the hessian matrix and the number of samples
        # TODO: this should probably be improved eventually!!
        mod.hessian = hessian
        mod.samples = 0

    @classmethod
    def _hook_compute_hessian(cls, mod: nn.Module, inp: torch.Tensor, out: torch.Tensor):
        with torch.inference_mode():
            # TODO: move the hessian matrix to GPU memory if possible
            if isinstance(inp, tuple):
                inp = inp[0]
            # use torch.float32 to avoid overflow in Hessian
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            # nn.Linear and *ParallelLinear in mcore
            if "Linear" in type(mod).__name__:
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp.t_()
            if isinstance(mod, nn.Conv2d):
                unfold = nn.Unfold(
                    mod.kernel_size,
                    dilation=mod.dilation,
                    stride=mod.stride,
                )
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
            mod.hessian *= mod.samples / (mod.samples + tmp)
            mod.samples += tmp
            inp = math.sqrt(2 / mod.samples) * inp.float()

            # the hessian matrix is calculated as X * X^T
            target_device = mod.hessian.device
            if mod.hessian.device.type == "cuda":
                if cls._is_memory_sufficient(mod.hessian.device.index, 0.8):
                    mod.hessian += inp.matmul(inp.t()).to(mod.hessian.device)
                else:
                    target_device = "cpu"
                    mod.hessian = mod.hessian.to("cpu")
            mod.hessian += inp.matmul(inp.t()).to(target_device)

            assert not torch.isinf(mod.hessian).any(), "Hessian contains inf"
