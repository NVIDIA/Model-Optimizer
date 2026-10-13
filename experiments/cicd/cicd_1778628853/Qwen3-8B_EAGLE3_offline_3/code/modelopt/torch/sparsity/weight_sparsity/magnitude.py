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

"""Magnitude-base sparsity inspired by NVIDIA ASP (Automatic SParsity)."""

import re
import warnings
from itertools import permutations

import torch
import torch.nn as nn

from .module import SparseModule
from .searcher import BaseSparseSearcher


def get_nmprune_info(pattern: str) -> tuple[bool, int, int]:
    """Gets the n:m sparsity pattern information from a given string."""
    nm_prune = re.search(r"(\d+):(\d+) sparsity", pattern)
    if nm_prune is not None:
        n, m = map(int, nm_prune.groups())
        return nm_prune is not None, n, m
    return False, 0, 0


def fill(x):
    """Calculates the ratio of non-zero elements in a tensor."""
    return float(x.nonzero().size(0)) / torch.numel(x)


def reshape_1d(matrix, m):
    """Reshapes a given matrix into m-dimensional vectors: (h,w) -> (hw/m, m)."""
    if matrix.shape[1] % m > 0:
        new_cols = matrix.shape[1] + (m - matrix.shape[1] % m)
        mat = matrix.new_empty(matrix.shape[0], new_cols).fill_(0)
        mat[:, : matrix.shape[1]] = matrix

        return mat.view(-1, m), mat.shape
    else:
        return matrix.view(-1, m), matrix.shape


def compute_valid_1d_patterns(m, n):
    """Computes all possible m:n patterns in a 1D vector.

    The function generates a tensor of size m with n ones and (m-n) zeros.
    It then generates all permutations of this tensor, removes duplicates,
    and returns the unique patterns as a tensor.
    """
    patterns = torch.zeros(m)
    patterns[:n] = 1
    valid_patterns = torch.tensor(list(set(permutations(patterns.tolist()))))
    return valid_patterns


def mn_1d_best(matrix, m, n):
    """Finds the best m:n pattern in a given matrix.

    The function computes all possible m:n patterns and selects the one
    that maximizes the sum of non-masked weights in the matrix. The selected
    pattern is then used to create a mask for the matrix.
    """
    patterns = compute_valid_1d_patterns(m, n).to(matrix.device)

    # Find the best m:n pattern (sum of non-masked weights).
    mask = torch.IntTensor(matrix.shape).fill_(1).view(-1, m)
    mat, _ = reshape_1d(matrix, m)
    pmax = torch.argmax(torch.matmul(mat.abs(), patterns.t()), dim=1)
    mask[:] = patterns[pmax[:]]
    mask = mask.view(matrix.shape)
    return mask


def m4n2_1d(mat):
    """Finds the best 2:4 pattern in a given matrix."""
    return mn_1d_best(mat, 4, 2)


def create_asp_mask(tensor: nn.Parameter, pattern: str) -> torch.BoolTensor:
    """Creates a mask for a given tensor based on a specified sparse pattern.

    The function reshapes the tensor and applies the specified pattern to create a sparse mask.
    The default pattern is m4n2_1d, which finds the best 2:4 sparsity pattern in the tensor.
    """
    pattern_method_lut = {BaseSparseSearcher._pattern_2_4: m4n2_1d}
    if pattern not in pattern_method_lut:
        raise NotImplementedError(f"Unsupported pattern {pattern} for ASP sparsity")
    func = pattern_method_lut[pattern]

    shape = tensor.shape
    tensor.type()
    t = tensor.float().contiguous()

    # 1d-tensor
    if len(shape) == 1:
        t = t.view(1, shape[0])
        mask = func(t)
    # 2d-tensor (K, C)
    elif len(shape) == 2:
        # linear
        t = t.view(shape[0], shape[1])
        mask = func(t)
    # 3d-tensor (K, C, R)
    elif len(shape) == 3:
        # 1d convs
        t = t.permute(0, 2, 1).contiguous().view(shape[0] * shape[2], shape[1])
        mask = func(t)
        mask = mask.view(shape[0], shape[2], shape[1]).permute(0, 2, 1).contiguous()
    # 4d-tensor (K, C, R, S)
    elif len(shape) == 4:
        # 2d convs
        t = t.permute(2, 3, 0, 1).contiguous().view(shape[2] * shape[3] * shape[0], shape[1])
        mask = func(t)
        mask = mask.view(shape[2], shape[3], shape[0], shape[1]).permute(2, 3, 0, 1).contiguous()

    return mask.view(shape).to(dtype=torch.bool)


class MagnitudeSearcher(BaseSparseSearcher):
    """Searcher for magnitude-based sparsity."""

    def _check_weight_size(self, weight: torch.nn.Parameter, mod_name: str) -> bool:
        """Check if the weight size is supported."""
        # rules from ASP
        if weight.size(0) % 8 != 0 or weight.size(1) % 16 != 0:
            warnings.warn(
                f"Skipping sparsifying {mod_name} of size={weight.size()!s} and"
                f" type={weight.dtype!s} for sparsity"
            )
            return False

        return True

    def _compute_mask(self, module: SparseModule) -> torch.BoolTensor:
        """Compute the mask (and weight update) for the given module."""
        return create_asp_mask(module.weight, self.config["pattern"])
