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

"""Guarded inverse policies for experimental GDN prefill studies."""

import torch

from .config import LinearAttentionSolveConfig

__all__ = []


def _check_lower(lower):
    if not lower.is_floating_point() or lower.ndim < 2 or lower.shape[-1] != lower.shape[-2]:
        raise ValueError("lower must contain floating square matrices")
    return lower


def _neumann_inverse_strict(lower, degree):
    identity = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype).expand_as(lower)
    series = identity + lower * 0
    power = -lower
    for step in range((degree + 1).bit_length() - 1):
        series = series + power @ series
        if step + 1 < (degree + 1).bit_length() - 1:
            power = power @ power
    return series


def neumann_inverse(lower, degree):
    """Evaluate degrees 3, 7, 15, or 31 in a fixed binary association order."""
    if degree not in (3, 7, 15, 31):
        raise ValueError("Neumann degree must be one of 3, 7, 15, or 31")
    return _neumann_inverse_strict(_check_lower(lower).tril(-1), degree)


def neumann_inverse_reference(lower, degree):
    """Sum powers sequentially for diagnostics, including deliberately unsafe degree 63."""
    if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= 63:
        raise ValueError("Reference Neumann degree must be an integer in [0, 63]")
    lower = _check_lower(lower).tril(-1)
    identity = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype).expand_as(lower)
    term = identity + lower * 0
    result = term
    for _ in range(degree):
        term = (-lower) @ term
        result = result + term
    return result


def _neumann_tail_bound_strict(lower, degree):
    norm = lower.detach().to(torch.float64).abs().sum(-1).amax(-1)
    return torch.where(
        norm < 1,
        norm.pow(degree + 1) / (1 - norm),
        torch.full_like(norm, torch.inf),
    )


def neumann_tail_bound(lower, degree):
    """Return the sufficient exact-arithmetic infinity-norm truncation bound.

    For ``r = ||L||_inf < 1``, the omitted tail after degree ``d`` is at most
    ``r**(d+1) / (1-r)``. Other matrices return infinity instead of relying on
    triangular nilpotence, which is numerically unsafe under cancellation.
    """
    if degree not in (3, 7, 15, 31):
        raise ValueError("Neumann degree must be one of 3, 7, 15, or 31")
    return _neumann_tail_bound_strict(_check_lower(lower).tril(-1), degree)


def _exact_inverse(lower):
    identity = torch.eye(lower.shape[-1], device=lower.device, dtype=lower.dtype).expand_as(lower)
    return torch.linalg.solve_triangular(
        identity + lower, identity, upper=False, unitriangular=True
    )


def triangular_inverse(lower, policy: LinearAttentionSolveConfig):
    """Apply an exact solve or guarded polynomial with exact per-matrix fallback."""
    lower = _check_lower(lower)
    with torch.autocast(device_type=lower.device.type, enabled=False):
        if policy.method == "exact":
            return _exact_inverse(lower)
        degree = policy.degree
        if degree is None:
            raise ValueError("Guarded Neumann solve requires an explicit degree")
        lower = lower.tril(-1)
        safe = _neumann_tail_bound_strict(lower, degree) <= policy.max_tail_bound
        if policy.unsafe_action == "error" and not bool(safe.all()):
            count = int((~safe).sum())
            raise ValueError(
                f"Neumann truncation bound rejected {count} matrix/matrices; "
                "use the exact solve or unsafe_action='exact'"
            )
        if bool(safe.all()):
            return _neumann_inverse_strict(lower, degree)
        flat_lower = lower.reshape(-1, lower.shape[-2], lower.shape[-1])
        flat_result = torch.empty_like(flat_lower)
        safe_indices = safe.reshape(-1).nonzero().flatten()
        unsafe = (~safe).reshape(-1).nonzero().flatten()
        if safe_indices.numel():
            approximation = _neumann_inverse_strict(
                flat_lower.index_select(0, safe_indices), degree
            )
            flat_result = flat_result.index_copy(0, safe_indices, approximation)
        exact = _exact_inverse(flat_lower.index_select(0, unsafe))
        return flat_result.index_copy(0, unsafe, exact).reshape_as(lower)
