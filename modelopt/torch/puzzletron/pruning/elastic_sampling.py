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

"""Prefix-size sampling for nested (Matryoshka) elastic bypass distillation.

Each prunable subblock samples one of its allowed sizes per minibatch with probability
``p ~ 1 / num_params(size)`` (normalized). Larger variants (more params, closer to the teacher) are
sampled less, so they get fewer recovery iterations; small variants get more. The full (teacher)
size is included.

The parameter counts MUST come from the canonical
``subblock_stats.calc_subblock_params_and_memory.calculate_subblock_params`` (which builds the meta
decoder layer and counts ``sum(p.numel())``, so GQA, gated-vs-not FFN, biases, and norms are all
exact) — not a hand-written formula. This module only consumes the counts (so it stays pure /
unit-testable); ``elastic_supernet.build_subblock_elastics`` supplies them via that function.
"""

from __future__ import annotations

import torch

__all__ = [
    "inverse_param_probs",
    "inverse_width_probs",
    "ElasticSizeSampler",
]


def inverse_param_probs(param_counts) -> torch.Tensor:
    """Normalized probabilities proportional to ``1 / param_count`` (float64).

    No-op elastic endpoints have zero parameters. Treat them as having the same
    effective size as the smallest non-zero candidate so they are sampled as a
    normal endpoint instead of dominating the distribution or producing ``inf``.
    """
    counts = torch.tensor([float(p) for p in param_counts], dtype=torch.float64)
    positives = counts[counts > 0]
    if positives.numel() == 0:
        return torch.full_like(counts, 1.0 / counts.numel())
    counts = torch.where(counts > 0, counts, positives.min())
    inv = 1.0 / counts
    return inv / inv.sum()


def inverse_width_probs(widths) -> torch.Tensor:
    """Normalized hidden-width probabilities proportional to ``1 / width``."""

    values = [int(width) for width in widths]
    if not values or any(width <= 0 for width in values):
        raise ValueError("hidden widths must be a non-empty sequence of positive integers")
    return inverse_param_probs(values)


class ElasticSizeSampler:
    """Samples a size for one subblock with ``p ~ 1/num_params`` (full/teacher size included).

    ``sizes`` is the allowed list (FFN intermediate ints, or ``(q, kv)`` tuples for attention);
    ``param_counts`` are the corresponding parameter counts. Use one sampler per prunable subblock.
    A shared ``torch.Generator`` makes a run reproducible.
    """

    def __init__(self, sizes, param_counts):
        if len(sizes) != len(param_counts):
            raise ValueError(f"sizes ({len(sizes)}) and param_counts ({len(param_counts)}) differ")
        if len(sizes) == 0:
            raise ValueError("ElasticSizeSampler needs at least one size")
        self.sizes = list(sizes)
        self.param_counts = [int(count) for count in param_counts]
        self.probs = inverse_param_probs(param_counts)

    def sample(self, generator: torch.Generator | None = None):
        idx = int(torch.multinomial(self.probs, 1, generator=generator).item())
        return self.sizes[idx]
