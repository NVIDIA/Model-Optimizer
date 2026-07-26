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

"""CPU tests for the 1/num_params elastic prefix-size sampler."""

import torch

from modelopt.torch.puzzletron.pruning.elastic_sampling import (
    ElasticSizeSampler,
    inverse_param_probs,
    inverse_width_probs,
)


def test_inverse_param_probs_favor_small():
    # Smaller params -> larger probability; normalized.
    probs = inverse_param_probs([100, 200, 400])
    assert torch.isclose(probs.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert probs[0] > probs[1] > probs[2]
    # p ~ 1/params: ratio of probs == inverse ratio of params.
    assert torch.isclose(probs[0] / probs[2], torch.tensor(4.0, dtype=torch.float64))


def test_sampler_distribution_matches_inverse_params():
    # Param counts proportional to size -> p ~ 1/size; biggest size sampled least.
    sizes = [2, 4, 8]
    params = [s * 48 for s in sizes]  # counts from the canonical fn in practice; proportional here
    sampler = ElasticSizeSampler(sizes, params)
    gen = torch.Generator().manual_seed(0)
    counts = dict.fromkeys(sizes, 0)
    for _ in range(20000):
        counts[sampler.sample(gen)] += 1
    total = sum(counts.values())
    for i, s in enumerate(sizes):
        assert abs(counts[s] / total - float(sampler.probs[i])) < 0.02, (s, counts)
    assert counts[2] > counts[4] > counts[8]  # smaller sampled more (more recovery iters)


def test_sampler_handles_tuple_sizes():
    # Attention sizes are (q, kv) tuples; sampler returns them verbatim.
    sizes = [(24, 2), (20, 4), (15, 3)]
    params = [1000, 2000, 1500]
    s = ElasticSizeSampler(sizes, params).sample(torch.Generator().manual_seed(1))
    assert s in sizes


def test_inverse_width_probs_favor_thinner_models():
    probs = inverse_width_probs([2688, 1344])

    assert torch.allclose(
        probs,
        torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=torch.float64),
    )
