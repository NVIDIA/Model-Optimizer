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

import itertools

import torch

from examples.local_quantization_distillation.hessian_refit import block_gn, block_ls


def test_block_refit_keeps_a_planted_optimum_and_never_increases_the_objective():
    torch.manual_seed(0)
    rows, n, group, local_max = 8, 512, 16, 15
    groups_per_block = block_ls.GGML_BLOCK // group
    levels = torch.randint(-3, 4, (rows, n)).float()
    scale = (torch.rand(rows, n // block_ls.GGML_BLOCK) * 0.1 + 0.05).half().double()
    local = torch.randint(0, local_max + 1, (rows, n // group))
    steps = scale.repeat_interleave(groups_per_block, 1) * (2 * local.double() + 1) / 8
    # The teacher weight is a legal student, so the planted fields are the optimum.
    teacher = block_ls.expand_steps(levels, steps, group)
    inputs = torch.randn(4096, n)
    gram = inputs.T @ inputs
    a, c = block_ls.group_quadratics(levels, gram, teacher @ gram.T, group)

    torch.testing.assert_close(
        block_ls.solve_continuous(a, c, damp=0.0), steps, rtol=1e-3, atol=1e-5
    )
    kept_scale, kept_local, _, _ = block_ls.refit_hierarchy(
        a, c, scale, local, groups_per_block, local_max, damp=0.0
    )
    assert torch.equal(kept_local, local)
    torch.testing.assert_close(kept_scale, scale, rtol=1e-3, atol=0.0)

    worse_local = (local + torch.randint(-2, 3, local.shape)).clamp(0, local_max)
    worse_scale = scale * (1 + 0.2 * torch.randn(scale.shape)).clamp_min(0.5)
    _, _, _, history = block_ls.refit_hierarchy(
        a, c, worse_scale, worse_local, groups_per_block, local_max, damp=0.0
    )
    rounds = [history["ptq"], *(history[f"round{i}"] for i in range(3))]
    slack = 1e-6 * abs(history["ptq"])
    assert all(later <= earlier + slack for earlier, later in itertools.pairwise(rounds))
    assert rounds[-1] < rounds[0]


def test_gauss_newton_gradient_matches_autograd():
    torch.manual_seed(0)
    tokens, n, group, rows, width = 64, 32, 16, 4, 6
    x = torch.randn(tokens, n, dtype=torch.float64)
    levels_gate = torch.randint(-3, 4, (rows, n)).double()
    levels_up = torch.randint(-3, 4, (rows, n)).double()
    steps_gate = (torch.rand(rows, n // group, dtype=torch.float64) * 0.1 + 0.05).requires_grad_()
    steps_up = (torch.rand(rows, n // group, dtype=torch.float64) * 0.1 + 0.05).requires_grad_()
    down = torch.randn(width, rows, dtype=torch.float64)
    target = torch.randn(tokens, width, dtype=torch.float64)

    gate = x @ (levels_gate * steps_gate.repeat_interleave(group, 1)).T
    up = x @ (levels_up * steps_up.repeat_interleave(group, 1)).T
    residual = (torch.nn.functional.silu(gate) * up) @ down.T - target
    expected = torch.autograd.grad(0.5 * residual.square().sum(), (steps_gate, steps_up))

    stats = block_gn.GaussNewtonStats(rows, n // group, "cpu")
    stats.add(x, gate.detach(), up.detach(), residual.detach(), down, levels_gate, levels_up, group)
    torch.testing.assert_close(stats.b_g, expected[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(stats.b_u, expected[1], rtol=1e-4, atol=1e-5)
