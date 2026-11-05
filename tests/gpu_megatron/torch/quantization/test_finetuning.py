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

"""Full-rank routed-expert training through a frozen HybridStack boundary."""

from functools import partial

import pytest
import torch
from _test_utils.torch.megatron.models import HAS_MAMBA, get_mcore_hybrid_model
from _test_utils.torch.megatron.utils import get_batch, initialize_for_megatron

import modelopt.torch.quantization as mtq
from examples.megatron_bridge.finetuning import configure_trainable_parameters


@pytest.mark.skipif(not HAS_MAMBA, reason="Hybrid model fixture requires Mamba")
@pytest.mark.parametrize("recompute", [False, True])
def test_full_rank_hybrid_routed_experts(dist_workers_size_1, recompute):
    """Train routed experts through a frozen HybridStack with optional recomputation."""
    dist_workers_size_1.run(partial(_test_full_rank_hybrid_routed_experts, recompute))


def _test_full_rank_hybrid_routed_experts(recompute, rank, size):
    """Check finite expert gradients and updates while preserving frozen weights."""
    initialize_for_megatron(tensor_model_parallel_size=1, seed=42)
    model = get_mcore_hybrid_model(
        num_layers=2,
        hybrid_layer_pattern="*E",
        transformer_impl="transformer_engine",
        moe_grouped_gemm=False,
        recompute_granularity="full" if recompute else None,
        recompute_method="uniform" if recompute else None,
        recompute_num_layers=1 if recompute else None,
    ).cuda()
    tokens, _, positions, attention_mask, _ = get_batch(model, batch_size=1)

    def forward(m):
        return m(tokens, positions, attention_mask)

    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward)
    configure_trainable_parameters(
        model, ["*.local_experts.*.linear_fc1.weight", "*.local_experts.*.linear_fc2.weight"]
    )
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".local_experts." in n for n in trainable)
    assert not any("lora_" in n for n in before)
    optimizer = torch.optim.Adam(trainable.values(), lr=0.01)
    forward(model).float().square().mean().backward()
    assert any(p.grad is not None and p.grad.count_nonzero() for p in trainable.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable.values())
    optimizer.step()
    assert any(not torch.equal(p, before[n]) for n, p in trainable.items())
    for name, parameter in model.named_parameters():
        if name not in trainable:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
