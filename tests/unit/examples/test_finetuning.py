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

"""Selective full-rank fake-quant training with frozen activation boundaries."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from examples.megatron_bridge.finetuning import configure_trainable_parameters
from modelopt.torch.quantization.qtensor import INT4QTensor, QTensorWrapper


@pytest.mark.parametrize("recompute", [False, True])
def test_full_rank_training_and_restore(recompute):
    """Preserve selective updates and Adam restoration with frozen embeddings."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Embedding(16, 16), nn.Linear(16, 16), nn.Linear(16, 16))
    inputs = torch.arange(8)
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(inputs))
    baseline = model(inputs).detach()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if n != "1.weight"}
    configure_trainable_parameters(model, ["1.weight"])
    # Reapplying setup must not nest wrappers or alter quantizer state.
    configure_trainable_parameters(model, ["1.weight"])
    torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
    initial_weight = model[1].weight.detach().clone()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.1)
    activation = model[0](inputs)
    assert activation.requires_grad
    output = (
        checkpoint(model[1], activation, use_reentrant=True) if recompute else model[1](activation)
    )
    model[2](output).square().mean().backward()
    assert model[1].weight.grad.count_nonzero() > 0
    optimizer.step()
    assert not torch.equal(model[1].weight, initial_weight)
    for name, value in frozen.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ["1.weight"]
    assert not any("lora_" in n for n, _ in model.named_parameters())
    restored = nn.Sequential(nn.Embedding(16, 16), nn.Linear(16, 16), nn.Linear(16, 16))
    restored = mto.restore_from_modelopt_state(restored, mto.modelopt_state(model))
    restored.load_state_dict(model.state_dict())
    configure_trainable_parameters(restored, ["1.weight"])
    restored_optimizer = torch.optim.Adam([restored[1].weight], lr=0.1)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for candidate, opt in [(model, optimizer), (restored, restored_optimizer)]:
        opt.zero_grad()
        candidate(inputs).square().mean().backward()
        opt.step()
    torch.testing.assert_close(model[1].weight, restored[1].weight, rtol=0, atol=0)
    torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
    with torch.no_grad():
        assert not restored[0](inputs).requires_grad


@pytest.mark.parametrize("invalid", ["unmatched", "shared", "packed", "adapters", "empty"])
def test_full_rank_validation_is_atomic(invalid):
    """Reject invalid selections without changing parameter trainability."""
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
    patterns = ["0.weight"]
    if invalid == "unmatched":
        patterns.append("missing.*")
    elif invalid == "shared":
        model[1].weight = model[0].weight
    elif invalid == "packed":
        model[0].weight = QTensorWrapper(
            INT4QTensor.quantize(model[0].weight.detach(), block_size=16)[0]
        )
    elif invalid == "adapters":
        mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(torch.randn(2, 16)))
        mtq.enable_quant_lora(model)
    else:
        patterns = []
    before = {n: p.requires_grad for n, p in model.named_parameters()}
    with pytest.raises(ValueError):
        configure_trainable_parameters(model, patterns)
    assert before == {n: p.requires_grad for n, p in model.named_parameters()}
