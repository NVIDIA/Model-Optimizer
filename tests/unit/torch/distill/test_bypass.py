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

"""Independent mathematical checks for Bypass updates."""

from types import SimpleNamespace

import pytest
import torch

from modelopt.torch.distill._bypass_capture import _MSE_CHUNK_ELEMENTS, mse_statistics
from modelopt.torch.distill.plugins.transformers.bypass import TransformersBypassAdapter


def test_lion_conflicting_gradients_and_effective_schedule():
    from modelopt.torch.distill import BypassConfig
    from modelopt.torch.distill._bypass_optimizer import Lion, learning_rate

    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = Lion([parameter], lr=0.001)
    parameter.grad = torch.tensor([2.0, -4.0])
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([0.999, -0.999]))
    torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.tensor([0.02, -0.04]))
    parameter.grad = torch.tensor([-0.1, 0.1])
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([0.998, -0.998]))
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg"], torch.tensor([0.0188, -0.0386])
    )
    config = BypassConfig(num_steps=23, max_lr=0.001, warmup_fraction=0.1)
    for step, expected in {1: 0.001 / 3, 2: 0.002 / 3, 3: 0.001, 13: 0.0005, 23: 0.0}.items():
        assert learning_rate(config, step) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("chunk_elements", [1 << 16, 1 << 24])
def test_masked_mse_keeps_narrow_saved_activations_and_fp32_vjp(monkeypatch, chunk_elements):
    monkeypatch.setitem(_MSE_CHUNK_ELEMENTS, "cpu", chunk_elements)
    torch.manual_seed(18)
    prediction = torch.randn(2, 513, 257, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn_like(prediction, requires_grad=True)
    mask = torch.arange(513).expand(2, -1) % 3 != 0
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda value: saved.append((value.numel(), value.dtype)) or value, lambda value: value
    ):
        actual, count = mse_statistics(prediction, target, mask)
    expected = torch.where(mask[..., None], prediction.float() - target.float(), 0).square().sum()
    torch.testing.assert_close(actual, expected)
    assert count == mask.sum() * 257
    actual_gradients = torch.autograd.grad(actual, (prediction, target))
    expected_gradients = torch.autograd.grad(expected, (prediction, target))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    assert (prediction.numel(), torch.float32) not in saved

    prediction = torch.empty(2, 3, 0, dtype=torch.bfloat16, requires_grad=True)
    target = torch.empty_like(prediction, dtype=torch.float32, requires_grad=True)
    loss, count = mse_statistics(prediction, target, torch.ones(2, 3, dtype=torch.bool))
    assert loss == count == 0
    for gradient in torch.autograd.grad(loss, (prediction, target)):
        assert gradient.shape == prediction.shape and gradient.numel() == 0


def test_packed_attention_refuses_backends_without_block_masks():
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "_packed_seq_ids": torch.tensor([[1, 1, 2]])}
    adapter = object.__new__(TransformersBypassAdapter)
    with pytest.raises(ValueError, match="attention backend"):
        adapter.prepare_batch(model, batch)


@pytest.mark.parametrize("kind", ["none", "boolean", "additive"])
def test_packed_attention_preserves_stronger_masks_and_positive_biases(kind):
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(_attn_implementation="sdpa")
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "_packed_seq_ids": torch.tensor([[1, 1, 2]])}
    smallest = torch.finfo(torch.float32).min
    expected = [[0, smallest, smallest], [0, 0, smallest], [smallest, smallest, 0]]
    if kind == "boolean":
        batch["attention_mask"] = torch.tensor(
            [[[[True, True, False], [True, True, True], [False, True, True]]]]
        )
        expected[0][2] = expected[2][0] = float("-inf")
    elif kind == "additive":
        batch["attention_mask"] = torch.tensor(
            [[[[0.5, 1, float("-inf")], [0.25, -0.5, 0], [float("-inf"), 0, 2]]]]
        )
        expected = [
            [0.5, smallest, float("-inf")],
            [0.25, -0.5, smallest],
            [float("-inf"), smallest, 2],
        ]
    adapter = object.__new__(TransformersBypassAdapter)
    actual = adapter.prepare_batch(model, batch)["attention_mask"]
    torch.testing.assert_close(actual, torch.tensor([[expected]]), rtol=0, atol=0)


def test_thinker_without_a_use_cache_argument_keeps_outer_hooks():
    model = torch.nn.Module()
    model.thinker = torch.nn.Linear(1, 1)
    calls = []
    model.register_forward_hook(lambda *_: calls.append(1))
    inputs = torch.ones(2, 1)
    adapter = object.__new__(TransformersBypassAdapter)
    actual = adapter.forward_teacher(model, (inputs,), {})
    torch.testing.assert_close(actual, model.thinker(inputs), rtol=0, atol=0)
    assert calls == [1]


@pytest.mark.parametrize("packed", [False, True])
def test_hf_refuses_ambiguous_floating_2d_attention_mask(packed):
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(_attn_implementation="sdpa")
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[0.0, 0.0, float("-inf")]]),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    if packed:
        batch["_packed_seq_ids"] = torch.tensor([[1, 1, 2]])
        del batch["valid_mask"]
    adapter = object.__new__(TransformersBypassAdapter)
    for microbatch in adapter.microbatches(batch, 1):
        with pytest.raises(ValueError, match="2D attention mask"):
            adapter.prepare_batch(model, microbatch)
