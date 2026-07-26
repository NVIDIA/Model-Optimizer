# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from modelopt.torch.puzzletron.plugins.automodel.local_kd_recipe import (
    AutoModelLocalDistillationRecipe,
)
from modelopt.torch.puzzletron.pruning.ple_pruning import PLEPruningSpec
from modelopt.torch.puzzletron.pruning.runtime_ple import ple_layer_context


def _spec() -> PLEPruningSpec:
    return PLEPruningSpec(
        language_prefix="model.language_model",
        layer_template="model.language_model.layers.{layer_idx}",
        num_layers=2,
        width=3,
    )


def _state() -> dict[str, torch.Tensor]:
    return {
        "model.language_model.embed_tokens_per_layer.weight": torch.arange(30).reshape(5, 6),
        "model.language_model.per_layer_model_projection.weight": torch.arange(24).reshape(6, 4),
        "model.language_model.per_layer_projection_norm.weight": torch.arange(3),
        "model.language_model.layers.0.per_layer_input_gate.weight": torch.arange(12).reshape(3, 4),
        "model.language_model.layers.0.per_layer_projection.weight": torch.arange(12).reshape(4, 3),
        "model.language_model.layers.1.per_layer_input_gate.weight": torch.arange(12, 24).reshape(3, 4),
        "model.language_model.layers.1.per_layer_projection.weight": torch.arange(12, 24).reshape(4, 3),
    }


def test_ple_ranking_is_global_sum_of_all_layer_projection_scores() -> None:
    spec = _spec()
    logs = {
        "model.language_model.layers.0.per_layer_projection": {
            "score": torch.tensor([9.0, 1.0, 3.0])
        },
        "model.language_model.layers.1.per_layer_projection": {
            "score": torch.tensor([0.0, 10.0, 2.0])
        },
    }

    order = spec.order_from_score_logs(logs)

    torch.testing.assert_close(order, torch.tensor([1, 0, 2]))


def test_ple_permutation_and_prefix_slice_cover_packed_and_per_layer_tensors() -> None:
    spec = _spec()
    original = _state()
    order = torch.tensor([2, 0, 1])

    permuted, handled = spec.permute_state_dict(original, order)
    sliced = spec.slice_state_dict(permuted, 2)

    expected_embedding = original[
        "model.language_model.embed_tokens_per_layer.weight"
    ].view(5, 2, 3)[:, :, order]
    torch.testing.assert_close(
        permuted["model.language_model.embed_tokens_per_layer.weight"],
        expected_embedding.reshape(5, 6),
    )
    torch.testing.assert_close(
        permuted["model.language_model.per_layer_projection_norm.weight"],
        original["model.language_model.per_layer_projection_norm.weight"][order],
    )
    torch.testing.assert_close(
        permuted["model.language_model.layers.0.per_layer_input_gate.weight"],
        original["model.language_model.layers.0.per_layer_input_gate.weight"][order],
    )
    torch.testing.assert_close(
        permuted["model.language_model.layers.0.per_layer_projection.weight"],
        original["model.language_model.layers.0.per_layer_projection.weight"][:, order],
    )
    assert handled == set(original)
    assert sliced["model.language_model.embed_tokens_per_layer.weight"].shape == (5, 4)
    assert sliced["model.language_model.per_layer_model_projection.weight"].shape == (4, 4)
    assert sliced["model.language_model.per_layer_projection_norm.weight"].shape == (2,)
    assert sliced["model.language_model.layers.1.per_layer_input_gate.weight"].shape == (2, 4)
    assert sliced["model.language_model.layers.1.per_layer_projection.weight"].shape == (4, 2)


def test_runtime_ple_prefix_matches_physical_layer_and_zeros_inactive_gradients() -> None:
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.per_layer_input_gate = torch.nn.Linear(4, 3, bias=False)
            self.per_layer_projection = torch.nn.Linear(3, 4, bias=False)

        def forward(self, hidden_states, per_layer_input=None):
            branch = torch.nn.functional.silu(self.per_layer_input_gate(hidden_states))
            branch = branch * per_layer_input
            return hidden_states + self.per_layer_projection(branch)

    torch.manual_seed(5)
    layer = Layer()
    physical = Layer()
    physical.per_layer_input_gate = torch.nn.Linear(4, 2, bias=False)
    physical.per_layer_projection = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        physical.per_layer_input_gate.weight.copy_(
            layer.per_layer_input_gate.weight[:2]
        )
        physical.per_layer_projection.weight.copy_(
            layer.per_layer_projection.weight[:, :2]
        )
    hidden = torch.randn(2, 3, 4)
    per_layer_input = torch.randn(2, 3, 3)

    with ple_layer_context(layer, spec=_spec(), width=2):
        virtual = layer(hidden, per_layer_input=per_layer_input)
    expected = physical(hidden, per_layer_input=per_layer_input[..., :2])
    torch.testing.assert_close(virtual, expected)

    with ple_layer_context(layer, spec=_spec(), width=2):
        layer(hidden, per_layer_input=per_layer_input).sum().backward()
    assert torch.count_nonzero(layer.per_layer_input_gate.weight.grad[2:]) == 0
    assert torch.count_nonzero(layer.per_layer_projection.weight.grad[:, 2:]) == 0


def test_local_kd_cycles_cartesian_hidden_and_ple_widths(monkeypatch) -> None:
    recipe = object.__new__(AutoModelLocalDistillationRecipe)
    recipe._ple_spec = _spec()
    recipe._ple_widths = (3, 2)
    recipe._ple_width_counts = {3: 0, 2: 0}
    recipe._hidden_widths = (8, 4)
    recipe._logical_dp_size = 1
    recipe._logical_dp_lane = 0
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "broadcast", lambda tensor, src: tensor)

    selected = [recipe._ple_width_for_step(step) for step in range(1, 5)]

    assert selected == [3, 3, 2, 2]
    assert recipe._ple_width_counts == {3: 2, 2: 2}
