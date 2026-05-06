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

"""Unit tests for ``_set_keys_to_learn`` in stitched_model_factory.py.

This function is the single source of truth for which parameters get trained
during a bypass run. Its branches (subblock_ffn / subblock_attention /
subblock_mamba / entire_block / list / regex) and its hybrid-model
``block_configs`` filter are all silent on misuse — a regression here would
freeze the wrong layers and produce a worse-than-teacher checkpoint with no
loud failure.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from modelopt.torch.puzzletron.bypass_distillation.stitched_model_factory import (
    _set_keys_to_learn,
)


# ---------------------------------------------------------------------------
# Fixtures: a minimal Llama-shaped model and a Llama-shaped descriptor stub
# ---------------------------------------------------------------------------


def _make_dense_model(num_layers: int = 2) -> nn.Module:
    """Build a tiny model whose named_parameters mimic Llama's naming.

    Parameters live under ``model.layers.{i}.self_attn.{q,k,v,o}_proj.weight``
    and ``model.layers.{i}.mlp.{up,down}_proj.weight``. The function never reads
    parameter shapes, so size doesn't matter — what matters is that the names
    match what `_set_keys_to_learn` expects to see in `named_parameters()` and
    `state_dict().keys()`.
    """
    model = nn.Module()
    model_inner = nn.Module()
    layers = nn.ModuleList()
    for _ in range(num_layers):
        layer = nn.Module()
        # attention
        layer.self_attn = nn.Module()
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, proj, nn.Linear(4, 4, bias=False))
        # feed-forward
        layer.mlp = nn.Module()
        for proj in ("up_proj", "down_proj"):
            setattr(layer.mlp, proj, nn.Linear(4, 4, bias=False))
        layers.append(layer)
    model_inner.layers = layers
    model.model = model_inner
    # `_set_keys_to_learn` reads `model.config` only to pass through to
    # `descriptor.get_language_model_config` — a SimpleNamespace is enough.
    model.config = SimpleNamespace()
    # Start with everything frozen so any True flag is something the function set.
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _make_descriptor(num_layers: int, *, block_configs=None):
    """Build a descriptor stub exposing only what ``_set_keys_to_learn`` calls.

    - ``get_language_model_config(config)`` returns an object with
      ``num_hidden_layers`` and (optionally) ``block_configs``.
    - ``get_weight_groups(state_dict_keys, num_hidden_layers)`` returns
      ``{"block_{i}_attention": [...], "block_{i}_ffn": [...]}``.
    """

    def get_language_model_config(_config):
        ns = SimpleNamespace(num_hidden_layers=num_layers)
        if block_configs is not None:
            ns.block_configs = block_configs
        return ns

    def get_weight_groups(state_dict_keys, n):
        groups: dict[str, list[str]] = {}
        for i in range(n):
            attn_prefix = f"model.layers.{i}.self_attn."
            ffn_prefix = f"model.layers.{i}.mlp."
            groups[f"block_{i}_attention"] = [
                k for k in state_dict_keys if k.startswith(attn_prefix)
            ]
            groups[f"block_{i}_ffn"] = [k for k in state_dict_keys if k.startswith(ffn_prefix)]
        return groups

    return SimpleNamespace(
        get_language_model_config=get_language_model_config,
        get_weight_groups=get_weight_groups,
    )


def _trainable_names(model: nn.Module) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad}


# ---------------------------------------------------------------------------
# Single-string subblock keys (dense model)
# ---------------------------------------------------------------------------


def test_subblock_ffn_trains_only_mlp():
    model = _make_dense_model(num_layers=2)
    descriptor = _make_descriptor(num_layers=2)
    _set_keys_to_learn(model, descriptor, "subblock_ffn")
    trainable = _trainable_names(model)
    assert all(".mlp." in n for n in trainable), trainable
    assert not any(".self_attn." in n for n in trainable), trainable
    # Both layers' mlp params must be trainable, not just one.
    assert any("model.layers.0.mlp." in n for n in trainable)
    assert any("model.layers.1.mlp." in n for n in trainable)


def test_subblock_attention_trains_only_self_attn():
    model = _make_dense_model(num_layers=2)
    descriptor = _make_descriptor(num_layers=2)
    _set_keys_to_learn(model, descriptor, "subblock_attention")
    trainable = _trainable_names(model)
    assert all(".self_attn." in n for n in trainable), trainable
    assert not any(".mlp." in n for n in trainable), trainable


def test_entire_block_trains_attention_and_mlp():
    model = _make_dense_model(num_layers=2)
    descriptor = _make_descriptor(num_layers=2)
    _set_keys_to_learn(model, descriptor, "entire_block")
    trainable = _trainable_names(model)
    # Both groups present.
    assert any(".self_attn." in n for n in trainable), trainable
    assert any(".mlp." in n for n in trainable), trainable
    # Equal to the union of every model parameter.
    assert trainable == {n for n, _ in model.named_parameters()}


# ---------------------------------------------------------------------------
# Hybrid model: subblock_mamba vs subblock_attention should partition by
# block_configs[i].attention.mamba — this is the path most likely to
# silently misroute training under future descriptor changes.
# ---------------------------------------------------------------------------


def _hybrid_block_configs():
    """Block 0: Mamba. Block 1: GQA. Detected via ``attention.mamba is not None``."""
    return [
        SimpleNamespace(attention=SimpleNamespace(mamba=SimpleNamespace())),  # Mamba
        SimpleNamespace(attention=SimpleNamespace(mamba=None)),  # GQA
    ]


def test_subblock_mamba_on_hybrid_trains_only_mamba_block():
    model = _make_dense_model(num_layers=2)
    descriptor = _make_descriptor(num_layers=2, block_configs=_hybrid_block_configs())
    _set_keys_to_learn(model, descriptor, "subblock_mamba")
    trainable = _trainable_names(model)
    # Block 0 (Mamba) attention-group params should be trainable; block 1 (GQA) must not.
    assert any("model.layers.0.self_attn." in n for n in trainable), trainable
    assert not any("model.layers.1.self_attn." in n for n in trainable), trainable
    # FFN params are never trainable under subblock_mamba.
    assert not any(".mlp." in n for n in trainable), trainable


def test_subblock_attention_on_hybrid_trains_only_gqa_block():
    model = _make_dense_model(num_layers=2)
    descriptor = _make_descriptor(num_layers=2, block_configs=_hybrid_block_configs())
    _set_keys_to_learn(model, descriptor, "subblock_attention")
    trainable = _trainable_names(model)
    # Block 1 (GQA) attention-group params are trainable; block 0 (Mamba) must not.
    assert any("model.layers.1.self_attn." in n for n in trainable), trainable
    assert not any("model.layers.0.self_attn." in n for n in trainable), trainable
    assert not any(".mlp." in n for n in trainable), trainable


# ---------------------------------------------------------------------------
# Free-form key forms: list, regex, no-match
# ---------------------------------------------------------------------------


def test_explicit_param_name_list_only_marks_listed_params():
    model = _make_dense_model(num_layers=2)
    descriptor = _make_descriptor(num_layers=2)
    target = "model.layers.0.self_attn.q_proj.weight"
    _set_keys_to_learn(model, descriptor, [target])
    assert _trainable_names(model) == {target}


def test_regex_string_uses_re_search():
    model = _make_dense_model(num_layers=2)
    descriptor = _make_descriptor(num_layers=2)
    _set_keys_to_learn(model, descriptor, r"q_proj")
    trainable = _trainable_names(model)
    # Both layers' q_proj must match; nothing else should.
    assert trainable == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
    }


def test_no_match_regex_returns_silently_with_no_trainable_params():
    model = _make_dense_model(num_layers=2)
    descriptor = _make_descriptor(num_layers=2)
    # Pipeline-parallel idle-rank case: a regex that matches nothing on this rank.
    _set_keys_to_learn(model, descriptor, r"nonexistent_param_pattern")
    assert _trainable_names(model) == set()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys_to_learn",
    ["subblock_ffn", "subblock_attention", "entire_block"],
)
def test_subblock_keys_skip_non_floating_point_params(keys_to_learn):
    """Integer / non-floating buffers exposed as parameters must stay frozen.

    The function explicitly guards on ``torch.is_floating_point(param)``; this
    test pins that guard so a future refactor doesn't accidentally try to
    enable grad on int tensors (which would raise at runtime).
    """
    model = _make_dense_model(num_layers=2)
    # Inject an int "param" alongside a real one.
    int_param = nn.Parameter(torch.zeros(2, dtype=torch.long), requires_grad=False)
    model.model.layers[0].self_attn.register_parameter("int_counter", int_param)
    descriptor = _make_descriptor(num_layers=2)
    # Should not raise even though the int param's name matches the attention group.
    _set_keys_to_learn(model, descriptor, keys_to_learn)
    # The int counter must remain frozen regardless.
    assert not model.model.layers[0].self_attn.int_counter.requires_grad
