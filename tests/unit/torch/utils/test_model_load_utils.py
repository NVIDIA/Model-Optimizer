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

"""Pure-function tests for ``modelopt.torch.utils.plugins.model_load_utils``."""

import pytest
import torch
from packaging.version import Version

pytest.importorskip("accelerate")

from modelopt.torch.utils.plugins.model_load_utils import (
    _conversion_plan,
    _convert_keys,
    _resolve_target,
    record_unplaced_source_keys,
)


def _build_tiny_qwen3_moe():
    """A tiny meta-init Qwen3-MoE (fused ``gate_up_proj`` + ``down_proj`` experts) for converter tests."""
    transformers = pytest.importorskip("transformers")
    if Version(transformers.__version__) < Version("5.0"):
        pytest.skip("multi-source fused-MoE conversion needs transformers>=5")
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3_moe",
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=6,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        max_position_embeddings=16,
    )
    try:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
    except Exception as e:  # modeling class unavailable in this transformers build
        pytest.skip(f"qwen3_moe modeling unavailable: {e}")
    return model, cfg


def test_checkpoint_key_converter_multisource_expert_fusion():
    """A multi-source fused MoE (Qwen3: gate_proj+up_proj -> gate_up_proj) converts correctly.

    Exercises the multi-source path (gate/up concatenated after expert stacking) AND the
    single-source path (down_proj is a plain expert stack) in one model.
    """
    model, cfg = _build_tiny_qwen3_moe()
    plan = _conversion_plan(model)
    assert plan is not None

    names = dict(model.named_parameters())
    gname = next(n for n in names if n.endswith("mlp.experts.gate_up_proj"))
    prefix = gname[: -len("mlp.experts.gate_up_proj")]
    n_exp, inter, hidden = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size

    gate = [torch.randn(inter, hidden) for _ in range(n_exp)]
    up = [torch.randn(inter, hidden) for _ in range(n_exp)]
    down = [torch.randn(hidden, inter) for _ in range(n_exp)]
    state = {}
    for e in range(n_exp):
        state[f"{prefix}mlp.experts.{e}.gate_proj.weight"] = gate[e]
        state[f"{prefix}mlp.experts.{e}.up_proj.weight"] = up[e]
        state[f"{prefix}mlp.experts.{e}.down_proj.weight"] = down[e]

    out = _convert_keys(plan, state)

    # Multi-source: experts stacked (dim 0) then gate|up concatenated (dim 1), gate first.
    gate_up = out[f"{prefix}mlp.experts.gate_up_proj"]
    assert gate_up.shape == (n_exp, 2 * inter, hidden) == tuple(names[gname].shape)
    assert torch.equal(gate_up, torch.cat([torch.stack(gate), torch.stack(up)], dim=1))

    # Single-source (regression): down_proj is a plain expert stack.
    down_proj = out[f"{prefix}mlp.experts.down_proj"]
    assert down_proj.shape == (n_exp, hidden, inter)
    assert torch.equal(down_proj, torch.stack(down))

    # Name-only mapping routes every source key to the fused target.
    for e in range(n_exp):
        assert _resolve_target(plan, f"{prefix}mlp.experts.{e}.gate_proj.weight")[0] == gname
        assert _resolve_target(plan, f"{prefix}mlp.experts.{e}.up_proj.weight")[0] == gname


# --- the loader's own accounting of what it could not place --------------------------------------


def test_record_unplaced_source_keys_stores_sorted_keys_and_provenance():
    model = torch.nn.Linear(2, 2)
    keys = record_unplaced_source_keys(model, "/ckpt/path", ["z.weight", "a.weight"])

    assert keys == ["a.weight", "z.weight"]
    assert model._modelopt_unplaced_source_keys == ["a.weight", "z.weight"]
    assert model._modelopt_source_checkpoint == "/ckpt/path"


def test_record_unplaced_source_keys_distinguishes_none_from_empty():
    """An empty list is an ANSWER -- the loader placed everything -- not "nobody asked".

    The export re-derives the set only when the attribute is absent, so recording [] has to stick;
    treating it as falsy-and-therefore-unknown would make every clean load pay a re-derivation.
    """
    model = torch.nn.Linear(2, 2)
    assert record_unplaced_source_keys(model, "/ckpt/path", None) == []
    assert model._modelopt_unplaced_source_keys == []
    assert model._modelopt_source_checkpoint == "/ckpt/path"


def test_record_unplaced_source_keys_accepts_any_iterable():
    model = torch.nn.Linear(2, 2)
    assert record_unplaced_source_keys(model, "/c", iter(["b", "a"])) == ["a", "b"]
