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

import json

import pytest
import torch
from packaging.version import Version
from safetensors.torch import save_file

pytest.importorskip("accelerate")

from modelopt.torch.utils.plugins.model_load_utils import (
    _conversion_rules,
    _convert_keys,
    _target_name,
    read_safetensors_subset,
    weight_map_for,
)


def test_weight_map_for_sharded(tmp_path):
    save_file({"a.weight": torch.zeros(2)}, str(tmp_path / "shard1.safetensors"))
    save_file({"b.weight": torch.zeros(2)}, str(tmp_path / "shard2.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a.weight": "shard1.safetensors", "b.weight": "shard2.safetensors"}}
        )
    )

    assert weight_map_for(str(tmp_path)) == {
        "a.weight": "shard1.safetensors",
        "b.weight": "shard2.safetensors",
    }


def test_weight_map_for_single_file(tmp_path):
    save_file(
        {"a.weight": torch.zeros(2), "b.weight": torch.zeros(2)},
        str(tmp_path / "model.safetensors"),
    )

    assert weight_map_for(str(tmp_path)) == {
        "a.weight": "model.safetensors",
        "b.weight": "model.safetensors",
    }


def test_weight_map_for_missing(tmp_path):
    with pytest.raises(RuntimeError, match="No safetensors checkpoint"):
        weight_map_for(str(tmp_path))


def test_read_safetensors_subset(tmp_path):
    save_file(
        {"a.weight": torch.tensor([1.0, 2.0]), "a.bias": torch.tensor([3.0])},
        str(tmp_path / "shard1.safetensors"),
    )
    save_file({"b.weight": torch.tensor([4.0])}, str(tmp_path / "shard2.safetensors"))
    weight_map = {
        "a.weight": "shard1.safetensors",
        "a.bias": "shard1.safetensors",
        "b.weight": "shard2.safetensors",
    }

    result = read_safetensors_subset(str(tmp_path), weight_map, lambda n: n.startswith("a."))

    assert set(result.keys()) == {"a.weight", "a.bias"}
    assert torch.equal(result["a.weight"], torch.tensor([1.0, 2.0]))
    assert torch.equal(result["a.bias"], torch.tensor([3.0]))


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
    rules = _conversion_rules(model)
    assert rules is not None

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

    out = _convert_keys(rules, state)

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
        assert _target_name(rules, f"{prefix}mlp.experts.{e}.gate_proj.weight") == gname
        assert _target_name(rules, f"{prefix}mlp.experts.{e}.up_proj.weight") == gname
