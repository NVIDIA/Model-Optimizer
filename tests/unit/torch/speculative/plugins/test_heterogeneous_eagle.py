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

"""EAGLE on heterogeneous (per-layer) targets and the first-batch hidden-state check."""

import json
from copy import deepcopy

import pytest
import torch

pytest.importorskip("transformers")
from _test_utils.torch.transformers_models import get_tiny_gemma4
from transformers import LlamaConfig

import modelopt.torch.speculative as mtsp
import modelopt.torch.utils.distributed as dist
from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG
from modelopt.torch.speculative.plugins.hf_eagle import (
    _reduce_per_layer_rope,
    _report_hidden_state_sanity,
)
from modelopt.torch.speculative.plugins.modeling_fakebase import _cfg_attr

_PER_LAYER_ROPE = {
    "full_attention": {"rope_theta": 1e6, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
}


def _tiny_gemma4():
    """Tiny Gemma 4 text model whose head_dim and kv heads differ between layer types."""
    pytest.importorskip("transformers.models.gemma4")
    model = get_tiny_gemma4()
    if not getattr(model.config, "is_heterogeneous", False):
        pytest.skip("transformers without per-layer Gemma 4 attributes")
    return model


def _convert_offline(model, **arch_overrides):
    """Convert ``model`` in place to an offline EAGLE-3 draft with a tiny recipe."""
    config = deepcopy(EAGLE3_DEFAULT_CFG["config"])
    config["eagle_architecture_config"].update(
        {
            "draft_vocab_size": model.config.vocab_size,
            "intermediate_size": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "head_dim": 2,
            **arch_overrides,
        }
    )
    config["eagle_offline"] = True
    config["eagle_use_torch_compile"] = False
    mtsp.convert(model, mode=[("eagle", config)])
    return model


def _offline_batch(model, seq_len=8):
    """A streamed-base batch: final and aux planes in the shape the offline collator emits."""
    hidden = model.config.hidden_size
    num_aux = len(model.eagle_config.eagle_aux_hidden_state_layer_ids)
    dtype = model.lm_head.weight.dtype
    return {
        "input_ids": torch.randint(0, model.config.vocab_size, (2, seq_len)),
        "base_model_outputs": {
            "base_model_hidden_states": torch.randn(2, seq_len, hidden, dtype=dtype),
            "aux_hidden_states": torch.randn(2, seq_len, hidden * num_aux, dtype=dtype),
        },
    }


def _aligned_logits(input_ids, vocab):
    """Logits whose argmax at position t is the token at t + 1."""
    logits = torch.zeros(*input_ids.shape, vocab)
    logits[:, :-1].scatter_(2, input_ids[:, 1:, None], 1.0)
    return logits


def test_cfg_attr():
    """Homogeneous configs read as getattr; per-layer attributes come from the requested layer."""
    llama = LlamaConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=2)
    assert _cfg_attr(llama, "num_key_value_heads") == 2
    assert _cfg_attr(llama, "missing", "fallback") == "fallback"
    cfg = _tiny_gemma4().config
    with pytest.raises(Exception, match="per-layer"):
        getattr(cfg, "head_dim")
    first, last = cfg.per_layer_config[0], cfg.per_layer_config[-1]
    assert first.num_key_value_heads != last.num_key_value_heads
    assert _cfg_attr(cfg, "num_key_value_heads") == last.num_key_value_heads
    assert _cfg_attr(cfg, "head_dim") == last.head_dim
    assert _cfg_attr(cfg, "head_dim", layer=0) == first.head_dim


def test_per_layer_rope_is_reduced_to_the_last_layer():
    """A flat rope_parameters is kept; a layer-type map is reduced and survives from_dict."""
    cfg = _tiny_gemma4().config
    flat = {"rope_theta": 5e5, "rope_type": "default"}
    arch = {"rope_parameters": dict(flat)}
    _reduce_per_layer_rope(arch, cfg)
    assert arch["rope_parameters"] == flat
    arch = {
        "rope_scaling": {"rope_type": "default", "rope_theta": 1e4},
        "rope_parameters": dict(_PER_LAYER_ROPE),
    }
    _reduce_per_layer_rope(arch, cfg)
    selected = _PER_LAYER_ROPE[cfg.layer_types[-1]]
    assert arch["rope_parameters"] == selected
    assert LlamaConfig.from_dict(arch).rope_parameters["rope_theta"] == selected["rope_theta"]


@pytest.mark.parametrize("per_layer_rope", [False, True])
def test_convert_builds_a_llama_draft_on_a_heterogeneous_base(per_layer_rope):
    """The draft keeps the recipe's attention shape; its RoPE is the recipe's or the reduced map."""
    overrides = {"rope_parameters": dict(_PER_LAYER_ROPE)} if per_layer_rope else {}
    model = _convert_offline(_tiny_gemma4(), **overrides)
    draft = model.eagle_config
    assert isinstance(draft, LlamaConfig)
    assert (draft.num_key_value_heads, draft.head_dim) == (16, 2)
    if per_layer_rope:
        expected = _PER_LAYER_ROPE[model.config.layer_types[-1]]["rope_theta"]
    else:
        expected = EAGLE3_DEFAULT_CFG["config"]["eagle_architecture_config"]["rope_theta"]
    assert draft.rope_parameters["rope_theta"] == expected
    assert len(model.eagle_module.layers) == 1


def test_offline_forward_reports_the_first_scored_batch(tmp_path, monkeypatch, capsys):
    """An unscored batch leaves the check pending; the first scored batch reports exactly once."""
    model = _convert_offline(_tiny_gemma4()).train()
    path = tmp_path / "sanity.json"
    monkeypatch.setenv("SPECDEC_HIDDEN_STATE_SANITY_PATH", str(path))
    torch.manual_seed(0)
    batch = _offline_batch(model)

    model(**batch, loss_mask=torch.zeros_like(batch["input_ids"]))
    assert not model._hidden_state_sanity_reported and not path.exists()

    model(**batch)
    assert model._hidden_state_sanity_reported
    assert json.loads(path.read_text())["tokens"] == batch["input_ids"][:, 1:].numel()
    assert capsys.readouterr().out.count("HIDDEN_STATE_SANITY") == 1

    path.unlink()
    model(**batch)
    assert "HIDDEN_STATE_SANITY" not in capsys.readouterr().out and not path.exists()


def test_offline_sanity_scores_the_streamed_hidden_states(tmp_path, monkeypatch):
    """Producer-supplied logits do not take part: aligned logits with random planes score low."""
    model = _convert_offline(_tiny_gemma4()).train()
    path = tmp_path / "sanity.json"
    monkeypatch.setenv("SPECDEC_HIDDEN_STATE_SANITY_PATH", str(path))
    torch.manual_seed(0)
    batch = _offline_batch(model)
    batch["base_model_outputs"]["base_model_logits"] = _aligned_logits(
        batch["input_ids"], model.config.vocab_size
    ).to(model.lm_head.weight.dtype)

    model(**batch)
    assert json.loads(path.read_text())["base_top1_next_token_agreement"] < 0.5


def test_hidden_state_sanity(tmp_path, monkeypatch, capsys):
    """Aligned planes score 1.0, shifted near 0; no scored token stays pending; only rank 0 reports."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, 16, (1, 64))
    loss_mask = torch.ones_like(input_ids)
    loss_mask[:, :8] = 0
    path = tmp_path / "sanity.json"
    monkeypatch.setenv("SPECDEC_HIDDEN_STATE_SANITY_PATH", str(path))
    monkeypatch.setenv("RANK", "1")  # a launcher variable; the process-group rank decides
    logits = _aligned_logits(input_ids, 16)

    record = _report_hidden_state_sanity(logits, input_ids, loss_mask)
    assert record["base_top1_next_token_agreement"] == 1.0
    assert record["tokens"] == int(loss_mask[:, 1:].sum())
    assert json.loads(path.read_text()) == record
    assert "HIDDEN_STATE_SANITY " + json.dumps(record) in capsys.readouterr().out

    shifted = _report_hidden_state_sanity(torch.roll(logits, 3, dims=1), input_ids, loss_mask)
    assert shifted["base_top1_next_token_agreement"] < 0.5
    assert _report_hidden_state_sanity(logits, input_ids, torch.zeros_like(input_ids)) is None

    monkeypatch.delenv("RANK")
    monkeypatch.setattr(dist, "rank", lambda group=None: 1)  # an initialized non-zero rank
    path.unlink()
    capsys.readouterr()
    assert _report_hidden_state_sanity(logits, input_ids, None)["tokens"] == 63
    assert capsys.readouterr().out == "" and not path.exists()
