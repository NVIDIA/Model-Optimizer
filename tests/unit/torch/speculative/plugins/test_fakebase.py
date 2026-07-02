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

"""Unit tests for FakeBaseModel and the fake-base / offline paths in load_vlm_or_llm."""

import json

import pytest
import safetensors.torch
import torch

pytest.importorskip("transformers")
import transformers

from modelopt.torch.speculative.plugins.modeling_fakebase import FakeBaseModel
from modelopt.torch.speculative.utils import load_vlm_or_llm

_HIDDEN_SIZE = 16
_VOCAB_SIZE = 32


@pytest.fixture
def fake_config(monkeypatch):
    """Monkeypatch AutoConfig.from_pretrained to return a minimal fake config."""
    cfg = transformers.PretrainedConfig()
    cfg.model_type = "llama"
    cfg.hidden_size = _HIDDEN_SIZE
    cfg.vocab_size = _VOCAB_SIZE
    cfg.num_hidden_layers = 2
    cfg.max_position_embeddings = 128
    cfg.tie_word_embeddings = False
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: cfg)
    return cfg


@pytest.fixture
def fake_checkpoint(tmp_path, fake_config):
    """Minimal local safetensors checkpoint loadable by FakeBaseModel."""
    tensors = {
        "lm_head.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        "embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, shard)
    index = {"weight_map": dict.fromkeys(tensors, shard.name)}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path


def test_fakebase_local_happy_path(fake_checkpoint):
    model = FakeBaseModel.from_source(str(fake_checkpoint))
    assert model.lm_head.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])
    assert model.embed_tokens.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])


def test_fakebase_vlm_dtype_from_top_level_config(tmp_path, monkeypatch):
    """Regression (OMNIML-4747): for a VLM the language-model sub-config often carries no
    torch_dtype — only the top-level config sets it (e.g. MiniMax-M3: top-level
    torch_dtype=bfloat16, text_config none). FakeBaseModel must resolve the base dtype from the
    top-level config rather than falling back to fp32; an fp32 head mismatches the bf16 offline
    hidden states in the draft forward ('mat1 and mat2 must have the same dtype')."""
    text_cfg = transformers.PretrainedConfig()
    text_cfg.hidden_size = _HIDDEN_SIZE
    text_cfg.vocab_size = _VOCAB_SIZE
    text_cfg.num_hidden_layers = 2
    text_cfg.max_position_embeddings = 128
    text_cfg.tie_word_embeddings = False
    text_cfg.torch_dtype = None  # the nested config carries no dtype (the M3 case)

    cfg = transformers.PretrainedConfig()
    cfg.model_type = "minimax_m3_vl"
    cfg.text_config = text_cfg  # marks this as a VLM -> base_cfg becomes text_cfg
    cfg.torch_dtype = "bfloat16"  # only the top-level config sets the dtype
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: cfg)

    safetensors.torch.save_file(
        {
            "lm_head.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
            "embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        },
        tmp_path / "model.safetensors",
    )

    model = FakeBaseModel.from_source(str(tmp_path))
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert model.embed_tokens.weight.dtype == torch.bfloat16


def test_fakebase_missing_index_raises(tmp_path, fake_config):
    with pytest.raises(FileNotFoundError, match="safetensors"):
        FakeBaseModel.from_source(str(tmp_path))


def test_fakebase_single_file_no_index(tmp_path, fake_config):
    """Small models often ship a single ``model.safetensors`` without an index.json."""
    tensors = {
        "lm_head.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        "embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
    }
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")
    model = FakeBaseModel.from_source(str(tmp_path))
    assert model.lm_head.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])
    assert model.embed_tokens.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])


def test_fakebase_tied_embeddings_falls_back_to_embed(tmp_path, fake_config):
    """Tied-embeddings models (e.g. Llama-3.2-1B) omit ``lm_head`` from safetensors;
    FakeBaseModel must reuse ``embed_tokens`` for both."""
    fake_config.tie_word_embeddings = True
    weight = torch.randn(_VOCAB_SIZE, _HIDDEN_SIZE)
    safetensors.torch.save_file({"embed_tokens.weight": weight}, tmp_path / "model.safetensors")
    model = FakeBaseModel.from_source(str(tmp_path))
    torch.testing.assert_close(model.lm_head.weight, weight)
    torch.testing.assert_close(model.embed_tokens.weight, weight)


def test_fakebase_missing_lm_head_without_tying_raises(tmp_path, fake_config):
    """Without ``tie_word_embeddings`` a missing ``lm_head`` is still an error."""
    safetensors.torch.save_file(
        {"embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE)},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(RuntimeError, match="lm_head"):
        FakeBaseModel.from_source(str(tmp_path))


def test_load_vlm_or_llm_returns_fakebase(fake_checkpoint):
    model = load_vlm_or_llm(str(fake_checkpoint), use_offline_training=True, use_fake_base=True)
    assert isinstance(model, FakeBaseModel)


def test_load_vlm_or_llm_offline_zero_layers(monkeypatch):
    cfg = transformers.PretrainedConfig()
    cfg.model_type = "llama"
    cfg.num_hidden_layers = 4
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: cfg)

    captured_kwargs = {}

    class _FakeModel:
        config = cfg

    def _fake_from_pretrained(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _fake_from_pretrained)

    model = load_vlm_or_llm("fake-model", use_offline_training=True, use_fake_base=False)
    assert captured_kwargs.get("num_hidden_layers") == 0
    assert model.config.num_orig_hidden_layers == 4
