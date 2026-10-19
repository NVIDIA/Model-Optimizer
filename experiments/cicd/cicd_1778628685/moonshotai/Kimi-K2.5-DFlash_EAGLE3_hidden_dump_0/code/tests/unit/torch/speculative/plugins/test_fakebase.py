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


def test_fakebase_missing_index_raises(tmp_path, fake_config):
    with pytest.raises(FileNotFoundError, match="safetensors"):
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
