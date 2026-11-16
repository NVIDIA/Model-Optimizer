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
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from modelopt.torch.speculative.plugins.modeling_fakebase import (
    FakeBaseConfig,
    FakeBaseModel,
    find_final_norm_path,
)
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
        # model_type "llama" is in the final-norm whitelist, so FakeBaseModel requires the norm.
        "norm.weight": torch.ones(_HIDDEN_SIZE),
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(tensors, shard)
    index = {"weight_map": dict.fromkeys(tensors, shard.name)}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path


@pytest.mark.parametrize("config_attr", [None, "text_config", "llm_config"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fakebase_local_happy_path(tmp_path, fake_config, monkeypatch, config_attr, dtype):
    fake_config.dtype = dtype
    fake_config.layer_norm_epsilon = 1e-5
    if config_attr is not None:
        fake_config.model_type = "nemotron_h"
        config = transformers.PretrainedConfig(layer_norm_epsilon=1e-3, rms_norm_eps=1e-3)
        setattr(config, config_attr, fake_config)
        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: config)
        embed_key = "language_model.backbone.embeddings.weight"
        head_key = "language_model.lm_head.weight"
        norm_key = "language_model.backbone.norm_f.weight"
        eps = fake_config.layer_norm_epsilon
    else:
        fake_config.rms_norm_eps = 1e-4
        embed_key, head_key, norm_key = "embed_tokens.weight", "lm_head.weight", "norm.weight"
        eps = fake_config.rms_norm_eps
    generator = torch.Generator().manual_seed(0)
    tensors = {
        embed_key: torch.randn(_VOCAB_SIZE, _HIDDEN_SIZE, generator=generator).to(dtype),
        head_key: torch.randn(_VOCAB_SIZE, _HIDDEN_SIZE, generator=generator).to(dtype),
        norm_key: torch.linspace(0.8, 1.2, _HIDDEN_SIZE).to(dtype),
    }
    safetensors.torch.save_file(tensors, tmp_path / "model.safetensors")

    model = FakeBaseModel.from_source(str(tmp_path))
    assert model.lm_head.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])
    assert model.embed_tokens.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])
    torch.testing.assert_close(model.embed_tokens.weight, tensors[embed_key], rtol=0, atol=0)
    reference_norm = LlamaRMSNorm(_HIDDEN_SIZE, eps=eps).to(dtype)
    reference_norm.weight.data.copy_(tensors[norm_key])
    # Small residuals make an incorrect epsilon materially change the reconstructed logits.
    hidden = (torch.randn(2, 4, _HIDDEN_SIZE, generator=generator) * 0.003).to(dtype)
    actual_logits = model.lm_head(model.norm(hidden))
    expected_logits = torch.nn.functional.linear(reference_norm(hidden), tensors[head_key])
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    assert model.config.rms_norm_eps == eps


def test_fakebase_missing_index_raises(tmp_path, fake_config):
    with pytest.raises(FileNotFoundError, match="safetensors"):
        FakeBaseModel.from_source(str(tmp_path))


def test_fakebase_single_file_no_index(tmp_path, fake_config):
    """Small models often ship a single ``model.safetensors`` without an index.json."""
    tensors = {
        "lm_head.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        "embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        "norm.weight": torch.ones(_HIDDEN_SIZE),
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
    safetensors.torch.save_file(
        {"embed_tokens.weight": weight, "norm.weight": torch.ones(_HIDDEN_SIZE)},
        tmp_path / "model.safetensors",
    )
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


def test_load_vlm_or_llm_uses_transformers5_vlm_auto_class(monkeypatch):
    """Transformers 5 loads VLMs through AutoModelForImageTextToText."""
    cfg = transformers.PretrainedConfig()
    cfg.model_type = "qwen3_vl"
    cfg.text_config = object()
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: cfg)

    captured = {}

    class _FakeVLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return object()

    # ``transformers`` exposes auto classes lazily, so deleting this attribute
    # lets its module-level ``__getattr__`` recreate the legacy class.  An
    # explicit ``None`` models its absence and reliably exercises the v5
    # fallback.
    monkeypatch.setattr(transformers, "AutoModelForVision2Seq", None, raising=False)
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", _FakeVLM)

    assert load_vlm_or_llm("qwen3-vl", dtype="auto") is not None
    assert captured["args"] == ("qwen3-vl",)
    assert captured["kwargs"]["dtype"] == "auto"


class TestFakeBaseRopeTheta:
    """The RoPE base has to survive the fake base, whichever shape the target stores it in.

    The draft injects the target's KV, so a mismatched base trains and exports without
    complaint and only misbehaves at serve time.
    """

    def test_from_source_carries_a_transformers_5_base_theta(self, tmp_path, monkeypatch):
        """The seam, not the reader.

        Reading rope_theta correctly is the exporter's ``_get_rope_theta`` and is tested
        there. What is pinned here is that this call site uses it: a plain
        ``getattr(base_cfg, "rope_theta")`` reads None from a transformers 5 config and
        silently builds a draft with no RoPE base at all.
        """
        base_cfg = transformers.PretrainedConfig(
            model_type="llama",
            hidden_size=_HIDDEN_SIZE,
            vocab_size=_VOCAB_SIZE,
            num_hidden_layers=2,
            max_position_embeddings=128,
            tie_word_embeddings=False,
            rope_parameters={"rope_theta": 1000000.0},
        )
        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: base_cfg)
        tensors = {
            "lm_head.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
            "embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
            "norm.weight": torch.ones(_HIDDEN_SIZE),
        }
        shard = tmp_path / "model-00001-of-00001.safetensors"
        safetensors.torch.save_file(tensors, shard)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(tensors, shard.name)})
        )

        model = FakeBaseModel.from_source(str(tmp_path))

        assert model.config.rope_theta == 1000000.0
        assert model.config.rope_parameters == {"rope_theta": 1000000.0, "rope_type": "default"}

    def test_config_publishes_both_shapes(self):
        """Consumers that prefer the dict must find it on a fake base too."""
        config = FakeBaseConfig(num_hidden_layers=2, hidden_size=32, rope_theta=1000000.0)
        assert config.rope_theta == 1000000.0
        assert config.rope_parameters == {"rope_theta": 1000000.0, "rope_type": "default"}

    def test_config_drives_a_transformers_rotary_embedding(self):
        """The published dict has to satisfy transformers, not just carry the number.

        This class is also the class the EAGLE draft config is built from, so the dict
        reaches `LlamaRotaryEmbedding`, which indexes ``rope_parameters["rope_type"]``
        unconditionally.
        """
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        config = FakeBaseConfig(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=1000000.0,
        )
        config.head_dim = 16

        rotary = LlamaRotaryEmbedding(config=config)
        cos, sin = rotary(torch.zeros(1, 4, 64), torch.arange(4).unsqueeze(0))

        assert rotary.rope_type == "default"
        assert cos.shape == (1, 4, 16)

    def test_unknown_theta_publishes_no_dict(self):
        """An absent base must stay absent rather than become a wrong default."""
        config = FakeBaseConfig(num_hidden_layers=2, hidden_size=32, rope_theta=None)
        assert config.rope_theta is None
        assert not getattr(config, "rope_parameters", None)


@pytest.mark.parametrize("path", ["model.norm", "language_model.backbone.norm_f"])
def test_find_final_norm_path_resolves_a_shared_norm_instance(path):
    """A norm instance also registered under an earlier path must still be found.

    ``named_modules()`` de-duplicates by default and yields a shared module only under
    the first name it reaches, so a dict built from it can be missing ``model.norm``
    entirely. The failure is silent: callers read ``None``, skip the norm, and feed an
    un-normed residual to lm_head -- a plausible loss curve against a garbage teacher.
    """
    shared = torch.nn.LayerNorm(8)

    class _Aliased(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # "aaa" sorts before "model", so traversal reaches the alias first.
            self.aaa = torch.nn.Module()
            self.aaa.norm = shared
            parent = self
            *parents, name = path.split(".")
            for part in parents:
                child = torch.nn.Module()
                setattr(parent, part, child)
                parent = child
            setattr(parent, name, shared)

    model = _Aliased()
    assert path not in dict(model.named_modules()), "alias no longer shadows"
    assert find_final_norm_path(model) == path


def test_find_final_norm_path_returns_none_without_a_norm():
    class _NoNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(4, 4)

    assert find_final_norm_path(_NoNorm()) is None
