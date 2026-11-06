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
import stat

import pytest
import safetensors.torch
import torch
from _test_utils.torch.transformers_models import get_tiny_tokenizer

pytest.importorskip("transformers")
import transformers

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


def test_fakebase_local_happy_path(fake_checkpoint):
    model = FakeBaseModel.from_source(str(fake_checkpoint))
    assert model.lm_head.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])
    assert model.embed_tokens.weight.shape == torch.Size([_VOCAB_SIZE, _HIDDEN_SIZE])


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


@pytest.mark.parametrize("layout", ["flat", "tied", "text_config", "llm_config", "unknown_norm"])
def test_export_source_slice_preserves_source_and_loader_parity(tmp_path, layout):
    source, destination = tmp_path / "source", tmp_path / "slice"
    source.mkdir()
    config = transformers.LlamaConfig(
        hidden_size=_HIDDEN_SIZE,
        vocab_size=_VOCAB_SIZE,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        tie_word_embeddings=layout == "tied",
        dtype=torch.bfloat16,
    )
    embedding_key, head_key, norm_key = (
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    )
    if layout == "text_config":
        config = transformers.LlavaConfig(text_config=config.to_dict())
    elif layout == "llm_config":
        remote_code = (
            "from transformers import PretrainedConfig, LlamaConfig\n"
            "class SliceConfig(PretrainedConfig):\n"
            "    model_type = 'slice_test'\n"
            "    def __init__(self, llm_config=None, **kwargs):\n"
            "        super().__init__(**kwargs)\n"
            "        self.llm_config = LlamaConfig(**(llm_config or {}))\n"
        )
        (source / "configuration_slice.py").write_text(remote_code)
        config = transformers.PretrainedConfig(
            llm_config=config.to_dict(),
            auto_map={"AutoConfig": "configuration_slice.SliceConfig"},
        )
        config.model_type = "slice_test"
    elif layout == "unknown_norm":
        config = transformers.GPT2Config(n_embd=_HIDDEN_SIZE, vocab_size=_VOCAB_SIZE)
    config.save_pretrained(source)
    if layout == "llm_config":
        raw = json.loads((source / "config.json").read_text())
        raw["model_type"] = "slice_test"
        (source / "config.json").write_text(json.dumps(raw))
    if layout in ("text_config", "llm_config"):
        embedding_key, head_key, norm_key = (
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "language_model.model.norm.weight",
        )
    generator = torch.Generator().manual_seed(0)
    tensors = {
        embedding_key: torch.randn(_VOCAB_SIZE, _HIDDEN_SIZE, generator=generator).to(
            torch.bfloat16
        )
    }
    if layout != "tied":
        tensors[head_key] = torch.randn(_VOCAB_SIZE, _HIDDEN_SIZE, generator=generator)
    if layout != "unknown_norm":
        tensors[norm_key] = torch.linspace(0.8, 1.2, _HIDDEN_SIZE)
    shard = "model-00001-of-00002.safetensors"
    safetensors.torch.save_file(tensors, source / shard)
    unused = "model.layers.0.unused.weight"
    safetensors.torch.save_file(
        {unused: torch.ones(16, 16)}, source / "model-00002-of-00002.safetensors"
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    **dict.fromkeys(tensors, shard),
                    unused: "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    tokenizer = get_tiny_tokenizer()
    tokenizer.save_pretrained(source)
    (source / "modeling_custom.py").write_text(
        "# Custom model code remains part of the snapshot.\n"
    )
    original_config = (source / "config.json").read_bytes()

    assert (
        FakeBaseModel.export_source_slice(
            source, destination, trust_remote_code=layout == "llm_config"
        )
        == destination
    )

    actual = safetensors.torch.load_file(destination / "model.safetensors")
    assert set(actual) == set(tensors)
    for key, weight in tensors.items():
        assert actual[key].dtype == weight.dtype
        torch.testing.assert_close(actual[key], weight, rtol=0, atol=0)
    assert (destination / "config.json").read_bytes() == original_config
    assert stat.S_IMODE(destination.stat().st_mode) == stat.S_IMODE(source.stat().st_mode)
    assert not (destination / "model.safetensors.index.json").exists()
    assert list(destination.glob("*.safetensors")) == [destination / "model.safetensors"]
    for path in source.glob("*.py"):
        assert (destination / path.name).read_bytes() == path.read_bytes()
    assert transformers.AutoTokenizer.from_pretrained(destination).encode(
        "Hello"
    ) == tokenizer.encode("Hello")
    original = FakeBaseModel.from_source(str(source), trust_remote_code=layout == "llm_config")
    restored = FakeBaseModel.from_source(str(destination), trust_remote_code=layout == "llm_config")
    for key, weight in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], weight, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        FakeBaseModel.export_source_slice(source, destination)
    assert (destination / "config.json").read_bytes() == original_config


@pytest.mark.parametrize("invalid", ["shape", "parent_shard", "absolute_shard"])
def test_export_source_slice_rejects_invalid_source(
    fake_checkpoint, fake_config, tmp_path, invalid
):
    fake_config.save_pretrained(fake_checkpoint)
    shard = fake_checkpoint / "model-00001-of-00001.safetensors"
    tensors = {
        "embed_tokens.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        "lm_head.weight": torch.zeros(_VOCAB_SIZE, _HIDDEN_SIZE),
        "norm.weight": torch.ones(_HIDDEN_SIZE),
    }
    if invalid == "shape":
        tensors["embed_tokens.weight"] = torch.zeros(1, _HIDDEN_SIZE)
    else:
        shard = tmp_path.parent / f"{tmp_path.name}-outside.safetensors"
        index = json.loads((fake_checkpoint / "model.safetensors.index.json").read_text())
        name = "../" + shard.name if invalid == "parent_shard" else str(shard)
        index["weight_map"] = dict.fromkeys(tensors, name)
        (fake_checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
    safetensors.torch.save_file(tensors, shard)
    destination = tmp_path / "slice"
    with pytest.raises(ValueError):
        FakeBaseModel.export_source_slice(fake_checkpoint, destination)
    assert not destination.exists()


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


def test_find_final_norm_path_resolves_a_shared_norm_instance():
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
            self.model = torch.nn.Module()
            self.model.norm = shared

    model = _Aliased()
    assert "model.norm" not in dict(model.named_modules()), "alias no longer shadows"
    assert find_final_norm_path(model) == "model.norm"


def test_find_final_norm_path_returns_none_without_a_norm():
    class _NoNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(4, 4)

    assert find_final_norm_path(_NoNorm()) is None
