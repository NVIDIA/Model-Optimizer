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
"""End-to-end unit tests for ``examples/hf_ptq/example_utils.load_mtp_weights``.

One test per supported on-disk MTP convention (inlined-orphaned, inlined-in-state-dict,
separate-file-standalone, separate-file-indexed) plus a negative case.
"""

import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from _test_utils.examples.hf_ptq_example_utils import example_utils
from safetensors.torch import save_file


class _FakeModel:
    """Stub exposing only the surface ``load_mtp_weights`` touches."""

    def __init__(self, config, state_dict_keys):
        self.config = config
        self._sd = {k: torch.zeros(1) for k in state_dict_keys}
        self.loaded = {}

    def state_dict(self):
        return dict(self._sd)

    def load_state_dict(self, state_dict, strict=True):
        self.loaded.update(state_dict)
        self._sd.update(state_dict)


def _write_safetensors(path, tensors):
    save_file(tensors, str(path), metadata={"format": "pt"})


def test_load_mtp_weights_inlined_orphaned(tmp_path):
    # GLM-5.1: HF builds only num_hidden decoders → MTP keys orphaned.
    main_keys = ["model.embed_tokens.weight", "model.layers.0.x.weight"]
    mtp_keys = ["model.layers.4.eh_proj.weight", "model.layers.4.enorm.weight"]
    _write_safetensors(
        tmp_path / "model.safetensors",
        {k: torch.zeros(2, 2) for k in main_keys + mtp_keys},
    )

    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=1)
    model = _FakeModel(cfg, state_dict_keys=main_keys)
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))

    assert prefixes == ["model.layers.4"]
    assert set(orphans) == set(mtp_keys)
    assert model.loaded == {}  # nothing matched the (MTP-less) state_dict


def test_load_mtp_weights_inlined_in_state_dict(tmp_path):
    # DeepSeek-V3 via trust_remote_code: MTP slots exist → keys loaded, no orphans.
    main_keys = ["model.embed_tokens.weight"]
    mtp_keys = ["model.layers.4.eh_proj.weight", "model.layers.4.enorm.weight"]
    _write_safetensors(
        tmp_path / "model.safetensors",
        {k: torch.ones(2, 2) for k in main_keys + mtp_keys},
    )

    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=1)
    model = _FakeModel(cfg, state_dict_keys=main_keys + mtp_keys)
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))

    assert prefixes == ["model.layers.4"]
    assert orphans == {}
    assert set(model.loaded) == set(mtp_keys)


def test_load_mtp_weights_separate_standalone_file(tmp_path):
    # GLM-4.7: standalone mtp.safetensors with no shard index.
    _write_safetensors(
        tmp_path / "model.safetensors", {"model.embed_tokens.weight": torch.zeros(2, 2)}
    )
    _write_safetensors(
        tmp_path / "mtp.safetensors",
        {
            "mtp.fc.weight": torch.zeros(2, 2),
            "mtp.layers.0.q_proj.weight": torch.zeros(2, 2),
        },
    )

    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=0)
    model = _FakeModel(cfg, state_dict_keys=["model.embed_tokens.weight"])
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))

    assert set(prefixes) == {"mtp", "mtp.layers.0"}
    assert set(orphans) == {"mtp.fc.weight", "mtp.layers.0.q_proj.weight"}


def test_load_mtp_weights_separate_indexed_shard(tmp_path):
    # Qwen3-Next: mtp.* keys in a dedicated indexed tail shard (filename has no "mtp").
    main_shard = "model-00001-of-00002.safetensors"
    mtp_shard = "model-00002-of-00002.safetensors"
    _write_safetensors(tmp_path / main_shard, {"model.embed_tokens.weight": torch.zeros(2, 2)})
    mtp_tensors = {
        "mtp.fc.weight": torch.zeros(2, 2),
        "mtp.norm.weight": torch.zeros(2),
        "mtp.layers.0.input_layernorm.weight": torch.zeros(2),
        "mtp.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
    }
    _write_safetensors(tmp_path / mtp_shard, mtp_tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": main_shard,
                    **dict.fromkeys(mtp_tensors, mtp_shard),
                }
            }
        )
    )

    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=0)
    model = _FakeModel(cfg, state_dict_keys=["model.embed_tokens.weight"])
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))

    assert set(prefixes) == {"mtp", "mtp.layers.0"}
    assert set(orphans) == set(mtp_tensors)


def test_keys_to_prefixes_drops_model_top_level():
    # nvbug 6108133: inlined keys like "model.layers.92.X" must NOT emit "model"
    # as a top-level prefix (would become "model*" excluding the whole backbone).
    out = example_utils._keys_to_prefixes(
        ["model.layers.92.eh_proj.weight", "mtp.fc.weight", "mtp.layers.0.q_proj.weight"]
    )
    assert "model" not in out
    assert out == {"mtp", "mtp.layers.0", "model.layers.92"}


def test_load_mtp_weights_no_mtp_returns_empty(tmp_path):
    # Also pins the ``num_nextn_predict_layers=None`` regression: some configs
    # set the field explicitly to None, which must not crash ``int(None)``.
    _write_safetensors(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": torch.zeros(2, 2),
            "model.layers.0.x.weight": torch.zeros(2, 2),
        },
    )
    cfg = SimpleNamespace(num_hidden_layers=4, num_nextn_predict_layers=None)
    model = _FakeModel(cfg, state_dict_keys=["model.embed_tokens.weight"])
    prefixes, orphans = example_utils.load_mtp_weights(model, str(tmp_path))
    assert prefixes == []
    assert orphans == {}


def test_get_checkpoint_weight_keys_reads_local_safetensors_index(tmp_path):
    weight_map = {
        "vision_tower.blocks.0.mlp.fc1.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.mlp.experts.0.gate_proj.weight_packed": (
            "model-00002-of-00002.safetensors"
        ),
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    assert example_utils._get_checkpoint_weight_keys(str(tmp_path)) == set(weight_map)


def test_get_checkpoint_weight_keys_falls_back_to_remote_single_safetensors(tmp_path, monkeypatch):
    tensors = {
        "vision_tower.blocks.0.mlp.fc1.weight": torch.zeros(2, 2),
        "model.layers.0.mlp.experts.0.gate_proj.weight_packed": torch.zeros(
            2, 2, dtype=torch.int32
        ),
    }
    safetensors_path = tmp_path / "model.safetensors"
    _write_safetensors(safetensors_path, tensors)
    requested_filenames = []

    def fake_hf_hub_download(repo_id, filename):
        assert repo_id == "owner/checkpoint"
        requested_filenames.append(filename)
        if filename == "model.safetensors.index.json":
            raise FileNotFoundError(filename)
        assert filename == "model.safetensors"
        return str(safetensors_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    assert example_utils._get_checkpoint_weight_keys("owner/checkpoint") == set(tensors)
    assert requested_filenames == ["model.safetensors.index.json", "model.safetensors"]


def test_get_model_passes_checkpoint_schema_to_compressed_tensors_patch(monkeypatch):
    checkpoint_weight_keys = {
        "vision_tower.blocks.0.mlp.fc1.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight_packed",
    }
    hf_config = SimpleNamespace(
        architectures=["KimiForCausalLM"],
        model_type="kimi",
        quantization_config={"format": "pack-quantized"},
    )
    calls = {}

    class FakeModel:
        def eval(self):
            calls["eval"] = True

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["from_pretrained"] = kwargs
            return FakeModel()

    @contextmanager
    def fake_loading_patch(weight_keys=None):
        calls["patch_weight_keys"] = weight_keys
        yield

    monkeypatch.setattr(
        example_utils.AutoConfig, "from_pretrained", lambda *args, **kwargs: hf_config
    )
    monkeypatch.setattr(example_utils, "AutoModelForCausalLM", FakeAutoModelForCausalLM)
    monkeypatch.setattr(
        example_utils, "_get_checkpoint_weight_keys", lambda ckpt_path: checkpoint_weight_keys
    )
    monkeypatch.setattr(example_utils, "_unpack_compressed_linear_weights", lambda *args: None)
    monkeypatch.setattr(
        "modelopt.torch.quantization.plugins.huggingface.patch_compressed_linear_loading",
        fake_loading_patch,
    )

    example_utils.get_model("checkpoint", device="cpu", trust_remote_code=True)

    assert calls["patch_weight_keys"] is checkpoint_weight_keys
    assert calls["eval"]


# ---------- get_original_hf_quant_method -------------------------------------
# get_model uses this to detect native MXFP4 checkpoints (e.g. openai/gpt-oss-*) and load
# them dequantized to BF16 GptOssExperts (so ModelOpt can quantize/export the experts).


def test_get_original_hf_quant_method_mxfp4_dict():
    # gpt-oss layout: quantization_config is a plain dict carrying quant_method.
    cfg = SimpleNamespace(
        quantization_config={"quant_method": "mxfp4", "modules_to_not_convert": []}
    )
    assert example_utils.get_original_hf_quant_method(cfg) == "mxfp4"


def test_get_original_hf_quant_method_object():
    # Some configs expose quantization_config as an object with a quant_method attribute.
    cfg = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="fp8"))
    assert example_utils.get_original_hf_quant_method(cfg) == "fp8"


def test_get_original_hf_quant_method_nested_text_config():
    # Multi-modal models nest the quantization_config under text_config.
    cfg = SimpleNamespace(
        text_config=SimpleNamespace(quantization_config={"quant_method": "mxfp4"})
    )
    assert example_utils.get_original_hf_quant_method(cfg) == "mxfp4"


def test_get_original_hf_quant_method_none_for_unquantized():
    assert example_utils.get_original_hf_quant_method(SimpleNamespace()) is None
    assert (
        example_utils.get_original_hf_quant_method(SimpleNamespace(quantization_config=None))
        is None
    )


# ---------- _resolve_init_config ---------------------------------------------


def _remote_config():
    # Config whose class module lives under "transformers_modules" (remote code).
    cls = type("_RemoteConfig", (), {"__module__": "transformers_modules.ckpt.config"})
    return cls()


def test_resolve_init_config_rederives_for_remote_config():
    builtin_cfg = SimpleNamespace()
    with patch.object(
        example_utils.AutoConfig, "from_pretrained", return_value=builtin_cfg
    ) as mock:
        out = example_utils._resolve_init_config(
            _remote_config(), object, "/ckpt", {"trust_remote_code": True}
        )
    assert out is builtin_cfg
    mock.assert_called_once_with("/ckpt")  # trust_remote_code stripped


def test_resolve_init_config_keeps_non_remote_config():
    cfg = SimpleNamespace()  # module is "types", not remote
    with patch.object(example_utils.AutoConfig, "from_pretrained") as mock:
        assert example_utils._resolve_init_config(cfg, object, "/ckpt", {}) is cfg
    mock.assert_not_called()


def test_resolve_init_config_falls_back_when_rederive_raises():
    cfg = _remote_config()
    with patch.object(example_utils.AutoConfig, "from_pretrained", side_effect=ValueError()):
        assert example_utils._resolve_init_config(cfg, object, "/ckpt", {}) is cfg


@pytest.mark.parametrize(
    (
        "architecture",
        "model_class_name",
        "expected_config_dtype_kwarg",
        "unexpected_config_dtype_kwarg",
    ),
    [
        ("DeciLMForCausalLM", "AutoModelForCausalLM", "torch_dtype", "dtype"),
        ("LlamaForCausalLM", "LlamaForCausalLM", "dtype", "torch_dtype"),
    ],
)
def test_get_model_uses_expected_dtype_kwarg(
    monkeypatch,
    architecture,
    model_class_name,
    expected_config_dtype_kwarg,
    unexpected_config_dtype_kwarg,
):
    calls = {}
    hf_config = SimpleNamespace(
        architectures=[architecture],
        dtype=torch.float16,
        model_type="llama",
        torch_dtype=torch.bfloat16,
    )

    class FakeModel:
        def eval(self):
            calls["eval"] = True

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_config(config, **kwargs):
            calls["from_config"] = kwargs
            assert config is hf_config
            assert kwargs[expected_config_dtype_kwarg] is torch.float16
            assert unexpected_config_dtype_kwarg not in kwargs
            assert "max_memory" not in kwargs
            return FakeModel()

        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["from_pretrained"] = kwargs
            assert "dtype" not in kwargs
            assert kwargs["torch_dtype"] is torch.float16
            return FakeModel()

    class FakeLlamaForCausalLM(FakeAutoModelForCausalLM):
        _from_config = FakeAutoModelForCausalLM.from_config

        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["from_pretrained"] = kwargs
            assert kwargs["dtype"] == "auto"
            assert "torch_dtype" not in kwargs
            return FakeModel()

    monkeypatch.setattr(
        example_utils.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: hf_config,
    )
    if model_class_name == "AutoModelForCausalLM":
        monkeypatch.setattr(example_utils, "AutoModelForCausalLM", FakeAutoModelForCausalLM)
        monkeypatch.delattr(example_utils.transformers, architecture, raising=False)
    else:
        monkeypatch.setattr(example_utils.transformers, model_class_name, FakeLlamaForCausalLM)
    monkeypatch.setattr(example_utils, "is_nemotron_vl", lambda config: False)
    monkeypatch.setattr(example_utils, "is_speculative", lambda config: False)
    monkeypatch.setattr(example_utils, "init_empty_weights", lambda include_buffers: nullcontext())
    monkeypatch.setattr(example_utils, "get_max_memory", lambda: {0: 1024})
    monkeypatch.setattr(example_utils, "infer_auto_device_map", lambda model, max_memory: {"": 0})

    model = example_utils.get_model("checkpoint", device="cpu", trust_remote_code=True)

    assert isinstance(model, FakeModel)
    assert calls["eval"]
    if expected_config_dtype_kwarg == "torch_dtype":
        assert calls["from_config"]["trust_remote_code"] is True
    else:
        assert "trust_remote_code" not in calls["from_config"]
    assert calls["from_pretrained"]["trust_remote_code"] is True


@pytest.mark.parametrize(
    ("model_type", "architecture", "device_count", "expected_device_map"),
    [
        # DiffusionGemma ties encoder/decoder weights; "auto" can split a tied pair
        # across GPUs, so multi-GPU loads must fall back to "sequential".
        ("diffusion_gemma", "DiffusionGemmaForConditionalGeneration", 2, "sequential"),
        # Detection must also work off ``architectures`` alone, without ``model_type``.
        (None, "DiffusionGemmaForConditionalGeneration", 2, "sequential"),
        # Single GPU cannot split a tied pair, so it keeps the unrestricted "auto" map.
        ("diffusion_gemma", "DiffusionGemmaForConditionalGeneration", 1, "auto"),
        # "gemma" is a substring of "diffusiongemma"; other Gemmas must not match.
        ("gemma3", "Gemma3ForCausalLM", 2, "auto"),
    ],
)
def test_get_model_device_map_for_diffusion_gemma(
    monkeypatch, model_type, architecture, device_count, expected_device_map
):
    calls = {}
    hf_config = SimpleNamespace(
        architectures=[architecture],
        dtype=torch.float16,
        model_type=model_type,
        torch_dtype=torch.bfloat16,
    )

    class FakeModel:
        def eval(self):
            calls["eval"] = True

        def parameters(self):
            return iter(())

    class FakeArchitecture:
        @staticmethod
        def _from_config(config, **kwargs):
            return FakeModel()

        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["from_pretrained"] = kwargs
            return FakeModel()

    monkeypatch.setattr(
        example_utils.AutoConfig, "from_pretrained", lambda *args, **kwargs: hf_config
    )
    # Set rather than delete: ``transformers`` lazy-imports, so a deleted real class
    # (e.g. Gemma3ForCausalLM) reappears on the next ``hasattr`` and the real one loads.
    # raising=False: DiffusionGemma may not exist in the installed transformers.
    monkeypatch.setattr(example_utils.transformers, architecture, FakeArchitecture, raising=False)
    monkeypatch.setattr(example_utils, "is_nemotron_vl", lambda config: False)
    monkeypatch.setattr(example_utils, "is_speculative", lambda config: False)
    monkeypatch.setattr(example_utils, "init_empty_weights", lambda include_buffers: nullcontext())
    monkeypatch.setattr(example_utils, "get_max_memory", lambda: {0: 1024})
    monkeypatch.setattr(example_utils, "infer_auto_device_map", lambda model, max_memory: {"": 0})
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)

    example_utils.get_model("checkpoint", device="cuda", trust_remote_code=True)

    assert calls["from_pretrained"]["device_map"] == expected_device_map
    # Sequential caps per-GPU memory; "auto" must stay unrestricted.
    if expected_device_map == "sequential":
        assert calls["from_pretrained"]["max_memory"] == {0: 1024 * 0.8}
    else:
        assert "max_memory" not in calls["from_pretrained"]


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        (SimpleNamespace(model_type="diffusion_gemma", architectures=None), True),
        (SimpleNamespace(model_type=None, architectures=["DiffusionGemmaForCausalLM"]), True),
        (SimpleNamespace(model_type="gemma3", architectures=["Gemma3ForCausalLM"]), False),
        # Multi-modal wrappers keep the family name on the nested ``text_config``.
        (
            SimpleNamespace(
                model_type="multimodal",
                architectures=["SomeWrapperForConditionalGeneration"],
                text_config=SimpleNamespace(model_type="diffusion_gemma"),
            ),
            True,
        ),
        (
            SimpleNamespace(
                model_type="multimodal",
                text_config=SimpleNamespace(architectures=["DiffusionGemmaForCausalLM"]),
            ),
            True,
        ),
        # A non-DiffusionGemma nested config must not match.
        (
            SimpleNamespace(
                model_type="multimodal", text_config=SimpleNamespace(model_type="gemma3")
            ),
            False,
        ),
        # Stub configs may omit either attribute entirely.
        (SimpleNamespace(), False),
    ],
)
def test_is_diffusion_gemma(hf_config, expected):
    assert example_utils.is_diffusion_gemma(hf_config) is expected
