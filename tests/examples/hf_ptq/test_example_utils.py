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
"""Unit tests for ``examples/hf_ptq/example_utils`` helpers.

The per-MTP-convention tests are gone with ``load_mtp_weights``: weights the loader could not
place are now identified from Transformers' own ``unexpected_keys`` rather than by recognising
storage layouts, so there is no convention matrix left to enumerate. What remains covers the
recording path, checkpoint-path resolution, and the sidecar copy.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from _test_utils.examples.hf_ptq_example_utils import example_utils
from safetensors.torch import save_file


def _write_safetensors(path, tensors):
    save_file(tensors, str(path), metadata={"format": "pt"})


def test_copy_custom_model_files_preserves_non_weight_sidecars(tmp_path):
    source_dir = tmp_path / "source"
    export_dir = tmp_path / "export"
    source_dir.mkdir()
    export_dir.mkdir()

    source_files = {
        "super_v3_reasoning_parser.py": "class Parser: pass\n",
        "modeling_custom.py": "class Model: pass\n",
        "README.md": "# Source model\n",
        "LICENSE": "license text\n",
        "chat_template.jinja": "{{ messages }}\n",
        "tokenizer_config.json": '{"chat_template": "source"}\n',
        "generation_config.json": '{"source": "generation"}\n',
        "config.json": '{"source": "config"}\n',
        "hf_quant_config.json": '{"source": "quant"}\n',
        "quant_config.json": '{"source": "stale quant"}\n',
        "quantize_config.json": '{"source": "stale quant"}\n',
        "recipe.yaml": "quantize: {}\n",
        "model.safetensors.index.json": '{"weight_map": {}}\n',
        "model-00001-of-00001.safetensors": "source weights\n",
        "model.gguf": "source weights\n",
    }
    for file_name, contents in source_files.items():
        (source_dir / file_name).write_text(contents)

    (export_dir / "config.json").write_text('{"export": "config"}\n')
    (export_dir / "generation_config.json").write_text('{"export": "generation"}\n')
    (export_dir / "hf_quant_config.json").write_text('{"export": "quant"}\n')
    (export_dir / "chat_template.jinja").write_text("{{ exported_messages }}\n")
    (export_dir / "tokenizer_config.json").write_text('{"chat_template": "export"}\n')

    example_utils.copy_custom_model_files(str(source_dir), str(export_dir), trust_remote_code=False)

    for file_name in [
        "super_v3_reasoning_parser.py",
        "modeling_custom.py",
        "README.md",
        "LICENSE",
        "chat_template.jinja",
        "generation_config.json",
    ]:
        assert (export_dir / file_name).read_text() == source_files[file_name]

    assert (export_dir / "config.json").read_text() == '{"export": "config"}\n'
    assert (export_dir / "hf_quant_config.json").read_text() == '{"export": "quant"}\n'
    assert (export_dir / "tokenizer_config.json").read_text() == '{"chat_template": "export"}\n'
    assert not (export_dir / "quant_config.json").exists()
    assert not (export_dir / "quantize_config.json").exists()
    assert not (export_dir / "recipe.yaml").exists()
    assert not (export_dir / "model.safetensors.index.json").exists()
    assert not (export_dir / "model-00001-of-00001.safetensors").exists()
    assert not (export_dir / "model.gguf").exists()

    (export_dir / "generation_config.json").write_text('{"export": "generation"}\n')
    example_utils.copy_custom_model_files(
        str(source_dir),
        str(export_dir),
        exclude_files={"generation_config.json"},
    )
    assert (export_dir / "generation_config.json").read_text() == '{"export": "generation"}\n'


def test_resolve_model_path_snapshot_download_stays_allowlisted(monkeypatch, tmp_path):
    snapshot_dir = tmp_path / "snapshot"

    def fake_snapshot_download(**kwargs):
        assert kwargs == {
            "repo_id": "org/model",
            "allow_patterns": example_utils._HF_SIDECAR_DOWNLOAD_ALLOW_PATTERNS,
        }
        return str(snapshot_dir)

    def fake_from_pretrained(*args, **kwargs):
        assert (args, kwargs) == (("org/model",), {"trust_remote_code": False})
        return SimpleNamespace(_name_or_path="org/model")

    monkeypatch.setattr(
        example_utils.AutoConfig,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(example_utils, "snapshot_download", fake_snapshot_download)

    assert example_utils._resolve_model_path("org/model", trust_remote_code=False) == str(
        snapshot_dir
    )


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
        def from_pretrained(*args, output_loading_info=False, **kwargs):
            calls["from_pretrained"] = kwargs
            assert "dtype" not in kwargs
            assert kwargs["torch_dtype"] is torch.float16
            m = FakeModel()
            return (m, {"unexpected_keys": []}) if output_loading_info else m

    class FakeLlamaForCausalLM(FakeAutoModelForCausalLM):
        _from_config = FakeAutoModelForCausalLM.from_config

        @staticmethod
        def from_pretrained(*args, output_loading_info=False, **kwargs):
            calls["from_pretrained"] = kwargs
            assert kwargs["dtype"] == "auto"
            assert "torch_dtype" not in kwargs
            m = FakeModel()
            return (m, {"unexpected_keys": []}) if output_loading_info else m

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
        def from_pretrained(*args, output_loading_info=False, **kwargs):
            calls["from_pretrained"] = kwargs
            m = FakeModel()
            return (m, {"unexpected_keys": []}) if output_loading_info else m

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


@pytest.mark.parametrize(
    ("trust_remote_code", "expect_bundled_code"),
    [(True, True), (False, False)],
)
def test_get_model_deepseek_honors_trust_remote_code(
    monkeypatch, trust_remote_code, expect_bundled_code
):
    """DeepSeek ships bundled modeling code; --trust_remote_code selects it, else built-in."""
    used = {}
    hf_config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        dtype=torch.bfloat16,
        model_type="deepseek_v3",
        torch_dtype=torch.bfloat16,
    )

    class FakeModel:
        def eval(self):
            return None

    def _record(tag):
        class Fake:
            @staticmethod
            def from_config(config, **kwargs):
                used["path"] = tag
                return FakeModel()

            _from_config = from_config

            @staticmethod
            def from_pretrained(*args, output_loading_info=False, **kwargs):
                used["path"] = tag
                m = FakeModel()
                return (m, {"unexpected_keys": []}) if output_loading_info else m

        return Fake

    monkeypatch.setattr(example_utils.AutoConfig, "from_pretrained", lambda *a, **k: hf_config)
    monkeypatch.setattr(example_utils, "AutoModelForCausalLM", _record("bundled"))
    monkeypatch.setattr(
        example_utils.transformers, "DeepseekV3ForCausalLM", _record("builtin"), raising=False
    )
    monkeypatch.setattr(example_utils, "is_nemotron_vl", lambda config: False)
    monkeypatch.setattr(example_utils, "is_speculative", lambda config: False)
    monkeypatch.setattr(example_utils, "init_empty_weights", lambda include_buffers: nullcontext())
    monkeypatch.setattr(example_utils, "get_max_memory", lambda: {0: 1024})
    monkeypatch.setattr(example_utils, "infer_auto_device_map", lambda model, max_memory: {"": 0})

    example_utils.get_model("checkpoint", device="cpu", trust_remote_code=trust_remote_code)

    assert used["path"] == ("bundled" if expect_bundled_code else "builtin")


def _layerwise(**kwargs):
    return {"enable": True, **kwargs}


def _blocks(quant_cfg):
    algorithm = quant_cfg["algorithm"]
    entries = algorithm if isinstance(algorithm, list) else [algorithm]
    return [e["layerwise"] for e in entries if "layerwise" in e]


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        pytest.param(
            {"method": "max", "layerwise": _layerwise(export_dir="/placeholder")},
            ["/out.layerwise_resume"],
            id="single-entry",
        ),
        pytest.param(
            [
                {"method": "awq", "layerwise": _layerwise()},
                {"method": "max", "layerwise": _layerwise(export_dir="/placeholder")},
            ],
            [None, "/out.layerwise_resume"],
            id="only-the-exporting-entry",
        ),
        pytest.param(
            [
                {"method": "awq", "layerwise": _layerwise(checkpoint_dir="/theirs")},
                {"method": "max", "layerwise": _layerwise(export_dir="/placeholder")},
            ],
            ["/theirs", "/out.layerwise_resume"],
            id="another-entrys-explicit-path-is-not-this-ones",
        ),
    ],
)
def test_default_layerwise_resume_dir_targets_the_exporting_entry(algorithm, expected):
    """Only the pass that exports gets a derived resume dir, and only if it lacks one."""
    updated, changed = example_utils.default_layerwise_resume_dir({"algorithm": algorithm}, "/out")

    assert [b.get("checkpoint_dir") for b in _blocks(updated)] == expected
    assert changed is True


def test_resolve_checkpoint_dir_keeps_each_entrys_base():
    """Two layerwise passes must not resolve onto one manifest."""
    algorithm = [
        {"method": "awq", "layerwise": _layerwise(checkpoint_dir="/theirs")},
        {"method": "max", "layerwise": _layerwise(checkpoint_dir="/ours", export_dir="/ph")},
    ]

    updated, resolved = example_utils.resolve_checkpoint_dir({"algorithm": algorithm}, "/m/Model")

    theirs, ours = (b["checkpoint_dir"] for b in _blocks(updated))
    assert theirs.startswith("/theirs/") and ours.startswith("/ours/")
    assert theirs != ours
    # The exporting pass owns the path the caller reports.
    assert resolved == ours


@pytest.mark.parametrize(
    ("algorithm", "match"),
    [
        pytest.param(
            [
                {"layerwise": _layerwise(export_dir="/a")},
                {"layerwise": _layerwise(export_dir="/b")},
            ],
            "only one calibration pass",
            id="two-exporting-entries",
        ),
        pytest.param(
            [{"layerwise": _layerwise(export_dir="/a")}, {"method": "max"}],
            "must be the last",
            id="a-later-pass-would-change-the-model",
        ),
    ],
)
def test_set_layerwise_export_dir_refuses_ambiguous_ownership(algorithm, match):
    with pytest.raises(ValueError, match=match):
        example_utils.set_layerwise_export_dir({"algorithm": algorithm}, "/out")


class _Block:
    """A config object, as the deprecated ``--auto_quantize_*`` path builds."""

    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self):
        return dict(self._fields)


class _Recipe:
    def __init__(self, algorithm):
        self.quantize = SimpleNamespace(algorithm=algorithm)


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        pytest.param(None, [], id="no-recipe"),
        pytest.param(_Recipe(None), [], id="no-algorithm"),
        pytest.param(_Recipe({"method": "max"}), [], id="algorithm-without-layerwise"),
        pytest.param(
            _Recipe({"method": "max", "layerwise": {"enable": True}}),
            [{"enable": True}],
            id="dict-entry",
        ),
        pytest.param(
            _Recipe(
                [
                    {"method": "awq", "layerwise": {"enable": True}},
                    {"method": "max", "layerwise": {"enable": True, "export_dir": "/x"}},
                ]
            ),
            [{"enable": True}, {"enable": True, "export_dir": "/x"}],
            id="list-keeps-algorithm-order",
        ),
        pytest.param(
            _Recipe(SimpleNamespace(layerwise=_Block(enable=True, export_dir="/x"))),
            [{"enable": True, "export_dir": "/x"}],
            id="config-object-entry",
        ),
    ],
)
def test_recipe_layerwise_blocks(recipe, expected):
    """Both recipe shapes normalize to dicts, so callers need no shape-aware access."""
    assert example_utils.recipe_layerwise_blocks(recipe) == expected


# --- carry-over of weights the loader could not place -------------------------------------------


class _StubAuto:
    """Minimal stand-in for an ``AutoModelFor*`` class."""

    def __init__(self, unexpected, model=None):
        self._unexpected = unexpected
        self._model = model if model is not None else SimpleNamespace()
        self.saw_output_loading_info = None
        self.saw_kwargs = None

    def from_pretrained(self, ckpt_path, output_loading_info=False, **kwargs):
        self.saw_output_loading_info = output_loading_info
        self.saw_kwargs = kwargs
        return self._model, {
            "missing_keys": [],
            "unexpected_keys": list(self._unexpected),
            "mismatched_keys": [],
            "error_msgs": [],
        }


def test_from_pretrained_recording_asks_the_loader_for_its_accounting(tmp_path):
    """The point of this path: take unexpected_keys from the loader rather than re-deriving it."""
    auto = _StubAuto(["mtp.layers.0.eh_proj.weight", "mtp.fc.weight"])
    model = example_utils._from_pretrained_recording(auto, str(tmp_path), device_map="cpu")

    assert auto.saw_output_loading_info is True, "must request the loading info"
    assert auto.saw_kwargs == {"device_map": "cpu"}, "caller kwargs must pass through untouched"
    assert model._modelopt_unplaced_source_keys == [
        "mtp.fc.weight",
        "mtp.layers.0.eh_proj.weight",
    ], "recorded sorted, so export order is stable"
    assert model._modelopt_source_checkpoint == str(tmp_path)


def test_from_pretrained_recording_is_quiet_when_everything_was_placed(tmp_path):
    """Record an empty answer too, so the exporter can tell 'nothing to carry' from 'never asked'."""
    auto = _StubAuto([])
    model = example_utils._from_pretrained_recording(auto, str(tmp_path))

    assert model._modelopt_unplaced_source_keys == []
    assert model._modelopt_source_checkpoint == str(tmp_path)


def test_recording_is_architecture_agnostic(tmp_path):
    """Nothing keys off the string 'mtp': an auxiliary tower is carried by the same rule."""
    auto = _StubAuto(["aux_tower.blocks.0.weight", "something_else.weight"])
    model = example_utils._from_pretrained_recording(auto, str(tmp_path))

    assert model._modelopt_unplaced_source_keys == [
        "aux_tower.blocks.0.weight",
        "something_else.weight",
    ]
