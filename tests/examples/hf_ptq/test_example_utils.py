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

import warnings
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from _test_utils.examples.hf_ptq_example_utils import example_utils
from _test_utils.torch.transformers_models import (
    create_tiny_qwen3_moe_dir,
    get_tiny_llama,
    get_tiny_nemotron_h,
)
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, GenerationMixin, LlamaConfig

from modelopt.torch.utils.distributed import DistributionPlan
from modelopt.torch.utils.plugins.model_load_utils import load_pretrained


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize(
    ("calib_with_images", "packed_location", "error"),
    [
        (False, None, None),
        (True, None, "text-only inputs"),
        (False, "root", "pack-quantized"),
        (False, "text", "pack-quantized"),
    ],
)
def test_fsdp2_qwen_vlm_requires_text_only_calibration(
    model_type, calib_with_images, packed_location, error
):
    args = SimpleNamespace(local_checkpoint_path="local-model", calib_with_images=calib_with_images)
    config = SimpleNamespace(model_type=model_type, vision_config={}, text_config=SimpleNamespace())
    if packed_location is not None:
        target = config if packed_location == "root" else config.text_config
        target.quantization_config = {"quant_method": "compressed-tensors"}
    expected = pytest.raises(NotImplementedError, match=error) if error else nullcontext()
    with expected:
        example_utils.validate_fsdp2_supported(args, config)


@pytest.mark.parametrize(
    ("ranks", "qformat", "recipe", "refused"),
    [
        (2, "w4a8_awq_beta", None, True),
        (2, "fp8", "general/auto_quantize/w4a8_awq_beta_fp8_at_6p0bits", True),
        (2, "fp8", None, False),
        (1, "w4a8_awq_beta", None, False),
    ],
)
def test_fsdp2_refuses_awq_on_several_ranks_before_loading(
    monkeypatch, ranks, qformat, recipe, refused
):
    """Each rank would fold its own AWQ scale into its weight shard, which the library refuses
    only once the model is loaded: the example refuses at argument validation."""
    monkeypatch.setattr(example_utils.dist_utils, "size", lambda: ranks)
    args = SimpleNamespace(local_checkpoint_path="local-model", qformat=qformat, recipe=recipe)
    expected = (
        pytest.raises(NotImplementedError, match="AWQ calibration") if refused else nullcontext()
    )
    with expected:
        example_utils.validate_fsdp2_supported(args, SimpleNamespace(model_type="llama"))


def _write_safetensors(path, tensors):
    save_file(tensors, str(path), metadata={"format": "pt"})


@pytest.mark.parametrize("loader", ["native", "auto"])
@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_mtp_preparation_matches_loader(tmp_path, loader, trust_remote_code):
    """A native architecture and a remote auto_map must prepare the class actually loaded."""
    native = pytest.importorskip("transformers.models.nemotron_h.modeling_nemotron_h")
    mtp_adapter = pytest.importorskip("modelopt.torch.models.nemotron_h.mtp")
    source = get_tiny_nemotron_h(
        layers_block_type=["attention" if "attention" in native.MIXER_TYPES else "full_attention"],
        num_hidden_layers=1,
        hybrid_override_pattern="*",
        use_mamba_kernels=False,
        attn_implementation="eager",
        dtype=torch.float32,
        num_nextn_predict_layers=1,
        mtp_layers_block_type=["attention", "moe"],
    )
    source.mtp = mtp_adapter._NemotronHMTP(source.config)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.uniform_(-0.1, 0.1)
    source.config.architectures = ["NemotronHForCausalLM"]
    source.config.auto_map = {"AutoModelForCausalLM": "modeling_remote.RemoteNemotronH"}
    source.config.save_pretrained(tmp_path)
    _write_safetensors(tmp_path / "model.safetensors", source.state_dict())
    (tmp_path / "modeling_remote.py").write_text(
        "from transformers import NemotronHForCausalLM\n"
        "class RemoteNemotronH(NemotronHForCausalLM):\n"
        "    pass\n"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if loader == "native":
            model = example_utils.get_model(
                str(tmp_path), device="cpu", trust_remote_code=trust_remote_code
            )
        else:
            model = load_pretrained(
                str(tmp_path),
                AutoModelForCausalLM,
                model_type=source.config.model_type,
                trust_remote_code=trust_remote_code,
            )
    expected_class = (
        "RemoteNemotronH" if loader == "auto" and trust_remote_code else "NemotronHForCausalLM"
    )
    assert type(model).__name__ == expected_class
    assert model.state_dict().keys() == source.state_dict().keys()
    for name, value in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert not any("MTP weights may remain unquantized" in str(w.message) for w in caught)


def test_get_model_loads_with_the_requested_experts_implementation(tmp_path):
    """hf_ptq's --experts_implementation reaches the fused experts: eager calibrates them as earlier
    releases did, where transformers would pick grouped_mm for this model."""
    path = create_tiny_qwen3_moe_dir(tmp_path)
    model = example_utils.get_model(str(path), device="cpu", experts_implementation="eager")
    assert model.config._experts_implementation == "eager"


@pytest.mark.parametrize("failure_at", [1, 2])
def test_get_model_preserves_preparation_error(monkeypatch, tmp_path, failure_at):
    config = LlamaConfig(architectures=["LlamaForCausalLM"])
    monkeypatch.setattr(example_utils.AutoConfig, "from_pretrained", lambda *a, **k: config)
    calls = 0

    def prepare(*args, model_class):
        nonlocal calls
        calls += 1
        assert args == (config.model_type, str(tmp_path), False)
        assert model_class.__name__ == "LlamaForCausalLM"
        if calls == failure_at:
            raise ValueError("MTP preparation failed")
        return nullcontext()

    monkeypatch.setattr(example_utils, "prepare_model_for_loading", prepare)
    # The second call, around the real load, is load_pretrained's, which imports it from here.
    monkeypatch.setattr("modelopt.torch.models.hf.prepare_model_for_loading", prepare)
    with pytest.raises(ValueError, match="MTP preparation failed"):
        example_utils.get_model(str(tmp_path), device="cpu")
    assert calls == failure_at


# ---------- get_original_hf_quant_method -------------------------------------
# get_model uses this to detect native MXFP4 checkpoints (e.g. openai/gpt-oss-*) and load
# them dequantized to BF16 GptOssExperts (so ModelOpt can quantize/export the experts).


@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        ("max", {"method": "max", "moe_calib_experts_ratio": 0.5}),
        ({"method": "gptq"}, {"method": "gptq", "moe_calib_experts_ratio": 0.5}),
        (
            [{"method": "gptq"}, {"method": "imatrix"}],
            [
                {"method": "gptq", "moe_calib_experts_ratio": 0.5},
                {"method": "imatrix", "moe_calib_experts_ratio": 0.5},
            ],
        ),
    ],
)
def test_build_quant_cfg_sets_moe_calib_experts_ratio_on_every_algorithm(algorithm, expected):
    quant_cfg = example_utils.build_quant_cfg({"quant_cfg": [], "algorithm": algorithm}, None, 0.5)
    assert quant_cfg["algorithm"] == expected


@pytest.mark.parametrize("wrapped", [False, True], ids=["the-model", "a-model-around-it"])
def test_a_preview_keeps_generate_from_switching_the_experts_implementation(monkeypatch, wrapped):
    """On a GPU, ``generate`` switches a ``grouped_mm`` MoE's experts to ``batched_mm`` for decoding,
    which ModelOpt's quantizers do not intercept (the preview after PTQ decodes unquantized experts)
    and which Transformers refuses for an FSDP2-wrapped class (``FSDPQwen3MoeForCausalLM does not
    support setting experts implementation``). The hook ``generate`` enters is out of the way for the
    preview, and back after it, also when the previewed model generates through a submodule, as a
    Nemotron-VL model's ``chat`` generates through its language model."""
    entered = []

    @contextmanager
    def switch(self):
        entered.append(type(self).__name__)
        raise ValueError(f"{type(self).__name__} does not support setting experts implementation.")
        yield

    monkeypatch.setattr(GenerationMixin, "_optimize_model_for_decode", switch)
    model = get_tiny_llama()
    # The model around it has no hook of its own; the preview generates through the inner one.
    previewed = torch.nn.ModuleDict({"language_model": model}) if wrapped else model
    input_ids = torch.tensor([[1, 2, 3]])

    # Some releases enter the hook only for the decode steps after the first token.
    with pytest.raises(ValueError, match="does not support setting experts implementation"):
        model.generate(input_ids, max_new_tokens=3)
    with example_utils.keep_the_experts_implementation(previewed):
        model.generate(input_ids, max_new_tokens=3)

    assert entered == ["LlamaForCausalLM"]  # only the first, unprotected call reached the hook
    assert "_optimize_model_for_decode" not in vars(model)


def test_a_model_without_the_decode_hook_is_previewed_as_it_is():
    model = torch.nn.Linear(1, 1)

    with example_utils.keep_the_experts_implementation(model):
        assert "_optimize_model_for_decode" not in vars(model)
    assert not hasattr(model, "_optimize_model_for_decode")


def test_a_decode_hook_on_the_model_itself_is_put_back_and_previews_nest():
    """A hook bound on the model instance is restored rather than dropped, and a preview inside
    another leaves the outer one's stand-in in place."""
    model = torch.nn.Linear(1, 1)
    own = model._optimize_model_for_decode = object()

    with example_utils.keep_the_experts_implementation(model):
        with example_utils.keep_the_experts_implementation(model):
            pass
        assert model._optimize_model_for_decode is nullcontext

    assert model._optimize_model_for_decode is own


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
    tmp_path,
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

    class FakeModel(torch.nn.Module):  # load_pretrained records its plan on every module
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

    model = example_utils.get_model(str(tmp_path), device="cpu", trust_remote_code=True)

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
    monkeypatch, tmp_path, model_type, architecture, device_count, expected_device_map
):
    calls = {}
    hf_config = SimpleNamespace(
        architectures=[architecture],
        dtype=torch.float16,
        model_type=model_type,
        torch_dtype=torch.bfloat16,
    )

    class FakeModel(torch.nn.Module):  # load_pretrained records its plan on every module
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

    example_utils.get_model(str(tmp_path), device="cuda", trust_remote_code=True)

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
    (
        "architecture",
        "remote_config",
        "builtin_config_error",
        "trust_remote_code",
        "expect_bundled",
    ),
    [
        ("DeepseekV3ForCausalLM", False, None, True, True),
        ("DeepseekV3ForCausalLM", False, None, False, False),
        # Dense Nemotron-H: the built-in class cannot parse the remote config -> bundled code.
        ("NemotronHForCausalLM", True, KeyError("-"), True, True),
        # MoE Nemotron-H (or a newer Transformers): the built-in config parses -> built-in class.
        ("NemotronHForCausalLM", True, None, True, False),
        # An unrelated failure (e.g. Hub retrieval) is not an incompatibility: it propagates.
        ("NemotronHForCausalLM", True, OSError("hub down"), True, None),
    ],
)
def test_get_model_selects_bundled_or_builtin_class(
    monkeypatch,
    tmp_path,
    *,
    architecture,
    remote_config,
    builtin_config_error,
    trust_remote_code,
    expect_bundled,
):
    """--trust_remote_code selects bundled code for DeepSeek and for a config the built-in
    class cannot represent; otherwise the built-in class is used."""
    used = {}
    config_cls = type(
        "_Config",
        (SimpleNamespace,),
        {"__module__": "transformers_modules.ckpt.config" if remote_config else __name__},
    )
    hf_config = config_cls(
        architectures=[architecture],
        dtype=torch.bfloat16,
        model_type="nemotron_h" if "NemotronH" in architecture else "deepseek_v3",
        torch_dtype=torch.bfloat16,
    )

    def _load_config(*args, **kwargs):
        # The re-derivation of a built-in config is the call made without trust_remote_code.
        if builtin_config_error and not kwargs.get("trust_remote_code"):
            raise builtin_config_error
        return hf_config

    class FakeModel(torch.nn.Module):  # load_pretrained records its plan on every module
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

    monkeypatch.setattr(example_utils.AutoConfig, "from_pretrained", _load_config)
    monkeypatch.setattr(example_utils, "AutoModelForCausalLM", _record("bundled"))
    monkeypatch.setattr(example_utils.transformers, architecture, _record("builtin"), raising=False)
    monkeypatch.setattr(example_utils, "prepare_model_for_loading", lambda *a, **k: nullcontext())
    monkeypatch.setattr(example_utils, "is_nemotron_vl", lambda config: False)
    monkeypatch.setattr(example_utils, "is_speculative", lambda config: False)
    monkeypatch.setattr(example_utils, "init_empty_weights", lambda include_buffers: nullcontext())
    monkeypatch.setattr(example_utils, "get_max_memory", lambda: {0: 1024})
    monkeypatch.setattr(example_utils, "infer_auto_device_map", lambda model, max_memory: {"": 0})

    if expect_bundled is None:
        with pytest.raises(OSError):
            example_utils.get_model(
                str(tmp_path), device="cpu", trust_remote_code=trust_remote_code
            )
        return
    example_utils.get_model(str(tmp_path), device="cpu", trust_remote_code=trust_remote_code)

    assert used["path"] == ("bundled" if expect_bundled else "builtin")


def test_nemotron_h_prefers_bundled_code_for_unsupported_block_type(monkeypatch):
    """A built-in config that parses but names a block type the built-in model lacks needs the
    checkpoint's bundled code."""
    modeling = pytest.importorskip("transformers.models.nemotron_h.modeling_nemotron_h")
    remote_cls = type("_Config", (SimpleNamespace,), {"__module__": "transformers_modules.c.cfg"})
    hf_config = remote_cls(model_type="nemotron_h")
    monkeypatch.setattr(
        example_utils.AutoConfig,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(layers_block_type=["mamba", "mlp"]),
    )

    monkeypatch.setattr(modeling, "MIXER_TYPES", {"mamba": object})
    assert example_utils.prefers_bundled_code("nemotron_h", hf_config, "ckpt", {})
    monkeypatch.setattr(modeling, "MIXER_TYPES", {"mamba": object, "mlp": object})
    assert not example_utils.prefers_bundled_code("nemotron_h", hf_config, "ckpt", {})


# ---------- get_tokenizer ----------------------------------------------------


class _StubTokenizer:
    """Minimal stand-in for a PreTrainedTokenizer."""

    def __init__(self, pad_token, eos_token="<|eos|>"):
        self.pad_token = pad_token
        self.eos_token = eos_token


@pytest.mark.parametrize(
    ("pad_token", "eos_token", "expected_pad"),
    [
        # Valid existing pad token must be preserved unchanged.
        ("<|endoftext|>", "<|im_end|>", "<|endoftext|>"),
        # <unk> as pad must also be preserved (only None triggers the fallback).
        ("<unk>", "<|im_end|>", "<unk>"),
        # Missing pad token falls back to eos_token.
        (None, "<|im_end|>", "<|im_end|>"),
    ],
)
def test_get_tokenizer_pad_token(monkeypatch, pad_token, eos_token, expected_pad):
    stub = _StubTokenizer(pad_token=pad_token, eos_token=eos_token)
    monkeypatch.setattr(example_utils.AutoTokenizer, "from_pretrained", lambda *a, **k: stub)
    tok = example_utils.get_tokenizer("/fake/path")
    assert tok.pad_token == expected_pad


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


# Small enough that the dense part fits any GPU.
_TINY = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
}


@pytest.mark.parametrize(
    ("device", "world_size", "device_map"),
    [
        ("cuda:1", 4, {"": "cuda:1"}),
        (torch.device("cuda:1"), 4, {"": "cuda:1"}),
        ("cuda:1", 1, "auto"),
        (torch.device("cpu"), 1, "cpu"),
    ],
    ids=["str", "torch_device", "single_process", "cpu"],
)
def test_a_rank_device_pins_the_whole_model_to_it(
    monkeypatch, tmp_path, device, world_size, device_map
):
    """Under torchrun each rank asks for its own cuda:<local_rank>; "auto" would let every rank's
    accelerate pick the same idle GPU. A single process spreads the model over the GPUs, whatever
    GPU it names, and keeps it on a CPU it names, as the torch.device hf_ptq's run hands in. Only a
    single process's load is held to a memory limit sized over the node's GPUs: the pinned map would
    ignore one."""
    LlamaConfig(architectures=["LlamaForCausalLM"], **_TINY).save_pretrained(tmp_path)
    # No GPU on the node holds the model, so sizing it over the node limits its memory.
    monkeypatch.setattr(example_utils, "get_max_memory", lambda: {0: 0, "cpu": 1 << 40})
    placed = []

    plan = DistributionPlan(world_size)

    def load_pretrained(ckpt_path, auto_class, **kwargs):
        assert kwargs["distribution"] == plan
        placed.append((kwargs["device_map"], "max_memory" in kwargs))
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(example_utils, "load_pretrained", load_pretrained)

    example_utils.get_model(str(tmp_path), device=device, distribution=plan)

    assert placed == [(device_map, world_size == 1)]


@pytest.mark.parametrize(
    "checkpoint",
    ["pack-quantized", "mxfp4", "diffusion-gemma"],
)
def test_every_load_path_keeps_the_rank_pin(monkeypatch, tmp_path, checkpoint):
    """A pack-quantized or MXFP4 checkpoint loads through a path of its own, and DiffusionGemma asks
    for a sequential map on a multi-GPU node; each would otherwise place every rank's copy over the
    same GPUs."""
    quantization_config = {
        "pack-quantized": {"quant_method": "compressed-tensors", "format": "pack-quantized"},
        "mxfp4": {"quant_method": "mxfp4"},
    }.get(checkpoint)
    LlamaConfig(
        architectures=["LlamaForCausalLM"], quantization_config=quantization_config, **_TINY
    ).save_pretrained(tmp_path)
    if checkpoint == "diffusion-gemma":
        monkeypatch.setattr(example_utils, "is_diffusion_gemma", lambda config: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(example_utils, "_unpack_compressed_linear_weights", lambda *args: None)
    placed = []

    def load_pretrained(ckpt_path, auto_class, **kwargs):
        placed.append(kwargs["device_map"])
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(example_utils, "load_pretrained", load_pretrained)

    example_utils.get_model(str(tmp_path), device="cuda:1", distribution=DistributionPlan(4))

    assert placed == [{"": "cuda:1"}]


@pytest.mark.parametrize(
    ("device", "kwargs", "match"),
    [
        ("cuda:1", {"use_seq_device_map": True}, "--use_seq_device_map"),
        # "auto" would let every rank's accelerate pick the same idle GPU.
        ("cuda", {}, "names no GPU"),
    ],
    ids=["sequential_map", "unindexed_device"],
)
def test_a_placement_over_shared_gpus_is_refused_where_each_rank_holds_its_own_copy(
    tmp_path, device, kwargs, match
):
    LlamaConfig(architectures=["LlamaForCausalLM"], **_TINY).save_pretrained(tmp_path)

    with pytest.raises(ValueError, match=match):
        example_utils.get_model(
            str(tmp_path), device=device, distribution=DistributionPlan(4), **kwargs
        )
