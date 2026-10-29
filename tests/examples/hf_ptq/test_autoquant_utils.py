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

"""Offline coverage for the Hugging Face AutoQuantize example helpers."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from _test_utils.torch.transformers_models import get_tiny_llama
from transformers import WhisperConfig, WhisperForConditionalGeneration

from modelopt.recipe import ModelOptAutoQuantizeRecipe, load_recipe
from modelopt.recipe.config import AutoQuantizeConfig, AutoQuantizeConstraints
from modelopt.recipe.presets import QUANT_CFG_CHOICES
from modelopt.torch.quantization.config import QuantizeConfig
from modelopt.torch.utils.dataset_utils import get_max_batch_size


@pytest.fixture
def autoquant_utils(monkeypatch):
    examples_dir = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"
    monkeypatch.syspath_prepend(str(examples_dir))
    return importlib.import_module("autoquant_utils")


def _validated_autoquant_config(**updates):
    config = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize
    values = {name: getattr(config, name) for name in type(config).model_fields}
    return AutoQuantizeConfig.model_validate({**values, **updates})


@pytest.mark.parametrize(
    ("recipe_name", "has_kl", "has_gradient"),
    [
        ("nvfp4_fp8_at_5p4bits", False, True),
        ("nvfp4_fp8_kl_div_at_5p4bits", True, False),
        ("kv_fp8_nvfp4_cast_kl_div_at_5p4bits", True, False),
        ("nvfp4_fp8_gradient_then_kv_fp8_nvfp4_cast_kl_div_at_5p4bits", True, True),
    ],
)
def test_autoquant_memory_probe_and_oom_retry(
    autoquant_utils, monkeypatch, recipe_name, has_kl, has_gradient
):
    model = get_tiny_llama()
    model.config.use_cache = True
    recipe = load_recipe("general/auto_quantize/" + recipe_name)
    probe = autoquant_utils._get_autoquant_memory_probe(recipe)
    assert (probe is not None) == has_kl
    forwards, kl_tokens, logit_pointers = [], [], []

    def record_forward(module, inputs):
        assert not module.config.use_cache
        forwards.append((inputs[0].shape[0], torch.is_grad_enabled()))

    model.register_forward_pre_hook(record_forward)
    model.register_forward_hook(
        lambda _m, _i, output: logit_pointers.append(output.logits.data_ptr())
    )
    real_kl = autoquant_utils._forward_kl

    def limited_kl(logits, log_prob_ref):
        score = real_kl(logits, log_prob_ref)
        assert not logits.requires_grad and not log_prob_ref.requires_grad
        assert log_prob_ref.dtype == torch.float32
        tokens = logits.numel() // logits.shape[-1]
        kl_tokens.append(tokens)
        assert (logits.data_ptr() == logit_pointers[-1]) == (
            recipe.auto_quantize.constraints.cost_model == "weight"
            and recipe.kv_auto_quantize is None
        )
        # Simulate a GPU limit reached only by scoring, not by the model forward.
        if tokens > 4 * 8:
            raise torch.cuda.OutOfMemoryError
        return score

    monkeypatch.setattr(autoquant_utils, "_forward_kl", limited_kl)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=1000)
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    free_memory = iter((1000, 900))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (next(free_memory), 1000))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)

    result = get_max_batch_size(
        model,
        sample_input_single_batch=torch.ones((1, 8), dtype=torch.long),
        enable_grad=has_gradient,
        forward_step=probe,
    )
    assert model.config.use_cache
    if has_kl:
        assert result == 2
        assert kl_tokens == [8, 8, 80, 40, 16, 16]
        assert [batch for batch, grad in forwards if grad] == (
            [1, 10, 5, 2] if has_gradient else []
        )
    else:
        assert result == 8
        assert forwards == [(1, True), (10, True)]
        assert not kl_tokens


def test_kl_memory_probe_preserves_whisper_generation(autoquant_utils):
    model = WhisperForConditionalGeneration(
        WhisperConfig(
            vocab_size=32,
            num_mel_bins=8,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_source_positions=4,
            max_target_positions=16,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
            suppress_tokens=[],
            begin_suppress_tokens=[],
        )
    ).eval()
    model.generation_config.max_new_tokens = 2
    outputs = []
    model.register_forward_hook(lambda _m, _i, output: outputs.append(output.logits))
    recipe = load_recipe("general/auto_quantize/nvfp4_fp8_kl_div_at_5p4bits")
    with torch.no_grad():
        autoquant_utils._get_autoquant_memory_probe(recipe)(model, torch.ones((1, 8, 8)))
    assert outputs and all(torch.isfinite(logits).all() for logits in outputs)


def test_autoquant_recipe_builds_mtq_inputs(autoquant_utils):
    """The recipe path maps an AutoQuantizeConfig to the expected mtq.auto_quantize inputs."""
    args = SimpleNamespace(kv_cache_qformat="none")
    aq = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)

    # The shared base cost-excluded unit is spliced into every general AutoQuantize recipe, so it
    # reaches mtq under constraints.cost (VL vision tower / MTP out of the bit-budget denominator).
    assert inputs["constraints"] == {
        "effective_bits": 5.4,
        "cost_model": "weight",
        "cost": {"excluded_module_name_patterns": ["*visual*", "*mtp*", "*vision_tower*"]},
    }
    assert inputs["kv_cache_quant_cfg"] is None
    assert inputs["method"] == "gradient"
    assert "method_options" not in inputs
    assert inputs["score_size"] == 128
    assert inputs["fixed_quantization_config"] is None
    assert inputs["module_search_spaces"] == []
    # disabled_layers come straight from the recipe (no model introspection).
    assert inputs["disabled_layers"] == aq.disabled_layers
    assert "*output_layer*" in inputs["disabled_layers"]
    # Candidates resolve to the exact preset dicts mtq expects (preset identity preserved).
    assert inputs["quantization_formats"][0] == QUANT_CFG_CHOICES["nvfp4"]
    assert inputs["quantization_formats"][1] == QUANT_CFG_CHOICES["fp8"]


def test_kv_autoquant_recipe_builds_kv_search_inputs(autoquant_utils):
    args = SimpleNamespace(kv_cache_qformat="fp8_cast")
    aq = load_recipe("general/auto_quantize/kv_fp8_nvfp4_cast_kl_div_at_5p4bits").auto_quantize
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["search_domain"] == "kv_cache"
    assert inputs["constraints"] == {"effective_bits": 5.4, "cost_model": "kv_cache"}
    assert inputs["method"] == "kl_div"
    assert [config["effective_bits"] for config in inputs["quantization_formats"]] == [8.0, 4.5]
    assert aq.cost_excluded_layers == []
    assert "*mtp*" in inputs["disabled_layers"]
    assert "kv_cache_quant_cfg" not in inputs


def test_followup_kv_autoquant_suppresses_uniform_kv_fallback(autoquant_utils):
    args = SimpleNamespace(kv_cache_qformat="fp8_cast")
    aq = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize

    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args, allow_uniform_kv=False)

    assert inputs["kv_cache_quant_cfg"] is None


def test_fixed_ptq_kv_precheck_does_not_widen_scoped_gemm_rule(autoquant_utils):
    fixed = QuantizeConfig(
        quant_cfg=[
            {
                "quantizer_name": "model.layers.*.mlp.*",
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
            }
        ],
        algorithm="max",
    )

    assert not autoquant_utils._quantize_config_explicitly_enables_kv(fixed.model_dump())


@pytest.mark.parametrize(
    "kv_pattern",
    [
        "model.layers.*.self_attn.*[kv]_bmm_quantizer",
        "*self_attn*k_bmm_quantizer",
        "*.language_model.*.attention.*_bmm_quantizer",
        "*k_bmm*",
        "*self_attn.*",
        "*[kv]_bmm*",
    ],
)
def test_fixed_ptq_kv_precheck_detects_scoped_kv_rules(autoquant_utils, kv_pattern):
    fixed = QuantizeConfig(
        quant_cfg=[
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": kv_pattern,
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
            },
            {
                "parent_class": "nn.Embedding",
                "quantizer_name": "*",
                "enable": False,
            },
        ],
        algorithm="max",
    )

    assert autoquant_utils._quantize_config_explicitly_enables_kv(fixed.model_dump())


def test_fixed_ptq_kv_precheck_detects_parent_scoped_kv_rule(autoquant_utils):
    fixed = QuantizeConfig(
        quant_cfg=[
            {"quantizer_name": "*", "enable": False},
            {
                "parent_class": "LlamaAttention",
                "quantizer_name": "*_bmm_quantizer",
                "cfg": {"num_bits": (4, 3), "constant_amax": 1.0},
            },
        ],
        algorithm="max",
    )

    assert autoquant_utils._quantize_config_explicitly_enables_kv(fixed.model_dump())


def test_kv_autoquant_kl_excludes_padding_positions(autoquant_utils):
    logits = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    attention_mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]])

    selected = autoquant_utils._select_unpadded_logits(logits, {"attention_mask": attention_mask})

    assert torch.equal(selected, logits[attention_mask.bool()])


def test_kv_autoquant_kl_rejects_misaligned_attention_mask(autoquant_utils):
    with pytest.raises(ValueError, match="matching token dimensions"):
        autoquant_utils._select_unpadded_logits(
            torch.zeros(2, 4, 3), {"attention_mask": torch.ones(2, 3)}
        )


@pytest.mark.parametrize(
    ("search_domain", "expected_shape"),
    [("weight", (2, 4, 3)), ("kv_cache", (4, 3))],
)
def test_kl_padding_exclusion_is_scoped_to_kv_autoquant(
    autoquant_utils, monkeypatch, search_domain, expected_shape
):
    inputs = {
        "search_domain": search_domain,
        "constraints": {"effective_bits": 8.0},
        "quantization_formats": [],
        "fixed_quantization_config": None,
        "module_search_spaces": [],
        "disabled_layers": [],
        "kv_cache_quant_cfg": None,
        "method": "kl_div",
        "score_size": 1,
    }
    monkeypatch.setattr(
        autoquant_utils, "_mtq_inputs_from_auto_quantize_config", lambda *_args, **_kwargs: inputs
    )
    logits = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    batch = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [0, 1, 1, 0]]),
    }

    class Model(torch.nn.Module):
        def forward(self, **_kwargs):
            return SimpleNamespace(logits=logits, loss=torch.tensor(0.0))

    observed = {}

    def fake_auto_quantize(search_model, **kwargs):
        observed["shape"] = tuple(kwargs["forward_step"](search_model, batch).shape)
        return search_model, {}

    monkeypatch.setattr(autoquant_utils.mtq, "auto_quantize", fake_auto_quantize)
    args = SimpleNamespace(
        calib_with_images=False,
        inference_pipeline_parallel=1,
        use_fsdp2=False,
        batch_size=1,
        auto_quantize_checkpoint=None,
    )
    model = Model()

    autoquant_utils.auto_quantize(args, model, [batch], SimpleNamespace(), full_model=model)

    assert observed["shape"] == expected_shape


@pytest.mark.parametrize("domain", ["weight", "kv_cache"])
def test_autoquant_retains_fsdp2_warning(autoquant_utils, monkeypatch, domain):
    model = torch.nn.Module()
    inputs = {
        "search_domain": domain,
        "constraints": {"effective_bits": 8.0},
        "quantization_formats": [],
        "fixed_quantization_config": None,
        "module_search_spaces": [],
        "disabled_layers": [],
        "kv_cache_quant_cfg": None,
        "method": "gradient" if domain == "weight" else "kl_div",
        "score_size": 8,
    }
    monkeypatch.setattr(
        autoquant_utils, "_mtq_inputs_from_auto_quantize_config", lambda *_args, **_kwargs: inputs
    )
    observed = {}

    def search(search_model, **kwargs):
        observed.update(kwargs)
        return search_model, {}

    monkeypatch.setattr(autoquant_utils.mtq, "auto_quantize", search)
    monkeypatch.setattr(autoquant_utils, "world_size", lambda: 2)
    args = SimpleNamespace(
        calib_with_images=False,
        inference_pipeline_parallel=1,
        use_fsdp2=True,
        batch_size=1,
        auto_quantize_checkpoint=None,
    )

    with pytest.warns(UserWarning, match="experimental"):
        assert autoquant_utils.auto_quantize(args, model, [None] * 8, SimpleNamespace()) is model
    assert observed["num_calib_steps"] == 8
    assert observed["num_score_steps"] == 4


def test_recipe_distinguishes_weight_and_kv_autoquant(autoquant_utils):
    assert autoquant_utils._recipe_is_kv_auto_quantize(
        "general/auto_quantize/kv_fp8_nvfp4_cast_kl_div_at_5p4bits"
    )
    assert autoquant_utils._recipe_is_kv_auto_quantize(
        "general/auto_quantize/nvfp4_fp8_gradient_then_kv_fp8_nvfp4_cast_kl_div_at_5p4bits"
    )
    assert not autoquant_utils._recipe_is_kv_auto_quantize(
        "general/auto_quantize/nvfp4_fp8_at_5p4bits"
    )


def test_autoquant_recipe_forwards_aumann_shapley_options(autoquant_utils):
    args = SimpleNamespace(kv_cache_qformat="none")
    aq = _validated_autoquant_config(
        auto_quantize_method="aumann_shapley",
        method_options={"num_path_nodes": 3, "damage_link": "additive"},
    )
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["constraints"]["effective_bits"] == 5.4
    assert inputs["method"] == {
        "method": "aumann_shapley",
        "num_path_nodes": 3,
        "damage_link": "additive",
    }
    assert "method_options" not in inputs


def test_autoquant_recipe_damage_bound_omits_default_effective_bits(autoquant_utils):
    args = SimpleNamespace(kv_cache_qformat="none")
    aq = _validated_autoquant_config(
        constraints={},
        cost_excluded_layers=[],
        auto_quantize_method="aumann_shapley",
        method_options={"max_predicted_damage": 0.05},
    )
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["constraints"] == {"cost_model": "weight"}
    assert inputs["method"] == {
        "method": "aumann_shapley",
        "max_predicted_damage": 0.05,
    }


@pytest.mark.parametrize(
    ("method", "method_options", "loss_func_is_none"),
    [
        ("kl_div", None, False),
        ("aumann_shapley", {"num_path_nodes": 3, "damage_link": "additive"}, True),
    ],
)
def test_autoquant_label_free_recipe_calls_mtq(
    autoquant_utils, monkeypatch, method, method_options, loss_func_is_none
):
    args = SimpleNamespace(
        calib_with_images=False,
        inference_pipeline_parallel=1,
        use_fsdp2=False,
        batch_size=1,
        kv_cache_qformat="none",
    )
    aq = _validated_autoquant_config(
        auto_quantize_method=method,
        method_options=method_options,
    )
    batch = {"input_ids": torch.tensor([[1.0, 2.0]])}

    class LogitsModel(torch.nn.Module):
        def forward(self, input_ids):
            return SimpleNamespace(logits=input_ids + 1)

    captured = {}

    def fake_auto_quantize(model, **kwargs):
        captured.update(kwargs)
        assert torch.equal(kwargs["forward_step"](model, batch), batch["input_ids"] + 1)
        return model, {}

    monkeypatch.setattr(autoquant_utils.mtq, "auto_quantize", fake_auto_quantize)
    model = LogitsModel()

    assert autoquant_utils.auto_quantize(args, model, [batch], aq) is model
    expected_method = {"method": method, **method_options} if method_options else method
    assert captured["method"] == expected_method
    assert "method_options" not in captured
    assert (captured["loss_func"] is None) is loss_func_is_none
    assert captured["constraints"]["effective_bits"] == 5.4


def test_autoquant_recipe_cost_excluded_layers_map_into_cost(autoquant_utils):
    """Top-level cost_excluded_layers maps to the mtq constraints.cost.excluded_module_name_patterns
    key (distinct from disabled_layers), so a cost-exclusion recipe matches the nested mtq dict."""
    args = SimpleNamespace(kv_cache_qformat="none")
    aq = load_recipe(
        "model_type/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_at_6p0bits-active_moe"
    ).auto_quantize
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)

    # cost-exclusion is hoisted to a sibling of disabled_layers but still reaches the mtq cost dict.
    assert aq.cost_excluded_layers == ["*visual*", "*mtp*", "*vision_tower*"]
    assert inputs["constraints"]["cost"] == {
        "active_moe_expert_ratio": 0.03125,
        "excluded_module_name_patterns": ["*visual*", "*mtp*", "*vision_tower*"],
    }
    # The two exclusions are independent: cost-excluded patterns are also disabled here, but the
    # roles (cost-accounting vs search) are tracked separately.
    assert "*visual*" in inputs["disabled_layers"]


def test_autoquant_recipe_maps_module_search_spaces(autoquant_utils):
    """Fixed PTQ baseline and explicit recipe candidates map to mtq inputs."""
    args = SimpleNamespace(kv_cache_qformat="none")
    recipe = load_recipe(
        "model_type/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_module_spaces_at_6p0bits-active_moe"
    )
    inputs = autoquant_utils._mtq_inputs_from_auto_quantize_config(
        recipe.auto_quantize, args, fixed_quantize_config=recipe.quantize
    )
    model_ptq = load_recipe("model_type/qwen3_5_moe/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast")

    assert inputs["quantization_formats"] == []
    assert inputs["fixed_quantization_config"] == model_ptq.quantize.model_dump()
    (searched,) = inputs["module_search_spaces"]
    assert searched["module_name_patterns"] == [
        "*mlp.shared_expert*",
        "*linear_attn*",
        "*self_attn*",
        "*lm_head*",
    ]
    assert searched["quantization_formats"] == [
        QUANT_CFG_CHOICES["w4a16_nvfp4"],
        QUANT_CFG_CHOICES["fp8"],
    ]
    assert searched["allow_no_quant"] is False


def test_autoquant_rejects_non_export_safe_candidate(autoquant_utils):
    """A candidate that resolves to a preset outside the export-safe set is rejected before search."""
    args = SimpleNamespace(kv_cache_qformat="none")
    non_safe = next(
        k for k in QUANT_CFG_CHOICES if k not in autoquant_utils._AUTO_QUANTIZE_QFORMATS
    )
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[
            QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]),
            QuantizeConfig(**QUANT_CFG_CHOICES[non_safe]),
        ],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_warns_on_custom_candidate(autoquant_utils):
    """A candidate matching no shipped preset can't be export-verified, so it warns (not blocks)."""
    args = SimpleNamespace(kv_cache_qformat="none")
    custom = QuantizeConfig(quant_cfg=[{"quantizer_name": "*", "enable": False}])
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), custom],
    )
    with pytest.warns(UserWarning, match="export compatibility cannot be verified"):
        autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_export_guard_not_bypassed_by_effective_bits(autoquant_utils):
    """A non-export-safe preset can't dodge the guard by adding a cost-only effective_bits override."""
    args = SimpleNamespace(kv_cache_qformat="none")
    non_safe = next(
        k for k in QUANT_CFG_CHOICES if k not in autoquant_utils._AUTO_QUANTIZE_QFORMATS
    )
    tampered = QuantizeConfig(**{**QUANT_CFG_CHOICES[non_safe], "effective_bits": 4.5})
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=5.4),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), tampered],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        autoquant_utils._mtq_inputs_from_auto_quantize_config(aq, args)


@pytest.mark.parametrize("automatic", [False, True, "layerwise"])
@pytest.mark.skipif(sys.platform == "win32", reason="Calibration workers require POSIX locks")
def test_hf_candidate_worker_and_coordinator_use_public_api(
    autoquant_utils, monkeypatch, tmp_path, automatic
):
    if automatic:
        tmp_path = tmp_path / "work" / "search"
        tmp_path.mkdir(parents=True)
    torch.manual_seed(19)
    source = get_tiny_llama(num_hidden_layers=1).eval()
    inputs = torch.randint(0, source.config.vocab_size, (1, 8))
    batches = [{"input_ids": inputs, "labels": inputs.clone()}]
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=8.0),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"])],
        score_size=1,
    )
    if automatic == "layerwise":
        aq.candidate_formats[0].algorithm = {"method": "max", "layerwise": {"enable": True}}
        aq.disabled_layers = ["*lm_head*"]
        aq.cost_excluded_layers = ["*lm_head*"]
        source.config._name_or_path = str(tmp_path / "source")
        source.save_pretrained(source.config._name_or_path)
        source = type(source).from_pretrained(source.config._name_or_path).eval()
    args = SimpleNamespace(
        calib_with_images=False,
        inference_pipeline_parallel=1,
        use_fsdp2=False,
        batch_size=1,
        kv_cache_qformat="none",
        auto_quantize_checkpoint=None,
        auto_quantize_calibration_dir=str(tmp_path / "candidates"),
        auto_quantize_calibration_candidate=0,
    )
    if automatic:
        recipe = ModelOptAutoQuantizeRecipe(auto_quantize=aq)
        payload = recipe.model_dump()
        payload["metadata"]["recipe_type"] = recipe.recipe_type.value
        args.recipe = str(tmp_path / "recipe.yaml")
        Path(args.recipe).write_text(yaml.safe_dump(payload))
        aq = load_recipe(args.recipe).auto_quantize
        args.auto_quantize_pipeline = True
        args.auto_quantize_resources = {
            "candidate_bytes": 1024**2,
            "checkpoint_bytes": 2 * 1024**2,
            "metadata_bytes": 1024**2,
        }
        args.auto_quantize_assignment = str(tmp_path / "assignment.json")
    if automatic == "layerwise":
        args.auto_quantize_final_layerwise = True
        args.auto_quantize_concurrency = 1
        args.auto_quantize_resources.update(
            activation_bytes=1024**2,
            final_checkpoint_bytes=2 * 1024**2,
            export_bytes=2 * 1024**2,
            host_bytes=1024**3,
        )
        args.export_path = str(tmp_path / "export")
        args.auto_quantize_export_available_bytes = 2 * 1024**2
    autoquant_utils.auto_quantize(args, copy.deepcopy(source), batches, aq)
    args.auto_quantize_calibration_candidate = None
    args.auto_quantize_checkpoint = str(tmp_path / "search")
    model = autoquant_utils.auto_quantize(
        args, copy.deepcopy(source), batches, aq, checkpoint=str(tmp_path / "search")
    )
    assert (tmp_path / "search" / "rank0.pth").is_file()
    if automatic:
        assignment = json.loads(Path(args.auto_quantize_assignment).read_text())
        assert assignment["provenance"]["source"]
        assert all(
            yaml.safe_load(group["configuration_yaml"])["algorithm"]
            == aq.candidate_formats[0].algorithm
            for group in assignment["groups"]
            if group["format_id"] is not None
        )
        assert any(group["final_layerwise"] for group in assignment["groups"]) == (
            automatic == "layerwise"
        )
    if automatic == "layerwise":
        assert args.auto_quantize_reload
        args.auto_quantize_reload = False
        pipeline = importlib.import_module("autoquant_pipeline")
        exports = importlib.import_module("autoquant_export")
        final = importlib.import_module("autoquant_final")
        with pipeline.pipeline_run(args):
            model = autoquant_utils.auto_quantize(args, copy.deepcopy(source), batches, aq)
            assert not args.auto_quantize_reload
            assert (tmp_path / "final/layer_0000").is_dir()
            with torch.no_grad():
                output = model(**batches[0]).logits
                assert torch.isfinite(output).all()
            retained = {
                p: (p.read_bytes(), p.stat().st_mtime_ns)
                for p in (tmp_path / "final/layer_0000").iterdir()
            }
            with pytest.raises(ValueError):
                final.reclaim_final_checkpoint(args)
            model = autoquant_utils.auto_quantize(args, copy.deepcopy(source), batches, aq)
            with torch.no_grad():
                torch.testing.assert_close(model(**batches[0]).logits, output, rtol=0, atol=0)
            assert retained == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in retained}
            exports.export_pipeline(args, model)
            with monkeypatch.context() as failure:

                def interrupt_delete(*_args, **_kwargs):
                    raise OSError("interrupted reclamation")

                failure.setattr(final.shutil, "rmtree", interrupt_delete)
                with pytest.raises(OSError):
                    final.reclaim_final_checkpoint(args)
            assert (tmp_path / ".final-reclaimed").is_dir()
            assert (tmp_path / "final.json").is_file()
            final.reclaim_final_checkpoint(args)
            assert not (tmp_path / ".final-reclaimed").exists()
        assert not (tmp_path / "final").exists()
        assert not (tmp_path / "final.json").exists()
    else:
        with torch.no_grad():
            assert torch.isfinite(model(**batches[0]).logits).all()
