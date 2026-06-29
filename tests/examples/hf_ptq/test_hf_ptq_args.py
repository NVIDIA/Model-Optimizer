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

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


def _import_hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


def _import_example_utils(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("example_utils")


def _parse_hf_ptq_args(monkeypatch, *args):
    hf_ptq = _import_hf_ptq(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py", *args])
    parsed_args = hf_ptq.parse_args()
    parsed_args.dataset = (
        parsed_args.dataset.split(",")
        if isinstance(parsed_args.dataset, str)
        else parsed_args.dataset
    )
    parsed_args.calib_size = [int(num_sample) for num_sample in parsed_args.calib_size.split(",")]
    return hf_ptq, parsed_args


def test_parse_args_rejects_autoquant_image_calibration(monkeypatch):
    hf_ptq = _import_hf_ptq(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf_ptq.py",
            "--pyt_ckpt_path",
            "nemotron-vl",
            "--auto_quantize_bits",
            "5.0",
            "--calib_with_images",
        ],
    )

    with pytest.raises(SystemExit) as error:
        hf_ptq.parse_args()

    assert error.value.code == 2


def test_load_model_keeps_nemotron_vl_text_calibration_for_autoquant(monkeypatch):
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "nemotron-vl",
        "--auto_quantize_bits",
        "5.0",
    )
    fake_model = SimpleNamespace(device="cpu")
    fake_tokenizer = SimpleNamespace(padding_side="right", pad_token="<pad>")

    monkeypatch.setattr(hf_ptq, "get_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(hf_ptq, "get_model_type", lambda model: "qwen2")
    monkeypatch.setattr(hf_ptq, "get_tokenizer", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(hf_ptq, "is_nemotron_vl", lambda model: True)

    full_model, language_model, _, _, _, tokenizer, _, _, _ = hf_ptq.load_model(args)

    assert args.calib_with_images is False
    assert full_model is fake_model
    assert language_model is fake_model
    assert tokenizer is fake_tokenizer


def test_qwen_autoquant_disabled_layers_are_scoped_to_qwen_models(monkeypatch):
    example_utils = _import_example_utils(monkeypatch)
    qwen_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))
    llama_model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    qwen_only_patterns = {
        "*shared_expert_gate*",
    }

    monkeypatch.setattr(example_utils, "is_multimodal_model", lambda model: False)

    qwen_disabled_layers = set(example_utils._get_auto_quantize_disabled_layers(qwen_model))
    llama_disabled_layers = set(example_utils._get_auto_quantize_disabled_layers(llama_model))

    assert qwen_only_patterns <= qwen_disabled_layers
    assert qwen_only_patterns.isdisjoint(llama_disabled_layers)


def test_autoquant_recipe_builds_canonical_mtq_inputs(monkeypatch):
    """Recipe input-building matches the CLI defaults it must stay equivalent to."""
    from modelopt.recipe.config import AutoQuantizeConfig, AutoQuantizeConstraints
    from modelopt.recipe.presets import QUANT_CFG_CHOICES
    from modelopt.torch.quantization.config import QuantizeConfig

    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "dummy",
        "--kv_cache_qformat",
        "none",
    )
    # Isolate the model-derived pieces so the test targets the recipe input-building.
    monkeypatch.setattr(hf_ptq, "_get_auto_quantize_disabled_layers", lambda m: ["*lm_head*"])
    monkeypatch.setattr(hf_ptq, "_get_auto_quantize_cost_excluded_patterns", lambda m: [])
    fake_model = SimpleNamespace()

    aq_config = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=6.0),
        candidate_formats=[
            QuantizeConfig(**QUANT_CFG_CHOICES["nvfp4"]),
            QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]),
        ],
    )
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq_config, args, fake_model)

    assert inputs["constraints"] == {"effective_bits": 6.0, "cost_model": "weight"}
    assert inputs["disabled_layers"] == ["*lm_head*"]
    assert inputs["kv_cache_quant_cfg"] is None
    assert inputs["method"] == "gradient"
    assert inputs["num_score_steps"] == 128
    # Candidates resolve to the exact preset dicts the CLI feeds mtq, so the search names
    # them identically (FP8_DEFAULT_CFG / NVFP4_DEFAULT_CFG) and checkpoints stay compatible.
    assert inputs["quantization_formats"][0] == QUANT_CFG_CHOICES["nvfp4"]
    assert inputs["quantization_formats"][1] == QUANT_CFG_CHOICES["fp8"]


def _recipe_config_from_cli_args(args):
    """Build the AutoQuantizeConfig a user would write to mirror the given CLI args."""
    from modelopt.recipe.config import AutoQuantizeConfig, AutoQuantizeConstraints, AutoQuantizeCost
    from modelopt.recipe.presets import QUANT_CFG_CHOICES
    from modelopt.torch.quantization.config import QuantizeConfig

    cost = None
    if args.auto_quantize_active_moe_expert_ratio is not None:
        cost = AutoQuantizeCost(active_moe_expert_ratio=args.auto_quantize_active_moe_expert_ratio)
    return AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(
            effective_bits=args.auto_quantize_bits,
            cost_model=args.auto_quantize_cost_model,
            cost=cost,
        ),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES[f]) for f in args.qformat.split(",")],
        auto_quantize_method=args.auto_quantize_method,
        num_score_steps=args.auto_quantize_score_size,
        # kv_cache omitted -> recipe path falls back to --kv_cache_qformat, like the CLI.
    )


def _cli_expected_mtq_inputs(hf_ptq, args, model):
    """Reconstruct the mtq.auto_quantize inputs the CLI helper builds from args.

    Uses the same building blocks the CLI helper uses (QUANT_CFG_CHOICES, the disabled/excluded
    helpers, KV presets), so it is the reference the recipe path must match field-for-field.
    """
    import copy

    from modelopt.recipe.presets import KV_CACHE_NONE, KV_QUANT_CFG_CHOICES, QUANT_CFG_CHOICES
    from modelopt.torch.quantization._auto_quantize_cost import EXCLUDED_MODULE_NAME_PATTERNS_KEY

    constraints = {
        "effective_bits": args.auto_quantize_bits,
        "cost_model": args.auto_quantize_cost_model,
    }
    cost = {}
    if args.auto_quantize_active_moe_expert_ratio is not None:
        cost["active_moe_expert_ratio"] = args.auto_quantize_active_moe_expert_ratio
    excluded = hf_ptq._get_auto_quantize_cost_excluded_patterns(model)
    if excluded:
        cost[EXCLUDED_MODULE_NAME_PATTERNS_KEY] = excluded
    if cost:
        constraints["cost"] = cost

    if args.kv_cache_qformat == KV_CACHE_NONE:
        kv = None
    else:
        kv = copy.deepcopy(KV_QUANT_CFG_CHOICES[args.kv_cache_qformat])

    return {
        "constraints": constraints,
        "quantization_formats": [QUANT_CFG_CHOICES[f] for f in args.qformat.split(",")],
        "disabled_layers": hf_ptq._get_auto_quantize_disabled_layers(model),
        "kv_cache_quant_cfg": kv,
        "method": args.auto_quantize_method,
        "num_score_steps": args.auto_quantize_score_size,
    }


@pytest.mark.parametrize(
    "cli_flags",
    [
        ["--qformat", "fp8,nvfp4", "--auto_quantize_bits", "6.0"],
        [
            "--qformat",
            "fp8,w4a16_nvfp4",
            "--auto_quantize_bits",
            "6.0",
            "--auto_quantize_cost_model",
            "active_moe",
            "--auto_quantize_active_moe_expert_ratio",
            "0.03125",
        ],
        ["--qformat", "fp8,nvfp4", "--auto_quantize_bits", "4.8", "--kv_cache_qformat", "fp8"],
        [
            "--qformat",
            "fp8,nvfp4",
            "--auto_quantize_bits",
            "5.0",
            "--auto_quantize_method",
            "kl_div",
        ],
    ],
)
def test_recipe_inputs_match_cli_inputs(monkeypatch, cli_flags):
    """Across the supported matrix, the recipe path feeds mtq the same inputs as the CLI."""
    hf_ptq, args = _parse_hf_ptq_args(monkeypatch, "--pyt_ckpt_path", "dummy", *cli_flags)
    monkeypatch.setattr(hf_ptq, "_get_auto_quantize_disabled_layers", lambda m: ["*lm_head*"])
    monkeypatch.setattr(hf_ptq, "_get_auto_quantize_cost_excluded_patterns", lambda m: [])
    model = SimpleNamespace()

    recipe_inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(
        _recipe_config_from_cli_args(args), args, model
    )
    cli_inputs = _cli_expected_mtq_inputs(hf_ptq, args, model)
    assert recipe_inputs == cli_inputs


def test_autoquant_cli_flags_have_recipe_mapping(monkeypatch):
    """Every autoquant spec CLI flag maps to a recipe field (or is intentionally runtime-only).

    Introspects the parsed args, so a newly added ``--auto_quantize_*`` flag that isn't mapped
    fails here — flagging that the recipe schema/dispatch needs updating.
    """
    from modelopt.recipe.config import AutoQuantizeConfig, AutoQuantizeConstraints, AutoQuantizeCost

    _, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "dummy",
        "--qformat",
        "fp8,nvfp4",
        "--auto_quantize_bits",
        "6.0",
    )
    spec_flags = {k for k in vars(args) if k.startswith("auto_quantize_")} | {
        "qformat",
        "kv_cache_qformat",
    }

    aq_fields = set(AutoQuantizeConfig.model_fields)
    constraint_fields = set(AutoQuantizeConstraints.model_fields)
    cost_fields = set(AutoQuantizeCost.model_fields)

    # CLI flag (args dest) -> True if covered by the recipe schema (or runtime-only by design).
    covered = {
        "auto_quantize_bits": "effective_bits" in constraint_fields,
        "auto_quantize_method": "auto_quantize_method" in aq_fields,
        "auto_quantize_score_size": "num_score_steps" in aq_fields,
        "auto_quantize_cost_model": "cost_model" in constraint_fields,
        "auto_quantize_active_moe_expert_ratio": "active_moe_expert_ratio" in cost_fields,
        "auto_quantize_checkpoint": True,  # runtime filesystem path, intentionally CLI-only
        "qformat": "candidate_formats" in aq_fields,
        "kv_cache_qformat": "kv_cache" in aq_fields,
    }
    unmapped = spec_flags - set(covered)
    assert not unmapped, (
        f"Unmapped autoquant CLI flags (add to recipe schema + mapping): {unmapped}"
    )
    assert all(covered.values()), f"A mapped recipe field is missing from the schema: {covered}"
