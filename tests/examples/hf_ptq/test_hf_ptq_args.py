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

import getpass
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from modelopt.recipe import load_recipe
from modelopt.recipe.config import AutoQuantizeConfig, AutoQuantizeConstraints
from modelopt.recipe.presets import QUANT_CFG_CHOICES
from modelopt.torch.quantization.config import QuantizeConfig

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


def _import_hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


@pytest.fixture
def example_utils(monkeypatch):
    """The MLflow wiring lives beside the other hf_ptq helpers."""
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


def _validated_autoquant_config(**updates):
    config = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize
    values = {name: getattr(config, name) for name in type(config).model_fields}
    return AutoQuantizeConfig.model_validate({**values, **updates})


def test_autoquant_recipe_builds_mtq_inputs(monkeypatch):
    """The recipe path maps an AutoQuantizeConfig to the expected mtq.auto_quantize inputs."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = load_recipe("general/auto_quantize/nvfp4_fp8_at_5p4bits").auto_quantize
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    # The shared base cost-excluded unit is spliced into every general AutoQuantize recipe, so it
    # reaches mtq under constraints.cost (VL vision tower / MTP out of the bit-budget denominator).
    assert inputs["constraints"] == {
        "effective_bits": 5.4,
        "cost_model": "weight",
        "cost": {"excluded_module_name_patterns": ["*visual*", "*mtp*", "*vision_tower*"]},
    }
    assert inputs["kv_cache_quant_cfg"] is None
    assert inputs["method"] == "gradient"
    assert inputs["method_options"] is None
    assert inputs["score_size"] == 128
    assert inputs["fixed_quantization_config"] is None
    assert inputs["module_search_spaces"] == []
    # disabled_layers come straight from the recipe (no model introspection).
    assert inputs["disabled_layers"] == aq.disabled_layers
    assert "*output_layer*" in inputs["disabled_layers"]
    # Candidates resolve to the exact preset dicts mtq expects (preset identity preserved).
    assert inputs["quantization_formats"][0] == QUANT_CFG_CHOICES["nvfp4"]
    assert inputs["quantization_formats"][1] == QUANT_CFG_CHOICES["fp8"]


def test_autoquant_recipe_forwards_aumann_shapley_options(monkeypatch):
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = _validated_autoquant_config(
        auto_quantize_method="aumann_shapley",
        method_options={"num_path_nodes": 3, "damage_link": "additive"},
    )
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["constraints"]["effective_bits"] == 5.4
    assert inputs["method"] == "aumann_shapley"
    assert inputs["method_options"] == {"num_path_nodes": 3, "damage_link": "additive"}


def test_autoquant_recipe_damage_bound_omits_default_effective_bits(monkeypatch):
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = _validated_autoquant_config(
        constraints={},
        cost_excluded_layers=[],
        auto_quantize_method="aumann_shapley",
        method_options={"max_predicted_damage": 0.05},
    )
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    assert inputs["constraints"] == {"cost_model": "weight"}
    assert inputs["method_options"] == {"max_predicted_damage": 0.05}


@pytest.mark.parametrize(
    ("method", "method_options", "loss_func_is_none"),
    [
        ("kl_div", None, False),
        ("aumann_shapley", {"num_path_nodes": 3, "damage_link": "additive"}, True),
    ],
)
def test_autoquant_label_free_recipe_calls_mtq(
    monkeypatch, method, method_options, loss_func_is_none
):
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "dummy",
        "--kv_cache_qformat",
        "none",
        "--batch_size",
        "1",
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

    monkeypatch.setattr(hf_ptq.mtq, "auto_quantize", fake_auto_quantize)
    model = LogitsModel()

    assert hf_ptq.auto_quantize(args, model, [batch], aq) is model
    assert captured["method"] == method
    assert captured["method_options"] == method_options
    assert (captured["loss_func"] is None) is loss_func_is_none
    assert captured["constraints"]["effective_bits"] == 5.4


def test_autoquant_recipe_cost_excluded_layers_map_into_cost(monkeypatch):
    """Top-level cost_excluded_layers maps to the mtq constraints.cost.excluded_module_name_patterns
    key (distinct from disabled_layers), so a cost-exclusion recipe matches the nested mtq dict."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    aq = load_recipe(
        "huggingface/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_at_6p0bits-active_moe"
    ).auto_quantize
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)

    # cost-exclusion is hoisted to a sibling of disabled_layers but still reaches the mtq cost dict.
    assert aq.cost_excluded_layers == ["*visual*", "*mtp*", "*vision_tower*"]
    assert inputs["constraints"]["cost"] == {
        "active_moe_expert_ratio": 0.03125,
        "excluded_module_name_patterns": ["*visual*", "*mtp*", "*vision_tower*"],
    }
    # The two exclusions are independent: cost-excluded patterns are also disabled here, but the
    # roles (cost-accounting vs search) are tracked separately.
    assert "*visual*" in inputs["disabled_layers"]


def test_autoquant_recipe_maps_module_search_spaces(monkeypatch):
    """Fixed PTQ baseline and explicit recipe candidates map to mtq inputs."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    recipe = load_recipe(
        "huggingface/qwen3_6_moe/auto_quantize/w4a16_nvfp4_fp8_module_spaces_at_6p0bits-active_moe"
    )
    inputs = hf_ptq._mtq_inputs_from_auto_quantize_config(
        recipe.auto_quantize, args, fixed_quantize_config=recipe.quantize
    )
    model_ptq = load_recipe("huggingface/qwen3_5_moe/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast")

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


def test_autoquant_rejects_non_export_safe_candidate(monkeypatch):
    """A candidate that resolves to a preset outside the export-safe set is rejected before search."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    non_safe = next(k for k in QUANT_CFG_CHOICES if k not in hf_ptq._AUTO_QUANTIZE_QFORMATS)
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[
            QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]),
            QuantizeConfig(**QUANT_CFG_CHOICES[non_safe]),
        ],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_warns_on_custom_candidate(monkeypatch):
    """A candidate matching no shipped preset can't be export-verified, so it warns (not blocks)."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    custom = QuantizeConfig(quant_cfg=[{"quantizer_name": "*", "enable": False}])
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=4.8),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), custom],
    )
    with pytest.warns(UserWarning, match="export compatibility cannot be verified"):
        hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)


def test_autoquant_export_guard_not_bypassed_by_effective_bits(monkeypatch):
    """A non-export-safe preset can't dodge the guard by adding a cost-only effective_bits override."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "dummy", "--kv_cache_qformat", "none"
    )
    non_safe = next(k for k in QUANT_CFG_CHOICES if k not in hf_ptq._AUTO_QUANTIZE_QFORMATS)
    tampered = QuantizeConfig(**{**QUANT_CFG_CHOICES[non_safe], "effective_bits": 4.5})
    aq = AutoQuantizeConfig(
        constraints=AutoQuantizeConstraints(effective_bits=5.4),
        candidate_formats=[QuantizeConfig(**QUANT_CFG_CHOICES["fp8"]), tampered],
    )
    with pytest.raises(ValueError, match="not supported for unified checkpoint export"):
        hf_ptq._mtq_inputs_from_auto_quantize_config(aq, args)


def test_mlflow_flag_defaults_the_experiment_name(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "/models/Qwen3-0.6B",
        "--recipe",
        "general/ptq/nvfp4_default-kv_fp8_cast",
        "--mlflow",
        "https://mlflow.example.com/",
    )

    assert args.mlflow == "https://mlflow.example.com"
    assert args.mlflow_experiment == "tester/hf_ptq/Qwen3-0.6B-nvfp4_default-kv_fp8_cast"
    assert args.mlflow_run_name is None


def test_mlflow_experiment_falls_back_to_qformat_without_a_recipe(monkeypatch):
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    _, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "nvidia/Llama-3.3-70B-Instruct",
        "--qformat",
        "nvfp4",
        "--mlflow",
        "https://mlflow.example.com",
    )

    assert args.mlflow_experiment == "tester/hf_ptq/Llama-3.3-70B-Instruct-nvfp4"


def test_mlflow_is_off_by_default(monkeypatch):
    _, args = _parse_hf_ptq_args(monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B")

    assert args.mlflow is None
    assert args.mlflow_experiment is None


def test_mlflow_rejects_a_bad_tracking_uri(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_hf_ptq_args(
            monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B", "--mlflow", "not-a-url"
        )


def test_mlflow_requires_a_value(monkeypatch):
    """The bare form meant "use $MLFLOW_TRACKING_URI", which is now what happens with no flag
    at all, so it is gone and argparse asks for the value."""
    with pytest.raises(SystemExit):
        _parse_hf_ptq_args(monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B", "--mlflow")


def test_the_environment_alone_enables_tracking(monkeypatch):
    """MLFLOW_TRACKING_URI is MLflow's own variable, so exporting it opts in on its own."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com/")
    monkeypatch.setattr(getpass, "getuser", lambda: "tester")
    _, args = _parse_hf_ptq_args(monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B")

    assert args.mlflow == "https://mlflow.example.com"
    assert args.mlflow_required is False
    assert args.mlflow_experiment == "tester/hf_ptq/Qwen3-0.6B-fp8"


def test_an_explicit_flag_beats_the_environment(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://from-env.example.com/")
    _, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "/m/x", "--mlflow", "https://from-flag.example.com"
    )

    assert args.mlflow == "https://from-flag.example.com"
    assert args.mlflow_required is True


def test_an_unusable_environment_uri_warns_instead_of_failing(monkeypatch):
    """The variable is commonly exported for other tooling, so it must not fail a run --
    unlike an explicit --mlflow, which is an unambiguous request."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///local/mlruns")

    with pytest.warns(UserWarning, match="Ignoring MLFLOW_TRACKING_URI"):
        _, args = _parse_hf_ptq_args(monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B")

    assert args.mlflow is None

    with pytest.raises(SystemExit):  # the same value passed explicitly still fails
        _parse_hf_ptq_args(
            monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B", "--mlflow", "file:///local/mlruns"
        )


def test_mlflow_provenance_is_not_logged_as_a_param(monkeypatch, example_utils):
    """mlflow_required describes the tracking setup, not the quantization."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com/")
    hf_ptq, args = _parse_hf_ptq_args(monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B")
    args.dist_state = SimpleNamespace(is_main=True, world_size=1)

    params, _ = example_utils._mlflow_run_inputs(args)

    assert "mlflow_required" not in params


def test_mlflow_run_inputs_carry_the_resolved_recipe(monkeypatch, example_utils):
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "/models/Qwen3-0.6B",
        "--recipe",
        "general/ptq/nvfp4_default-kv_fp8_cast",
    )
    args.dist_state = SimpleNamespace(is_main=True, world_size=1)

    params, texts = example_utils._mlflow_run_inputs(args)

    assert params["pyt_ckpt_path"] == "/models/Qwen3-0.6B"
    assert params["recipe"] == "general/ptq/nvfp4_default-kv_fp8_cast"
    # $imports are expanded, so the artifact stands alone.
    recipe = yaml.safe_load(texts["recipe/resolved_recipe.yaml"])
    assert recipe["metadata"]["recipe_type"] == "ptq"
    assert recipe["quantize"]["quant_cfg"]


def test_mlflow_run_inputs_omit_the_recipe_when_unused(monkeypatch, example_utils):
    hf_ptq, args = _parse_hf_ptq_args(monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B")
    args.dist_state = SimpleNamespace(is_main=True, world_size=1)

    params, texts = example_utils._mlflow_run_inputs(args)

    assert texts == {}
    assert params["recipe"] is None


def test_mlflow_run_outputs_name_the_summaries(monkeypatch, example_utils):
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B", "--export_path", "/tmp/out"
    )

    files = example_utils._mlflow_run_outputs(args)

    assert files["summary/quant_summary.txt"] == Path("/tmp/out/.quant_summary.txt")
    assert files["summary/moe.html"] == Path("/tmp/out/.moe.html")


def test_untracked_runs_do_not_gather_mlflow_inputs(monkeypatch, example_utils):
    """Without --mlflow the recipe must not be re-read: it is parsed again in quantize_main,
    and the extra load prints a second '[load_recipe] loading:' line on every default run."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "/models/Qwen3-0.6B",
        "--recipe",
        "general/ptq/nvfp4_default-kv_fp8_cast",
    )
    args.dist_state = SimpleNamespace(is_main=True, world_size=1)
    calls = []
    monkeypatch.setattr(example_utils, "_mlflow_run_inputs", lambda a: calls.append(a) or ({}, {}))

    with example_utils.mlflow_run(args):
        pass

    assert not example_utils._mlflow_logger(args).enabled
    assert calls == []


def test_non_main_ranks_do_not_open_a_run(monkeypatch, example_utils):
    """Under torchrun only rank 0 uploads, so the other ranks must not touch the server."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "/models/Qwen3-0.6B",
        "--mlflow",
        "https://mlflow.example.com",
    )
    args.dist_state = SimpleNamespace(is_main=False, world_size=8)
    calls = []
    monkeypatch.setattr(example_utils, "_mlflow_run_inputs", lambda a: calls.append(a) or ({}, {}))

    with example_utils.mlflow_run(args):
        pass

    assert not example_utils._mlflow_logger(args).enabled
    assert calls == []


def test_mlflow_params_track_every_cli_argument(monkeypatch, example_utils):
    """Params are derived from the parsed args, so a new flag needs no bookkeeping here."""
    hf_ptq, args = _parse_hf_ptq_args(monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B")
    args.dist_state = SimpleNamespace(is_main=True, world_size=4)

    params, _ = example_utils._mlflow_run_inputs(args)

    tracked = set(vars(args)) - example_utils._MLFLOW_NON_PARAM_ARGS
    assert tracked <= set(params)
    # The tracking settings describe the destination, not the run, and dist_state is an object.
    assert not {"mlflow", "mlflow_experiment", "mlflow_run_name", "dist_state"} & set(params)
    assert params["world_size"] == 4
    # A flag added to the parser later is picked up without editing _mlflow_run_inputs.
    args.some_future_flag = "future"
    assert example_utils._mlflow_run_inputs(args)[0]["some_future_flag"] == "future"


def test_mlflow_tags_identify_the_produced_checkpoint(monkeypatch, example_utils, tmp_path):
    """checkpoint_path must name what the run *writes*: an evaluation is pointed at the
    exported checkpoint, so tagging the input would never join the two."""
    export = tmp_path / "exports" / "Qwen3-0.6B-nvfp4"
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B", "--export_path", str(export)
    )

    assert example_utils._mlflow_run_tags(args) == {
        "model": "Qwen3-0.6B",
        "checkpoint_path": str(export),
        "source_checkpoint_path": "/models/Qwen3-0.6B",
    }


def test_mlflow_checkpoint_tag_is_absolute(monkeypatch, example_utils):
    """--export_path defaults to a relative path, which is useless as a join key."""
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch, "--pyt_ckpt_path", "/models/Qwen3-0.6B", "--export_path", "exported_model"
    )

    assert Path(example_utils._mlflow_run_tags(args)["checkpoint_path"]).is_absolute()
