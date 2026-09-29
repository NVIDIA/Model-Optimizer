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
"""Smoke tests for ``modelopt.recipe.presets`` preset discovery.

Guards the eager import-time load shared by the PTQ example scripts: every preset
under the model/KV dirs must load into a usable ``quant_cfg`` dict, and the KV
``none`` sentinel must not collide with a discovered preset. A single malformed
preset YAML would otherwise break ``import modelopt.recipe.presets`` (and every
PTQ example).
"""

import argparse
import subprocess
import sys
import textwrap

import pytest

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe, presets
from modelopt.recipe.presets import RecipeSupersededAction
from modelopt.torch.opt.config_loader import BUILTIN_CONFIG_ROOT
from modelopt.torch.quantization.config import LocalHessianCalibConfig, QuantizeConfig
from modelopt.torch.quantization.ggml import (
    IQ1_S_BLOCK_SIZE,
    IQ1_S_EFFECTIVE_BITS,
    IQ2_XS_BLOCK_SIZE,
    IQ2_XS_EFFECTIVE_BITS,
    IQ2_XXS_BLOCK_SIZE,
    IQ2_XXS_EFFECTIVE_BITS,
)


def _yaml_basenames(subdir: str) -> set[str]:
    return {
        entry.name.rsplit(".", 1)[0]
        for entry in BUILTIN_CONFIG_ROOT.joinpath(subdir).iterdir()
        if entry.name.endswith((".yaml", ".yml"))
    }


@pytest.mark.parametrize(
    ("choices", "preset_dir"),
    [
        (presets.QUANT_CFG_CHOICES, presets.MODEL_QUANT_PRESET_DIR),
        (presets.KV_QUANT_CFG_CHOICES, presets.KV_QUANT_PRESET_DIR),
    ],
    ids=["model", "kv"],
)
def test_every_discovered_preset_loads(choices, preset_dir):
    # Configs load eagerly at import, so a malformed preset would already have raised.
    # Assert discovery is non-empty, covers every YAML on disk, and that each resolved
    # entry is a usable quant_cfg dict.
    basenames = _yaml_basenames(preset_dir)
    assert basenames, f"no preset YAMLs discovered under {preset_dir}"
    assert basenames <= set(choices), "a preset YAML is missing from the discovered choices"
    for name, cfg in choices.items():
        assert isinstance(cfg, dict), f"{name} did not resolve to a dict"
        assert "quant_cfg" in cfg, f"{name} is missing the 'quant_cfg' key"


def test_kv_none_sentinel_is_not_a_discovered_preset():
    # The scripts branch on ``kv_cache_qformat != KV_CACHE_NONE``; a real preset named
    # "none" would make that branch ambiguous.
    assert presets.KV_CACHE_NONE not in presets.KV_QUANT_CFG_CHOICES


def test_w4a16_nvfp4_preset_disables_vllm_marlin_incompatible_projections():
    disabled_quantizers = {
        entry["quantizer_name"]
        for entry in presets.QUANT_CFG_CHOICES["w4a16_nvfp4"]["quant_cfg"]
        if entry.get("enable") is False
    }

    assert {
        "*linear_attn.in_proj_a*",
        "*linear_attn.in_proj_b*",
        "*visual*",
        "*vision_tower*",
    } <= disabled_quantizers


def test_local_hessian_layerwise_is_scoped_to_the_model_recipe():
    # Layerwise calibration needs identifiable decoder layers, so it belongs to the model
    # recipe the published numbers come from -- not to the shared preset, which also backs
    # mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG and --qformat and therefore has to stay usable
    # on models with no decoder layers (a plain nn.Sequential, say).
    recipe = load_recipe("models/Qwen/Qwen3.8-27B/ptq/nvfp4_w4a4_mlp_fp8_attn_local_hessian")
    layerwise = LocalHessianCalibConfig(**recipe.quantize.model_dump()["algorithm"]).layerwise

    assert layerwise.enable
    assert layerwise.get_qdq_activations_from_prev_layer

    # ``--qformat`` has no argparse ``choices=``; hf_ptq.py gates it on membership in
    # QUANT_CFG_CHOICES, so this mapping is the CLI allowlist the preset basename lands in.
    qformat = "nvfp4_w4a4_weight_local_hessian"
    assert qformat in presets.QUANT_CFG_CHOICES
    preset_cfg = LocalHessianCalibConfig(**presets.QUANT_CFG_CHOICES[qformat]["algorithm"])

    assert not preset_cfg.layerwise.enable

    # The exported constant is a separately-loaded dict, so pin it rather than trusting it to
    # track the CLI mapping.
    exported = LocalHessianCalibConfig(**mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG["algorithm"])

    assert not exported.layerwise.enable


@pytest.mark.parametrize(
    ("recipe_name", "cfg_name"),
    [
        ("general/ptq/mxfp4_mlp_weight_only", "MXFP4_MLP_WEIGHT_ONLY_CFG"),
        ("general/ptq/nvfp4_mlp_weight_only", "NVFP4_MLP_WEIGHT_ONLY_CFG"),
    ],
)
def test_mlp_weight_only_recipe_matches_its_mtq_cfg(recipe_name, cfg_name):
    # examples/gpt-oss migrated from --quant_cfg <CFG> to --recipe <recipe>; pin the
    # equality so the recipe and the mtq constant cannot drift apart silently.
    recipe_cfg = load_recipe(recipe_name).quantize.model_dump(exclude_unset=True)
    mtq_cfg = QuantizeConfig(**getattr(mtq, cfg_name)).model_dump(exclude_unset=True)
    assert recipe_cfg == mtq_cfg


@pytest.mark.parametrize(
    ("qformat", "block_size", "effective_bits"),
    [
        ("iq1_s", IQ1_S_BLOCK_SIZE, IQ1_S_EFFECTIVE_BITS),
        ("iq2_xxs", IQ2_XXS_BLOCK_SIZE, IQ2_XXS_EFFECTIVE_BITS),
        ("iq2_xs", IQ2_XS_BLOCK_SIZE, IQ2_XS_EFFECTIVE_BITS),
    ],
)
def test_iq_recipe_matches_packing_contract(qformat, block_size, effective_bits):
    recipe = load_recipe(f"general/ptq/{qformat}")
    quant_cfg = recipe.quantize.model_dump(exclude_unset=True)["quant_cfg"]
    weight_cfg = next(
        entry["cfg"] for entry in quant_cfg if entry.get("quantizer_name") == "*weight_quantizer"
    )

    assert qformat in presets.QUANT_CFG_CHOICES
    assert weight_cfg["backend"] == "ggml"
    assert weight_cfg["num_bits"] == qformat
    assert weight_cfg["block_sizes"][-1] == block_size
    assert weight_cfg["effective_bits"] == effective_bits


# --- RecipeSupersededAction: the flags --recipe replaces ----------------------------------------


def _one_flag_parser(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_only", action=RecipeSupersededAction, **kwargs)
    return parser


def test_store_true_style_flag_defaults_without_warning():
    """``nargs=0`` flags default to False and stay silent -- argparse skips absent options."""
    args = _one_flag_parser(nargs=0, const=True, default=False).parse_args([])
    assert args.weight_only is False


def test_store_true_style_flag_stores_const_not_an_empty_list():
    """A ``nargs=0`` flag is handed ``[]``, so the action has to store ``const`` instead.

    ``--weight_only`` on megatron_bridge is the only caller of this branch; storing the empty list
    would leave a falsy value and silently turn weight-only quantization off.
    """
    parser = _one_flag_parser(nargs=0, const=True, default=False)
    with pytest.warns(FutureWarning, match="--weight_only is deprecated"):
        args = parser.parse_args(["--weight_only"])
    assert args.weight_only is True


def test_deprecation_reaches_stderr_under_the_real_default_filters():
    """The warning has to reach an actual CLI user, not just a test run.

    pytest enables every warning, so a category CPython suppresses looks healthy here and says
    nothing in production. argparse invokes the action from its own module, so a
    ``DeprecationWarning`` would be dropped by the default ``ignore::DeprecationWarning`` filter --
    the flag would keep working with nothing said. A subprocess is the only honest check: it uses
    the interpreter's real filters rather than a reconstruction of them.
    """
    script = textwrap.dedent(
        """
        import argparse
        from modelopt.recipe.presets import RecipeSupersededAction

        parser = argparse.ArgumentParser()
        parser.add_argument("--qformat", action=RecipeSupersededAction)
        parser.parse_args(["--qformat", "nvfp4"])
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert "--qformat is deprecated" in proc.stderr, (
        "the deprecation is filtered out under Python's default filters, so a CLI user would "
        f"never see it; stderr was: {proc.stderr!r}"
    )
