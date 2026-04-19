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

"""Unit tests for LAQ PTQ recipe YAML files in configs/quantize/."""

from pathlib import Path

import pytest
import yaml

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "examples" / "llm_qat" / "configs" / "quantize"

# (filename, expected learnable_amax, expected tied_amax)
_LAQ_RECIPES = [
    ("nvfp4_laq_post-mse_init-fp8_kv.yml", ["post"], False),
    ("nvfp4_laq_pre-mse_init-fp8_kv.yml", ["pre"], False),
    ("nvfp4_laq_pre_post-mse_init-fp8_kv.yml", ["pre", "post"], False),
    ("nvfp4_laq_pre_post_tied-mse_init-fp8_kv.yml", ["pre", "post"], True),
    ("nvfp4_laq_frozen-mse_init-fp8_kv.yml", [], False),
]


def _load_yaml(filename):
    path = CONFIGS_DIR / filename
    with open(path) as f:
        return yaml.safe_load(f)


def _find_entry(quant_cfg, quantizer_name):
    """Find entry by quantizer_name in the quant_cfg list."""
    for entry in quant_cfg:
        if entry.get("quantizer_name") == quantizer_name:
            return entry
    raise KeyError(f"No entry with quantizer_name={quantizer_name!r}")


# ---------------------------------------------------------------------------
# Parametrized load & parse test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("filename", "_", "__"), _LAQ_RECIPES, ids=[r[0] for r in _LAQ_RECIPES])
def test_recipe_loads_and_has_required_sections(filename, _, __):
    """Each LAQ recipe YAML is parseable and has metadata + quantize."""
    data = _load_yaml(filename)
    assert "metadata" in data
    assert data["metadata"]["recipe_type"] == "ptq"
    assert "quantize" in data
    assert "algorithm" in data["quantize"]
    assert "quant_cfg" in data["quantize"]


# ---------------------------------------------------------------------------
# Algorithm structure test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected_learnable", "expected_tied"),
    _LAQ_RECIPES,
    ids=[r[0] for r in _LAQ_RECIPES],
)
def test_algorithm_has_correct_laq_params(filename, expected_learnable, expected_tied):
    """Algorithm section has correct method, learnable_amax, tied_amax, and scale_algorithm."""
    algo = _load_yaml(filename)["quantize"]["algorithm"]
    assert algo["method"] == "laq"
    assert algo["learnable_amax"] == expected_learnable
    assert algo["tied_amax"] is expected_tied
    assert algo["scale_algorithm"] == {"method": "mse", "fp8_scale_sweep": True}


# ---------------------------------------------------------------------------
# Weight quantizer uses static type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("filename", "_", "__"), _LAQ_RECIPES, ids=[r[0] for r in _LAQ_RECIPES])
def test_weight_quantizer_is_static(filename, _, __):
    """Weight quantizer must use static block type for LAQ learnable scales."""
    qcfg = _load_yaml(filename)["quantize"]["quant_cfg"]
    w = _find_entry(qcfg, "*weight_quantizer")
    assert w["enable"] is True
    assert w["cfg"]["block_sizes"]["type"] == "static"
    assert w["cfg"]["num_bits"] == "e2m1"


# ---------------------------------------------------------------------------
# Input quantizer uses dynamic type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("filename", "_", "__"), _LAQ_RECIPES, ids=[r[0] for r in _LAQ_RECIPES])
def test_input_quantizer_is_dynamic(filename, _, __):
    """Input/activation quantizer uses dynamic block type."""
    qcfg = _load_yaml(filename)["quantize"]["quant_cfg"]
    inp = _find_entry(qcfg, "*input_quantizer")
    assert inp["enable"] is True
    assert inp["cfg"]["block_sizes"]["type"] == "dynamic"
    assert inp["cfg"]["num_bits"] == "e2m1"


# ---------------------------------------------------------------------------
# KV cache quantizer enabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("filename", "_", "__"), _LAQ_RECIPES, ids=[r[0] for r in _LAQ_RECIPES])
def test_kv_cache_quantizer_enabled(filename, _, __):
    """FP8 KV cache quantizer is present and enabled."""
    qcfg = _load_yaml(filename)["quantize"]["quant_cfg"]
    kv = _find_entry(qcfg, "*[kv]_bmm_quantizer")
    assert kv["enable"] is True
    assert kv["cfg"]["num_bits"] == "e4m3"
