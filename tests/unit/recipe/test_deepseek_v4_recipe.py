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

"""Tests for the DeepSeek-V4-Pro-0813 checkpoint-mirror PTQ recipe.

``examples/deepseek/deepseek_v4/ptq.py`` keeps ``_build_nvfp4_experts_cfg()`` as its
default, so the recipe and that builder can drift apart without anything failing --
the symptom would only show up as a difference between two amax dumps. Pin the
equivalence here, the same way ``test_presets.py`` and ``test_kimi_k3_recipe.py`` do.
"""

import fnmatch
import importlib.util
from pathlib import Path

import pytest

from modelopt.recipe import load_recipe

RECIPE = "huggingface/models/deepseek-ai/DeepSeek-V4-Pro-0813/ptq/nvfp4_experts_only"

_PTQ_PY = Path(__file__).parents[3] / "examples" / "deepseek" / "deepseek_v4" / "ptq.py"

# Names spanning every branch of the config: routed experts (enabled), and the
# groups that must stay untouched -- shared expert, attention, MTP, lm_head.
_PROBES = [
    "model.layers.3.ffn.experts.17.w1_weight_quantizer",
    "model.layers.3.ffn.experts.17.w2_input_quantizer",
    "model.layers.3.ffn.shared_experts.w1_weight_quantizer",
    "model.layers.3.attn.wq_weight_quantizer",
    "mtp.0.ffn.experts.2.w1_weight_quantizer",
    "lm_head_weight_quantizer",
]


def _load_example_module():
    spec = importlib.util.spec_from_file_location("_dsv4_ptq", _PTQ_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve(quant_cfg, name):
    """Effective (enabled, numeric format) for ``name``; later rules win, as mtq applies them in order."""
    state = (False, None)
    for entry in quant_cfg:
        if not isinstance(entry, dict) or "quantizer_name" not in entry:
            continue
        if fnmatch.fnmatch(name, entry["quantizer_name"]):
            cfg = entry.get("cfg")
            fmt = None
            if cfg:
                # Compare only the fields PTQ acts on. The recipe additionally carries
                # effective_bits from configs/numerics/nvfp4, which is autoquant-only.
                fmt = (tuple(cfg["num_bits"]), cfg["block_sizes"][-1], cfg["block_sizes"]["type"])
            state = (entry.get("enable", True), fmt)
    return state


def test_recipe_matches_the_builtin_quant_cfg():
    recipe_cfg = load_recipe(RECIPE).quantize.model_dump()
    builtin_cfg = _load_example_module()._build_nvfp4_experts_cfg()

    for name in _PROBES:
        assert _resolve(recipe_cfg["quant_cfg"], name) == _resolve(
            builtin_cfg["quant_cfg"], name
        ), f"recipe and _build_nvfp4_experts_cfg() disagree for {name}"


def test_recipe_uses_the_calibration_free_max_algorithm():
    algorithm = load_recipe(RECIPE).quantize.model_dump()["algorithm"]
    method = algorithm.get("method") if isinstance(algorithm, dict) else algorithm
    assert method == "max"


def test_routed_experts_are_block16_nvfp4():
    quant_cfg = load_recipe(RECIPE).quantize.model_dump()["quant_cfg"]
    enabled = [e for e in quant_cfg if isinstance(e, dict) and e.get("enable") is True]
    assert enabled, "recipe enables no quantizers"
    for entry in enabled:
        assert "ffn.experts." in entry["quantizer_name"]
        cfg = entry["cfg"]
        assert tuple(cfg["num_bits"]) == (2, 1)
        assert cfg["block_sizes"][-1] == 16


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda c: c.update(algorithm="awq_lite"), "max"),
        (
            lambda c: c["quant_cfg"].append(
                {
                    "quantizer_name": "*shared_experts*weight_quantizer",
                    "enable": True,
                    "cfg": {"num_bits": (2, 1), "block_sizes": {-1: 16}},
                }
            ),
            "routed-expert",
        ),
        (
            lambda c: c["quant_cfg"].append(
                {
                    "quantizer_name": "*ffn.experts.*.w*_weight_quantizer",
                    "enable": True,
                    "cfg": {"num_bits": (4, 3), "block_sizes": {-1: 128}},
                }
            ),
            "block-16 NVFP4",
        ),
    ],
    ids=["wrong-algorithm", "quantizer-outside-experts", "wrong-numeric-format"],
)
def test_guard_rejects_recipes_the_export_path_cannot_represent(monkeypatch, mutate, match):
    """The manifest is hardcoded to NVFP4_W4A4; a deviating recipe must fail loudly."""
    module = _load_example_module()
    base = load_recipe(RECIPE).quantize.model_dump()
    mutate(base)

    class _Stub:
        quantize = type("Q", (), {"model_dump": staticmethod(lambda: base)})()

    monkeypatch.setattr(module, "load_recipe", lambda _path: _Stub())
    with pytest.raises(ValueError, match=match):
        module._quant_cfg_from_recipe("ignored")
