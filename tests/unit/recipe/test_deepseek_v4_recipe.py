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


"""Tests for the DeepSeek-V4-Pro-0813 checkpoint-mirror PTQ recipe and its guard.

``examples/deepseek/deepseek_v4/ptq.py`` keeps ``_build_nvfp4_experts_cfg()`` as its
default, so the recipe and that builder can drift apart without anything failing --
the symptom would only show up as a difference between two amax dumps. Kept beside
the other recipe tests rather than under ``tests/examples/``: CI runs ``tests/unit``
on every change, while the example lanes cover an allowlist that excludes deepseek.
"""

import copy
import fnmatch
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "deepseek" / "deepseek_v4" / "ptq.py"
_SPEC = importlib.util.spec_from_file_location("deepseek_v4_ptq", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
dsv4_ptq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dsv4_ptq)

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
                block_sizes = cfg["block_sizes"]
                fmt = (
                    tuple(cfg["num_bits"]),
                    block_sizes[-1],
                    block_sizes["type"],
                    tuple(block_sizes["scale_bits"]),
                )
            state = (entry.get("enable", True), fmt)
    return state


def test_recipe_matches_the_builtin_quant_cfg():
    """Loads via ``_quant_cfg_from_recipe``, so the guard's own checks (max algorithm,
    experts-only scope, NVFP4 encoding) run against the shipped recipe too."""
    recipe_cfg = dsv4_ptq._quant_cfg_from_recipe(dsv4_ptq._PUBLISHED_RECIPE)
    builtin_cfg = dsv4_ptq._build_nvfp4_experts_cfg()

    for name in _PROBES:
        assert _resolve(recipe_cfg["quant_cfg"], name) == _resolve(
            builtin_cfg["quant_cfg"], name
        ), f"recipe and _build_nvfp4_experts_cfg() disagree for {name}"


_NVFP4 = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}


def _entry(name, **cfg_overrides):
    cfg = copy.deepcopy(_NVFP4)
    cfg["block_sizes"].update(cfg_overrides.pop("block_sizes", {}))
    cfg.update(cfg_overrides)
    return {"quantizer_name": name, "enable": True, "cfg": cfg}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda c: c.update(algorithm="awq_lite"), "max"),
        (
            lambda c: c["quant_cfg"].append(_entry("*shared_experts*weight_quantizer")),
            "routed-expert",
        ),
        (
            lambda c: c["quant_cfg"].append(
                _entry("*ffn.experts.*.w*_weight_quantizer", num_bits=(4, 3))
            ),
            "block-16 NVFP4",
        ),
        (
            lambda c: c["quant_cfg"].append(
                _entry("*ffn.experts.*.w*_weight_quantizer", block_sizes={"type": "static"})
            ),
            "block-16 NVFP4",
        ),
        (
            lambda c: c["quant_cfg"].append(
                _entry("*ffn.experts.*.w*_weight_quantizer", block_sizes={"scale_bits": (8, 0)})
            ),
            "block-16 NVFP4",
        ),
        (
            lambda c: c["quant_cfg"].append(
                {
                    "quantizer_name": "*ffn.experts.*.w*_weight_quantizer",
                    "enable": True,
                    "cfg": [_NVFP4],
                }
            ),
            "non-dict 'cfg'",
        ),
    ],
    ids=[
        "wrong-algorithm",
        "quantizer-outside-experts",
        "wrong-num-bits",
        "static-block-quantization",
        "wrong-scale-bits",
        "list-valued-cfg",
    ],
)
def test_guard_rejects_recipes_the_export_path_cannot_represent(monkeypatch, mutate, match):
    """The manifest is hardcoded to NVFP4_W4A4; a deviating recipe must fail loudly."""
    base = dsv4_ptq.load_recipe(dsv4_ptq._PUBLISHED_RECIPE).quantize.model_dump()
    mutate(base)

    class _Stub:
        quantize = type("Q", (), {"model_dump": staticmethod(lambda: base)})()

    monkeypatch.setattr(dsv4_ptq, "load_recipe", lambda _path: _Stub())
    with pytest.raises(ValueError, match=match):
        dsv4_ptq._quant_cfg_from_recipe("ignored")


def test_guard_rejects_a_non_ptq_recipe():
    """``--recipe`` takes any path; a speculative-decoding recipe has no ``quantize``."""
    with pytest.raises(ValueError, match="no 'quantize' section"):
        dsv4_ptq._quant_cfg_from_recipe("general/speculative_decoding/eagle3")
