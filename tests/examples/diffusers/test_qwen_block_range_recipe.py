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

"""Unit tests for the transformer-block-range quantization recipe (e.g. Qwen-Image).

The recipe must quantize only the linears under ``transformer_blocks`` while
excluding the first/last N blocks, and it must be expressible as ``quant_cfg``
rules applied before calibration (so SVDQuant never mutates the excluded blocks).
"""

import re
import sys
from pathlib import Path

import pytest

# Importing the example module pulls in diffusers/torch/datasets/modelopt.
pytest.importorskip("diffusers")
pytest.importorskip("torch")

# Make the diffusers quantization example importable.
_EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "diffusers" / "quantization"
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from models_utils import build_block_range_quant_cfg  # noqa: E402

_BLOCK_RULE_RE = re.compile(r"\*transformer_blocks\.(\d+)\.\*(?:weight|input)_quantizer")


class _StubBackbone:
    """Minimal stand-in exposing a ``transformer_blocks`` sequence of length n."""

    def __init__(self, num_blocks: int):
        self.transformer_blocks = list(range(num_blocks))


def _disabled_block_indices(rules):
    """Indices of transformer blocks explicitly disabled by per-block rules."""
    indices = set()
    for rule in rules:
        if rule.get("enable") is False:
            match = _BLOCK_RULE_RE.fullmatch(rule["quantizer_name"])
            if match:
                indices.add(int(match.group(1)))
    return indices


def test_recipe_excludes_first_and_last_two_blocks():
    rules = build_block_range_quant_cfg(_StubBackbone(6), exclude_first_n=2, exclude_last_n=2)

    # 1. disable-all rules come first (weight + input).
    assert rules[0] == {"quantizer_name": "*weight_quantizer", "enable": False}
    assert rules[1] == {"quantizer_name": "*input_quantizer", "enable": False}
    # 2. then re-enable only the transformer_blocks (top-level `enable`; a `None` cfg
    #    keeps the base preset's quant params).
    assert {"quantizer_name": "*transformer_blocks.*weight_quantizer", "enable": True} in rules
    assert {"quantizer_name": "*transformer_blocks.*input_quantizer", "enable": True} in rules
    # 3. then disable the first 2 and last 2 of the 6 blocks -> {0, 1, 4, 5}; quantize {2, 3}.
    assert _disabled_block_indices(rules) == {0, 1, 4, 5}


def test_recipe_block_count_scales_with_model():
    # For a 60-block model (Qwen-Image), exclude {0, 1, 58, 59}; quantize 2..57.
    rules = build_block_range_quant_cfg(_StubBackbone(60), exclude_first_n=2, exclude_last_n=2)
    assert _disabled_block_indices(rules) == {0, 1, 58, 59}


@pytest.mark.parametrize("num_blocks", [5, 4, 3])
def test_recipe_rejects_too_few_blocks(num_blocks):
    # A 2 + 2 exclusion needs at least 6 blocks (>= 2 quantized middle blocks).
    # A 5-block model leaves only 1 middle block and must be rejected too.
    with pytest.raises(ValueError, match="at least"):
        build_block_range_quant_cfg(
            _StubBackbone(num_blocks), exclude_first_n=2, exclude_last_n=2
        )


def test_recipe_missing_block_module_raises():
    class _NoBlocks:
        pass

    with pytest.raises(ValueError, match="transformer_blocks"):
        build_block_range_quant_cfg(_NoBlocks(), exclude_first_n=2, exclude_last_n=2)


def test_svdquant_recipe_leaves_excluded_blocks_bit_identical():
    """The block-range recipe must keep the excluded first/last blocks bit-identical
    through SVDQuant (whose calibration subtracts a residual from every *enabled*
    linear), while the middle blocks receive LoRA."""
    import torch
    import torch.nn as nn

    import modelopt.torch.quantization as mtq

    class _Block(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            return self.proj(x)

    class _Backbone(nn.Module):
        def __init__(self, num_blocks: int = 6, dim: int = 32):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(_Block(dim) for _ in range(num_blocks))

        def forward(self, x):
            for block in self.transformer_blocks:
                x = block(x)
            return x

    torch.manual_seed(0)
    model = _Backbone(num_blocks=6, dim=32)
    weights_before = {
        i: model.transformer_blocks[i].proj.weight.detach().clone() for i in range(6)
    }

    # Base rules quantize every linear weight/input quantizer; the recipe then
    # disables all and re-enables only the middle transformer blocks (2, 3).
    quant_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
            *build_block_range_quant_cfg(model, exclude_first_n=2, exclude_last_n=2),
        ],
        "algorithm": {"method": "svdquant", "lowrank": 4},
    }
    calib_data = [torch.randn(2, 32) for _ in range(2)]
    mtq.quantize(model, quant_cfg, lambda m: [m(batch) for batch in calib_data])

    excluded = {0, 1, 4, 5}
    for idx in range(6):
        proj = model.transformer_blocks[idx].proj
        lora_a = getattr(getattr(proj, "weight_quantizer", None), "svdquant_lora_a", None)
        if idx in excluded:
            # Never calibrated -> weight bit-identical, no LoRA residual.
            assert torch.equal(proj.weight, weights_before[idx]), (
                f"excluded block {idx} weight was modified"
            )
            assert lora_a is None, f"excluded block {idx} unexpectedly has SVDQuant LoRA"
        else:
            # Calibrated -> LoRA present and the residual was subtracted from the weight.
            assert lora_a is not None, f"middle block {idx} is missing SVDQuant LoRA"
            assert not torch.equal(proj.weight, weights_before[idx]), (
                f"middle block {idx} weight was not modified by SVDQuant"
            )
