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
excluding the first/last N blocks, and it must be expressible as pre-calibration
``quant_cfg`` rules (so SVDQuant never mutates the excluded blocks' weights).
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
        if rule["cfg"].get("enable") is False:
            match = _BLOCK_RULE_RE.fullmatch(rule["quantizer_name"])
            if match:
                indices.add(int(match.group(1)))
    return indices


def test_recipe_excludes_first_and_last_two_blocks():
    rules = build_block_range_quant_cfg(_StubBackbone(6), exclude_first_n=2, exclude_last_n=2)

    # 1. disable-all rules come first (weight + input).
    assert rules[0] == {"quantizer_name": "*weight_quantizer", "cfg": {"enable": False}}
    assert rules[1] == {"quantizer_name": "*input_quantizer", "cfg": {"enable": False}}
    # 2. then enable only the transformer_blocks.
    assert {"quantizer_name": "*transformer_blocks.*weight_quantizer", "cfg": {"enable": True}} in rules
    assert {"quantizer_name": "*transformer_blocks.*input_quantizer", "cfg": {"enable": True}} in rules
    # 3. then disable the first 2 and last 2 of the 6 blocks -> {0, 1, 4, 5}; quantize {2, 3}.
    assert _disabled_block_indices(rules) == {0, 1, 4, 5}


def test_recipe_block_count_scales_with_model():
    # For a 60-block model (Qwen-Image), exclude {0, 1, 58, 59}; quantize 2..57.
    rules = build_block_range_quant_cfg(_StubBackbone(60), exclude_first_n=2, exclude_last_n=2)
    assert _disabled_block_indices(rules) == {0, 1, 58, 59}


def test_recipe_rejects_too_few_blocks():
    # 2 + 2 exclusion needs at least 5 blocks; 4 blocks must raise a clear error.
    with pytest.raises(ValueError, match="at least"):
        build_block_range_quant_cfg(_StubBackbone(4), exclude_first_n=2, exclude_last_n=2)


def test_recipe_missing_block_module_raises():
    class _NoBlocks:
        pass

    with pytest.raises(ValueError, match="transformer_blocks"):
        build_block_range_quant_cfg(_NoBlocks(), exclude_first_n=2, exclude_last_n=2)
