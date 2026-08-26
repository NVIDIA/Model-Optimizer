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

"""Tests for ModelDescriptor.truncate_pattern_for_subblock.

Validates that the base descriptor method selects the correct pattern
character when building a 1-layer model for per-subblock param counting.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from modelopt.torch.puzzletron.anymodel.model_descriptor import ModelDescriptor

NEMOTRON_H_PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"


class TestTruncatePatternForSubblock:
    """Test ModelDescriptor.truncate_pattern_for_subblock."""

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (0, "M"),
            (1, "-"),
            (7, "*"),
        ],
        ids=["mamba", "ffn", "attention"],
    )
    def test_index_selects_correct_layer_type(self, index, expected):
        """Parent layer index selects the matching character from the pattern."""
        cfg = _make_config()

        ModelDescriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=index)

        assert cfg.hybrid_override_pattern == expected

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (1, "-"),
            (2, "*"),
        ],
        ids=["ffn_after_strip", "attention_after_strip"],
    )
    def test_pipe_separators_stripped_before_indexing(self, index, expected):
        """Pipe-delimited patterns like 'M|-|*' are normalised to 'M-*' before lookup."""
        cfg = _make_config("M|-|*")

        ModelDescriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=index)

        assert cfg.hybrid_override_pattern == expected

    def test_missing_attribute_is_noop(self):
        """Config without hybrid_override_pattern is left unchanged."""
        cfg = SimpleNamespace()

        ModelDescriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=0)

        assert not hasattr(cfg, "hybrid_override_pattern")

    def test_empty_pattern_is_noop(self):
        """Empty pattern string is left unchanged."""
        cfg = _make_config("")

        ModelDescriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=0)

        assert cfg.hybrid_override_pattern == ""

    def test_pipes_only_pattern_raises(self):
        """Pattern with only pipe separators has no layer-type characters and should error."""
        cfg = _make_config("|||")

        with pytest.raises(ValueError, match="no layer-type characters"):
            ModelDescriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=0)

    def test_none_index_defaults_to_first_char(self):
        """Without an explicit index, defaults to pattern[0]."""
        cfg = _make_config("*-M")

        ModelDescriptor.truncate_pattern_for_subblock(cfg)

        assert cfg.hybrid_override_pattern == "*"

    @pytest.mark.parametrize(
        "index",
        [999, -1],
        ids=["above_range", "negative"],
    )
    def test_out_of_range_index_defaults_to_first_char(self, index):
        """Out-of-range index defaults to pattern[0]."""
        cfg = _make_config("*-M")

        ModelDescriptor.truncate_pattern_for_subblock(cfg, parent_layer_index=index)

        assert cfg.hybrid_override_pattern == "*"


def _make_config(pattern=NEMOTRON_H_PATTERN):
    return SimpleNamespace(hybrid_override_pattern=pattern)
