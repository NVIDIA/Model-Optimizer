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

"""Tests for hybrid_override_pattern handling in calculate_subblock_params.

Covers _infer_hybrid_pattern_char, the fallback used when parent_layer_index
is not available.  End-to-end validation with the real model is in
tests/gpu/puzzletron/test_nemotron_h_gpu_validation.py.
"""

import pytest

pytest.importorskip("transformers")

from modelopt.torch.puzzletron.block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
)
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_params_and_memory import (
    _infer_hybrid_pattern_char,
)

NEMOTRON_H_PATTERN = "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"


class TestInferHybridPatternChar:
    """Test _infer_hybrid_pattern_char picks the right character for the subblock type."""

    def test_ffn_picks_dash(self):
        bc = BlockConfig(
            attention=AttentionConfig(no_op=True), ffn=FFNConfig(intermediate_size=4096)
        )
        assert _infer_hybrid_pattern_char(NEMOTRON_H_PATTERN, bc) == "-"

    def test_ffn_picks_e_when_no_dash(self):
        bc = BlockConfig(
            attention=AttentionConfig(no_op=True), ffn=FFNConfig(intermediate_size=4096)
        )
        assert _infer_hybrid_pattern_char("MMM*EE", bc) == "E"

    def test_mamba_picks_m(self):
        bc = BlockConfig(
            attention=AttentionConfig(
                mamba=MambaConfig(state_dim=128, num_heads=64, head_dim=64, num_groups=8)
            ),
            ffn=FFNConfig(no_op=True),
        )
        assert _infer_hybrid_pattern_char(NEMOTRON_H_PATTERN, bc) == "M"

    def test_attention_picks_star(self):
        bc = BlockConfig(
            attention=AttentionConfig(num_key_value_heads=8),
            ffn=FFNConfig(no_op=True),
        )
        assert _infer_hybrid_pattern_char(NEMOTRON_H_PATTERN, bc) == "*"

    def test_fallback_to_first_char(self):
        bc = BlockConfig(
            attention=AttentionConfig(num_key_value_heads=8),
            ffn=FFNConfig(no_op=True),
        )
        assert _infer_hybrid_pattern_char("MMM", bc) == "M"

    def test_pipe_separator_stripped(self):
        """Patterns with pipe separators should still match after stripping."""
        bc = BlockConfig(
            attention=AttentionConfig(no_op=True), ffn=FFNConfig(intermediate_size=4096)
        )
        assert _infer_hybrid_pattern_char("M|-|*", bc) == "-"
