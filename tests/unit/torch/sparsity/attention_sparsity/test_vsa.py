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

"""CPU-only unit tests for Video Sparse Attention (VSA).

Tests cover:
- vsa_utils.py: tile/untile index logic, variable block sizes
- vsa.py: VSA method init, metadata computation, validation, caching
- config.py: VSAAttributeConfig validation
- plugins/ltx2.py: model/module detection helpers
"""

import math

import pytest
import torch
from pydantic import ValidationError

from modelopt.torch.sparsity.attention_sparsity.config import VSAAttributeConfig, VSAConfig
from modelopt.torch.sparsity.attention_sparsity.methods.vsa import VSA
from modelopt.torch.sparsity.attention_sparsity.methods.vsa_utils import (
    construct_variable_block_sizes,
    get_non_pad_index,
    get_reverse_tile_partition_indices,
    get_tile_partition_indices,
)

# ---------------------------------------------------------------------------
# vsa_utils: tile partition indices
# ---------------------------------------------------------------------------


class TestTilePartitionIndices:
    """Tests for get_tile_partition_indices."""

    def test_evenly_divisible(self):
        """Tiles cover full volume with no remainder."""
        video_shape = (8, 8, 8)
        tile_size = (4, 4, 4)
        idx = get_tile_partition_indices(video_shape, tile_size, torch.device("cpu"))
        assert idx.shape == (8 * 8 * 8,)
        # Every original index appears exactly once
        assert torch.equal(idx.sort().values, torch.arange(512))

    def test_non_divisible(self):
        """Edge tiles are smaller when dims don't divide evenly."""
        video_shape = (5, 6, 7)
        tile_size = (4, 4, 4)
        seq_len = 5 * 6 * 7
        idx = get_tile_partition_indices(video_shape, tile_size, torch.device("cpu"))
        assert idx.shape == (seq_len,)
        assert torch.equal(idx.sort().values, torch.arange(seq_len))

    def test_round_trip(self):
        """tile then reverse_tile is identity."""
        video_shape = (6, 10, 8)
        tile_size = (4, 4, 4)
        device = torch.device("cpu")
        fwd = get_tile_partition_indices(video_shape, tile_size, device)
        rev = get_reverse_tile_partition_indices(video_shape, tile_size, device)
        # Applying forward then reverse should yield the original order
        assert torch.equal(fwd[rev], torch.arange(6 * 10 * 8))


# ---------------------------------------------------------------------------
# vsa_utils: variable block sizes
# ---------------------------------------------------------------------------


class TestVariableBlockSizes:
    """Tests for construct_variable_block_sizes."""

    def test_evenly_divisible(self):
        """All tiles have full size when dims divide evenly."""
        video_shape = (8, 8, 8)
        tile_size = (4, 4, 4)
        num_tiles = (2, 2, 2)
        sizes = construct_variable_block_sizes(
            video_shape, num_tiles, tile_size, torch.device("cpu")
        )
        assert sizes.shape == (8,)  # 2*2*2 tiles
        assert (sizes == 64).all()  # every tile is full 4*4*4

    def test_non_divisible_sum(self):
        """Sum of variable sizes equals original sequence length."""
        video_shape = (5, 6, 7)
        tile_size = (4, 4, 4)
        num_tiles = (
            math.ceil(5 / 4),
            math.ceil(6 / 4),
            math.ceil(7 / 4),
        )
        sizes = construct_variable_block_sizes(
            video_shape, num_tiles, tile_size, torch.device("cpu")
        )
        assert sizes.sum().item() == 5 * 6 * 7

    def test_partial_tile_smaller(self):
        """Last tile along a non-divisible dim should be smaller."""
        video_shape = (5, 4, 4)
        tile_size = (4, 4, 4)
        num_tiles = (2, 1, 1)
        sizes = construct_variable_block_sizes(
            video_shape, num_tiles, tile_size, torch.device("cpu")
        )
        # First tile: 4*4*4=64, second tile: 1*4*4=16
        assert sizes[0].item() == 64
        assert sizes[1].item() == 16


# ---------------------------------------------------------------------------
# vsa_utils: non-pad index
# ---------------------------------------------------------------------------


class TestNonPadIndex:
    """Tests for get_non_pad_index."""

    def test_full_blocks(self):
        """All blocks full size → non_pad covers everything."""
        sizes = torch.tensor([64, 64, 64])
        npi = get_non_pad_index(sizes, 64)
        assert npi.shape == (192,)  # 3 * 64

    def test_partial_blocks(self):
        """Partial blocks → non_pad skips padding positions."""
        sizes = torch.tensor([64, 16])
        npi = get_non_pad_index(sizes, 64)
        assert npi.shape == (80,)  # 64 + 16


# ---------------------------------------------------------------------------
# VSA method: init and config
# ---------------------------------------------------------------------------


class TestVSAInit:
    """Tests for VSA.__init__ and basic properties."""

    def test_defaults(self):
        vsa = VSA()
        assert vsa.block_size_3d == (4, 4, 4)
        assert vsa.block_elements == 64
        assert vsa.top_k_ratio == 0.5
        assert vsa.video_shape is None
        assert vsa.name == "vsa"

    def test_custom_config(self):
        vsa = VSA({"block_size_3d": [2, 2, 2], "top_k_ratio": 0.3, "video_shape": (8, 8, 8)})
        assert vsa.block_size_3d == (2, 2, 2)
        assert vsa.block_elements == 8
        assert vsa.top_k_ratio == 0.3
        assert vsa.video_shape == (8, 8, 8)

    def test_set_video_shape(self):
        vsa = VSA()
        vsa.set_video_shape((4, 8, 12))
        assert vsa.video_shape == (4, 8, 12)

    def test_get_threshold_info(self):
        vsa = VSA({"top_k_ratio": 0.7, "video_shape": (4, 4, 4)})
        info = vsa.get_threshold_info()
        assert info["type"] == "vsa"
        assert info["top_k_ratio"] == 0.7


# ---------------------------------------------------------------------------
# VSA method: metadata computation and validation
# ---------------------------------------------------------------------------


class TestVSAMetadata:
    """Tests for VSA._compute_metadata validation and caching."""

    def test_no_video_shape_raises(self):
        vsa = VSA()
        with pytest.raises(ValueError, match="video_shape must be provided"):
            vsa._compute_metadata(100, torch.device("cpu"))

    def test_seq_len_mismatch_raises(self):
        vsa = VSA({"video_shape": (4, 4, 4)})
        with pytest.raises(ValueError, match="does not match video shape"):
            vsa._compute_metadata(100, torch.device("cpu"))  # expected 64

    def test_valid_metadata(self):
        vsa = VSA({"video_shape": (8, 8, 8)})
        meta = vsa._compute_metadata(512, torch.device("cpu"))
        assert meta["video_shape"] == (8, 8, 8)
        assert meta["num_tiles"] == (2, 2, 2)
        assert meta["total_tiles"] == 8

    def test_metadata_caching(self):
        vsa = VSA({"video_shape": (8, 8, 8)})
        m1 = vsa._compute_metadata(512, torch.device("cpu"))
        m2 = vsa._compute_metadata(512, torch.device("cpu"))
        assert m1 is m2  # same object, not recomputed


# ---------------------------------------------------------------------------
# VSA method: abstract stubs raise
# ---------------------------------------------------------------------------


class TestVSAStubs:
    """calculate_sparsity and apply_sparsity should raise NotImplementedError."""

    def test_calculate_sparsity_raises(self):
        vsa = VSA()
        with pytest.raises(NotImplementedError, match="softmax-patching"):
            vsa.calculate_sparsity(torch.zeros(1))

    def test_apply_sparsity_raises(self):
        vsa = VSA()
        with pytest.raises(NotImplementedError, match="softmax-patching"):
            vsa.apply_sparsity(torch.zeros(1))


# ---------------------------------------------------------------------------
# VSAAttributeConfig validation
# ---------------------------------------------------------------------------


class TestVSAAttributeConfig:
    """Tests for VSAAttributeConfig pydantic validation."""

    def test_valid_defaults(self):
        cfg = VSAAttributeConfig()
        assert cfg.method == "vsa"
        assert cfg.block_size_3d == (4, 4, 4)
        assert cfg.top_k_ratio == 0.5

    def test_top_k_ratio_out_of_range(self):
        with pytest.raises(ValidationError, match="top_k_ratio"):
            VSAAttributeConfig(top_k_ratio=0.0)
        with pytest.raises(ValidationError, match="top_k_ratio"):
            VSAAttributeConfig(top_k_ratio=1.5)

    def test_video_shape_wrong_length(self):
        with pytest.raises(ValidationError, match="3 elements"):
            VSAAttributeConfig(video_shape=(4, 4))

    def test_video_shape_negative(self):
        with pytest.raises(ValidationError, match="positive"):
            VSAAttributeConfig(video_shape=(4, -1, 4))

    def test_video_shape_none_allowed(self):
        cfg = VSAAttributeConfig(video_shape=None)
        assert cfg.video_shape is None

    def test_vsa_config_defaults(self):
        cfg = VSAConfig()
        assert "*attention*" in cfg.sparse_cfg
        assert cfg.sparse_cfg["*attention*"]["method"] == "vsa"


# ---------------------------------------------------------------------------
# LTX-2 plugin: detection helpers
# ---------------------------------------------------------------------------


class TestLTX2Detection:
    """Tests for _is_ltx2_model and _is_ltx2_attention_module."""

    def test_non_ltx2_model(self):
        from modelopt.torch.sparsity.attention_sparsity.plugins.ltx2 import _is_ltx2_model

        model = torch.nn.Linear(10, 10)
        assert _is_ltx2_model(model) is False

    def test_ltx2_model_by_class_name(self):
        from modelopt.torch.sparsity.attention_sparsity.plugins.ltx2 import _is_ltx2_model

        # Fake a class named LTXModel
        class LTXModel(torch.nn.Module):
            pass

        assert _is_ltx2_model(LTXModel()) is True

    def test_ltx2_attention_by_class_name(self):
        from modelopt.torch.sparsity.attention_sparsity.plugins.ltx2 import (
            _is_ltx2_attention_module,
        )

        class LTXSelfAttention(torch.nn.Module):
            pass

        assert _is_ltx2_attention_module(LTXSelfAttention()) is True

    def test_ltx2_attention_by_structure(self):
        from modelopt.torch.sparsity.attention_sparsity.plugins.ltx2 import (
            _is_ltx2_attention_module,
        )

        # Module with LTX-2 attribute signature
        m = torch.nn.Module()
        m.to_q = torch.nn.Linear(8, 8)
        m.to_k = torch.nn.Linear(8, 8)
        m.to_v = torch.nn.Linear(8, 8)
        m.q_norm = torch.nn.LayerNorm(8)
        m.k_norm = torch.nn.LayerNorm(8)
        assert _is_ltx2_attention_module(m) is True

    def test_non_attention_module(self):
        from modelopt.torch.sparsity.attention_sparsity.plugins.ltx2 import (
            _is_ltx2_attention_module,
        )

        assert _is_ltx2_attention_module(torch.nn.Linear(10, 10)) is False
