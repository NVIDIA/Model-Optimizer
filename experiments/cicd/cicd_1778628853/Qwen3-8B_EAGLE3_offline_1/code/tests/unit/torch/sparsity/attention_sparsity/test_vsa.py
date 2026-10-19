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
- vsa.py: VSA method init, metadata computation, validation, caching, forward_attention
- config.py: VSAAttributeConfig validation
- HF integration: registration, sparsify, forward dispatch
"""

import math
import sys
from unittest.mock import patch

import pytest

# The attention_sparsity package transitively imports the HF plugin, which
# requires transformers.  Skip the entire module when it is not installed.
pytest.importorskip("transformers")

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
        """All blocks full size -> non_pad covers everything."""
        sizes = torch.tensor([64, 64, 64])
        npi = get_non_pad_index(sizes, 64)
        assert npi.shape == (192,)  # 3 * 64

    def test_partial_blocks(self):
        """Partial blocks -> non_pad skips padding positions."""
        sizes = torch.tensor([64, 16])
        npi = get_non_pad_index(sizes, 64)
        assert npi.shape == (80,)  # 64 + 16


# ---------------------------------------------------------------------------
# VSA: tile/untile round-trip
# ---------------------------------------------------------------------------


class TestTileUntileRoundTrip:
    """Test _tile_tensor / _untile_tensor preserve data."""

    @pytest.mark.parametrize(
        "video_shape",
        [(8, 8, 8), (5, 6, 7), (4, 4, 4)],
        ids=["even", "non-divisible", "single-tile"],
    )
    def test_round_trip(self, video_shape):
        """tile then untile recovers the original tensor."""
        seq_len = video_shape[0] * video_shape[1] * video_shape[2]
        vsa = VSA({"video_shape": video_shape})
        meta = vsa._compute_metadata(seq_len, torch.device("cpu"))

        x = torch.randn(2, 4, seq_len, 16)  # [batch, heads, seq, dim]
        tiled = vsa._tile_tensor(x, meta)
        recovered = vsa._untile_tensor(tiled, meta, seq_len)

        assert recovered.shape == x.shape
        assert torch.allclose(recovered, x)


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
# VSA: forward_attention (kernel import guard)
# ---------------------------------------------------------------------------


class TestVSAForwardAttention:
    """Tests for VSA.forward_attention."""

    def test_missing_kernel_raises(self):
        """forward_attention raises ImportError when fastvideo_kernel is missing."""
        vsa = VSA({"video_shape": (4, 4, 4), "top_k_ratio": 0.5})
        seq_len = 4 * 4 * 4
        q = torch.randn(1, 2, seq_len, 16)
        k = torch.randn(1, 2, seq_len, 16)
        v = torch.randn(1, 2, seq_len, 16)
        with (
            patch.dict(sys.modules, {"fastvideo_kernel": None}),
            pytest.raises(ImportError, match="fastvideo_kernel"),
        ):
            vsa.forward_attention(q, k, v)

    def test_video_shape_override(self):
        """forward_attention accepts video_shape kwarg to override instance shape."""
        vsa = VSA({"video_shape": (4, 4, 4), "top_k_ratio": 0.5})
        new_shape = (8, 8, 8)
        seq_len = 8 * 8 * 8
        q = torch.randn(1, 2, seq_len, 16)
        with (
            patch.dict(sys.modules, {"fastvideo_kernel": None}),
            pytest.raises(ImportError),
        ):
            vsa.forward_attention(q, q, q, video_shape=new_shape)
        assert vsa.video_shape == new_shape


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
        assert "*attn*" in cfg.sparse_cfg
        assert cfg.sparse_cfg["*attn*"]["method"] == "vsa"


# ---------------------------------------------------------------------------
# ModelOpt integration: sparsify() with VSA config
# ---------------------------------------------------------------------------

from _test_utils.torch.sparsity.sparse_attention_common import SimpleAttentionModel

import modelopt.torch.opt as mto
import modelopt.torch.sparsity.attention_sparsity as sparse_attn
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule

VSA_TEST_CFG = {
    "sparse_cfg": {
        "*attention*": {
            "method": "vsa",
            "block_size_3d": (4, 4, 4),
            "top_k_ratio": 0.5,
            "enable": True,
        },
        "default": {"enable": False},
    },
}


class TestVSASparsifyIntegration:
    """Test VSA integration with modelopt sparsify() API."""

    def test_sparsify_creates_sparse_modules(self):
        """sparsify() with VSA config replaces attention modules."""
        model = SimpleAttentionModel()
        sparse_model = sparse_attn.sparsify(model, VSA_TEST_CFG)

        sparse_modules = [m for m in sparse_model.modules() if isinstance(m, SparseAttentionModule)]
        assert len(sparse_modules) > 0

    def test_sparse_module_has_vsa_method(self):
        """Replaced modules are configured with VSA method."""
        model = SimpleAttentionModel()
        sparse_model = sparse_attn.sparsify(model, VSA_TEST_CFG)

        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                assert module._method == "vsa"
                assert isinstance(module._sparse_method_instance, VSA)
                assert module._sparse_method_instance.block_size_3d == (4, 4, 4)
                assert module._sparse_method_instance.top_k_ratio == 0.5

    def test_enable_disable(self):
        """Enable/disable works on VSA sparse modules."""
        model = SimpleAttentionModel()
        sparse_model = sparse_attn.sparsify(model, VSA_TEST_CFG)

        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                assert module.is_enabled
                module.disable()
                assert not module.is_enabled
                module.enable()
                assert module.is_enabled

    def test_threshold_info(self):
        """VSA sparse modules report correct threshold info."""
        model = SimpleAttentionModel()
        sparse_model = sparse_attn.sparsify(model, VSA_TEST_CFG)

        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                info = module.get_threshold_info()
                assert info["type"] == "vsa"
                assert info["top_k_ratio"] == 0.5

    def test_save_restore(self):
        """VSA modelopt_state can be saved and restored."""
        model = SimpleAttentionModel()
        sparse_model = sparse_attn.sparsify(model, VSA_TEST_CFG)

        state = mto.modelopt_state(sparse_model)

        # Restore to a fresh model
        model_restored = SimpleAttentionModel()
        mto.restore_from_modelopt_state(model_restored, state)

        # Verify VSA method is restored
        for module in model_restored.modules():
            if isinstance(module, SparseAttentionModule):
                assert module._method == "vsa"
                assert isinstance(module._sparse_method_instance, VSA)

    def test_pattern_matching(self):
        """Pattern-based config selectively applies VSA."""
        model = SimpleAttentionModel()

        # Pattern that won't match anything
        config = {
            "sparse_cfg": {
                "*nonexistent*": {
                    "method": "vsa",
                    "enable": True,
                },
                "default": {"enable": False},
            },
        }
        sparse_model = sparse_attn.sparsify(model, config)

        # No modules should have VSA enabled
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule):
                assert not module.is_enabled

    def test_forward_patches_sdpa(self):
        """VSA patches F.scaled_dot_product_attention during forward.

        SimpleAttentionModel uses nn.MultiheadAttention which calls SDPA.
        VSA intercepts the SDPA call. Without fastvideo_kernel, this raises
        ImportError — proving the interception works.
        """
        model = SimpleAttentionModel()
        sparse_model = sparse_attn.sparsify(model, VSA_TEST_CFG)

        # Set video_shape so metadata can be computed.
        # seq_len=64, video_shape (4,4,4) -> T*H*W=64
        for module in sparse_model.modules():
            if isinstance(module, SparseAttentionModule) and module.is_enabled:
                module._sparse_method_instance.set_video_shape((4, 4, 4))

        with (
            patch.dict(sys.modules, {"fastvideo_kernel": None}),
            pytest.raises(ImportError, match="fastvideo_kernel"),
        ):
            sparse_model(torch.randn(1, 64, 256))
