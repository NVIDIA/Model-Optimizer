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

"""CPU unit tests for DFlash speculative decoding plugin.

GPU-dependent tests (training forward, module forward) are in tests/gpu/.
"""

import os
from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import (
    get_tiny_llama,
    tf_modelopt_state_and_output_tester,
)
from transformers import AutoModelForCausalLM

import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG
from modelopt.torch.speculative.plugins.hf_dflash import (
    DFlashModule,
    HFDFlashModel,
    build_target_layer_ids,
    create_dflash_attention_mask,
    create_dflash_loss_mask,
)

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be multiple of BLOCK_SIZE


def _get_dflash_config(block_size=BLOCK_SIZE, num_layers=NUM_DRAFT_LAYERS):
    """Create a DFlash config for testing."""
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config["dflash_block_size"] = block_size
    config["dflash_use_torch_compile"] = False
    config["dflash_architecture_config"] = {
        "num_hidden_layers": num_layers,
        "mask_token_id": 0,  # use token 0 as mask for tiny model
    }
    return config


class TestDFlashConvert:
    """Test DFlash model conversion."""

    def test_convert_creates_dflash_model(self):
        """Test that convert produces an HFDFlashModel."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert isinstance(model, HFDFlashModel)

    def test_convert_creates_dflash_module(self):
        """Test that convert attaches a DFlashModule."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "dflash_module")
        assert isinstance(model.dflash_module, DFlashModule)

    def test_convert_freezes_base_model(self):
        """Test that base model parameters are frozen after convert."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        for name, param in model.named_parameters():
            if "dflash_module" not in name:
                assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_convert_dflash_module_trainable(self):
        """Test that DFlash module parameters are trainable after convert."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        dflash_params = [(n, p) for n, p in model.named_parameters() if "dflash_module" in n]
        assert len(dflash_params) > 0
        for name, param in dflash_params:
            assert param.requires_grad, f"DFlash param {name} should be trainable"

    def test_convert_sets_target_layer_ids(self):
        """Test that target layer IDs are set correctly."""
        model = get_tiny_llama(num_hidden_layers=8)
        config = _get_dflash_config(num_layers=3)
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "target_layer_ids")
        assert len(model.target_layer_ids) == 3
        for lid in model.target_layer_ids:
            assert 0 <= lid < 8

    def test_convert_sets_mask_token_id(self):
        """Test that mask_token_id is set from config."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "mask_token_id")
        assert model.mask_token_id == 0

    def test_convert_missing_mask_token_id_errors(self):
        """Test that missing mask_token_id raises ValueError for unknown model."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        del config["dflash_architecture_config"]["mask_token_id"]
        with pytest.raises(ValueError, match="Cannot auto-detect mask_token_id"):
            mtsp.convert(model, [("dflash", config)])


class TestDFlashSaveRestore:
    """Test DFlash model save and restore."""

    def test_save_and_restore(self, tmp_path):
        """Test round-trip save/load preserves modelopt state and outputs."""
        mto.enable_huggingface_checkpointing()
        model_ref = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model_ref, [("dflash", config)])

        model_ref.save_pretrained(tmp_path / "modelopt_model")
        assert os.path.exists(tmp_path / "modelopt_model/modelopt_state.pth")

        model_test = AutoModelForCausalLM.from_pretrained(tmp_path / "modelopt_model")
        assert isinstance(model_test, HFDFlashModel)
        tf_modelopt_state_and_output_tester(model_ref, model_test)


class TestDFlashApplyMetaFix:
    """Test DFlashModule._apply handles meta-tensor rotary buffers.

    During checkpoint restore, rotary inv_freq buffers may be on meta device
    (they are computed, not saved). _apply should re-create them on CPU.
    """

    def test_apply_recreates_meta_rotary(self):
        """Test that .to() recreates rotary_emb when buffers are on meta device."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])

        dflash_mod = model.dflash_module
        # Simulate meta buffers (as happens during checkpoint restore)
        for name, buf in list(dflash_mod.rotary_emb.named_buffers()):
            dflash_mod.rotary_emb._buffers[name] = torch.empty_like(buf, device="meta")

        assert any(b.is_meta for b in dflash_mod.rotary_emb.buffers())

        # .to() triggers _apply which should fix meta buffers
        dflash_mod.to("cpu")

        assert not any(b.is_meta for b in dflash_mod.rotary_emb.buffers())

    def test_apply_noop_when_no_meta(self):
        """Test that .to() does not recreate rotary_emb when buffers are normal."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])

        dflash_mod = model.dflash_module
        rotary_id_before = id(dflash_mod.rotary_emb)
        dflash_mod.to("cpu")
        assert id(dflash_mod.rotary_emb) == rotary_id_before


class TestDFlashAttentionMask:
    """Test DFlash attention mask construction."""

    def test_mask_shape(self):
        mask = create_dflash_attention_mask(SEQ_LEN, BLOCK_SIZE, "cpu", torch.float32)
        assert mask.shape == (1, 1, SEQ_LEN, 2 * SEQ_LEN)

    def test_mask_context_strictly_previous_blocks(self):
        """Context (left half): block B can only see blocks 0..B-1."""
        mask = create_dflash_attention_mask(8, 4, "cpu", torch.float32)
        mask_2d = mask[0, 0]
        ctx_mask = mask_2d[:, :8]
        assert (ctx_mask[:4, :] < 0).all()
        assert (ctx_mask[4:8, :4] == 0).all()
        assert (ctx_mask[4:8, 4:8] < 0).all()

    def test_mask_noise_causal_within_block(self):
        """Noise (right half): reverse-causal within same block (j >= i)."""
        mask = create_dflash_attention_mask(8, 4, "cpu", torch.float32)
        noise_mask = mask[0, 0, :, 8:]
        assert (noise_mask[0, :4] == 0).all()
        assert (noise_mask[3, :3] < 0).all()
        assert noise_mask[3, 3] == 0
        assert (noise_mask[4:8, :4] < 0).all()

    def test_mask_values_are_zero_or_neg_inf(self):
        mask = create_dflash_attention_mask(SEQ_LEN, BLOCK_SIZE, "cpu", torch.float32)
        unique_vals = mask.unique()
        assert len(unique_vals) == 2
        assert 0.0 in unique_vals
        assert unique_vals.min() == torch.finfo(torch.float32).min


class TestDFlashLossMask:
    """Test DFlash loss mask construction."""

    def test_loss_mask_shape(self):
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        assert mask.shape == (SEQ_LEN,)

    def test_loss_mask_excludes_block_zero(self):
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        assert (mask[:BLOCK_SIZE] == 0).all()

    def test_loss_mask_excludes_block_starts(self):
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        for i in range(0, SEQ_LEN, BLOCK_SIZE):
            assert mask[i] == 0

    def test_loss_mask_includes_non_start_positions(self):
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        for b in range(1, SEQ_LEN // BLOCK_SIZE):
            for offset in range(1, BLOCK_SIZE):
                pos = b * BLOCK_SIZE + offset
                assert mask[pos] == 1

    def test_loss_mask_count(self):
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        num_blocks = SEQ_LEN // BLOCK_SIZE
        expected = (num_blocks - 1) * (BLOCK_SIZE - 1)
        assert mask.sum().item() == expected


class TestBuildTargetLayerIds:
    """Test target layer selection."""

    def test_single_draft_layer(self):
        ids = build_target_layer_ids(32, 1)
        assert len(ids) == 1
        assert ids[0] == 16

    def test_multiple_draft_layers(self):
        ids = build_target_layer_ids(36, 5)
        assert len(ids) == 5
        assert ids == sorted(ids)
        assert all(1 <= lid <= 33 for lid in ids)

    def test_layer_ids_no_duplicates(self):
        ids = build_target_layer_ids(32, 5)
        assert len(set(ids)) == 5

    def test_layer_ids_match_zlab(self):
        """Test layer IDs match z-lab reference for Qwen3-8B (36 layers, 5 draft)."""
        ids = build_target_layer_ids(36, 5)
        assert ids == [1, 9, 17, 25, 33]


class TestDFlashSlidingWindow:
    """Test sliding window attention support."""

    def test_sliding_window_from_config(self):
        """Test DFlashAttention reads sliding_window from config.layer_types."""
        from modelopt.torch.speculative.plugins.hf_dflash import DFlashAttention
        from transformers import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=256,
            _attn_implementation="sdpa",
        )
        attn_full = DFlashAttention(config, layer_idx=0)
        attn_sliding = DFlashAttention(config, layer_idx=1)
        assert attn_full.sliding_window is None
        assert attn_sliding.sliding_window == 256

    def test_no_sliding_window_without_config(self):
        """Test DFlashAttention defaults to no sliding window."""
        from modelopt.torch.speculative.plugins.hf_dflash import DFlashAttention
        from transformers import PretrainedConfig

        config = PretrainedConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            _attn_implementation="sdpa",
        )
        attn = DFlashAttention(config, layer_idx=0)
        assert attn.sliding_window is None
