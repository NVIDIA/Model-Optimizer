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


class TestDFlashAttentionMask:
    """Test DFlash attention mask construction."""

    def test_mask_shape(self):
        """Test mask has shape [1, 1, L, 2L]."""
        mask = create_dflash_attention_mask(SEQ_LEN, BLOCK_SIZE, "cpu", torch.float32)
        assert mask.shape == (1, 1, SEQ_LEN, 2 * SEQ_LEN)

    def test_mask_context_strictly_previous_blocks(self):
        """Context (left half): block B can only see blocks 0..B-1."""
        mask = create_dflash_attention_mask(8, 4, "cpu", torch.float32)
        mask_2d = mask[0, 0]  # [8, 16]
        ctx_mask = mask_2d[:, :8]  # context part

        # Block 0 (rows 0-3) should NOT see any context
        assert (ctx_mask[:4, :] < 0).all()

        # Block 1 (rows 4-7) should see block 0 context only
        assert (ctx_mask[4:8, :4] == 0).all()  # can see block 0
        assert (ctx_mask[4:8, 4:8] < 0).all()  # cannot see own block

    def test_mask_noise_causal_within_block(self):
        """Noise (right half): reverse-causal within same block, matching SpecForge.

        SpecForge uses j >= i: position 0 (anchor) sees all positions in block,
        position B-1 sees only itself. Cross-block noise is fully masked.
        """
        mask = create_dflash_attention_mask(8, 4, "cpu", torch.float32)
        mask_2d = mask[0, 0]
        noise_mask = mask_2d[:, 8:]  # noise part

        # Block 0, position 0: can see all positions in block (0-3)
        assert (noise_mask[0, :4] == 0).all()

        # Block 0, position 3: can only see position 3
        assert (noise_mask[3, :3] < 0).all()
        assert noise_mask[3, 3] == 0

        # Block 1 cannot see block 0 noise
        assert (noise_mask[4:8, :4] < 0).all()

    def test_mask_values_are_zero_or_neg_inf(self):
        """Test mask contains only 0 (attend) and -inf (mask)."""
        mask = create_dflash_attention_mask(SEQ_LEN, BLOCK_SIZE, "cpu", torch.float32)
        unique_vals = mask.unique()
        assert len(unique_vals) == 2
        assert 0.0 in unique_vals
        assert unique_vals.min() == torch.finfo(torch.float32).min


class TestDFlashLossMask:
    """Test DFlash loss mask construction."""

    def test_loss_mask_shape(self):
        """Test loss mask has shape [L]."""
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        assert mask.shape == (SEQ_LEN,)

    def test_loss_mask_excludes_block_zero(self):
        """Test all positions in block 0 are masked out."""
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        assert (mask[:BLOCK_SIZE] == 0).all()

    def test_loss_mask_excludes_block_starts(self):
        """Test block start positions are masked."""
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        for i in range(0, SEQ_LEN, BLOCK_SIZE):
            assert mask[i] == 0, f"Block start position {i} should be masked"

    def test_loss_mask_includes_non_start_positions(self):
        """Test non-start positions in non-zero blocks are included."""
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        for b in range(1, SEQ_LEN // BLOCK_SIZE):
            for offset in range(1, BLOCK_SIZE):
                pos = b * BLOCK_SIZE + offset
                assert mask[pos] == 1, f"Position {pos} should be in loss"

    def test_loss_mask_count(self):
        """Test total active positions matches expected count."""
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        num_blocks = SEQ_LEN // BLOCK_SIZE
        expected = (num_blocks - 1) * (BLOCK_SIZE - 1)
        assert mask.sum().item() == expected


class TestBuildTargetLayerIds:
    """Test target layer selection."""

    def test_single_draft_layer(self):
        """Test single draft layer selects middle target layer."""
        from modelopt.torch.speculative.plugins.hf_dflash import build_target_layer_ids

        ids = build_target_layer_ids(32, 1)
        assert len(ids) == 1
        assert ids[0] == 16  # middle layer

    def test_multiple_draft_layers(self):
        """Test multiple draft layers are monotonically increasing and in bounds."""
        from modelopt.torch.speculative.plugins.hf_dflash import build_target_layer_ids

        ids = build_target_layer_ids(36, 5)
        assert len(ids) == 5
        assert ids == sorted(ids)
        assert all(1 <= lid <= 33 for lid in ids)

    def test_layer_ids_spread(self):
        """Test layer IDs have no duplicates."""
        from modelopt.torch.speculative.plugins.hf_dflash import build_target_layer_ids

        ids = build_target_layer_ids(32, 5)
        assert len(ids) == 5
        assert len(set(ids)) == 5
