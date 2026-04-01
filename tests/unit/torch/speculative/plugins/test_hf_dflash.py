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

"""Unit tests for DFlash speculative decoding plugin."""

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
    create_dflash_attention_mask,
    create_dflash_loss_mask,
)

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be multiple of BLOCK_SIZE


def _get_dflash_config(block_size=BLOCK_SIZE, num_layers=NUM_DRAFT_LAYERS):
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
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert isinstance(model, HFDFlashModel)

    def test_convert_creates_dflash_module(self):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "dflash_module")
        assert isinstance(model.dflash_module, DFlashModule)

    def test_convert_freezes_base_model(self):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        # Base model params should be frozen
        for name, param in model.named_parameters():
            if "dflash_module" not in name:
                assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_convert_dflash_module_trainable(self):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        # DFlash module params should be trainable
        dflash_params = [(n, p) for n, p in model.named_parameters() if "dflash_module" in n]
        assert len(dflash_params) > 0
        for name, param in dflash_params:
            assert param.requires_grad, f"DFlash param {name} should be trainable"

    def test_convert_sets_target_layer_ids(self):
        model = get_tiny_llama(num_hidden_layers=8)
        config = _get_dflash_config(num_layers=3)
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "target_layer_ids")
        assert len(model.target_layer_ids) == 3
        # Layer IDs should be within target model range
        for lid in model.target_layer_ids:
            assert 0 <= lid < 8

    def test_convert_sets_mask_token_id(self):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        assert hasattr(model, "mask_token_id")
        assert model.mask_token_id == 0


class TestDFlashSaveRestore:
    """Test DFlash model save and restore."""

    def test_save_and_restore(self, tmp_path):
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
        """Noise (right half): causal within same block, blocked across blocks."""
        mask = create_dflash_attention_mask(8, 4, "cpu", torch.float32)
        mask_2d = mask[0, 0]
        noise_mask = mask_2d[:, 8:]  # noise part

        # Block 0, position 0: can only see position 0
        assert noise_mask[0, 0] == 0
        assert (noise_mask[0, 1:4] < 0).all()

        # Block 0, position 3: can see positions 0-3
        assert (noise_mask[3, :4] == 0).all()

        # Block 1 cannot see block 0 noise
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
        # All positions in block 0 should be masked out
        assert (mask[:BLOCK_SIZE] == 0).all()

    def test_loss_mask_excludes_block_starts(self):
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        # Block start positions (every BLOCK_SIZE) should be masked
        for i in range(0, SEQ_LEN, BLOCK_SIZE):
            assert mask[i] == 0, f"Block start position {i} should be masked"

    def test_loss_mask_includes_non_start_positions(self):
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        # Non-start positions in non-zero blocks should be included
        for b in range(1, SEQ_LEN // BLOCK_SIZE):
            for offset in range(1, BLOCK_SIZE):
                pos = b * BLOCK_SIZE + offset
                assert mask[pos] == 1, f"Position {pos} should be in loss"

    def test_loss_mask_count(self):
        mask = create_dflash_loss_mask(SEQ_LEN, BLOCK_SIZE, "cpu")
        num_blocks = SEQ_LEN // BLOCK_SIZE
        # Block 0 excluded entirely (BLOCK_SIZE positions)
        # Each remaining block excludes 1 start position
        expected = (num_blocks - 1) * (BLOCK_SIZE - 1)
        assert mask.sum().item() == expected


class TestDFlashModule:
    """Test DFlash draft module forward pass."""

    @pytest.fixture
    def model_and_config(self):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        return model

    def test_dflash_module_forward_shape(self, model_and_config):
        model = model_and_config
        bsz = 2
        hidden_size = model.config.hidden_size
        num_layers = len(model.target_layer_ids)

        # Create inputs matching training forward
        target_hidden = torch.randn(bsz, SEQ_LEN, num_layers * hidden_size)
        noise_emb = torch.randn(bsz, SEQ_LEN, hidden_size)
        pos_ids = (
            torch.cat([torch.arange(SEQ_LEN), torch.arange(SEQ_LEN)]).unsqueeze(0).expand(bsz, -1)
        )

        output = model.dflash_module(
            noise_embedding=noise_emb,
            target_hidden=target_hidden,
            position_ids=pos_ids,
            attention_mask=None,
        )
        assert output.shape == (bsz, SEQ_LEN, hidden_size)

    def test_dflash_module_deterministic(self, model_and_config):
        model = model_and_config
        model.eval()
        bsz = 1
        hidden_size = model.config.hidden_size
        num_layers = len(model.target_layer_ids)

        target_hidden = torch.randn(bsz, SEQ_LEN, num_layers * hidden_size)
        noise_emb = torch.randn(bsz, SEQ_LEN, hidden_size)
        pos_ids = torch.cat([torch.arange(SEQ_LEN), torch.arange(SEQ_LEN)]).unsqueeze(0)

        with torch.no_grad():
            out1 = model.dflash_module(
                noise_embedding=noise_emb,
                target_hidden=target_hidden,
                position_ids=pos_ids,
            )
            out2 = model.dflash_module(
                noise_embedding=noise_emb,
                target_hidden=target_hidden,
                position_ids=pos_ids,
            )
        assert torch.allclose(out1, out2)


class TestDFlashTrainingForward:
    """Test DFlash training forward pass end-to-end."""

    @pytest.fixture
    def model(self):
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        model.train()
        return model

    def test_training_forward_returns_loss(self, model):
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN))
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(output, "loss")
        assert output.loss.requires_grad

    def test_training_forward_returns_accuracy(self, model):
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN))
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(output, "train_acc")

    def test_training_forward_with_labels(self, model):
        """Test that labels are used for response-only loss masking."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN))
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long)

        # Labels with -100 for first half (masked), real labels for second half
        labels = torch.full((bsz, SEQ_LEN), -100, dtype=torch.long)
        labels[:, SEQ_LEN // 2 :] = input_ids[:, SEQ_LEN // 2 :]

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert hasattr(output, "loss")
        assert output.loss.requires_grad

    def test_training_forward_all_masked_labels(self, model):
        """Test that all-masked labels produce zero loss without crashing."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN))
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long)
        labels = torch.full((bsz, SEQ_LEN), -100, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert output.loss.item() == 0.0

    def test_training_backward(self, model):
        """Test that gradients flow to dflash_module."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN))
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        output.loss.backward()

        # Check dflash_module has gradients
        has_grad = False
        for name, param in model.dflash_module.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "DFlash module should receive gradients"

    def test_eval_forward_uses_base_model(self, model):
        """In eval mode, forward should use base model (not DFlash training)."""
        model.eval()
        bsz = 1
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN))

        with torch.no_grad():
            output = model(input_ids=input_ids)
        # Should return base model output (logits over vocab)
        assert output.logits.shape == (bsz, SEQ_LEN, model.config.vocab_size)


class TestBuildTargetLayerIds:
    """Test target layer selection."""

    def test_single_draft_layer(self):
        from modelopt.torch.speculative.plugins.hf_dflash import build_target_layer_ids

        ids = build_target_layer_ids(32, 1)
        assert len(ids) == 1
        assert ids[0] == 16  # middle layer

    def test_multiple_draft_layers(self):
        from modelopt.torch.speculative.plugins.hf_dflash import build_target_layer_ids

        ids = build_target_layer_ids(36, 5)
        assert len(ids) == 5
        # Should be monotonically increasing
        assert ids == sorted(ids)
        # Should be within [1, 33] for 36-layer model
        assert all(1 <= lid <= 33 for lid in ids)

    def test_layer_ids_spread(self):
        from modelopt.torch.speculative.plugins.hf_dflash import build_target_layer_ids

        ids = build_target_layer_ids(32, 5)
        assert len(ids) == 5
        # No duplicates
        assert len(set(ids)) == 5
