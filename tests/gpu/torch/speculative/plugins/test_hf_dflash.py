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

"""GPU tests for DFlash speculative decoding plugin.

These tests require a CUDA GPU. CPU-only tests are in tests/unit/.
"""

from copy import deepcopy

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_llama

import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import DFLASH_DEFAULT_CFG

BLOCK_SIZE = 4
NUM_DRAFT_LAYERS = 2
SEQ_LEN = 16  # must be multiple of BLOCK_SIZE


def _get_dflash_config(block_size=BLOCK_SIZE, num_layers=NUM_DRAFT_LAYERS):
    """Create a DFlash config for testing."""
    config = deepcopy(DFLASH_DEFAULT_CFG["config"])
    config["dflash_block_size"] = block_size
    config["dflash_use_torch_compile"] = False
    config["dflash_mask_token_id"] = 0
    config["dflash_architecture_config"] = {
        "num_hidden_layers": num_layers,
    }
    return config


@pytest.fixture
def dflash_model():
    """Create a tiny DFlash model on GPU."""
    model = get_tiny_llama(num_hidden_layers=4)
    config = _get_dflash_config()
    mtsp.convert(model, [("dflash", config)])
    model = model.cuda()
    return model


class TestDFlashModuleGPU:
    """Test DFlash draft module forward pass on GPU."""

    def test_dflash_module_forward_shape(self, dflash_model):
        """Test that draft module produces correct output shape."""
        model = dflash_model
        bsz = 2
        hidden_size = model.config.hidden_size
        num_layers = len(model.target_layer_ids)

        dtype = next(model.dflash_module.parameters()).dtype
        target_hidden = torch.randn(
            bsz, SEQ_LEN, num_layers * hidden_size, device="cuda", dtype=dtype
        )
        noise_emb = torch.randn(bsz, SEQ_LEN, hidden_size, device="cuda", dtype=dtype)
        pos_ids = (
            torch.cat([torch.arange(SEQ_LEN), torch.arange(SEQ_LEN)])
            .unsqueeze(0)
            .expand(bsz, -1)
            .cuda()
        )

        output = model.dflash_module(
            noise_embedding=noise_emb,
            target_hidden=target_hidden,
            position_ids=pos_ids,
            attention_mask=None,
        )
        assert output.shape == (bsz, SEQ_LEN, hidden_size)

    def test_dflash_module_deterministic(self, dflash_model):
        """Test that draft module produces identical outputs for same input."""
        model = dflash_model
        model.eval()
        bsz = 1
        hidden_size = model.config.hidden_size
        num_layers = len(model.target_layer_ids)

        dtype = next(model.dflash_module.parameters()).dtype
        target_hidden = torch.randn(
            bsz, SEQ_LEN, num_layers * hidden_size, device="cuda", dtype=dtype
        )
        noise_emb = torch.randn(bsz, SEQ_LEN, hidden_size, device="cuda", dtype=dtype)
        pos_ids = torch.cat([torch.arange(SEQ_LEN), torch.arange(SEQ_LEN)]).unsqueeze(0).cuda()

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


class TestDFlashTrainingForwardGPU:
    """Test DFlash training forward pass end-to-end on GPU."""

    @pytest.fixture
    def model(self):
        """Create a tiny DFlash model in training mode on GPU."""
        model = get_tiny_llama(num_hidden_layers=4)
        config = _get_dflash_config()
        mtsp.convert(model, [("dflash", config)])
        model = model.cuda()
        model.train()
        return model

    def test_training_forward_returns_loss(self, model):
        """Test that training forward returns a differentiable loss."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(output, "loss")
        assert output.loss.requires_grad

    def test_training_forward_returns_accuracy(self, model):
        """Test that training forward returns train_acc."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(output, "train_acc")

    def test_training_forward_with_labels(self, model):
        """Test that labels are used for response-only loss masking."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")

        # Labels with -100 for first half (masked), real labels for second half
        labels = torch.full((bsz, SEQ_LEN), -100, dtype=torch.long, device="cuda")
        labels[:, SEQ_LEN // 2 :] = input_ids[:, SEQ_LEN // 2 :]

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert hasattr(output, "loss")
        assert output.loss.requires_grad

    def test_training_forward_all_masked_labels(self, model):
        """Test that all-masked labels produce zero loss without crashing."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")
        labels = torch.full((bsz, SEQ_LEN), -100, dtype=torch.long, device="cuda")

        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert output.loss.item() == 0.0

    def test_training_backward(self, model):
        """Test that gradients flow to dflash_module."""
        bsz = 2
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")
        attention_mask = torch.ones(bsz, SEQ_LEN, dtype=torch.long, device="cuda")

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        output.loss.backward()

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
        input_ids = torch.randint(0, model.config.vocab_size, (bsz, SEQ_LEN), device="cuda")

        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (bsz, SEQ_LEN, model.config.vocab_size)
