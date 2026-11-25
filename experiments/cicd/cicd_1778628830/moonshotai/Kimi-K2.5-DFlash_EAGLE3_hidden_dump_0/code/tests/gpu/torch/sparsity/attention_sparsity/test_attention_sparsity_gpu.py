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

"""GPU tests for attention sparsity module."""

import pytest
import torch
from _test_utils.torch.sparsity.sparse_attention_common import (
    FLASH_SKIP_SOFTMAX_DEFAULT_CFG,
    SimpleAttentionModel,
    SimpleTransformerEncoder,
    SimpleTransformerEncoderLayer,
    get_test_configs,
    save_restore_test,
    sparsify_model_and_forward,
)

import modelopt.torch.sparsity.attention_sparsity as sparse_attn


class TestAttentionSparsityGPU:
    """GPU tests for attention sparsity."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        self.device = torch.device("cuda")
        torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "model_cls",
        [SimpleAttentionModel, SimpleTransformerEncoderLayer, SimpleTransformerEncoder],
    )
    @pytest.mark.parametrize("config", get_test_configs())
    def test_gpu_forward(self, model_cls, config):
        """Test sparse attention forward pass on GPU."""
        model = model_cls().to(self.device)
        calib_data = [model.get_input().to(self.device) for _ in range(2)]

        sparsify_model_and_forward(model, config, calib_data)

        # Additional GPU-specific checks
        for batch in calib_data:
            with torch.no_grad():
                output = model(batch)
            assert output.device.type == "cuda"

    @pytest.mark.parametrize(
        "model_cls",
        [SimpleAttentionModel, SimpleTransformerEncoderLayer, SimpleTransformerEncoder],
    )
    def test_save_restore(self, model_cls):
        """Test save and restore on GPU."""
        save_restore_test(model_cls, "cuda", FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_different_dtypes(self, dtype):
        """Test sparse attention with different dtypes."""
        model = SimpleTransformerEncoderLayer(d_model=256, nhead=8).to(self.device).to(dtype)
        calib_data = [model.get_input(d_model=256).to(self.device).to(dtype) for _ in range(2)]

        sparse_model = sparsify_model_and_forward(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG, calib_data)

        # Test forward
        x = model.get_input(d_model=256).to(self.device).to(dtype)
        with torch.no_grad():
            output = sparse_model(x)

        assert output.dtype == dtype
        assert not torch.isnan(output).any()
        if dtype != torch.bfloat16:  # bfloat16 can have inf
            assert not torch.isinf(output).any()

    def test_backward_pass(self):
        """Test that gradients flow correctly through sparse attention."""
        model = SimpleAttentionModel(hidden_size=128, num_heads=4).to(self.device)
        model = sparse_attn.sparsify(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

        # Enable training mode
        model.train()

        x = model.get_input(hidden_size=128, seq_len=32).to(self.device)
        x.requires_grad = True

        # Forward
        output = model(x)
        loss = output.sum()

        # Backward
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check model gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.parametrize("seq_len", [1, 1024, 2048])
    def test_various_sequence_lengths(self, seq_len):
        """Test sparse attention with various sequence lengths."""
        model = SimpleAttentionModel(hidden_size=128, num_heads=4).to(self.device)
        model = sparse_attn.sparsify(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

        x = model.get_input(hidden_size=128, seq_len=seq_len, batch_size=1).to(self.device)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, seq_len, 128)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("batch_size", [1, 8, 16])
    def test_various_batch_sizes(self, batch_size):
        """Test sparse attention with various batch sizes."""
        model = SimpleTransformerEncoderLayer(d_model=128, nhead=4).to(self.device)
        model = sparse_attn.sparsify(model, FLASH_SKIP_SOFTMAX_DEFAULT_CFG)

        x = model.get_input(d_model=128, seq_len=64, batch_size=batch_size).to(self.device)

        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 64, 128)
        assert not torch.isnan(output).any()
