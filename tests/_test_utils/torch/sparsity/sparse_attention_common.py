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

"""Common utilities for sparse attention testing."""

import torch
import torch.nn as nn

import modelopt.torch.opt as mto
import modelopt.torch.sparsity.attention_sparsity as sparse_attn
from modelopt.torch.sparsity.attention_sparsity.sparse_attention import SparseAttentionModule


# Test models for sparse attention
class SimpleAttentionModel(nn.Module):
    """Simple attention model for testing."""

    def __init__(self, hidden_size=256, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x, need_weights=False)
        return self.fc(attn_output)

    @classmethod
    def get_input(cls, hidden_size=256, seq_len=10, batch_size=2):
        """Get input tensor for testing."""
        return torch.randn(batch_size, seq_len, hidden_size)


class SimpleTransformerEncoderLayer(nn.Module):
    """Simple TransformerEncoderLayer wrapper for testing."""

    def __init__(self, d_model=128, nhead=4, dim_feedforward=256):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

    def forward(self, x):
        return self.layer(x)

    @classmethod
    def get_input(cls, d_model=128, seq_len=20, batch_size=2):
        """Get input tensor for testing."""
        return torch.randn(batch_size, seq_len, d_model)


class SimpleTransformerEncoder(nn.Module):
    """Simple TransformerEncoder wrapper for testing."""

    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers,
        )

    def forward(self, x):
        return self.encoder(x)

    @classmethod
    def get_input(cls, d_model=128, seq_len=10, batch_size=2):
        """Get input tensor for testing."""
        return torch.randn(batch_size, seq_len, d_model)


# Test configurations
FLASH_SKIP_SOFTMAX_DEFAULT_CFG = {
    "sparse_cfg": {
        "*attention*": {
            "method": "flash_skip_softmax",
            "threshold": 1e-4,
            "br": 128,
            "bc": 128,
            "enable": True,
        }
    },
}

FLASH_SKIP_SOFTMAX_PHASE_AWARE_CFG = {
    "sparse_cfg": {
        "*attention*": {
            "method": "flash_skip_softmax",
            "threshold": {"prefill": 1e-3, "decode": 1e-5},
            "br": 128,
            "bc": 128,
            "enable": True,
        }
    },
}


def get_test_configs():
    """Get test configurations for parameterized tests.

    Note: Calibration config excluded (requires GPU and real tokenizers).
    """
    return [FLASH_SKIP_SOFTMAX_DEFAULT_CFG, FLASH_SKIP_SOFTMAX_PHASE_AWARE_CFG]


def sparsify_model_and_forward(model, config, calib_data):
    """Apply sparse attention and run forward passes.

    Args:
        model: Model to sparsify
        config: Sparse attention configuration
        calib_data: List of calibration data tensors

    Returns:
        Sparsified model
    """

    def forward_loop(model):
        for batch in calib_data:
            model(batch)

    # Apply sparse attention
    model = sparse_attn.sparsify(model, config, forward_loop=forward_loop)

    # Verify sparse attention modules were inserted
    assert any(isinstance(m, SparseAttentionModule) for m in model.modules()), (
        "No sparse attention modules found"
    )

    # Test forward passes
    model.eval()
    with torch.no_grad():
        for batch in calib_data:
            output = model(batch)
            assert not torch.isnan(output).any(), "NaN in output"
            assert output is not None, "Output is None"

    return model


def save_restore_test(model_cls, device, sparse_config):
    """Test save and restore of sparse attention state.

    Args:
        model_cls: Model class to test
        device: Device to run on ('cpu' or 'cuda')
        sparse_config: Sparse attention configuration
    """
    # Create and sparsify reference model
    model_sparse = model_cls().to(device)
    calib_data = [model_sparse.get_input().to(device) for _ in range(2)]

    sparsify_model_and_forward(model_sparse, sparse_config, calib_data)

    # Save state
    state_dict = mto.modelopt_state(model_sparse)

    # Restore to new model
    model_restored = model_cls().to(device)
    mto.restore_from_modelopt_state(model_restored, state_dict)
    model_restored.load_state_dict(model_sparse.state_dict())

    # Verify outputs match
    test_input = calib_data[0]
    model_sparse.eval()
    model_restored.eval()

    with torch.no_grad():
        output_sparse = model_sparse(test_input)
        output_restored = model_restored(test_input)

    assert torch.allclose(output_sparse, output_restored, atol=1e-6), (
        "Restored model output doesn't match original"
    )
