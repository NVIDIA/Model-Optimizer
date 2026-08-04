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

"""Tests for PyTorch Geometric quantization plugin."""

import pytest
import torch
import torch.nn as nn
from _test_utils.torch.misc import set_seed
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv

import modelopt.torch.quantization as mtq


class TestPyTorchGeometricPlugin:
    """Test PyTorch Geometric quantization support."""

    @pytest.fixture(autouse=True)
    def setup_seed(self):
        """Set seed before each test function."""
        set_seed()

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_graph_data(self, batch_size=2, num_nodes=20, in_channels=16, device="cpu"):
        """Create sample graph data for testing."""
        x = torch.randn(batch_size * num_nodes, in_channels, device=device)
        # Create batch assignment
        batch = torch.cat([torch.full((num_nodes,), i, device=device) for i in range(batch_size)])

        # Create edge indices for each graph
        edge_list = []
        offset = 0
        for _ in range(batch_size):
            # Create random edges within each graph
            src = torch.randint(0, num_nodes, (50,), device=device) + offset
            dst = torch.randint(0, num_nodes, (50,), device=device) + offset
            edge_list.append(torch.stack([src, dst]))
            offset += num_nodes

        edge_index = torch.cat(edge_list, dim=1)
        edge_attr = torch.randn(edge_index.size(1), 32, device=device)

        return x, edge_index, edge_attr, batch

    def test_gat_conv_quantization(self, device):
        """Test GATConv layer quantization."""

        class GATModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.gat1 = GATConv(16, 64, heads=4, edge_dim=32)
                self.gat2 = GATConv(256, 32, heads=1, edge_dim=32)

            def forward(self, x, edge_index, edge_attr):
                x = torch.relu(self.gat1(x, edge_index, edge_attr))
                return self.gat2(x, edge_index, edge_attr)

        model = GATModel().to(device)

        # Calibration function
        def calibrate(m):
            m.eval()
            with torch.no_grad():
                for _ in range(5):
                    x, edge_index, edge_attr, _ = self.create_graph_data(device=device)
                    _ = m(x, edge_index, edge_attr)

        # Quantize model
        quantized = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, calibrate)

        # Verify quantization
        quantizer_count = sum(
            1 for _, m in quantized.named_modules() if "quantizer" in type(m).__name__.lower()
        )
        assert quantizer_count > 0, "No quantizers were inserted"

        # Test forward pass
        x, edge_index, edge_attr, _ = self.create_graph_data(device=device)
        with torch.no_grad():
            output = quantized(x, edge_index, edge_attr)
        assert output is not None

    def test_multiple_layer_types(self, device):
        """Test quantization of multiple PyG layer types."""

        class MultiLayerGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.gcn = GCNConv(16, 32)
                self.sage = SAGEConv(32, 64)
                self.transformer = TransformerConv(64, 32, heads=2)

            def forward(self, x, edge_index):
                x = torch.relu(self.gcn(x, edge_index))
                x = torch.relu(self.sage(x, edge_index))
                return self.transformer(x, edge_index)

        model = MultiLayerGNN().to(device)

        # Calibration
        def calibrate(m):
            m.eval()
            with torch.no_grad():
                for _ in range(3):
                    x = torch.randn(50, 16, device=device)
                    edge_index = torch.randint(0, 50, (2, 100), device=device)
                    _ = m(x, edge_index)

        # Quantize
        quantized = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, calibrate)

        # Check that PyG Linear layers were quantized
        pyg_linear_count = 0
        for name, module in model.named_modules():
            if hasattr(module, "lin") and "torch_geometric" in str(type(module.lin)):
                pyg_linear_count += 1

        quantizer_count = sum(
            1 for _, m in quantized.named_modules() if "quantizer" in type(m).__name__.lower()
        )

        # Each PyG linear should have at least 2 quantizers (input, weight)
        assert quantizer_count >= pyg_linear_count * 2, (
            f"Expected at least {pyg_linear_count * 2} quantizers, got {quantizer_count}"
        )

    def test_quantization_accuracy(self, device):
        """Test that quantization maintains reasonable accuracy."""
        # Set seed for this test specifically to ensure reproducibility
        set_seed()

        model = GATConv(16, 32, heads=2, edge_dim=16).to(device)

        # Create test data
        x, edge_index, edge_attr, _ = self.create_graph_data(
            batch_size=1, in_channels=16, device=device
        )
        edge_attr = edge_attr[:, :16]  # Match edge_dim

        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(x, edge_index, edge_attr)

        # Calibration with multiple samples for more stable quantization
        def calibrate(m):
            m.eval()
            with torch.no_grad():
                # Use multiple calibration samples for better stability
                for _ in range(5):
                    x_cal, edge_index_cal, edge_attr_cal, _ = self.create_graph_data(
                        batch_size=1, in_channels=16, device=device
                    )
                    edge_attr_cal = edge_attr_cal[:, :16]  # Match edge_dim
                    _ = m(x_cal, edge_index_cal, edge_attr_cal)

        # Quantize
        quantized = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, calibrate)

        # Get quantized output
        with torch.no_grad():
            quantized_output = quantized(x, edge_index, edge_attr)

        # Check relative error
        abs_diff = torch.abs(original_output - quantized_output)
        relative_error = abs_diff / (torch.abs(original_output) + 1e-8)
        mean_relative_error = relative_error.mean().item()

        assert mean_relative_error < 0.1, f"Quantization error too large: {mean_relative_error:.2%}"
