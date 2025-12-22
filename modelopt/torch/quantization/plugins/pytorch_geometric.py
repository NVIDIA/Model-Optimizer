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

"""PyTorch Geometric quantization plugin.

This plugin enables quantization support for PyTorch Geometric (PyG) layers by registering
PyG's custom Linear layer with ModelOpt's quantization registry.

Example:
    >>> import modelopt.torch.quantization as mtq
    >>> from torch_geometric.nn import GATConv
    >>>
    >>> # Create a model with PyG layers
    >>> class GATModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.gat1 = GATConv(10, 64, heads=4)
    ...         self.gat2 = GATConv(64 * 4, 32, heads=1)
    >>> model = GATModel()
    >>> # PyG layers are now automatically quantizable!
    >>> quantized_model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, calibrate)
"""

from torch_geometric.nn.dense.linear import Linear as PyGLinear

from modelopt.torch.quantization.nn.modules.quant_module import (
    QuantLinearConvBase,
    QuantModuleRegistry,
)
from modelopt.torch.quantization.tensor_quant import QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW


class QuantPyGLinear(QuantLinearConvBase):
    """Quantized version of PyTorch Geometric's Linear layer.

    PyTorch Geometric uses a custom Linear layer that is functionally equivalent to
    torch.nn.Linear but has a different API (in_channels/out_channels instead of
    in_features/out_features). This class enables quantization of PyG Linear layers
    by inheriting from QuantLinearConvBase, which handles all quantization logic.

    The quantization is handled automatically by the base classes:
    - Input quantization: Handled by QuantInputBase.forward()
    - Weight quantization: Handled by QuantLinearConvBase's dynamic weight attribute
    - Output quantization: Handled by QuantInputBase.forward()

    Note:
        Many PyTorch Geometric layers (GCNConv, GATConv, SAGEConv, TransformerConv, etc.)
        internally use PyG Linear layers, so registering this class enables quantization
        for a wide range of graph neural network layers.
    """

    default_quant_desc_weight = QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW


QuantModuleRegistry.register({PyGLinear: "torch_geometric.nn.dense.linear.Linear"})(QuantPyGLinear)
