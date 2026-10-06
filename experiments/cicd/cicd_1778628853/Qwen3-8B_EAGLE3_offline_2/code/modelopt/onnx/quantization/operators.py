# Adapted from https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/operators/conv.py
# and https://github.com/microsoft/onnxruntime/blob/baeece44ba075009c6bfe95891a8c1b3d4571cb3/onnxruntime/python/tools/quantization/operators/gather.py
#
# MIT License
#
# Copyright (c) Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT
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

"""Additional or modified QDQ operators on top of ORT quantized operators."""

from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase

from ..op_types import is_normalization_op


class QDQNormalization(QDQOperatorBase):
    """By default, ORT does not quantize Normalization ops. This module is intended to help with that.

    Note. QDQOperatorBase is not sufficient for dynamic input only quantization.
    """

    def __init__(self, onnx_quantizer, onnx_node):
        """Normalization quantizer init."""
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Main function to quantize the Normalization ops."""
        node = self.node
        assert is_normalization_op(node.op_type)

        # Quantize only the dynamic input (first input to the op)
        self.quantizer.quantize_activation_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_activation_tensor(node.output[0])


class QDQConvTranspose(QDQOperatorBase):
    """QDQ for ConvTranspose operator."""

    def __init__(self, onnx_quantizer, onnx_node):
        """ConvTranspose quantizer init."""
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Main function to quantize the ConvTranspose ops."""
        node = self.node
        assert node.op_type == "ConvTranspose"

        self.quantizer.quantize_activation_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_activation_tensor(node.output[0])

        is_weight_per_channel, weight_axis = self.quantizer.is_tensor_per_channel(
            node.input[1], default_axis=1
        )
        if is_weight_per_channel:
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], weight_axis)
        else:
            self.quantizer.quantize_weight_tensor(node.input[1])

        if len(node.input) == 3:
            self.quantizer.quantize_bias_tensor(
                node.name, node.input[2], node.input[0], node.input[1]
            )


class QDQCustomOp(QDQOperatorBase):
    """By default, ORT does not quantize custom ops. This module is intended to help with that.

    Note. QDQOperatorBase is not sufficient for dynamic input and output only quantization.
    """

    def __init__(self, onnx_quantizer, onnx_node):
        """Normalization quantizer init."""
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        """Main function to quantize the custom ops."""
        node = self.node

        # Quantize only the dynamic inputs with type FLOAT or FLOAT16
        for inp in node.input:
            if self.quantizer._is_tensor_quantizable(inp):
                self.quantizer.quantize_activation_tensor(inp)

        # Quantize the outputs with type FLOAT or FLOAT16
        for out in node.output:
            if self.quantizer._is_tensor_quantizable(out):
                self.quantizer.quantize_activation_tensor(out)
