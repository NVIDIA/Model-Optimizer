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

"""Create a simple identity neural network with a custom IdentityConv operator.

This script generates an ONNX model consisting of three convolutional layers where the
second Conv node is replaced with a custom ``IdentityConv`` operator. The custom operator
is not defined in the standard ONNX operator set and requires a TensorRT plugin to parse.

Based on https://github.com/leimao/TensorRT-Custom-Plugin-Example.
"""

import argparse
import os

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def create_identity_neural_network(output_path: str) -> None:
    """Create and save an ONNX model with a custom IdentityConv operator."""
    opset_version = 15

    input_shape = (1, 3, 480, 960)
    input_channels = input_shape[1]

    # Configure identity convolution weights (depthwise, 1x1 kernel with all ones).
    weights_shape = (input_channels, 1, 1, 1)
    num_groups = input_channels
    weights_data = np.ones(weights_shape, dtype=np.float32)

    # Build the ONNX graph using onnx-graphsurgeon.
    x0 = gs.Variable(name="X0", dtype=np.float32, shape=input_shape)
    w0 = gs.Constant(name="W0", values=weights_data)
    x1 = gs.Variable(name="X1", dtype=np.float32, shape=input_shape)
    w1 = gs.Constant(name="W1", values=weights_data)
    x2 = gs.Variable(name="X2", dtype=np.float32, shape=input_shape)
    w2 = gs.Constant(name="W2", values=weights_data)
    x3 = gs.Variable(name="X3", dtype=np.float32, shape=input_shape)

    conv_attrs = {
        "kernel_shape": [1, 1],
        "strides": [1, 1],
        "pads": [0, 0, 0, 0],
        "group": num_groups,
    }

    node_1 = gs.Node(name="Conv-1", op="Conv", inputs=[x0, w0], outputs=[x1], attrs=conv_attrs)

    # The second node uses the custom IdentityConv operator instead of standard Conv.
    # This operator requires a TensorRT plugin to be loaded at runtime.
    node_2 = gs.Node(
        name="Conv-2",
        op="IdentityConv",
        inputs=[x1, w1],
        outputs=[x2],
        attrs={
            **conv_attrs,
            "plugin_version": "1",
            "plugin_namespace": "",
        },
    )

    node_3 = gs.Node(name="Conv-3", op="Conv", inputs=[x2, w2], outputs=[x3], attrs=conv_attrs)

    graph = gs.Graph(
        nodes=[node_1, node_2, node_3],
        inputs=[x0],
        outputs=[x3],
        opset=opset_version,
    )
    model = gs.export_onnx(graph)
    # Shape inference does not work with the custom operator.
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    onnx.save(model, output_path)
    print(f"Saved ONNX model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an ONNX model with a custom IdentityConv operator."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="identity_neural_network.onnx",
        help="Path to save the generated ONNX model.",
    )
    args = parser.parse_args()
    create_identity_neural_network(args.output_path)
