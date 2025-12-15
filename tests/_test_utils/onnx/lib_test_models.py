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

import numpy as np
import onnx
import torch
import torch.nn as nn
import torchvision
from onnx import helper


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.down1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bn = self.bottleneck(self.pool3(d3))

        u3 = self.up3(bn)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.dec3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1)

        return self.final(u1)


class SimpleMLP(nn.Module):
    """Simple toy model."""

    def __init__(self, fi=16, f1=18, f2=20, fo=22, bias_add=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fi, f1, bias=False),
            nn.ReLU(),
            nn.Linear(f1, f2, bias=False),
            nn.ReLU(),
            nn.Linear(f2, fo, bias=False),
        )
        self.bias_add = bias_add

    def forward(self, x):
        for mod in self.net:
            x = mod(x)
        if self.bias_add:
            x += 1e-4
        return x


class NonSimplifiedModel(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

    def forward(self, x):
        for mod in self.net:
            x = mod(x)
        return x


def export_as_onnx(
    model,
    input_tensor,
    onnx_filename,
    input_names=["input"],
    output_names=["output"],
    opset=13,
    do_constant_folding=True,
):
    model.eval()
    torch.onnx.export(
        model,
        input_tensor,
        onnx_filename,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=do_constant_folding,
        dynamo=False,
    )


def build_r1a_model():
    # Define your model inputs and outputs
    input_names = ["input_1"]
    output_names = ["resnet50/conv1_conv/BiasAdd:0"]
    input_shapes = [(1, 3, 224, 224)]
    output_shapes = [(1, 64, 112, 112)]

    inputs = [
        helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        for input_name, input_shape in zip(input_names, input_shapes)
    ]
    outputs = [
        helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
        for output_name, output_shape in zip(output_names, output_shapes)
    ]

    # Create the ONNX graph with the nodes
    nodes = [
        helper.make_node(
            op_type="Conv",
            inputs=["input_1", "resnet50/conv1_conv/Conv2D/ReadVariableOp:0"],
            outputs=["resnet50/conv1_conv/Conv2D:0"],
            name="resnet50/conv1_conv/Conv2D",
            dilations=[1, 1],
            strides=[2, 2],
            kernel_shape=[7, 7],
            group=1,
            auto_pad="NOTSET",
            pads=[3, 3, 3, 3],
        ),
        helper.make_node(
            op_type="Add",
            inputs=["resnet50/conv1_conv/Conv2D:0", "Reshape__7:0"],
            outputs=["resnet50/conv1_conv/BiasAdd:0"],
            name="resnet50/conv1_conv/BiasAdd",
        ),
    ]

    # Create the ONNX initializers
    initializers = [
        helper.make_tensor(
            name="resnet50/conv1_conv/Conv2D/ReadVariableOp:0",
            data_type=onnx.TensorProto.FLOAT,
            dims=(64, 3, 7, 7),
            vals=np.random.uniform(low=0.5, high=1.0, size=64 * 3 * 7 * 7),
        ),
        helper.make_tensor(
            name="Reshape__7:0",
            data_type=onnx.TensorProto.FLOAT,
            dims=(64, 1, 1),
            vals=np.random.uniform(low=0.5, high=1.0, size=64 * 1 * 1),
        ),
    ]

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(nodes, "r1a", inputs, outputs, initializer=initializers)

    # Create the ONNX model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    # TODO: Remove all manual ir_version changes in tests once ORT supports ir_version 11
    model.ir_version = 10

    # Check the ONNX model
    model_inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model_inferred)

    return model_inferred


def build_resnet_block(feature_dim=16, num_blocks=1, final_conv=False, out_dim=None):
    """
    Construct a network block based on the BasicBlock used in ResNet 18 and 34.
    """
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    for i in range(num_blocks):
        odim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        feat_layers.append(torchvision.models.resnet.BasicBlock(feature_dim, odim))
    input_tensor = torch.randn(1, feature_dim, 28, 28)
    return nn.Sequential(*feat_layers), input_tensor


def build_resnet_block_with_downsample(
    feature_dim=16, num_blocks=1, final_conv=False, out_dim=None
):
    """
    Construct a network block based on the BasicBlock used in ResNet 18 and 34.
    """
    if out_dim is None:
        out_dim = feature_dim
    feat_layers = []
    for i in range(num_blocks):
        output_dim = feature_dim if i < num_blocks - 1 + int(final_conv) else out_dim
        downsample = nn.Sequential(
            torchvision.models.resnet.conv3x3(feature_dim, output_dim), nn.BatchNorm2d(output_dim)
        )
        feat_layers.append(
            torchvision.models.resnet.BasicBlock(feature_dim, output_dim, downsample=downsample)
        )
    input_tensor = torch.randn(1, feature_dim, 28, 28)
    return nn.Sequential(*feat_layers), input_tensor


def find_init(onnx_model: onnx.ModelProto, init_name: str) -> np.ndarray:
    ret = None
    for init in onnx_model.graph.initializer:
        if init.name == init_name:
            ret = init
            break

    assert ret is not None
    return onnx.numpy_helper.to_array(ret)


#
#                 |
#              conv_1
#             /   |
#         conv_2  |
#          /  |   |
#     conv_3  |   |
#        |    |   |
#       concatenation
#             |
#           conv_4
#             |
#
def build_conv_concat_model():
    # Define your model inputs and outputs
    input_names = ["input_0"]
    output_names = ["conv4_conv/Conv2D:0"]
    input_shapes = [(1, 128, 240, 64)]
    output_shapes = [(1, 256, 240, 64)]

    inputs = [
        helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        for input_name, input_shape in zip(input_names, input_shapes)
    ]
    outputs = [
        helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
        for output_name, output_shape in zip(output_names, output_shapes)
    ]

    # Create the ONNX graph with the nodes
    nodes = [
        helper.make_node(
            op_type="Conv",
            inputs=["input_0", "ReadVariableOp:0"],
            outputs=["conv1_conv/Conv2D:0"],
            name="conv1_conv/Conv2D",
            dilations=[1, 1],
            strides=[1, 1],
            kernel_shape=[3, 3],
            group=1,
            auto_pad="NOTSET",
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            op_type="Conv",
            inputs=["conv1_conv/Conv2D:0", "ReadVariableOp:1"],
            outputs=["conv2_conv/Conv2D:0"],
            name="conv2_conv/Conv2D",
            dilations=[1, 1],
            strides=[1, 1],
            kernel_shape=[3, 3],
            group=1,
            auto_pad="NOTSET",
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            op_type="Conv",
            inputs=["conv2_conv/Conv2D:0", "ReadVariableOp:2"],
            outputs=["conv3_conv/Conv2D:0"],
            name="conv3_conv/Conv2D",
            dilations=[1, 1],
            strides=[1, 1],
            kernel_shape=[3, 3],
            group=1,
            auto_pad="NOTSET",
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            op_type="Concat",
            inputs=["conv1_conv/Conv2D:0", "conv2_conv/Conv2D:0", "conv3_conv/Conv2D:0"],
            outputs=["concat0:0"],
            name="concat0",
            axis=1,
        ),
        helper.make_node(
            op_type="Conv",
            inputs=["concat0:0", "ReadVariableOp:3"],
            outputs=["conv4_conv/Conv2D:0"],
            name="conv4_conv/Conv2D",
            dilations=[1, 1],
            strides=[1, 1],
            kernel_shape=[1, 1],
            group=1,
            auto_pad="NOTSET",
            pads=[0, 0, 0, 0],
        ),
    ]

    # Create the ONNX initializers
    initializers = [
        helper.make_tensor(
            name="ReadVariableOp:0",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128, 128, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=128 * 128 * 3 * 3),
        ),
        helper.make_tensor(
            name="ReadVariableOp:1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128, 128, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=128 * 128 * 3 * 3),
        ),
        helper.make_tensor(
            name="ReadVariableOp:2",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128, 128, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=128 * 128 * 3 * 3),
        ),
        helper.make_tensor(
            name="ReadVariableOp:3",
            data_type=onnx.TensorProto.FLOAT,
            dims=(256, 384, 1, 1),
            vals=np.random.uniform(low=0.5, high=1.0, size=256 * 384 * 1 * 1),
        ),
    ]

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(nodes, "r1a", inputs, outputs, initializer=initializers)

    # Create the ONNX model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    model.ir_version = 10

    # Check the ONNX model
    model_inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model_inferred)

    return model_inferred


def build_convtranspose_conv_residual_model():
    # Define your model inputs and outputs
    input_names = ["input_0"]
    output_names = ["output_0"]
    input_shapes = [(2, 39, 96, 192)]
    output_shapes = [(2, 32, 192, 384)]

    inputs = [
        helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        for input_name, input_shape in zip(input_names, input_shapes)
    ]
    outputs = [
        helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
        for output_name, output_shape in zip(output_names, output_shapes)
    ]

    # Create the ONNX graph with the nodes
    nodes = [
        helper.make_node(
            op_type="ConvTranspose",
            inputs=["input_0", "weights_1", "bias_1"],
            outputs=["convtranspose1_convtranspose/ConvTranspose:0"],
            name="convtranspose1_convtranspose/ConvTranspose",
            dilations=[1, 1],
            group=1,
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            strides=[2, 2],
        ),
        helper.make_node(
            op_type="Relu",
            inputs=["convtranspose1_convtranspose/ConvTranspose:0"],
            outputs=["relu1_relu/Relu:0"],
            name="relu1_relu/Relu",
        ),
        helper.make_node(
            op_type="Conv",
            inputs=["relu1_relu/Relu:0", "weights_2"],
            outputs=["conv2_conv/Conv2D:0"],
            name="conv2_conv/Conv2D",
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            op_type="BatchNormalization",
            inputs=["conv2_conv/Conv2D:0", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var"],
            outputs=["bn1_batchnorm/BatchNormalization:0"],
            name="bn1_batchnorm/BatchNormalization",
        ),
        helper.make_node(
            op_type="Relu",
            inputs=["bn1_batchnorm/BatchNormalization:0"],
            outputs=["relu2_relu/Relu:0"],
            name="relu2_relu/Relu",
        ),
        helper.make_node(
            op_type="Conv",
            inputs=["relu2_relu/Relu:0", "weights_3"],
            outputs=["conv3_conv/Conv2D:0"],
            name="conv3_conv/Conv2D",
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            op_type="BatchNormalization",
            inputs=["conv3_conv/Conv2D:0", "bn2_scale", "bn2_bias", "bn2_mean", "bn2_var"],
            outputs=["bn2_batchnorm/BatchNormalization:0"],
            name="bn2_batchnorm/BatchNormalization",
        ),
        helper.make_node(
            op_type="Add",
            inputs=["relu1_relu/Relu:0", "bn2_batchnorm/BatchNormalization:0"],
            outputs=["add1_add/Add:0"],
            name="add1_add/Add",
        ),
        helper.make_node(
            op_type="Relu",
            inputs=["add1_add/Add:0"],
            outputs=["output_0"],
            name="relu3_relu/Relu",
        ),
    ]

    # Create the ONNX initializers
    initializers = [
        helper.make_tensor(
            name="weights_1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(39, 32, 2, 2),
            vals=np.random.uniform(low=0.5, high=1.0, size=39 * 32 * 2 * 2),
        ),
        helper.make_tensor(
            name="bias_1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32,),
            vals=np.random.uniform(low=0.5, high=1.0, size=32),
        ),
        helper.make_tensor(
            name="weights_2",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32, 32, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=32 * 32 * 3 * 3),
        ),
        helper.make_tensor(
            name="bn1_scale",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32,),
            vals=np.random.uniform(low=0.5, high=1.0, size=32),
        ),
        helper.make_tensor(
            name="bn1_bias",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32,),
            vals=np.random.uniform(low=0.5, high=1.0, size=32),
        ),
        helper.make_tensor(
            name="bn1_mean",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32,),
            vals=np.random.uniform(low=0.5, high=1.0, size=32),
        ),
        helper.make_tensor(
            name="bn1_var",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32,),
            vals=np.random.uniform(low=0.5, high=1.0, size=32),
        ),
        helper.make_tensor(
            name="weights_3",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32, 32, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=32 * 32 * 3 * 3),
        ),
        helper.make_tensor(
            name="bn2_scale",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32,),
            vals=np.random.uniform(low=0.5, high=1.0, size=32),
        ),
        helper.make_tensor(
            name="bn2_bias",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32,),
            vals=np.random.uniform(low=0.5, high=1.0, size=32),
        ),
        helper.make_tensor(
            name="bn2_mean",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32,),
            vals=np.random.uniform(low=0.5, high=1.0, size=32),
        ),
        helper.make_tensor(
            name="bn2_var",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32,),
            vals=np.random.uniform(low=0.5, high=1.0, size=32),
        ),
    ]

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(
        nodes, "convtranspose_conv_residual", inputs, outputs, initializer=initializers
    )

    # Create the ONNX model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    model.ir_version = 10

    # Check the ONNX model
    model_inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model_inferred)

    return model_inferred


def build_conv_batchnorm_sig_mul_model():
    # Define your model inputs and outputs
    input_names = ["input_0"]
    output_names = ["output_0"]
    input_shapes = [(6, 48, 64, 176)]
    output_shapes = [(6, 48, 64, 176)]

    inputs = [
        helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        for input_name, input_shape in zip(input_names, input_shapes)
    ]
    outputs = [
        helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
        for output_name, output_shape in zip(output_names, output_shapes)
    ]

    # Create the ONNX graph with the nodes
    nodes = [
        helper.make_node(
            op_type="Relu",
            inputs=["input_0"],
            outputs=["relu0_relu/Relu:0"],
            name="relu0_relu/Relu",
        ),
        helper.make_node(
            op_type="Conv",
            inputs=["relu0_relu/Relu:0", "weights_1"],
            outputs=["conv1_conv/Conv2D:0"],
            name="conv1_conv/Conv2D",
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            op_type="BatchNormalization",
            inputs=["conv1_conv/Conv2D:0", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var"],
            outputs=["bn1_batchnorm/BatchNormalization:0"],
            name="bn1_batchnorm/BatchNormalization",
        ),
        helper.make_node(
            op_type="Sigmoid",
            inputs=["bn1_batchnorm/BatchNormalization:0"],
            outputs=["sig1_sigmoid/Sigmoid:0"],
            name="sig1_sigmoid/Sigmoid",
        ),
        helper.make_node(
            op_type="Mul",
            inputs=["sig1_sigmoid/Sigmoid:0", "bn1_batchnorm/BatchNormalization:0"],
            outputs=["mul1_mul/Mul:0"],
            name="mul1_mul/Mul",
        ),
        helper.make_node(
            op_type="Add",
            inputs=["relu0_relu/Relu:0", "mul1_mul/Mul:0"],
            outputs=["add1_add/Add:0"],
            name="add1_add/Add",
        ),
        helper.make_node(
            op_type="Relu",
            inputs=["add1_add/Add:0"],
            outputs=["output_0"],
            name="relu2_relu/Relu",
        ),
    ]

    # Create the ONNX initializers
    initializers = [
        helper.make_tensor(
            name="weights_1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(48, 48, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=48 * 48 * 3 * 3),
        ),
        helper.make_tensor(
            name="bn1_scale",
            data_type=onnx.TensorProto.FLOAT,
            dims=(48,),
            vals=np.random.uniform(low=0.5, high=1.0, size=48),
        ),
        helper.make_tensor(
            name="bn1_bias",
            data_type=onnx.TensorProto.FLOAT,
            dims=(48,),
            vals=np.random.uniform(low=0.5, high=1.0, size=48),
        ),
        helper.make_tensor(
            name="bn1_mean",
            data_type=onnx.TensorProto.FLOAT,
            dims=(48,),
            vals=np.random.uniform(low=0.5, high=1.0, size=48),
        ),
        helper.make_tensor(
            name="bn1_var",
            data_type=onnx.TensorProto.FLOAT,
            dims=(48,),
            vals=np.random.uniform(low=0.5, high=1.0, size=48),
        ),
    ]

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(
        nodes, "conv_batchnorm_sig_mul", inputs, outputs, initializer=initializers
    )

    # Create the ONNX model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    model.ir_version = 10

    # Check the ONNX model
    model_inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model_inferred)

    return model_inferred


def build_conv_act_pool_model(include_reshape_node=False):
    # Define your model inputs and outputs
    input_names = ["input_0"]
    output_names = ["output_0"]
    input_shapes = [(32, 64, 256, 256)]
    output_shapes = [(32, 128, 128, 128)]

    inputs = [
        helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        for input_name, input_shape in zip(input_names, input_shapes)
    ]
    outputs = [
        helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
        for output_name, output_shape in zip(output_names, output_shapes)
    ]

    # Create the ONNX graph with the nodes
    nodes = [
        helper.make_node(
            op_type="Conv",
            inputs=["input_0", "weights_1", "bias_1"],
            outputs=["conv1_conv/Conv2D:0"],
            name="conv1_conv/Conv2D",
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            op_type="BatchNormalization",
            inputs=["conv1_conv/Conv2D:0", "bn1_scale", "bn1_bias", "bn1_mean", "bn1_var"],
            outputs=["bn1_batchnorm/BatchNormalization:0"],
            name="bn1_batchnorm/BatchNormalization",
        ),
        helper.make_node(
            op_type="Relu",
            inputs=["bn1_batchnorm/BatchNormalization:0"],
            outputs=["relu1_relu/Relu:0"],
            name="relu1_relu/Relu",
        ),
    ]
    if include_reshape_node:
        nodes.append(
            helper.make_node(
                op_type="Reshape",
                inputs=["relu1_relu/Relu:0", "shape_1"],
                outputs=["reshape1_reshape/Reshape:0"],
                name="reshape1_reshape/Reshape",
            ),
        )
    nodes.extend(
        [
            helper.make_node(
                op_type="MaxPool",
                inputs=[
                    "reshape1_reshape/Reshape:0" if include_reshape_node else "relu1_relu/Relu:0"
                ],
                outputs=["maxpool1_maxpool/MaxPool2D:0"],
                name="maxpool1_maxpool/MaxPool2D",
                ceil_mode=False,
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[2, 2],
            ),
            helper.make_node(
                op_type="Conv",
                inputs=["maxpool1_maxpool/MaxPool2D:0", "weights_2"],
                outputs=["output_0"],
                name="conv2_conv/Conv2D",
                dilations=[1, 1],
                group=1,
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            ),
        ]
    )

    # Create the ONNX initializers
    initializers = [
        helper.make_tensor(
            name="weights_1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128, 64, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=128 * 64 * 3 * 3),
        ),
        helper.make_tensor(
            name="bias_1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128,),
            vals=np.random.uniform(low=0.5, high=1.0, size=128),
        ),
        helper.make_tensor(
            name="bn1_scale",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128,),
            vals=np.random.uniform(low=0.5, high=1.0, size=128),
        ),
        helper.make_tensor(
            name="bn1_bias",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128,),
            vals=np.random.uniform(low=0.5, high=1.0, size=128),
        ),
        helper.make_tensor(
            name="bn1_mean",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128,),
            vals=np.random.uniform(low=0.5, high=1.0, size=128),
        ),
        helper.make_tensor(
            name="bn1_var",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128,),
            vals=np.random.uniform(low=0.5, high=1.0, size=128),
        ),
        helper.make_tensor(
            name="weights_2",
            data_type=onnx.TensorProto.FLOAT,
            dims=(128, 128, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=128 * 128 * 3 * 3),
        ),
    ]
    if include_reshape_node:
        initializers.append(
            helper.make_tensor(
                name="shape_1",
                data_type=onnx.TensorProto.INT64,
                dims=(4,),
                vals=(32, 128, 256, 256),
            ),
        )

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(nodes, "conv_act_pool", inputs, outputs, initializer=initializers)

    # Create the ONNX model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    model.ir_version = 10

    # Check the ONNX model
    model_inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model_inferred)

    return model_inferred


def build_conv_isinf_model(opset_version=13):
    # Define your model inputs and outputs
    input_names = ["input_0"]
    output_names = ["output_0"]
    input_shapes = [(6, 32, 900, 256)]
    output_shapes = [(6, 32, 900, 256)]

    inputs = [
        helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, input_shape)
        for input_name, input_shape in zip(input_names, input_shapes)
    ]
    outputs = [
        helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, output_shape)
        for output_name, output_shape in zip(output_names, output_shapes)
    ]

    # Create the ONNX graph with the nodes
    nodes = [
        helper.make_node(
            op_type="Conv",
            inputs=["input_0", "weights_1"],
            outputs=["conv1_conv/Conv2D:0"],
            name="conv1_conv/Conv2D",
            dilations=[1, 1],
            group=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            op_type="Cast",
            inputs=["conv1_conv/Conv2D:0"],
            outputs=["cast1_cast/Cast:0"],
            name="cast1_cast/Cast",
            to=onnx.TensorProto.DOUBLE,
        ),
        helper.make_node(
            op_type="IsInf",
            inputs=["cast1_cast/Cast:0"],
            outputs=["isinf1_isinf/IsInf:0"],
            name="isinf1_isinf/IsInf",
        ),
        helper.make_node(
            op_type="Greater",
            inputs=["conv1_conv/Conv2D:0", "greater_const1"],
            outputs=["greater1_greater/Greater:0"],
            name="greater1_greater/Greater",
        ),
        helper.make_node(
            op_type="And",
            inputs=["isinf1_isinf/IsInf:0", "greater1_greater/Greater:0"],
            outputs=["and1_and/And:0"],
            name="and1_and/And",
        ),
        helper.make_node(
            op_type="Where",
            inputs=["and1_and/And:0", "conv1_conv/Conv2D:0", "where_const1"],
            outputs=["output_0"],
            name="where1_where/Where",
        ),
    ]

    # Create the ONNX initializers
    initializers = [
        helper.make_tensor(
            name="weights_1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(32, 32, 3, 3),
            vals=np.random.uniform(low=0.5, high=1.0, size=32 * 32 * 3 * 3),
        ),
        helper.make_tensor(
            name="greater_const1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(1,),
            vals=[0],
        ),
        helper.make_tensor(
            name="where_const1",
            data_type=onnx.TensorProto.FLOAT,
            dims=(1,),
            vals=[10000],
        ),
    ]

    # Create the ONNX graph with the nodes and initializers
    graph = helper.make_graph(nodes, "conv_isinf", inputs, outputs, initializer=initializers)

    # Create the ONNX model
    model = helper.make_model(graph)
    model.opset_import[0].version = opset_version
    model.ir_version = 10

    # Check the ONNX model
    model_inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model_inferred)

    return model_inferred
