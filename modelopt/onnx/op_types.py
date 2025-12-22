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

"""Utility functions to categorize onnx ops."""


def is_unary_op(op_type: str):
    """Returns whether the given op is a unary operator or not."""
    return op_type in [
        "Neg",
        "Sqrt",
        "Abs",
        "Log",
        "Exp",
        "Not",
        "Cast",
        "Floor",
        "Ceil",
        "Round",
        "Erf",
        "Gelu",
        "Sin",
        "Cos",
        "Atan",
        "Sign",
        "IsNaN",
        "IsInf",
        "Log",
        "LeakyRelu",
        # "Relu",
        "Elu",
        "Tanh",
        "Sigmoid",
        # "BatchNormalization",
        "Softmax",
        "Softplus",
        "InstanceNormalization",
        "CumSum",
    ]


def is_binary_op(op_type: str):
    """Returns whether the given op is a binary operator or not."""
    return op_type in [
        "Add",
        "Sub",
        "Mul",
        "Pow",
        "Div",
        "Min",
        "Max",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Equal",
        "BitwiseOr",
        "BitwiseAnd",
        "BitwiseXor",
        "BitShift",
    ]


def is_fusible_reduction_op(op_type: str):
    """Returns whether the given op type is of reduction category and fusible by the compiler."""
    return op_type in [
        "ReduceMax",
        "ReduceMin",
        "ReduceMean",
        "ReduceProd",
        "ReduceSum",
        "TopK",  # Transformed to BottomK based on `largest` param
    ]


def is_fusible_scaling_op(op_type: str):
    """Returns whether the given op type is of scaling category and fusible with input."""
    return op_type in [
        "SkipSimplifiedLayerNormalization",
        "SimplifiedLayerNormalization",
        "MatMul",
        "Gemm",
        "Mul",
    ]


def get_copy_ops():
    """Returns list of copy operators."""
    return [
        "Flatten",
        "Transpose",
        "Concat",
        "Split",
        "Squeeze",
        "Expand",
        "ReverseSequence",
        "Reshape",
        "Tile",
        "Gather",
        "Slice",
        "GatherElements",
        "GatherND",
        "ScatterElements",
        "ScatterND",
        "OneHot",
    ]


def is_copy_op(op_type: str):
    """Returns whether the given op is a copy operator or not."""
    return op_type in get_copy_ops()


def is_linear_op(op_type: str):
    """Returns whether the given op type is of Linear category or not."""
    return op_type in ["Conv", "ConvTranspose", "Gemm", "MatMul"]


def is_pointwise_or_elementwise_op(op_type: str):
    """Returns whether the given op type is of Pointwise or Elementwise category or not.

    This considers only the fusible types.
    """
    return is_unary_op(op_type) or is_binary_op(op_type)


def is_pooling_or_window_op(op_type: str):
    """Returns whether the given op type is of Pooling/Window category or not."""
    return op_type in [
        "AveragePool",
        "GlobalAveragePool",
        "MaxPool",
        "GlobalMaxPool",
        "GlobalLpPool",
        "LpPool",
        "MaxPoolGridSample",
        "HammingWindow",
        "BlackmanWindow",
        "HannWindow",
    ]


def is_normalization_op(op_type: str):
    """Returns whether the given op type is of Normalization category or not."""
    return op_type in [
        "BatchNormalization",
        "InstanceNormalization",
        "LRN",
        "LpNormalization",
        "GroupNormalization",
        "LayerNormalization",
    ]


def is_conversion_op(op_type: str):
    """Returns whether the given op type is of Conversion category or not."""
    return op_type in ["Cast", "QuantizeLinear", "DequantizeLinear"]


def is_non_reshape_copy_op(op_type: str):
    """Returns whether the given op is a non-reshape copy op or not."""
    return is_copy_op(op_type) and (op_type != "Reshape")


def is_irregular_mem_access_op(op_type: str):
    """Returns whether the given op type is of Irreggular mem access category or not."""
    return op_type in [
        "Gather",
        "GatherElements",
        "GatherND",
        "MaxRoiPool",
        "RoiAlign",
        "ScatterND",
        "ScatterElements",
        "NonMaxSuppression",
    ]


def is_generator_op(op_type: str):
    """Returns whether the given op type is of Generator category or not."""
    return op_type in [
        "Const",
        "ConstOfShape",
        "EyeLike",
        "OneHot",
        "Multinomial",
        "RandomNormal",
        "RandomUniform",
        "Bernoulli",
    ]


def is_modifier_op(op_type: str):
    """Returns whether the given op type is of Modifier category or not."""
    return op_type in [
        "Identity",
        "Trilu",
        "Expand",
        "Pad",
        "Dropout",
        "TileDropout",
        "Col2Im",
        "MaxUnpool",
    ]


def is_sequence_op(op_type: str):
    """Returns whether the given op type is of Sequence category or not."""
    return op_type in [
        "SequenceAt",
        "SequenceConstruct",
        "SequenceEmpty",
        "SequenceErase",
        "SequenceInsert",
        "SequenceLength",
    ]


def is_selection_op(op_type: str):
    """Returns whether the given op type is of Selection category or not."""
    return op_type in ["Where", "Compress"]


def is_control_flow_op(op_type: str):
    """Returns whether the given op type is of Control Flow category or not."""
    return op_type in ["If", "Loop"]


def is_multiclass_op(op_type: str):
    """Returns whether the given op type is of Multiclass category or not."""
    return op_type in ["Einsum"]


def is_recurrent_op(op_type: str):
    """Returns whether the given op type is of Recurrent category or not."""
    return op_type in ["LSTM", "RNN", "GRU"]


def is_shape_op(op_type: str):
    """Returns whether the given op type is of Shape category or not."""
    return op_type in ["Shape", "Size"]


def is_default_quantizable_op_by_ort(op_type: str):
    """Returns if ORT quantizes the op type by default.

    Note. Subject to change with different ORT versions.
    Note. Users can use nodes_to_quantize and/or op_types_to_quantize arguments to quantize
    non-default operations.
    Reference: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/registry.py
    """
    return op_type in [
        "Conv",
        "Gemm",
        "ArgMax",
        "Relu",
        "Split",
        "MaxPool",
        "InstanceNormalization",
        "Softmax",
        "Where",
        "Squeeze",
        "GlobalAveragePool",
        "Pad",
        "Resize",
        "ConvTranspose",
        "Gather",
        "Sigmoid",
        "EmbedLayerNormalization",
        "Reshape",
        "Unsqueeze",
        "Transpose",
        "MatMul",
        "Concat",
        "Mul",
        "Clip",
        "Add",
        "LeakyRelu",
        "AveragePool",
    ]


def is_data_dependent_shape_op(op_type: str):
    """Returns whether the op type has data-dependent shapes (DDS).

    DDS ops have output shapes that are only determined at runtime based on its input data.
    Source: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#data-dependant-shape-dds-ops
    """
    return op_type in [
        "NonMaxSuppression",
        "NonZero",
        "RoiAlign",
    ]
