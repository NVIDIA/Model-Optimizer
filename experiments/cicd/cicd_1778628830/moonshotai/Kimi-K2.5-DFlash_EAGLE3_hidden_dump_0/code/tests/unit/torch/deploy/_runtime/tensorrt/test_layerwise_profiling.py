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

import pytest

from modelopt.torch._deploy._runtime.tensorrt.layerwise_profiling import (
    _merge_reformatters,
    map_trt_layers_to_onnx,
)


@pytest.mark.parametrize(
    ("layerwise_result", "onnx_node_names", "expected_output"),
    [
        (
            {
                "Slice_39": 0.01,
                "PWN(Clip_589 + (Unnamed Layer* 16) [Shuffle], PWN(PWN(PWN(Add_586 + (Unnamed Layer* 11) [Shuffle] + \
Add_43, Clip_46), Mul_47), Div_49))": 0.02,
            },
            ["Slice_39", "Conv_41", "Clip_589", "Add_586", "Add_43", "Clip_46", "Mul_47", "Div_49"],
            {"Slice_39": 0.01, "Clip_589 + Add_586 + Add_43 + Clip_46 + Mul_47 + Div_49": 0.02},
        ),
        (
            {"Conv_88 || Conv_59": 0.01},
            ["Conv_88", "Conv_59"],
            {"Conv_88 + Conv_59": 0.01},
        ),
        (
            {"Reshape_625 + Transpose_626": 0.01},
            ["Reshape_625", "Transpose_626"],
            {"Reshape_625 + Transpose_626": 0.01},
        ),
        (
            {
                "Conv_41 + PWN(Clip_589 + (Unnamed Layer* 16) [Shuffle], PWN(PWN(PWN(Add_586 + (Unnamed Layer* 11) \
[Shuffle] + Add_43, PWN(Clip_46)), Mul_47), Div_49))": 0.01
            },
            ["Conv_41", "Clip_589", "Add_586", "Add_43", "Clip_46", "Mul_47", "Div_49"],
            {"Conv_41 + Clip_589 + Add_586 + Add_43 + Clip_46 + Mul_47 + Div_49": 0.01},
        ),
        (
            {"Concat_636 copy": 0.01},
            ["Concat_636"],
            {"Concat_636": 0.01},
        ),
        (
            {"PWN(LeakyRelu_341)": 0.01},
            ["LeakyRelu_341"],
            {"LeakyRelu_341": 0.01},
        ),
        (
            {"2-layer MLP: Conv_68 + Relu_69 -> Conv_70": 0.01},
            ["Conv_68", "Relu_69", "Conv_70"],
            {"Conv_68 + Relu_69 + Conv_70": 0.01},
        ),
        (
            {"3-layer MLP: Conv_68 + Relu_69 -> Conv_72": 0.01},
            ["Conv_68", "Relu_69", "Conv_70", "Relu_71", "Conv_72"],
            {"Conv_68 + Relu_69 + Conv_70 + Relu_71 + Conv_72": 0.01},
        ),
        (
            {"Conv_785 (32, 32, 3)": 0.01},
            ["Conv_785"],
            {"Conv_785": 0.01},
        ),
        (
            {
                "Conv_0": 0.01,
                "{ForeignNode[Reshape_8 + Transpose_9...(Unnamed Layer* 1096) [Shuffle]]}": 0.02,
                "input_data": 0.03,
                "output_data": 0.04,
            },
            ["Conv_0", "Reshape_8", "Transpose_9"],
            {
                "Conv_0": 0.01,
                "Reshape_8 + Transpose_9": 0.02,
                "input_data": 0.03,
                "output_data": 0.04,
            },
        ),
        (
            {
                "Conv_0": 0.01,
                "{ForeignNode[Reshape_10 + Transpose_900...(Unnamed Layer* 1096) [Shuffle]]}": 0.02,
                "input_data": 0.03,
                "output_data": 0.04,
            },
            ["Conv_0", "Reshape_10", "Reshape_100", "Transpose_90", "Transpose_900"],
            {
                "Conv_0": 0.01,
                "Reshape_10 + Transpose_900": 0.02,
                "input_data": 0.03,
                "output_data": 0.04,
            },
        ),
        (
            {"Conv_0": 0.01, "(Unnamed Layer* 1096) [Shuffle]": 0.02},
            ["Conv_0"],
            {"Conv_0": 0.01, "other": 0.02},
        ),
        (
            {
                "PWN(Clip_589 + (Unnamed Layer* 16) [Shuffle], PWN(PWN(PWN(Add_586 + (Unnamed Layer* 11) [Shuffle] + \
Add_43, Clip_46), Mul_47), Div_49))": 0.02,
                "3-layer MLP: Conv_68 + Relu_69 -> Conv_72": 0.01,
                "{ForeignNode[Reshape_8 + Transpose_9...(Unnamed Layer* 1096) [Shuffle]]}": 0.02,
                "Concat_636 copy": 0.01,
                "input_data": 0.03,
            },
            [],
            {
                "Clip_589 + (Unnamed Layer* 16) [Shuffle] + Add_586 + (Unnamed Layer* 11) [Shuffle] + Add_43 + \
Clip_46 + Mul_47 + Div_49": 0.02,
                "Conv_68 + Relu_69 + Conv_72": 0.01,
                "{ForeignNode[Reshape_8 + Transpose_9...(Unnamed Layer* 1096) [Shuffle]]}": 0.02,
                "Concat_636 copy": 0.01,
                "input_data": 0.03,
            },
        ),
    ],
)
def test_map_trt_layers_to_onnx(layerwise_result, onnx_node_names, expected_output):
    output = map_trt_layers_to_onnx(layerwise_result, onnx_node_names)
    assert output == expected_output


@pytest.mark.parametrize(
    ("layer_latency_dict", "expected_output"),
    [
        ({"Conv_41 input reformatter 0": 0.01, "Conv_41": 0.02}, {"Conv_41": 0.03}),
        (
            {"Conv_757 || Conv_635": 0.01, "Conv_757 || Conv_635 output reformatter 0": 0.02},
            {"Conv_757 || Conv_635": 0.03},
        ),
        (
            {
                "Slice_39": 0.01,
                "Reformatting CopyNode for Input Tensor 0 to Conv_41 + PWN(Clip_589 + (Unnamed Layer* 16) [Shuffle], \
PWN(PWN(PWN(Add_586 + (Unnamed Layer* 11) [Shuffle] + Add_43, PWN(Clip_46)), Mul_47), Div_49))": 0.01,
                "Conv_41 + PWN(Clip_589 + (Unnamed Layer* 16) [Shuffle], PWN(PWN(PWN(Add_586 + (Unnamed Layer* 11) \
[Shuffle] + Add_43, PWN(Clip_46)), Mul_47), Div_49))": 0.02,
            },
            {
                "Slice_39": 0.01,
                "Conv_41 + PWN(Clip_589 + (Unnamed Layer* 16) [Shuffle], PWN(PWN(PWN(Add_586 + (Unnamed Layer* 11) \
[Shuffle] + Add_43, PWN(Clip_46)), Mul_47), Div_49))": 0.03,
            },
        ),
        (
            {
                "Conv_705 || Conv_733": 0.01,
                "Reformatting CopyNode for Output Tensor 0 to Conv_705 || Conv_733": 0.02,
            },
            {"Conv_705 || Conv_733": 0.03},
        ),
        (
            {"Reshape_625 + Transpose_626": 0.01, "Reshape_634 output reformatter 0": 0.02},
            {"Reshape_625 + Transpose_626": 0.01, "Reshape_634": 0.02},
        ),
    ],
)
def test__merge_reformatters(layer_latency_dict, expected_output):
    layer_latency_dict = _merge_reformatters(layer_latency_dict)
    assert layer_latency_dict == expected_output
