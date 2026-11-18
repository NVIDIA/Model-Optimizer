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

"""Tests for the FP16 Q/DQ scale cast-folding helpers in ``modelopt.onnx.utils``."""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.utils import fold_dq_fp32_to_fp16_casts, fold_q_fp16_to_fp32_casts


def _dq_cast_model(opset):
    """``DQ → Cast(FP32→FP16) → MatMul(x)`` with FP32 scale."""
    nodes = [
        helper.make_node("DequantizeLinear", ["w_q", "w_scale", "w_zp"], ["dq_out"], "dq"),
        helper.make_node("Cast", ["dq_out"], ["cast_out"], "cast", to=TensorProto.FLOAT16),
        helper.make_node("MatMul", ["x", "cast_out"], ["y"], "matmul"),
    ]
    inits = [
        numpy_helper.from_array(np.ones((4, 4), dtype=np.int8), "w_q"),
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "w_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "w_zp"),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [None, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [None, 4])],
            initializer=inits,
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )


def _cast_q_model(opset):
    """``Cast(FP16→FP32) → Q → DQ → MatMul`` with FP32 scale."""
    nodes = [
        helper.make_node("Cast", ["x"], ["c_out"], "cast", to=TensorProto.FLOAT),
        helper.make_node("QuantizeLinear", ["c_out", "scale", "zp"], ["q_out"], "q"),
        helper.make_node("DequantizeLinear", ["q_out", "scale", "zp"], ["dq_out"], "dq"),
        helper.make_node("MatMul", ["dq_out", "w"], ["y"], "matmul"),
    ]
    inits = [
        numpy_helper.from_array(np.ones((4, 4), dtype=np.float16), "w"),
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "zp"),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [None, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 4])],
            initializer=inits,
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )


@pytest.mark.parametrize(
    ("fold_fn", "build_model", "scale_name"),
    [
        (fold_dq_fp32_to_fp16_casts, _dq_cast_model, "w_scale"),
        (fold_q_fp16_to_fp32_casts, _cast_q_model, "scale"),
    ],
)
def test_fold_rewrites_cast_and_scale_at_opset_19(fold_fn, build_model, scale_name):
    folded = fold_fn(build_model(opset=19))
    assert "Cast" not in {n.op_type for n in folded.graph.node}
    scale = next(i for i in folded.graph.initializer if i.name == scale_name)
    assert scale.data_type == TensorProto.FLOAT16


@pytest.mark.parametrize(
    ("fold_fn", "build_model", "scale_name"),
    [
        (fold_dq_fp32_to_fp16_casts, _dq_cast_model, "w_scale"),
        (fold_q_fp16_to_fp32_casts, _cast_q_model, "scale"),
    ],
)
def test_fold_is_noop_below_min_opset(fold_fn, build_model, scale_name):
    folded = fold_fn(build_model(opset=18))
    assert "Cast" in {n.op_type for n in folded.graph.node}
    scale = next(i for i in folded.graph.initializer if i.name == scale_name)
    assert scale.data_type == TensorProto.FLOAT
