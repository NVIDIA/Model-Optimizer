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
import onnx
import pytest
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from onnxconverter_common import convert_float_to_float16

import modelopt.torch.quantization as mtq
from modelopt.onnx.utils import (
    fold_dq_fp32_to_fp16_casts,
    fold_q_fp16_to_fp32_casts,
    fold_qdq_scale_fp16_to_fp32_casts,
)
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata


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


def test_fold_q_ignores_non_fp16_source_cast():
    """``Cast(INT64→FP32) → Q`` must not be bypassed by ``fold_q_fp16_to_fp32_casts``."""
    nodes = [
        helper.make_node("Cast", ["idx"], ["c_out"], "cast_i64", to=TensorProto.FLOAT),
        helper.make_node("QuantizeLinear", ["c_out", "scale", "zp"], ["q_out"], "q"),
    ]
    inits = [
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "zp"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "g_i64",
            [helper.make_tensor_value_info("idx", TensorProto.INT64, [4])],
            [helper.make_tensor_value_info("q_out", TensorProto.INT8, [4])],
            initializer=inits,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    folded = fold_q_fp16_to_fp32_casts(model)
    onnx.checker.check_model(folded, full_check=True)
    assert "Cast" in {n.op_type for n in folded.graph.node}


def test_fold_qdq_scale_preserves_fp32_scale_when_q_input_remains_fp32():
    """When ``Div(FP32) → Q`` stays FP32, ``fold_qdq_scale_fp16_to_fp32_casts`` must keep Q's scale FP32."""
    nodes = [
        helper.make_node("Div", ["x", "denom"], ["div_out"], "div0"),
        helper.make_node("QuantizeLinear", ["div_out", "scale", "zp"], ["q_out"], "q"),
        helper.make_node("DequantizeLinear", ["q_out", "scale", "zp"], ["y"], "dq"),
    ]
    inits = [
        numpy_helper.from_array(np.array(2.0, dtype=np.float32), "denom"),
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "zp"),
    ]
    fp32_model = helper.make_model(
        helper.make_graph(
            nodes,
            "g_div_q",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
            initializer=inits,
        ),
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    fp16_model = convert_float_to_float16(
        fp32_model,
        keep_io_types=False,
        disable_shape_infer=True,
        check_fp16_ready=False,
        op_block_list=["QuantizeLinear", "DequantizeLinear", "Div"],
    )
    fp16_model = fold_dq_fp32_to_fp16_casts(fp16_model)
    fp16_model = fold_q_fp16_to_fp32_casts(fp16_model)
    fp16_model = fold_qdq_scale_fp16_to_fp32_casts(fp16_model)
    fp16_model.ir_version = 10
    onnx.checker.check_model(fp16_model, full_check=True)


def test_int8_torch_onnx_fp16_export_passes_full_check(tmp_path):
    """Regression test: INT8 PyTorch model exported with weights_dtype='fp16' must pass ONNX full_check."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 8, bias=False)).eval()
    x = torch.randn(2, 16, dtype=torch.float32)
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=lambda m: m(x))

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model, (x,), weights_dtype="fp16", model_name="int8_fp16"
    )
    OnnxBytes.from_bytes(onnx_bytes).write_to_disk(str(tmp_path), clean_dir=True)
    exported = onnx.load(str(tmp_path / "int8_fp16.onnx"))
    onnx.checker.check_model(exported, full_check=True)
