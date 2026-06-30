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

"""Tests for DQ transpose surgery (transpose_dequantize_linear_weights)."""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.graph_surgery.dq_transpose import transpose_dequantize_linear_weights

IN_FEATURES = 16
OUT_FEATURES = 32
BATCH = 2
SEQ = 4


def _build_dq_matmul_model():
    """Build minimal model: input -> MatMul(input, DQ(weight, scale, zp)) -> output.

    Uses per-channel INT8 quantization on axis 0 of the weight [IN, OUT].
    """
    rng = np.random.RandomState(123)

    # Quantized weight [IN_FEATURES, OUT_FEATURES] int8
    w_int8 = rng.randint(-127, 127, size=(IN_FEATURES, OUT_FEATURES)).astype(np.int8)
    # Per-channel scale along axis 0 -> shape [IN_FEATURES] (one per row)
    scale = rng.rand(IN_FEATURES).astype(np.float32) * 0.01 + 0.001
    # Per-channel zero point
    zp = np.zeros(IN_FEATURES, dtype=np.int8)

    w_init = numpy_helper.from_array(w_int8, name="weight_quantized")
    s_init = numpy_helper.from_array(scale, name="weight_scale")
    zp_init = numpy_helper.from_array(zp, name="weight_zp")

    dq_node = helper.make_node(
        "DequantizeLinear",
        inputs=["weight_quantized", "weight_scale", "weight_zp"],
        outputs=["weight_dequantized"],
        name="dq_weight",
        axis=0,
    )

    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "weight_dequantized"],
        outputs=["output"],
        name="matmul_0",
    )

    graph = helper.make_graph(
        [dq_node, matmul_node],
        "test_dq_transpose",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [BATCH, SEQ, IN_FEATURES]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [BATCH, SEQ, OUT_FEATURES]),
        ],
        initializer=[w_init, s_init, zp_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def _run_session(model_proto, feeds):
    """Run inference on in-memory model."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(
        model_proto.SerializeToString(), opts, providers=["CPUExecutionProvider"]
    )
    output_names = [o.name for o in sess.get_outputs()]
    return sess.run(output_names, feeds)


@pytest.fixture(scope="class")
def models():
    """Build original model, apply DQ transpose surgery, return both."""
    orig = _build_dq_matmul_model()

    with tempfile.TemporaryDirectory() as tmp:
        orig_path = os.path.join(tmp, "original.onnx")
        out_path = os.path.join(tmp, "transposed.onnx")

        onnx.save(orig, orig_path)
        transpose_dequantize_linear_weights(
            model_path=orig_path,
            output_path=out_path,
            use_external_data=False,
            verbose=True,
        )
        modified = onnx.load(out_path)

    return orig, modified


class TestDqTransposeSurgery:
    def test_transpose_node_added(self, models):
        _, modified = models
        transpose_nodes = [n for n in modified.graph.node if n.op_type == "Transpose"]
        assert len(transpose_nodes) == 1, f"Expected 1 Transpose node, got {len(transpose_nodes)}"

    def test_dq_node_preserved(self, models):
        _, modified = models
        dq_nodes = [n for n in modified.graph.node if n.op_type == "DequantizeLinear"]
        assert len(dq_nodes) == 1, f"Expected 1 DQ node, got {len(dq_nodes)}"

    def test_weight_transposed(self, models):
        orig, modified = models

        orig_w = None
        for init in orig.graph.initializer:
            if init.name == "weight_quantized":
                orig_w = numpy_helper.to_array(init)

        mod_w = None
        for init in modified.graph.initializer:
            if "weight_quantized" in init.name:
                mod_w = numpy_helper.to_array(init)

        assert orig_w is not None and mod_w is not None
        assert mod_w.shape == (OUT_FEATURES, IN_FEATURES), (
            f"Expected transposed shape, got {mod_w.shape}"
        )
        np.testing.assert_array_equal(mod_w, orig_w.T)

    def test_axis_updated(self, models):
        _, modified = models
        dq_node = next(n for n in modified.graph.node if n.op_type == "DequantizeLinear")
        axis_val = None
        for attr in dq_node.attribute:
            if attr.name == "axis":
                axis_val = attr.i
        assert axis_val == 1, f"Expected axis=1 after transpose, got {axis_val}"

    def test_output_matches(self, models):
        orig, modified = models
        rng = np.random.RandomState(999)
        x = rng.randn(BATCH, SEQ, IN_FEATURES).astype(np.float32)
        feeds = {"input": x}

        orig_out = _run_session(orig, feeds)[0]
        mod_out = _run_session(modified, feeds)[0]

        diff = np.abs(orig_out - mod_out)

        print(f"\n  Original shape:  {orig_out.shape}")
        print(f"  Original[0,:4]:  {orig_out[0, 0, :4]}")
        print(f"  Modified[0,:4]:  {mod_out[0, 0, :4]}")
        print(f"  Max  abs diff:   {diff.max():.6f}")
        print(f"  Mean abs diff:   {diff.mean():.6f}")

        np.testing.assert_allclose(orig_out, mod_out, atol=1e-5, rtol=1e-5)
