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

"""Unit tests: FP8 ONNX export shape inference.

Two complementary tests:
  1. Prove that TRT custom FP8 ops lose shape info (root-cause regression guard).
  2. Prove that standard ONNX QDQ ops preserve shape info after the fix.
"""

import io

import pytest

onnx = pytest.importorskip("onnx")

import torch  # noqa: E402

import modelopt.torch.quantization as mtq  # noqa: E402

from _test_utils.torch.quantization.models import SimpleConv  # noqa: E402


# ---------------------------------------------------------------------------
# Part 1 — root-cause: TRT custom ops have no ONNX shape inference function
# ---------------------------------------------------------------------------


def test_trt_fp8_ops_unsupported_by_onnx_inference():
    """ONNX shape inference raises or produces no shape for trt::TRT_FP8QuantizeLinear.

    This documents the root cause: TRT custom ops are not registered with ONNX
    shape inference, so any graph containing them cannot have shapes propagated
    through those nodes. ONNX either raises InferenceError (newer versions) or
    silently leaves the output shape empty (older versions).
    """
    from onnx import TensorProto, helper

    # Build a minimal ONNX graph: Input → trt::TRT_FP8QuantizeLinear → Output
    input_vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [1], [1.0])
    output_vi = helper.make_tensor_value_info("y", TensorProto.UINT8, None)

    node = helper.make_node(
        "TRT_FP8QuantizeLinear",
        inputs=["x", "scale"],
        outputs=["y"],
        domain="trt",
    )

    graph = helper.make_graph([node], "trt_fp8_q_test", [input_vi], [output_vi], [scale_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9

    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        # Newer ONNX rejects unknown domains outright — shape inference is impossible.
        return

    # Older ONNX silently skips unknown ops, leaving the output shape empty.
    output_shape = inferred.graph.output[0].type.tensor_type.shape
    assert not output_shape.dim, (
        "Expected TRT_FP8QuantizeLinear output to have no shape (op unknown to ONNX), "
        f"but got dims: {list(output_shape.dim)}"
    )


# ---------------------------------------------------------------------------
# Part 2 — fix: standard ONNX QDQ ops preserve shape after export
# ---------------------------------------------------------------------------


def test_fp8_onnx_export_shape_preserved():
    """FP8-quantized SimpleConv exported with opset 19 retains shape on all QDQ outputs."""
    model = SimpleConv().eval()
    dummy_input = SimpleConv.get_input()

    def forward_loop(m):
        m(dummy_input)

    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop=forward_loop)
    # Disable output quantizers to avoid export errors (they produce FP8 outputs that
    # downstream non-quantized ops can't accept in the TorchScript exporter).
    mtq.disable_quantizer(model, lambda name: "output_quantizer" in name)

    buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buf,
        opset_version=19,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    buf.seek(0)
    onnx_model = onnx.load_from_string(buf.read())

    # No TRT custom FP8 ops should remain.
    trt_fp8_ops = [
        n.op_type for n in onnx_model.graph.node if n.domain == "trt" and "FP8" in n.op_type
    ]
    assert not trt_fp8_ops, (
        f"Found TRT custom FP8 ops after export: {trt_fp8_ops}. "
        "These have no ONNX shape inference and will cause shape loss."
    )

    # Run shape inference and collect QDQ output shapes.
    inferred = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=False)
    shape_by_name: dict[str, list] = {}
    for vi in inferred.graph.value_info:
        shape_by_name[vi.name] = [
            d.dim_value if d.HasField("dim_value") else -1
            for d in vi.type.tensor_type.shape.dim
        ]

    missing = []
    for node in inferred.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            for out in node.output:
                shape = shape_by_name.get(out)
                if not shape:
                    missing.append(out)

    assert not missing, (
        f"The following QDQ outputs are missing shape info after shape inference: {missing}. "
        "This indicates the FP8 export still uses ops without ONNX shape inference support."
    )
