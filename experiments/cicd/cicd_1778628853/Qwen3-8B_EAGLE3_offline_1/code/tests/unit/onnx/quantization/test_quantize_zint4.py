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

import os
import tempfile as _tempfile
from collections.abc import Sequence

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from _test_utils.onnx.lib_test_models import find_init

import modelopt.onnx.quantization as moq
from modelopt.onnx.quantization.int4 import quantize as quantize_int4
from modelopt.onnx.utils import save_onnx

# TODO: Rename this script to *_int4.py
#       For that, we need to investigate failure in 'pytest tests/unit/onnx'.
#       test_quantize_int8.py::test_int8[bf16] fails if this script runs after the int4 test, but not before.


def _matmul_model(w: np.ndarray, in_shape: Sequence[int], out_shape: Sequence[int], tmp_path):
    # Assumes
    w = gs.Constant("w", w)
    x = gs.Variable("x", dtype=np.float32, shape=in_shape)
    y = gs.Variable("y", dtype=np.float32, shape=out_shape)
    mm = gs.Node("MatMul", "mm", inputs=[x, w], outputs=[y])
    g = gs.Graph([mm], inputs=[x, w], outputs=[y])

    onnx_model = gs.export_onnx(g)
    onnx_path = os.path.join(tmp_path, "model.onnx")
    save_onnx(onnx_model, onnx_path)

    return onnx_path


def _gather_model(rows, cols, tmp_path):
    data_const = np.arange(rows * cols, dtype=np.float16).reshape(rows, cols)
    data_node = gs.Constant("data", values=data_const)

    indices_var = gs.Variable("indices", dtype=np.int64, shape=())

    gather_out = gs.Variable("output", dtype=np.float16, shape=(cols,))
    gather_node = gs.Node(
        op="Gather", inputs=[data_node, indices_var], outputs=[gather_out], attrs={"axis": 0}
    )

    graph = gs.Graph(nodes=[gather_node], inputs=[indices_var], outputs=[gather_out])

    onnx_model = gs.export_onnx(graph)
    onnx_path = os.path.join(tmp_path, "gather_base_model.onnx")
    save_onnx(onnx_model, onnx_path)

    return onnx_path


def test_int4_rtn(tmp_path):
    # Test scale factor computation.
    # Use moq.quantize once to check that path doesnt have any bugs
    onnx_path = _matmul_model(
        w=np.asarray([[0.5, 1.5], [0.875, 1.75]]),
        in_shape=(1, 2),
        out_shape=(1, 2),
        tmp_path=tmp_path,
    )
    output_path = os.path.join(tmp_path, "model_int4.onnx")
    moq.quantize(onnx_path, "int4", calibration_method="rtn", output_path=output_path, block_size=8)
    onnx_model = onnx.load(output_path)

    node_names = [node.name for node in onnx_model.graph.node]
    assert "w_QuantizeLinear" in node_names
    assert "w_DequantizeLinear" in node_names

    s = find_init(onnx_model, "w_scale")
    assert np.array_equal(s, np.asarray([[0.125, 0.25]]))

    # Test multiple blocks.
    onnx_path = _matmul_model(
        w=np.asarray(
            [
                # Block 0.
                [0.5],
                [-0.5],
                [0.75],
                [-0.75],
                [0.875],
                [-0.875],
                [0.5],
                [-0.5],
                # Block 1.
                [0.25],
                [-0.25],
                [0.4375],
                [-0.4375],
                [0.0],
                [0.0],
                [0.25],
                [-0.25],
            ]
        ),
        in_shape=(1, 16),
        out_shape=(1, 1),
        tmp_path=tmp_path,
    )
    onnx_model = quantize_int4(onnx_path, "rtn", block_size=8)

    s = find_init(onnx_model, "w_scale")
    assert np.array_equal(s, np.asarray([[0.125], [0.0625]]))

    # Test shape compatibility
    onnx_path = _matmul_model(
        w=np.random.rand(288, 16), in_shape=(96, 288), out_shape=(96, 16), tmp_path=tmp_path
    )
    onnx_model = quantize_int4(onnx_path, "rtn", block_size=8)  # Ensure it passes.

    onnx_path = _matmul_model(
        w=np.random.rand(577, 3), in_shape=(8, 557), out_shape=(8, 3), tmp_path=tmp_path
    )
    onnx_model = quantize_int4(onnx_path, "rtn", block_size=8)  # Ensure it passes.


def test_shape_rtn(tmp_path):
    # Test shape compatibility
    onnx_dataloader = [{"x": np.random.rand(96, 288)}]
    onnx_path = _matmul_model(
        w=np.random.rand(288, 16).astype(np.float32),
        in_shape=(96, 288),
        out_shape=(96, 16),
        tmp_path=tmp_path,
    )
    quantize_int4(
        onnx_path,
        "rtn",
        onnx_dataloader,
        block_size=8,
        use_external_data_format=False,
    )  # Ensure it passes.


def test_shape_awq(tmp_path):
    # Test shape compatibility
    onnx_dataloader = [{"x": np.random.rand(96, 288).astype(np.float32)}]
    onnx_path = _matmul_model(
        w=np.random.rand(288, 16).astype(np.float32),
        in_shape=(96, 288),
        out_shape=(96, 16),
        tmp_path=tmp_path,
    )
    quantize_int4(
        onnx_path,
        "awq_clip",
        onnx_dataloader,
        block_size=8,
        use_external_data_format=False,
    )  # Ensure it passes.
    quantize_int4(
        onnx_path,
        "awq_lite",
        onnx_dataloader,
        block_size=8,
        use_external_data_format=False,
    )  # Ensure it passes.


def test_int4_gather(tmp_path):
    gather_rows = 8
    gather_cols = 16
    gather_block_size = 8

    m = _gather_model(gather_rows, gather_cols, tmp_path=tmp_path)

    m1 = quantize_int4(
        m, calibration_method="rtn_dq", gather_block_size=gather_block_size, gather_quantize_axis=1
    )
    m2 = quantize_int4(
        m, calibration_method="rtn_dq", gather_block_size=gather_block_size, gather_quantize_axis=0
    )

    def is_gather_quantized(model):
        g = gs.import_onnx(model)
        for node in g.nodes:
            if node.op != "Gather":
                continue
            for inp in node.inputs:
                if inp.name != "indices":
                    # print(f"inp={inp}, p={inp.inputs[0].op}")
                    assert inp.inputs[0].op == "DequantizeLinear", (
                        f"Input '{inp.name}' of node '{node.name}' is not quantized but should be!"
                    )
        return True

    assert is_gather_quantized(m1), "Failure in rtn_dq quantization of Gather node, quant-axis: 1"
    assert is_gather_quantized(m2), "Failure in rtn_dq quantization of Gather node, quant-axis: 0"

    def is_quant_scale_with_right_shape(model, quant_axis, block_size):
        assert quant_axis in [0, 1], "Incorrect quant-axis"  # used for 0/1 indexing below
        orig_shape = [gather_rows, gather_cols]
        graph = gs.import_onnx(model)
        for node in graph.nodes:
            if node.op == "DequantizeLinear":
                for inp in node.inputs:
                    if inp.name == "x_scale":
                        print(f"\nname={inp.name}, shape={inp.shape}\n")
                        c1 = (orig_shape[quant_axis] // block_size) == inp.shape[quant_axis]
                        c2 = orig_shape[1 - quant_axis] == inp.shape[1 - quant_axis]
                        assert c1 and c2, "Incorrect scale shape in DQ node for Gather"
        return True

    assert is_quant_scale_with_right_shape(m1, 1, gather_block_size), (
        "DQ Scale Error in rtn_dq quantization, axis 1"
    )
    assert is_quant_scale_with_right_shape(m2, 0, gather_block_size), (
        "DQ Scale Error in rtn_dq quantization, axis 0"
    )

    # Ensure above tests pass.


@pytest.mark.parametrize("calibration_method", ["awq_lite", "awq_clip"])
@pytest.mark.parametrize("use_external_data_format", [True, False])
def test_awq_no_temp_file_leak(tmp_path, monkeypatch, calibration_method, use_external_data_format):
    """Test that tmp*.onnx and tmp*.onnx_data are written to the
    system temp directory must be removed even when quantization fails mid-run.

    Simulates the real-world failure window (OOM, bad EP, driver error) by injecting
    a RuntimeError at ORT session creation — which happens after the augmented ONNX
    has already been written to disk but before the original cleanup code was reached.

    Thread-safe: tracks the exact paths created by mkstemp during this test rather
    than glob-snapshotting the temp directory, so parallel test runs cannot interfere.
    """
    onnx_path = _matmul_model(
        w=np.random.rand(288, 16).astype(np.float32),
        in_shape=(96, 288),
        out_shape=(96, 16),
        tmp_path=tmp_path,
    )

    # Intercept mkstemp to record the exact augmented-model temp path(s) created.
    created_paths = []
    real_mkstemp = _tempfile.mkstemp

    def _tracking_mkstemp(*args, **kwargs):
        fd, path = real_mkstemp(*args, **kwargs)
        created_paths.append(path)
        return fd, path

    monkeypatch.setattr("modelopt.onnx.quantization.int4.tempfile.mkstemp", _tracking_mkstemp)

    def _raise_session_error(*args, **kwargs):
        raise RuntimeError("injected ORT session failure")

    monkeypatch.setattr(
        "modelopt.onnx.quantization.int4.create_inference_session",
        _raise_session_error,
    )

    with pytest.raises(RuntimeError, match="injected ORT session failure"):
        quantize_int4(
            onnx_path,
            calibration_method=calibration_method,
            use_external_data_format=use_external_data_format,
            block_size=8,
        )

    assert created_paths, "Expected mkstemp to be called but it was not"
    for augmented_path in created_paths:
        assert not os.path.exists(augmented_path), f"Leaked: {augmented_path}"
        if use_external_data_format:
            assert not os.path.exists(augmented_path + "_data"), f"Leaked: {augmented_path}_data"
