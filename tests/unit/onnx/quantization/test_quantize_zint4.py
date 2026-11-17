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
import onnxruntime as ort
import pytest
from _test_utils.onnx.lib_test_models import find_init

import modelopt.onnx.quantization as moq
import modelopt.onnx.quantization.int4 as int4
from modelopt.onnx.quantization.int4 import get_weight_scale, get_x_w_mean_for_subgraph
from modelopt.onnx.quantization.int4 import quantize as quantize_int4
from modelopt.onnx.utils import save_onnx

# TODO: Rename this script to *_int4.py
#       For that, we need to investigate failure in 'pytest tests/unit/onnx'.
#       test_quantize_int8.py::test_int8[bf16] fails if this script runs after the int4 test, but not before.


def _matmul_model(
    w: np.ndarray, in_shape: Sequence[int], out_shape: Sequence[int], tmp_path, opset=21
):
    # Assumes
    w = gs.Constant("w", w)
    x = gs.Variable("x", dtype=np.float32, shape=in_shape)
    y = gs.Variable("y", dtype=np.float32, shape=out_shape)
    mm = gs.Node("MatMul", "mm", inputs=[x, w], outputs=[y])
    g = gs.Graph([mm], inputs=[x, w], outputs=[y], opset=opset)

    onnx_model = gs.export_onnx(g)
    onnx_path = os.path.join(tmp_path, "model.onnx")
    save_onnx(onnx_model, onnx_path)

    return onnx_path


def _gather_model(rows, cols, tmp_path, opset=21):
    data_const = np.arange(rows * cols, dtype=np.float16).reshape(rows, cols)
    data_node = gs.Constant("data", values=data_const)

    indices_var = gs.Variable("indices", dtype=np.int64, shape=())

    gather_out = gs.Variable("output", dtype=np.float16, shape=(cols,))
    gather_node = gs.Node(
        op="Gather", inputs=[data_node, indices_var], outputs=[gather_out], attrs={"axis": 0}
    )

    graph = gs.Graph(nodes=[gather_node], inputs=[indices_var], outputs=[gather_out], opset=opset)

    onnx_model = gs.export_onnx(graph)
    onnx_path = os.path.join(tmp_path, "gather_base_model.onnx")
    save_onnx(onnx_model, onnx_path)

    return onnx_path


def _upgrade_opset_after_quantize(model: onnx.ModelProto) -> onnx.ModelProto:
    # Mirror examples/windows/onnx_ptq/genai_llm/quantize.py, which quantizes first and
    # only then raises the default-domain opset to 21 for INT4 Q/DQ.
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx") and opset.version < 21:
            opset.version = 21
    return model


def test_int4_rtn(tmp_path):
    # Test scale factor computation.
    # Use moq.quantize once to check that path doesnt have any bugs
    onnx_path = _matmul_model(
        w=np.asarray([[0.5, 1.5], [0.875, 1.75]]),
        in_shape=(1, 2),
        out_shape=(1, 2),
        tmp_path=tmp_path,
        opset=20,
    )
    output_path = os.path.join(tmp_path, "model_int4.onnx")
    moq.quantize(onnx_path, "int4", calibration_method="rtn", output_path=output_path, block_size=8)
    onnx_model = onnx.load(output_path)
    assert onnx_model.opset_import[0].version >= 21

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


@pytest.mark.parametrize(("block_size", "layers_8bit"), [(8, None), (-1, None), (8, "mm")])
@pytest.mark.parametrize("opset", [20, 21])
def test_shape_rtn(tmp_path, block_size, layers_8bit, opset):
    # Test shape compatibility
    rng = np.random.default_rng(0)
    onnx_dataloader = [{"x": rng.random((96, 288), dtype=np.float32)}]
    onnx_path = _matmul_model(
        w=rng.random((288, 16), dtype=np.float32),
        in_shape=(96, 288),
        out_shape=(96, 16),
        tmp_path=tmp_path,
        opset=opset,
    )
    outputs = []
    for method in ["rtn", "rtn_dq"]:
        model = quantize_int4(
            onnx_path,
            method,
            onnx_dataloader,
            block_size=block_size,
            layers_8bit=layers_8bit,
            use_external_data_format=False,
        )
        model = _upgrade_opset_after_quantize(model)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        outputs.append(session.run(None, onnx_dataloader[0])[0])
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)


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


@pytest.mark.parametrize("calibration_method", ["awq_lite", "awq_full"])
@pytest.mark.parametrize(
    ("reuse_activations", "random_activation"),
    [(None, False), (False, False), (True, False), (None, True), (False, True), (True, True)],
)
def test_awq_reuses_adjacent_activations(
    tmp_path, monkeypatch, calibration_method, reuse_activations, random_activation
):
    """Opt in to reuse while preserving independent stochastic captures by default."""
    rng = np.random.default_rng(37)
    weights = [
        onnx.numpy_helper.from_array(rng.normal(size=(16, 16)).astype(np.float32), name)
        for name in ("wa", "wb", "wc", "wd")
    ]
    nodes = [
        onnx.helper.make_node("MatMul", ["x", "wa"], ["a"], name="a"),
        onnx.helper.make_node("MatMul", ["x", "wb"], ["b"], name="b"),
        onnx.helper.make_node("Relu", ["a"], ["hidden"], name="hidden"),
        onnx.helper.make_node("MatMul", ["hidden", "wc"], ["c"], name="c"),
        onnx.helper.make_node("MatMul", ["x", "wd"], ["d"], name="d"),
        onnx.helper.make_node("Add", ["b", "c"], ["sum"], name="sum"),
        onnx.helper.make_node("Add", ["sum", "d"], ["y"], name="y"),
    ]
    activation_name = "sample" if random_activation else "x"
    if random_activation:
        for node in nodes:
            if node.input[0] == "x":
                node.input[0] = activation_name
        nodes[:0] = [
            onnx.helper.make_node("RandomNormalLike", ["x"], ["noise"], seed=1.0),
            onnx.helper.make_node("Add", ["x", "noise"], [activation_name]),
        ]
    graph = onnx.helper.make_graph(
        nodes,
        "shared_activations",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4, 16])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [4, 16])],
        weights,
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 21)])
    model.ir_version = 10
    path = str(tmp_path / "shared.onnx")
    onnx.save(model, path)
    inputs = [{"x": rng.normal(size=(4, 16)).astype(np.float32)} for _ in range(2)]
    kwargs = {
        "calibration_method": calibration_method,
        "calibration_data_reader": inputs,
        "calibration_eps": ["cpu"],
        "block_size": 8,
        "use_external_data_format": False,
    }
    reference = None
    if not (random_activation and reuse_activations):
        reference = quantize_int4(
            path,
            awqlite_reuse_activations=False,
            enable_fast_path_using_high_sysram=not random_activation,
            **kwargs,
        )
    session_calls = []
    captured_samples = []
    weight_samples = []
    create_session = moq.int4.create_inference_session
    awq_init = moq.int4.AWQLiteHelper.__init__

    def capture_weight_samples(self, x, *args, **kwargs):
        weight_samples.append(x.get() if moq.int4.has_cupy else x.copy())
        awq_init(self, x, *args, **kwargs)

    def counted_session(*args, **kwargs):
        session = create_session(*args, **kwargs)
        run = session.run

        def counted_run(names, feed):
            session_calls.append(tuple(names))
            outputs = run(names, feed)
            if random_activation and names == [activation_name]:
                captured_samples.append(outputs[0].copy())
            return outputs

        session.run = counted_run
        return session

    monkeypatch.setattr(moq.int4, "create_inference_session", counted_session)
    if random_activation:
        monkeypatch.setattr(moq.int4.AWQLiteHelper, "__init__", capture_weight_samples)
    if reuse_activations is not None:
        kwargs["awqlite_reuse_activations"] = reuse_activations
    actual = quantize_int4(path, **kwargs)
    if reference is not None:
        assert actual.SerializeToString() == reference.SerializeToString()
    first_group_calls = 2 if reuse_activations else 4
    assert session_calls == (
        [(activation_name,)] * first_group_calls + [("hidden",)] * 2 + [(activation_name,)] * 2
    )
    if random_activation:
        assert not np.array_equal(captured_samples[0], captured_samples[2])
        assert len(weight_samples) == 4
        if reuse_activations:
            np.testing.assert_array_equal(weight_samples[0], weight_samples[1])
        else:
            assert not np.array_equal(weight_samples[0], weight_samples[1])
        assert not np.array_equal(weight_samples[0], weight_samples[3])


@pytest.mark.parametrize("calibration_method", ["rtn", "rtn_dq"])
@pytest.mark.parametrize("opset", [20, 21])
def test_int4_gather(tmp_path, calibration_method, opset):
    gather_rows = 8
    gather_cols = 16
    gather_block_size = 8

    m = _gather_model(gather_rows, gather_cols, tmp_path=tmp_path, opset=opset)

    m1 = quantize_int4(
        m,
        calibration_method=calibration_method,
        gather_block_size=gather_block_size,
        gather_quantize_axis=1,
    )
    m2 = quantize_int4(
        m,
        calibration_method=calibration_method,
        gather_block_size=gather_block_size,
        gather_quantize_axis=0,
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
        found_scale = False
        for node in graph.nodes:
            if node.op == "DequantizeLinear":
                for inp in node.inputs:
                    if inp.name == "data_scale":
                        found_scale = True
                        c1 = (orig_shape[quant_axis] // block_size) == inp.shape[quant_axis]
                        c2 = orig_shape[1 - quant_axis] == inp.shape[1 - quant_axis]
                        assert c1 and c2, "Incorrect scale shape in DQ node for Gather"
        assert found_scale, "Gather DQ scale not found"
        return True

    assert is_quant_scale_with_right_shape(m1, 1, gather_block_size), (
        "DQ Scale Error in rtn_dq quantization, axis 1"
    )
    assert is_quant_scale_with_right_shape(m2, 0, gather_block_size), (
        "DQ Scale Error in rtn_dq quantization, axis 0"
    )

    data = np.arange(gather_rows * gather_cols, dtype=np.float32).reshape(gather_rows, gather_cols)
    # Symmetric INT4 RTN: the error is at most half a step, amax / 7 / 2, plus fp16 rounding.
    atol = np.abs(data).max() / 7 / 2 + 0.1
    for model in [m1, m2]:
        model = _upgrade_opset_after_quantize(model)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        output = session.run(None, {"indices": np.asarray(3, dtype=np.int64)})[0]
        assert output.shape == (gather_cols,)
        np.testing.assert_allclose(output.astype(np.float32), data[3], rtol=0, atol=atol)


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


def _reference_weight_scale(weight: np.ndarray, block_size: int) -> np.ndarray:
    """Per-block-max scale using consecutive, non-overlapping row blocks."""
    ci, co = weight.shape
    n_blocks = -(-ci // block_size)  # ceil division, matches the padding in get_weight_scale
    padded = np.pad(weight, ((0, n_blocks * block_size - ci), (0, 0)))
    scale = np.zeros_like(padded)
    for b in range(n_blocks):
        block = padded[b * block_size : (b + 1) * block_size]
        amax = np.abs(block).max(axis=0, keepdims=True)
        scale[b * block_size : (b + 1) * block_size] = np.abs(block) / (
            amax + np.finfo(amax.dtype).tiny
        )
    return scale[:ci].mean(1)


@pytest.mark.parametrize(
    ("ci", "co"),
    [
        (16, 3),  # ci divisible by block_size
        (10, 3),  # ci not divisible by block_size, needs padding
        (4, 3),  # ci == block_size: the per-channel route mixed-precision INT8 layers take
    ],
)
def test_get_weight_scale_matches_reference(ci, co):
    rng = np.random.RandomState(0)
    weight = rng.randn(ci, co)
    block_size = 4

    scale = get_weight_scale(weight.copy(), block_size)

    assert scale.shape == (ci,)
    np.testing.assert_allclose(scale, _reference_weight_scale(weight, block_size))


def test_get_weight_scale_without_block_size_is_unchanged():
    rng = np.random.RandomState(2)
    weight = rng.randn(16, 3)

    scale = get_weight_scale(weight.copy(), block_size=None)

    amax = np.abs(weight).max(axis=0, keepdims=True)
    expected = (np.abs(weight) / (amax + np.finfo(amax.dtype).tiny)).mean(1)
    np.testing.assert_allclose(scale, expected)


def _reference_subgraph_w_mean(weight: np.ndarray, block_size: int) -> np.ndarray:
    """Per-block-max mean using consecutive, non-overlapping row blocks, 1e-6 damping."""
    ci, co = weight.shape
    n_blocks = -(-ci // block_size)
    padded = np.pad(weight, ((0, n_blocks * block_size - ci), (0, 0)))
    out = np.zeros_like(padded)
    for b in range(n_blocks):
        block = padded[b * block_size : (b + 1) * block_size]
        amax = np.abs(block).max(axis=0, keepdims=True)
        out[b * block_size : (b + 1) * block_size] = np.abs(block) / (amax + 1e-6)
    return out[:ci].mean(1)


@pytest.mark.parametrize(
    ("ci", "co_parts"),
    [
        (16, [3, 2]),  # ci divisible by block_size
        (10, [3, 2]),  # ci not divisible by block_size, needs padding
    ],
)
def test_get_x_w_mean_for_subgraph_matches_reference(ci, co_parts):
    """get_x_w_mean_for_subgraph must block rows the same way get_weight_scale does.

    Regression test: it reshaped the concatenated weight straight into
    (block_size, -1), which interleaves rows from different blocks instead of
    keeping each block's `block_size` consecutive rows together, and never
    padded a `ci` that doesn't divide evenly into `block_size`.
    """
    rng = np.random.RandomState(1)
    block_size = 4
    weight_parts = [rng.randn(ci, co) for co in co_parts]
    wa_pack = [(None, onnx.numpy_helper.from_array(w), False, 0, "") for w in weight_parts]
    x = rng.randn(5, ci)

    # get_x_w_mean_for_subgraph uses int4.py's own module-level `np`, which is cupy
    # when CUDA is available: pass x through the same name so it is a cupy array in
    # that case, like production callers already do, and convert the outputs back
    # for comparison with plain numpy.
    x_m, w_m = get_x_w_mean_for_subgraph(
        wa_pack, list(range(len(wa_pack))), "unused_path", int4.np.asarray(x), block_size
    )
    if int4.has_cupy:
        x_m = int4.np.asnumpy(x_m)
        w_m = int4.np.asnumpy(w_m)

    concatenated = np.concatenate(weight_parts, axis=1)
    expected_x_m = (np.abs(x).sum(axis=0) / x.shape[0]).reshape(-1)
    np.testing.assert_allclose(x_m, expected_x_m, rtol=1e-6)
    assert w_m.shape == (ci,)
    np.testing.assert_allclose(w_m, _reference_subgraph_w_mean(concatenated, block_size), rtol=1e-5)
