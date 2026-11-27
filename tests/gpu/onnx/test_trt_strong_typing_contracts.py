# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Native TensorRT strongly-typed parse + build + execute gates for ModelOpt ONNX outputs."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from _test_utils.import_helper import skip_if_no_tensorrt_python
from _test_utils.onnx.trt_contract import assert_onnx_trt_producer_contract
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx.autocast import convert_to_mixed_precision
from modelopt.onnx.quantization import quantize

skip_if_no_tensorrt_python()
import tensorrt as trt


def _build_conv_residual_branch_model(opset: int = 21) -> onnx.ModelProto:
    rng = np.random.default_rng(7)
    w1 = rng.normal(0.0, 0.2, size=(6, 3, 3, 3)).astype(np.float32)
    b1 = rng.normal(0.0, 0.05, size=(6,)).astype(np.float32)
    w2 = rng.normal(0.0, 0.2, size=(6, 6, 1, 1)).astype(np.float32)

    nodes = [
        helper.make_node("Conv", ["x", "w1", "b1"], ["conv1"], pads=[1, 1, 1, 1], name="Conv_1"),
        helper.make_node("Relu", ["conv1"], ["relu1"], name="Relu_1"),
        helper.make_node("Conv", ["relu1", "w2"], ["conv2"], name="Conv_2"),
        helper.make_node("Add", ["relu1", "conv2"], ["res_out"], name="Add_res"),
        helper.make_node("GlobalAveragePool", ["res_out"], ["pooled_out"], name="GAP"),
    ]
    graph = helper.make_graph(
        nodes,
        "conv_residual_dual_output",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 8, 8])],
        [
            helper.make_tensor_value_info("res_out", TensorProto.FLOAT, [2, 6, 8, 8]),
            helper.make_tensor_value_info("pooled_out", TensorProto.FLOAT, [2, 6, 1, 1]),
        ],
        [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(b1, "b1"),
            numpy_helper.from_array(w2, "w2"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return model


def _trt_strong_typed_infer(onnx_path: Path, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for native TensorRT engine build and execution")

    trt_logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(trt_logger, "")
    builder = trt.Builder(trt_logger)
    flags = (
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")
        else 0
    )
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt_logger)
    assert parser.parse_from_file(str(onnx_path)), "TRT OnnxParser failed:\n" + "\n".join(
        str(parser.get_error(i)) for i in range(parser.num_errors)
    )

    config = builder.create_builder_config()
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, f"TensorRT failed to build strongly-typed engine for {onnx_path}"

    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None

    trt_to_torch = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }

    device_buffers: dict[str, torch.Tensor] = {}
    output_names: list[str] = []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            mode = engine.get_tensor_mode(name)
            t_dtype = trt_to_torch[engine.get_tensor_dtype(name)]
            if mode == trt.TensorIOMode.INPUT:
                arr = feed[name]
                context.set_input_shape(name, tuple(arr.shape))
                buf = torch.from_numpy(arr).to(device="cuda", dtype=t_dtype).contiguous()
            else:
                shape = tuple(context.get_tensor_shape(name))
                buf = torch.empty(shape, device="cuda", dtype=t_dtype)
                output_names.append(name)
            device_buffers[name] = buf
            context.set_tensor_address(name, buf.data_ptr())

        assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    return {name: device_buffers[name].float().cpu().numpy() for name in output_names}


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision", ["fp16", "bf16"])
def test_autocast_builds_and_matches_in_trt_strongly_typed(tmp_path, keep_io_types, low_precision):
    """AutoCast outputs must parse, build, and numerically match FP32 in native TRT strong typing."""
    if (
        low_precision == "bf16"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() < (8, 0)
    ):
        pytest.skip("BF16 requires Ampere (SM80) or newer")

    orig = _build_conv_residual_branch_model()
    orig_path = tmp_path / "orig.onnx"
    onnx.save(orig, orig_path)

    converted = convert_to_mixed_precision(
        onnx_path=str(orig_path), low_precision_type=low_precision, keep_io_types=keep_io_types
    )
    conv_path = tmp_path / f"autocast_{low_precision}_{keep_io_types}.onnx"
    onnx.save(converted, conv_path)

    assert_onnx_trt_producer_contract(conv_path, orig, keep_io_types=keep_io_types)

    x = np.random.default_rng(11).standard_normal((2, 3, 8, 8), dtype=np.float32)
    ref_outputs = _trt_strong_typed_infer(orig_path, {"x": x})
    cand_outputs = _trt_strong_typed_infer(conv_path, {"x": x})

    assert set(cand_outputs.keys()) == {"res_out", "pooled_out"}
    for name in ("res_out", "pooled_out"):
        assert np.all(np.isfinite(cand_outputs[name]))
        np.testing.assert_allclose(cand_outputs[name], ref_outputs[name], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("quantize_mode", ["int8", "fp8"])
def test_quantize_builds_and_matches_in_trt_strongly_typed(tmp_path, quantize_mode):
    """Quantized ONNX graphs must parse and execute under native TRT strong typing with bounded error."""
    if (
        quantize_mode == "fp8"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() < (8, 9)
    ):
        pytest.skip("FP8 requires Ada/Hopper (SM89) or newer")

    orig = _build_conv_residual_branch_model()
    orig_path = tmp_path / "orig.onnx"
    quant_path = tmp_path / f"quant_{quantize_mode}.onnx"
    onnx.save(orig, orig_path)

    x = np.random.default_rng(19).uniform(-1.0, 1.0, size=(2, 3, 8, 8)).astype(np.float32)
    quantize(
        str(orig_path),
        quantize_mode=quantize_mode,
        calibration_data={"x": x},
        calibration_eps=["cpu"],
        output_path=str(quant_path),
    )

    assert_onnx_trt_producer_contract(quant_path, orig, keep_io_types=True)

    ref_outputs = _trt_strong_typed_infer(orig_path, {"x": x})
    q_outputs = _trt_strong_typed_infer(quant_path, {"x": x})

    assert set(q_outputs.keys()) == {"res_out", "pooled_out"}
    for name in ("res_out", "pooled_out"):
        assert np.all(np.isfinite(q_outputs[name]))
        np.testing.assert_allclose(q_outputs[name], ref_outputs[name], rtol=0.15, atol=0.15)
