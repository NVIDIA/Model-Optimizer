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

"""Fast CPU producer-contract tests catching TensorRT ONNX parser and boundary escapes early."""

from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from _test_utils.onnx.trt_contract import assert_onnx_trt_producer_contract
from onnx import TensorProto, helper, numpy_helper

from modelopt.onnx import trt_utils
from modelopt.onnx.autocast import convert_to_mixed_precision
from modelopt.onnx.quantization import quantize


def _build_asymmetric_gemm_branch_model(trans_b: int = 1, opset: int = 21) -> onnx.ModelProto:
    rng = np.random.default_rng(42)
    # Deliberately asymmetric M=3, K=13, N=7 so transposed vs non-transposed axes cannot coincide
    w_shape = (7, 13) if trans_b else (13, 7)
    w = rng.standard_normal(w_shape, dtype=np.float32)
    b = rng.standard_normal((7,), dtype=np.float32)

    nodes = [
        helper.make_node("Gemm", ["x", "w", "b"], ["gemm_out"], name="Gemm_0", transB=trans_b),
        helper.make_node("Relu", ["gemm_out"], ["pub_branch"], name="Relu_0"),
        # pub_branch is ALSO consumed internally while exposed as a public graph output
        helper.make_node("Add", ["pub_branch", "gemm_out"], ["pub_sum"], name="Add_0"),
    ]
    graph = helper.make_graph(
        nodes,
        "asymmetric_gemm_fanout",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 13])],
        [
            helper.make_tensor_value_info("pub_branch", TensorProto.FLOAT, ["batch", 7]),
            helper.make_tensor_value_info("pub_sum", TensorProto.FLOAT, ["batch", 7]),
        ],
        [
            numpy_helper.from_array(w, "w"),
            numpy_helper.from_array(b, "b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return model


def test_is_static_plugin_resolves_bare_soname(monkeypatch, tmp_path):
    """Bare SONAMEs from ctypes.util.find_library must resolve via linker paths before lief.parse."""
    lib_dir = tmp_path / "libs"
    conf_dir = tmp_path / "ld.so.conf.d"
    pkg_dir = tmp_path / "pkg_libs"
    lib_dir.mkdir()
    conf_dir.mkdir()
    pkg_dir.mkdir()
    fake_so = lib_dir / "libfake_plugin.so.11"
    fake_so.write_bytes(b"\x7fELF")
    (conf_dir / "trt.conf").write_text(f"# comment\n{lib_dir}\n", encoding="utf-8")

    seen_paths: list[str] = []

    def fake_lief_parse(path: str):
        seen_paths.append(path)
        if path in ("libfake_plugin.so.11", "libmissing_plugin.so.11"):
            raise RuntimeError("bare SONAME does not exist in cwd")
        return SimpleNamespace(libraries=["libnvinfer.so.11", "libc.so.6"])

    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(trt_utils, "_LD_CONF_GLOB", str(conf_dir / "*.conf"))
    monkeypatch.setattr(
        trt_utils.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=[str(pkg_dir)])
            if name == "tensorrt_libs"
            else None
        ),
    )
    monkeypatch.setattr(trt_utils.lief, "parse", fake_lief_parse)

    assert trt_utils._is_static_plugin("libfake_plugin.so.11") is True
    assert trt_utils._is_static_plugin(str(fake_so)) is True
    assert trt_utils._is_static_plugin("libmissing_plugin.so.11") is False
    assert seen_paths == [str(fake_so), str(fake_so), "libmissing_plugin.so.11"]


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("low_precision_type", ["fp16", "bf16"])
def test_autocast_preserves_public_io_and_trt_contract(tmp_path, keep_io_types, low_precision_type):
    """AutoCast must preserve exact public output names and ordering across fan-out outputs."""
    orig = _build_asymmetric_gemm_branch_model(trans_b=1, opset=21)
    orig_path = tmp_path / "orig.onnx"
    onnx.save(orig, orig_path)
    converted = convert_to_mixed_precision(
        onnx_path=str(orig_path),
        low_precision_type=low_precision_type,
        keep_io_types=keep_io_types,
    )
    out_path = tmp_path / f"autocast_{low_precision_type}_{keep_io_types}.onnx"
    onnx.save(converted, out_path)

    expected_float = (
        {TensorProto.FLOAT}
        if keep_io_types
        else ({TensorProto.FLOAT16} if low_precision_type == "fp16" else {TensorProto.BFLOAT16})
    )
    assert_onnx_trt_producer_contract(
        out_path,
        orig,
        keep_io_types=keep_io_types,
        allowed_io_float_types=expected_float,
    )


@pytest.mark.parametrize("trans_b", [0, 1])
def test_int8_quantize_asymmetric_gemm_obeys_trt_contract(tmp_path, trans_b):
    """Quantized asymmetric Gemm(transB=0/1) with fan-out outputs must satisfy TRT Q/DQ contracts."""
    orig = _build_asymmetric_gemm_branch_model(trans_b=trans_b, opset=21)
    in_path = tmp_path / f"in_transb_{trans_b}.onnx"
    out_path = tmp_path / f"quant_transb_{trans_b}.onnx"
    onnx.save(orig, in_path)

    calib = {"x": np.ones((2, 13), dtype=np.float32)}
    quantize(
        str(in_path),
        quantize_mode="int8",
        calibration_data=calib,
        calibration_eps=["cpu"],
        output_path=str(out_path),
    )
    assert_onnx_trt_producer_contract(out_path, orig, keep_io_types=True)


@pytest.mark.parametrize(
    ("defect_kind", "expected_match"),
    [
        ("transposed_blocked_dq_axis", "Blocked DequantizeLinear .* scale dim mismatch"),
        ("renamed_public_output", "Public graph.output names/order changed"),
        ("non_positive_scale", "Non-positive scale values"),
        ("raw_fp8_into_maxpool", "MaxPool .* receives raw quantized tensor"),
        ("inverted_fp8_scale", "G7: FP8 E4M3 weight .* uses max \\|code\\|=36\\.0"),
        ("scrambled_block_scales", "G7: Reconstructed weight (cosine|peak ratio)"),
    ],
)
def test_producer_contract_catches_known_trt_escape_classes(tmp_path, defect_kind, expected_match):
    """Verify the shared contract helper goes red on real TensorRT escape classes."""
    orig = _build_asymmetric_gemm_branch_model(trans_b=1, opset=21)

    if defect_kind == "renamed_public_output":
        mutated = onnx.ModelProto()
        mutated.CopyFrom(orig)
        mutated.graph.output[0].name = "pub_branch_cast_to_fp32"
        mutated.graph.node[1].output[0] = "pub_branch_cast_to_fp32"
        mutated.graph.node[2].input[0] = "pub_branch_cast_to_fp32"
        bad_path = tmp_path / "renamed.onnx"
        onnx.save(mutated, bad_path)
        with pytest.raises(AssertionError, match=expected_match):
            assert_onnx_trt_producer_contract(bad_path, orig)
        return

    if defect_kind == "transposed_blocked_dq_axis":
        # Transposed blocked-DQ mismatch: weight is [7, 16] (block_size=8 along axis 1 -> scale [7, 2]),
        # while DQ declares axis=0 against scale shape [7, 2].
        w_q = np.ones((7, 16), dtype=np.int8)
        scale = np.full((7, 2), 0.1, dtype=np.float32)
        zp = np.zeros((7, 2), dtype=np.int8)
        dq = helper.make_node(
            "DequantizeLinear",
            ["w_q", "scale", "zp"],
            ["w_dq"],
            name="DQ_bad_axis",
            axis=0,
            block_size=8,
        )
        gemm = helper.make_node("Gemm", ["x", "w_dq"], ["y"], transB=1)
        graph = helper.make_graph(
            [dq, gemm],
            "bad_blocked_dq",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 16])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 7])],
            [
                numpy_helper.from_array(w_q, "w_q"),
                numpy_helper.from_array(scale, "scale"),
                numpy_helper.from_array(zp, "zp"),
            ],
        )
        bad_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
        )
        bad_path = tmp_path / "bad_blocked_dq.onnx"
        onnx.save(bad_model, bad_path)
        with pytest.raises(
            (AssertionError, onnx.checker.ValidationError),
            match=expected_match + "|ShapeInferenceError",
        ):
            assert_onnx_trt_producer_contract(bad_path)
        return

    if defect_kind == "non_positive_scale":
        w_q = np.ones((7, 13), dtype=np.int8)
        scale = np.zeros((7,), dtype=np.float32)
        zp = np.zeros((7,), dtype=np.int8)
        dq = helper.make_node(
            "DequantizeLinear", ["w_q", "scale", "zp"], ["w_dq"], name="DQ_zero", axis=0
        )
        gemm = helper.make_node("Gemm", ["x", "w_dq"], ["y"], transB=1)
        graph = helper.make_graph(
            [dq, gemm],
            "zero_scale",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 13])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 7])],
            [
                numpy_helper.from_array(w_q, "w_q"),
                numpy_helper.from_array(scale, "scale"),
                numpy_helper.from_array(zp, "zp"),
            ],
        )
        bad_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
        )
        bad_path = tmp_path / "zero_scale.onnx"
        onnx.save(bad_model, bad_path)
        with pytest.raises(AssertionError, match=expected_match):
            assert_onnx_trt_producer_contract(bad_path)
        return

    if defect_kind == "raw_fp8_into_maxpool":
        q = helper.make_node("QuantizeLinear", ["x", "s", "zp"], ["x_fp8"], name="Q_fp8")
        mp = helper.make_node("MaxPool", ["x_fp8"], ["y_fp8"], name="MP_bad", kernel_shape=[2, 2])
        dq = helper.make_node("DequantizeLinear", ["y_fp8", "s", "zp"], ["y"], name="DQ_fp8")
        zp_tensor = helper.make_tensor("zp", TensorProto.FLOAT8E4M3FN, [], [0])
        graph = helper.make_graph(
            [q, mp, dq],
            "fp8_maxpool",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 3])],
            [numpy_helper.from_array(np.array(1.0, dtype=np.float32), "s"), zp_tensor],
        )
        bad_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
        )
        bad_path = tmp_path / "fp8_maxpool.onnx"
        onnx.save(bad_model, bad_path)
        with pytest.raises(
            (AssertionError, onnx.checker.ValidationError, onnx.shape_inference.InferenceError)
        ):
            assert_onnx_trt_producer_contract(bad_path)
        return

    if defect_kind == "inverted_fp8_scale":
        # Valid ONNX schema & positive scale, but INT8->FP8 scale multiplied by 448/127 instead of 127/448,
        # capping |W / scale| at 36.00 instead of utilizing [-448, 448].
        w_f32 = np.linspace(-10.0, 10.0, 64, dtype=np.float32).reshape(8, 8)
        bad_fp8_scale = np.array((10.0 / 127.0) * (448.0 / 127.0), dtype=np.float32)
        zp_fp8 = helper.make_tensor("zp", TensorProto.FLOAT8E4M3FN, [], [0])
        q = helper.make_node("QuantizeLinear", ["w", "s", "zp"], ["w_q"], name="Q_w")
        dq = helper.make_node("DequantizeLinear", ["w_q", "s", "zp"], ["w_dq"], name="DQ_w")
        mm = helper.make_node("MatMul", ["x", "w_dq"], ["y"], name="MM")
        graph = helper.make_graph(
            [q, dq, mm],
            "inverted_fp8_scale",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 8])],
            [
                numpy_helper.from_array(w_f32, "w"),
                numpy_helper.from_array(bad_fp8_scale, "s"),
                zp_fp8,
            ],
        )
        bad_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
        )
        bad_path = tmp_path / "inverted_fp8.onnx"
        onnx.save(bad_model, bad_path)
        with pytest.raises(AssertionError, match=expected_match):
            assert_onnx_trt_producer_contract(bad_path, bad_model)
        return

    if defect_kind == "scrambled_block_scales":
        # Valid ONNX schema & shapes, but 2D axis-0 block scales were computed with C-order strided reshape,
        # corrupting reconstructed weights relative to the original FP32 tensor.
        w_orig = np.ones((8, 4), dtype=np.float32)
        w_orig[4:, 0] = 100.0
        orig_mm = helper.make_model(
            helper.make_graph(
                [helper.make_node("MatMul", ["x", "w"], ["y"])],
                "orig_mm",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
                [numpy_helper.from_array(w_orig, "w")],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        # Scrambled scales assign 100/127 to block 0 col 0 where true weight is 1.0, and 1/127 where true is 100.0
        bad_scales = np.array(
            [
                [100.0 / 127.0, 1.0 / 127.0, 1.0 / 127.0, 1.0 / 127.0],
                [1.0 / 127.0, 1.0 / 127.0, 1.0 / 127.0, 1.0 / 127.0],
            ],
            dtype=np.float32,
        )
        w_q = np.clip(np.round(w_orig / np.repeat(bad_scales, 4, axis=0)), -127, 127).astype(
            np.int8
        )
        zp = np.zeros((2, 4), dtype=np.int8)
        dq = helper.make_node(
            "DequantizeLinear",
            ["w", "s", "zp"],
            ["w_dq"],
            name="DQ_scrambled",
            axis=0,
            block_size=4,
        )
        mm = helper.make_node("MatMul", ["x", "w_dq"], ["y"])
        bad_model = helper.make_model(
            helper.make_graph(
                [dq, mm],
                "scrambled_mm",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 8])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
                [
                    numpy_helper.from_array(w_q, "w"),
                    numpy_helper.from_array(bad_scales, "s"),
                    numpy_helper.from_array(zp, "zp"),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        bad_path = tmp_path / "scrambled_mm.onnx"
        onnx.save(bad_model, bad_path)
        with pytest.raises(AssertionError, match=expected_match):
            assert_onnx_trt_producer_contract(bad_path, orig_mm)
