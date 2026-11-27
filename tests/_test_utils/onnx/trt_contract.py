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

"""Reusable ONNX producer contracts for catching TensorRT parser and strong-typing escapes early."""

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

_FLOAT_ELEM_TYPES = {
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    TensorProto.BFLOAT16,
    TensorProto.DOUBLE,
}

_LOW_PRECISION_QUANT_ELEM_TYPES = {
    TensorProto.INT8,
    TensorProto.UINT8,
    TensorProto.INT4,
    TensorProto.UINT4,
    TensorProto.FLOAT8E4M3FN,
    TensorProto.FLOAT8E4M3FNUZ,
    TensorProto.FLOAT8E5M2,
    TensorProto.FLOAT8E5M2FNUZ,
    getattr(TensorProto, "FLOAT4E2M1", -1),
}

# Standard ONNX operators that must receive dequantized floating tensors rather than raw FP8/INT4/FP4
_FLOAT_ONLY_COMPUTE_OPS = {
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Resize",
    "Softmax",
    "LayerNormalization",
    "BatchNormalization",
    "InstanceNormalization",
    "GroupNormalization",
    "Sigmoid",
    "Tanh",
    "Relu",
    "Gelu",
}


def _collect_tensor_shapes_and_types(
    model: onnx.ModelProto,
) -> tuple[dict[str, list[int | str]], dict[str, int]]:
    shapes: dict[str, list[int | str]] = {}
    dtypes: dict[str, int] = {}

    def _record_vi(vi: onnx.ValueInfoProto) -> None:
        if not vi.type.HasField("tensor_type"):
            return
        tt = vi.type.tensor_type
        if tt.elem_type:
            dtypes[vi.name] = tt.elem_type
        if tt.HasField("shape"):
            dims: list[int | str] = []
            for d in tt.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append(-1)
            shapes[vi.name] = dims

    for vi in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        _record_vi(vi)
    for init in model.graph.initializer:
        dtypes[init.name] = init.data_type
        shapes[init.name] = [int(d) for d in init.dims]
    return shapes, dtypes


def _extract_initializers(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    result: dict[str, onnx.TensorProto] = {}

    def _walk(graph: onnx.GraphProto) -> None:
        for init in graph.initializer:
            result[init.name] = init
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _walk(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for sub in attr.graphs:
                        _walk(sub)

    _walk(model.graph)
    return result


def _iter_all_nodes(graph: onnx.GraphProto):
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _iter_all_nodes(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    yield from _iter_all_nodes(sub)


def _vi_dims(vi: onnx.ValueInfoProto) -> list[int | str]:
    if not vi.type.HasField("tensor_type") or not vi.type.tensor_type.HasField("shape"):
        return []
    out: list[int | str] = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            out.append(d.dim_param)
        else:
            out.append(-1)
    return out


def _expand_qdq_param(
    param: np.ndarray, target_shape: tuple[int, ...], axis: int, block_size: int
) -> np.ndarray:
    if param.ndim == 0 or param.size == 1:
        return np.broadcast_to(param.reshape(()), target_shape)
    rank = len(target_shape)
    norm_axis = axis + rank if axis < 0 else axis
    if block_size > 0 and param.ndim == rank:
        expanded = np.repeat(param, block_size, axis=norm_axis)
        slices = tuple(slice(0, d) for d in target_shape)
        return expanded[slices]
    reshape = [1] * rank
    reshape[norm_axis] = param.size
    return np.broadcast_to(param.reshape(reshape), target_shape)


def _decode_quant_initializer(tensor: onnx.TensorProto) -> np.ndarray | None:
    if tensor.data_type == TensorProto.FLOAT8E4M3FN:
        try:
            import ml_dtypes

            raw = (
                np.frombuffer(tensor.raw_data, dtype=ml_dtypes.float8_e4m3fn)
                if tensor.raw_data
                else numpy_helper.to_array(tensor)
            )
            return raw.astype(np.float64).reshape(tuple(int(d) for d in tensor.dims))
        except ImportError:
            return None
    try:
        return numpy_helper.to_array(tensor).astype(np.float64)
    except Exception:
        return None


def _verify_weight_dequant_fidelity(
    inferred: onnx.ModelProto,
    initializers: dict[str, onnx.TensorProto],
    orig_initializers: dict[str, onnx.TensorProto],
    *,
    min_weight_cosine: float,
    check_fp8_range_utilization: bool,
) -> None:
    """G7: CPU-only independent NumPy dequantization & dynamic-range fidelity check."""
    q_producers = {
        node.output[0]: node
        for node in _iter_all_nodes(inferred.graph)
        if (node.domain or "") in ("", "onnx.ai")
        and node.op_type == "QuantizeLinear"
        and node.output
    }

    for node in _iter_all_nodes(inferred.graph):
        if (node.domain or "") not in ("", "onnx.ai") or node.op_type != "DequantizeLinear":
            continue
        if len(node.input) < 2 or node.input[1] not in initializers:
            continue
        x_name, scale_name = node.input[0], node.input[1]
        zp_name = node.input[2] if len(node.input) >= 3 and node.input[2] else None
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        axis = int(attrs.get("axis", 1))
        block_size = int(attrs.get("block_size", 0))
        scale_arr = numpy_helper.to_array(initializers[scale_name]).astype(np.float64)

        ref_w: np.ndarray | None = None
        is_fp8_e4m3 = False
        if x_name in initializers:
            q_arr = _decode_quant_initializer(initializers[x_name])
            if q_arr is None or q_arr.ndim < 2:
                continue
            is_fp8_e4m3 = initializers[x_name].data_type == TensorProto.FLOAT8E4M3FN
            zp_arr = (
                (
                    _decode_quant_initializer(initializers[zp_name])
                    if zp_name in initializers
                    else None
                )
                if zp_name
                else None
            )
            zp_b = (
                _expand_qdq_param(zp_arr, q_arr.shape, axis, block_size)
                if zp_arr is not None
                else 0.0
            )
            sc_b = _expand_qdq_param(scale_arr, q_arr.shape, axis, block_size)
            w_hat = (q_arr - zp_b) * sc_b
            base_key = x_name.removesuffix("_quantized").removesuffix("_q")
            if base_key in orig_initializers:
                ref_w = _decode_quant_initializer(orig_initializers[base_key])
        elif x_name in q_producers and q_producers[x_name].input[0] in initializers:
            q_node = q_producers[x_name]
            w_init = initializers[q_node.input[0]]
            ref_w = _decode_quant_initializer(w_init)
            if ref_w is None or ref_w.ndim < 2:
                continue
            q_zp_name = q_node.input[2] if len(q_node.input) >= 3 and q_node.input[2] else None
            is_fp8_e4m3 = (
                q_zp_name in initializers
                and initializers[q_zp_name].data_type == TensorProto.FLOAT8E4M3FN
            )
            sc_b = _expand_qdq_param(scale_arr, ref_w.shape, axis, block_size)
            scaled = ref_w / sc_b
            q_arr = (
                np.clip(scaled, -448.0, 448.0)
                if is_fp8_e4m3
                else np.clip(np.round(scaled), -128.0, 127.0)
            )
            w_hat = q_arr * sc_b
        else:
            continue

        assert np.all(np.isfinite(w_hat)), (
            f"G7: Reconstructed weight for '{node.name}' contains NaN/Inf"
        )
        if check_fp8_range_utilization and is_fp8_e4m3 and np.max(np.abs(q_arr)) > 0:
            max_code = float(np.max(np.abs(q_arr)))
            assert max_code >= 64.0, (
                f"G7: FP8 E4M3 weight in '{node.name}' uses max |code|={max_code:.2f} << 448.0 "
                "(likely inverted 448/127 scale conversion)"
            )
        if ref_w is not None and ref_w.shape == w_hat.shape:
            ref_norm = float(np.linalg.norm(ref_w))
            hat_norm = float(np.linalg.norm(w_hat))
            if ref_norm > 1e-12:
                assert hat_norm > 1e-12, (
                    f"G7: Reconstructed weight for '{node.name}' collapsed to zero"
                )
                cos = float(np.sum(ref_w * w_hat) / (ref_norm * hat_norm))
                assert cos >= min_weight_cosine, (
                    f"G7: Reconstructed weight cosine={cos:.4f} < {min_weight_cosine} in '{node.name}'"
                )
                peak_ratio = float(np.max(np.abs(w_hat)) / np.max(np.abs(ref_w)))
                assert 0.5 <= peak_ratio <= 1.5, (
                    f"G7: Reconstructed weight peak ratio={peak_ratio:.3f} out of [0.5, 1.5] in '{node.name}'"
                )


def assert_onnx_trt_producer_contract(
    saved_model_path: str | Path,
    original_model: onnx.ModelProto | None = None,
    *,
    keep_io_types: bool | None = None,
    allowed_io_float_types: set[int] | None = None,
    min_weight_cosine: float = 0.90,
    check_fp8_range_utilization: bool = True,
) -> onnx.ModelProto:
    """Validate that a ModelOpt-produced ONNX file obeys TensorRT parser and boundary contracts.

    Checks:
      1. File-backed ``onnx.checker.check_model(path, full_check=True)`` (schema + C++ shape/type inference).
      2. Public graph I/O names, count, ordering, shapes, and non-floating dtypes match ``original_model``.
      3. Q/DQ scale tensors are non-empty, finite, and strictly positive; zero-point dtypes are valid;
         per-axis and blocked Q/DQ scale shapes match the actual input tensor rank/dimensions per
         TensorRT's ``onnxOpImporters.cpp::validateScaleDimensions`` contract.
      4. Raw quantized element types (FP8/INT4/FP4) do not leak directly into float-only compute ops.
    """
    path_str = str(saved_model_path)
    onnx.checker.check_model(path_str, full_check=True)

    loaded = onnx.load(path_str, load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(loaded, check_type=True, strict_mode=True)

    if original_model is not None:
        for field in ("input", "output"):
            orig_ios = list(getattr(original_model.graph, field))
            curr_ios = list(getattr(inferred.graph, field))
            orig_names = [io.name for io in orig_ios]
            curr_names = [io.name for io in curr_ios]
            assert curr_names == orig_names, (
                f"Public graph.{field} names/order changed: expected {orig_names}, got {curr_names}"
            )
            for orig_io, curr_io in zip(orig_ios, curr_ios):
                orig_elem = orig_io.type.tensor_type.elem_type
                curr_elem = curr_io.type.tensor_type.elem_type
                orig_dims = _vi_dims(orig_io)
                curr_dims = _vi_dims(curr_io)
                if orig_dims and curr_dims:
                    assert orig_dims == curr_dims, (
                        f"Public graph.{field} '{orig_io.name}' shape changed: {orig_dims} -> {curr_dims}"
                    )
                if keep_io_types is True or orig_elem not in _FLOAT_ELEM_TYPES:
                    assert curr_elem == orig_elem, (
                        f"Public graph.{field} '{orig_io.name}' elem_type changed from "
                        f"{orig_elem} to {curr_elem} (keep_io_types={keep_io_types})"
                    )
                elif keep_io_types is False and allowed_io_float_types is not None:
                    assert curr_elem in allowed_io_float_types, (
                        f"Public graph.{field} '{orig_io.name}' elem_type {curr_elem} "
                        f"not in allowed {allowed_io_float_types}"
                    )

    shapes, dtypes = _collect_tensor_shapes_and_types(inferred)
    initializers = _extract_initializers(inferred)

    for node in _iter_all_nodes(inferred.graph):
        domain = node.domain or ""
        if domain in ("", "onnx.ai") and node.op_type in (
            "QuantizeLinear",
            "DequantizeLinear",
        ):
            assert len(node.input) >= 2, f"{node.op_type} '{node.name}' missing scale input"
            x_name, scale_name = node.input[0], node.input[1]
            zp_name = node.input[2] if len(node.input) >= 3 and node.input[2] else None

            if scale_name in initializers:
                scale_arr = numpy_helper.to_array(initializers[scale_name]).astype(np.float64)
                assert scale_arr.size > 0, (
                    f"Empty scale initializer '{scale_name}' in '{node.name}'"
                )
                assert np.all(np.isfinite(scale_arr)), (
                    f"Non-finite scale values in '{scale_name}' of '{node.name}'"
                )
                assert np.all(scale_arr > 0), (
                    f"Non-positive scale values in '{scale_name}' of '{node.name}': min={scale_arr.min()}"
                )
                scale_shape = list(scale_arr.shape)
            else:
                scale_shape = [
                    d for d in shapes.get(scale_name, []) if isinstance(d, int) and d > 0
                ]

            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            axis = int(attrs.get("axis", 1))
            block_size = int(attrs.get("block_size", 0))

            x_shape = shapes.get(x_name)
            if x_shape and scale_name in initializers:
                rank = len(x_shape)
                norm_axis = axis + rank if axis < 0 else axis
                if scale_shape not in ([], [1]):
                    assert 0 <= norm_axis < rank, (
                        f"Invalid axis {axis} for rank-{rank} input '{x_name}' in '{node.name}'"
                    )
                    if block_size > 0:
                        assert len(scale_shape) == rank, (
                            f"Blocked {node.op_type} '{node.name}' requires scale rank {rank}, "
                            f"got {scale_shape} for input shape {x_shape}"
                        )
                        for idx, (in_dim, sc_dim) in enumerate(zip(x_shape, scale_shape)):
                            if not isinstance(in_dim, int) or in_dim <= 0:
                                continue
                            expected = (
                                math.ceil(in_dim / block_size) if idx == norm_axis else in_dim
                            )
                            assert sc_dim == expected, (
                                f"Blocked {node.op_type} '{node.name}' scale dim mismatch at axis {idx}: "
                                f"scale_shape={scale_shape}, input_shape={x_shape}, "
                                f"axis={norm_axis}, block_size={block_size}"
                            )
                    else:
                        assert len(scale_shape) == 1, (
                            f"Per-axis {node.op_type} '{node.name}' expects 1D scale, got {scale_shape}"
                        )
                        target_dim = x_shape[norm_axis]
                        if isinstance(target_dim, int) and target_dim > 0:
                            assert scale_shape[0] == target_dim, (
                                f"Per-axis {node.op_type} '{node.name}' scale length {scale_shape[0]} "
                                f"!= input dim {target_dim} at axis {norm_axis} (shape={x_shape})"
                            )

            if zp_name and zp_name in dtypes and node.op_type == "DequantizeLinear":
                x_dtype = dtypes.get(x_name)
                zp_dtype = dtypes.get(zp_name)
                if x_dtype is not None and zp_dtype is not None:
                    assert x_dtype == zp_dtype, (
                        f"DequantizeLinear '{node.name}' input dtype {x_dtype} != zero_point dtype {zp_dtype}"
                    )

        if domain in ("", "onnx.ai") and node.op_type in _FLOAT_ONLY_COMPUTE_OPS and node.input:
            first_in = node.input[0]
            in_dtype = dtypes.get(first_in)
            assert in_dtype not in _LOW_PRECISION_QUANT_ELEM_TYPES, (
                f"Operator {node.op_type} '{node.name}' receives raw quantized tensor "
                f"'{first_in}' (elem_type={in_dtype}) without DequantizeLinear"
            )

    orig_initializers = _extract_initializers(original_model) if original_model is not None else {}
    _verify_weight_dequant_fidelity(
        inferred,
        initializers,
        orig_initializers,
        min_weight_cosine=min_weight_cosine,
        check_fp8_range_utilization=check_fp8_range_utilization,
    )

    return inferred
