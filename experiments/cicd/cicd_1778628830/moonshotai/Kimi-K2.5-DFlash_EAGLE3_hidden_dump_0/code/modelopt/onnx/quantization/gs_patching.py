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

"""Patches onnx_graphsurgeon to support explicitly setting a dtype."""

from collections.abc import Sequence

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import LazyValues

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.quant_utils import pack_float32_to_4bit_cpp_based

ONNX_MAJOR, ONNX_MINOR = tuple(map(int, onnx.__version__.split(".")[:2]))


def _onnx_supports_int4():
    return ONNX_MAJOR > 1 or (ONNX_MAJOR == 1 and ONNX_MINOR >= 16)


def _make_constant(
    name: str, values: np.ndarray | LazyValues, dtype: onnx.TensorProto.DataType
) -> gs.Constant:
    """Creates a constant with a specified dtype."""
    converted_dtype = (
        dtype if isinstance(values, LazyValues) else onnx.helper.tensor_dtype_to_np_dtype(dtype)
    )

    # Allow int8/uint8 as intermediate representation for INT4/UINT4
    # INT4/UINT4 values are stored as int8/uint8 in numpy arrays and packed during export
    is_valid_int4_intermediate = (dtype == onnx.TensorProto.INT4 and values.dtype == np.int8) or (
        dtype == onnx.TensorProto.UINT4 and values.dtype == np.uint8
    )

    if not is_valid_int4_intermediate and values.dtype != converted_dtype:
        logger.error(
            f"Trying to create tensor with incompatible types: `{values.dtype}`, `{dtype}`"
        )

    t = gs.Constant(name, values)
    setattr(t, "explicit_dtype", dtype)
    return t


def _make_variable(
    name: str, dtype: onnx.TensorProto.DataType, shape: Sequence[int | str]
) -> gs.Constant:
    """Creates a variable with a specified dtype."""
    x = gs.Variable(name, onnx.helper.tensor_dtype_to_np_dtype(dtype), shape)
    setattr(x, "explicit_dtype", dtype)
    return x


def _export_tensor_proto(tensor: gs.Constant) -> onnx.TensorProto:
    if isinstance(tensor._values, LazyValues):
        onnx_tensor = tensor._values.tensor
    else:
        # is numpy array.
        dtype = getattr(
            tensor, "explicit_dtype", onnx.helper.np_dtype_to_tensor_dtype(tensor.values.dtype)
        )

        vals = tensor.values
        if _onnx_supports_int4() and dtype in [onnx.TensorProto.INT4, onnx.TensorProto.UINT4]:
            signed = dtype == onnx.TensorProto.INT4
            packed_dtype = np.int8 if signed else np.uint8
            vals = pack_float32_to_4bit_cpp_based(tensor.values, signed=signed).astype(packed_dtype)

        onnx_tensor = onnx.helper.make_tensor(
            tensor.name,
            dtype,
            dims=tensor.values.shape,
            vals=vals.tobytes(),
            raw=True,
        )
        if tensor.data_location is not None:
            onnx_tensor.data_location = tensor.data_location
    onnx_tensor.name = tensor.name
    return onnx_tensor


def _export_value_info_proto(tensor: gs.Variable, do_type_check: bool) -> onnx.ValueInfoProto:
    if do_type_check and tensor.dtype is None:
        logger.error(
            "Graph input and output tensors must include dtype information. Please set the dtype"
            f" attribute for: {tensor}"
        )

    if tensor.dtype is not None:
        dtype = getattr(
            tensor, "explicit_dtype", onnx.helper.np_dtype_to_tensor_dtype(np.dtype(tensor.dtype))
        )
        onnx_tensor = onnx.helper.make_tensor_value_info(tensor.name, dtype, tensor.shape)
    else:
        onnx_tensor = onnx.helper.make_empty_tensor_value_info(tensor.name)
    return onnx_tensor


def patch_gs_modules():
    """Dynamically patch graphsurgeon modules."""
    logger.debug("Patching ONNX GraphSurgeon modules")
    gs.make_constant = _make_constant
    gs.make_variable = _make_variable
    gs.exporters.onnx_exporter.OnnxExporter.export_tensor_proto = _export_tensor_proto
    gs.exporters.onnx_exporter.OnnxExporter.export_value_info_proto = _export_value_info_proto
