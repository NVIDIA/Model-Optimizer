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

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest

from modelopt.onnx.quantization.gs_patching import _export_tensor_proto, _export_value_info_proto


def test_export_constant_uses_explicit_bf16_dtype_without_fallback(monkeypatch):
    tensor = gs.Constant("scale", np.array([1.0], dtype=ml_dtypes.bfloat16))
    tensor.explicit_dtype = onnx.TensorProto.BFLOAT16

    def fail_on_fallback(_):
        raise AssertionError("NumPy dtype fallback should not be evaluated")

    monkeypatch.setattr(onnx.helper, "np_dtype_to_tensor_dtype", fail_on_fallback)

    tensor_proto = _export_tensor_proto(tensor)

    assert tensor_proto.data_type == onnx.TensorProto.BFLOAT16
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(tensor_proto), tensor.values)


def test_export_value_info_uses_explicit_onnx_dtype_without_numpy_conversion():
    tensor = gs.Variable("input", dtype=onnx.TensorProto.BFLOAT16, shape=[1])
    tensor.explicit_dtype = onnx.TensorProto.BFLOAT16

    value_info = _export_value_info_proto(tensor, do_type_check=True)

    assert value_info.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16


def test_export_value_info_accepts_onnx_dtype_without_explicit_dtype():
    tensor = gs.Variable("input", dtype=onnx.TensorProto.BFLOAT16, shape=[1])

    value_info = _export_value_info_proto(tensor, do_type_check=True)

    assert value_info.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16


def test_export_rejects_unknown_integer_dtype():
    invalid_dtype = max(onnx.TensorProto.DataType.values()) + 1
    constant = gs.Constant("scale", np.array([1.0], dtype=np.float32))
    constant.explicit_dtype = invalid_dtype

    with pytest.raises(ValueError):
        _export_tensor_proto(constant)

    variable = gs.Variable("input", dtype=invalid_dtype, shape=[1])
    with pytest.raises(ValueError):
        _export_value_info_proto(variable, do_type_check=True)
