/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensor_quant.h"

at::Tensor NF4_dequantize(torch::Tensor quantized_data, torch::Tensor scales, int block_size) {
  TORCH_CHECK(quantized_data.is_cuda());
  return NF4_dequantize_cuda(quantized_data, scales, block_size);
}
at::Tensor NF4_quantize(torch::Tensor quantized_data, torch::Tensor scales, int block_size) {
  TORCH_CHECK(quantized_data.is_cuda());
  return NF4_quantize_cuda(quantized_data, scales, block_size);
}
at::Tensor INT4_dequantize(torch::Tensor quantized_data, torch::Tensor scales, int block_size) {
  TORCH_CHECK(quantized_data.is_cuda());
  return INT4_dequantize_cuda(quantized_data, scales, block_size);
}
at::Tensor INT4_quantize(torch::Tensor quantized_data, torch::Tensor scales, int block_size) {
  TORCH_CHECK(quantized_data.is_cuda());
  return INT4_quantize_cuda(quantized_data, scales, block_size);
}

void fake_tensor_quant_(at::Tensor inputs, at::Tensor amax, int num_bits = 8,
                        bool is_unsigned = false, bool narrow_range = true) {
  TORCH_CHECK(inputs.is_cuda());
  TORCH_CHECK(inputs.is_contiguous()) // in-place on non-contiguous tensor is
                                      // more difficult
  TORCH_CHECK(amax.numel(), 1);
  fake_tensor_quant_cuda_inplace(inputs, amax, num_bits, is_unsigned, narrow_range);
}
// TODO: Can we add support for CPU tensors here?
at::Tensor fake_tensor_quant(at::Tensor inputs, at::Tensor amax, int num_bits = 8,
                             bool is_unsigned = false, bool narrow_range = true) {
  TORCH_CHECK(inputs.is_cuda());
  TORCH_CHECK(amax.numel(), 1);
  return fake_tensor_quant_cuda(inputs.contiguous(), amax.contiguous(), num_bits, is_unsigned,
                                narrow_range);
}

at::Tensor fake_tensor_quant_with_axis(at::Tensor inputs, at::Tensor amax, int axis,
                                       int num_bits = 8, bool is_unsigned = false,
                                       bool narrow_range = true) {
  TORCH_CHECK(inputs.is_cuda());
  TORCH_CHECK(amax.numel(), inputs.size(axis));
  return fake_tensor_quant_with_axis_cuda(inputs.contiguous(), amax.contiguous(), axis, num_bits,
                                          is_unsigned, narrow_range);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_tensor_quant_", &fake_tensor_quant_, "Fake Tensor Quant Inplace", py::arg("inputs"),
        py::arg("amax"), py::arg("num_bits") = 8, py::arg("unsigned") = false,
        py::arg("narrow_range") = true);
  m.def("fake_tensor_quant", &fake_tensor_quant, "Fake Tensor Quant", py::arg("inputs"),
        py::arg("amax"), py::arg("num_bits") = 8, py::arg("unsigned") = false,
        py::arg("narrow_range") = true);
  m.def("fake_tensor_quant_with_axis", &fake_tensor_quant_with_axis, "Fake Tensor Quant with axis",
        py::arg("inputs"), py::arg("amax"), py::arg("axis"), py::arg("num_bits") = 8,
        py::arg("unsigned") = false, py::arg("narrow_range") = true);
  m.def("NF4_dequantize", &NF4_dequantize, "Dequantize NF4 weights.");
  m.def("NF4_quantize", &NF4_quantize, "Quantize NF4 weights.");
  m.def("INT4_dequantize", &INT4_dequantize, "Dequantize INT4 weights.");
  m.def("INT4_quantize", &INT4_quantize, "Quantize INT4 weights.");
}
