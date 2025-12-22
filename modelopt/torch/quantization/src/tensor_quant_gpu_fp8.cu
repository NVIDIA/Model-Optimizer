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

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <optional>
#include <torch/extension.h>

#define BLOCK_SIZE 128

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                                                       \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                                            \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                                             \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                                              \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                                                \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template <typename T>
__global__ void fake_e4m3fy_kernel(const T *inputs, size_t n, const float *scale,
                                   const float *inv_scale, T *outputs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
    outputs[idx] = static_cast<T>(
        static_cast<float>(static_cast<__nv_fp8_e4m3>(static_cast<float>(inputs[idx]) * scale[0])) *
        inv_scale[0]);
  }
}

template <typename T>
__global__ void fake_e4m3fy_with_axis_cuda_kernel(const T *inputs, size_t n, const float *scale,
                                                  const float *inv_scale, int axis_size,
                                                  int outer_size, T *outputs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
    float x = static_cast<float>(inputs[idx]);

    int axis_id = (idx / outer_size) % axis_size;

    // compute the output
    float output =
        static_cast<float>(static_cast<__nv_fp8_e4m3>(scale[axis_id] * x)) * inv_scale[axis_id];

    outputs[idx] = output;
  }
}

at::Tensor fake_e4m3fy_with_axis(at::Tensor inputs, at::Tensor amax, int axis) {
  inputs = inputs.contiguous();
  amax = amax.contiguous().to(at::kFloat);
  auto outputs = torch::empty_like(inputs);
  size_t numel = inputs.numel();
  int axis_size = inputs.size(axis);
  int outer_size = inputs.stride(axis);

  auto scale = 448.f / amax;
  auto inv_scale = 1.f / scale;

  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "fake_e4m3fy_with_axis", [&] {
    fake_e4m3fy_with_axis_cuda_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
        inputs.data_ptr<scalar_t>(), numel, scale.data_ptr<float>(), inv_scale.data_ptr<float>(),
        axis_size, outer_size, outputs.data_ptr<scalar_t>());
  });

  return outputs;
}

at::Tensor fake_e4m3fy(at::Tensor inputs, at::Tensor amax) {
  inputs = inputs.contiguous();
  amax = amax.view(-1).to(at::kFloat);
  size_t numel = inputs.numel();
  at::Tensor scale = 448.f / amax;
  auto inv_scale = 1.f / scale;
  auto outputs = torch::empty_like(inputs);
  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "fake_e4m3fy", [&] {
    fake_e4m3fy_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0, stream>>>(
        inputs.data_ptr<scalar_t>(), numel, scale.data_ptr<float>(), inv_scale.data_ptr<float>(),
        outputs.data_ptr<scalar_t>());
  });
  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_e4m3fy", &fake_e4m3fy, "Reduce precision to E4M3", py::arg("inputs"),
        py::arg("amax"));
  m.def("fake_e4m3fy_with_axis", &fake_e4m3fy_with_axis, "Reduce precision to E4M3 (fused)",
        py::arg("inputs"), py::arg("amax"), py::arg("axis"));
}
