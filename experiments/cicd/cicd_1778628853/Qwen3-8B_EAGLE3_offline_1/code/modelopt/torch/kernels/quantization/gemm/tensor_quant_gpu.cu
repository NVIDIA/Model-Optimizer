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
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

#define BLOCK_SIZE 128
#define EPSILON (1. / (1 << 24)) // Minimum representable of fp16

#define AT_DISPATCH_CASE_FLOATING_TYPES(...)                                                       \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                                            \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                                             \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                                              \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                                                \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

__host__ __device__ float bits_to_bound(int num_bits, int is_unsigned) {
  float bound = (1 << (num_bits - 1 + int(is_unsigned))) - 1;
  return bound;
}

__device__ float fake_tensor_quant_device(float input, float amax, int min_bound, int max_bound) {
  CUDA_KERNEL_ASSERT(amax >= 0);

  if (amax < EPSILON) {
    return 0.f;
  }

  float scale = max_bound / amax;
  float output = rint(input * scale);
  output = output > max_bound ? max_bound : output;
  output = output < min_bound ? min_bound : output;

  return output / scale;
}

template <typename T>
__global__ void fake_tensor_quant_kernel(const T *inputs, size_t n, T *outputs, const float *amax,
                                         int num_bits = 8, bool is_unsigned = false,
                                         bool narrow_range = true) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    if (is_unsigned) {
      CUDA_KERNEL_ASSERT(inputs[tid] >= 0);
    }
    float bound = bits_to_bound(num_bits, is_unsigned);
    float max_bound = bound;
    float min_bound = -(bound + !narrow_range);
    outputs[tid] = fake_tensor_quant_device((float)inputs[tid], amax[0], min_bound, max_bound);
  }
}

void fake_tensor_quant_cuda_inplace(at::Tensor inputs, at::Tensor amax, int num_bits = 8,
                                    bool is_unsigned = false, bool narrow_range = true) {
  size_t numel = inputs.numel();
  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "fake_tensor_quant_cuda_inplace", [&] {
    fake_tensor_quant_kernel<<<numel / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(
        inputs.data_ptr<scalar_t>(), numel, inputs.data_ptr<scalar_t>(),
        amax.to(at::ScalarType::Float).data_ptr<float>(), num_bits, is_unsigned, narrow_range);
  });
}

at::Tensor fake_tensor_quant_cuda(at::Tensor inputs, at::Tensor amax, int num_bits = 8,
                                  bool is_unsigned = false, bool narrow_range = true) {
  size_t numel = inputs.numel();
  auto outputs = torch::empty_like(inputs);
  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "fake_tensor_quant_cuda", [&] {
    fake_tensor_quant_kernel<<<numel / BLOCK_SIZE + 1, BLOCK_SIZE, 0, stream>>>(
        inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<scalar_t>(),
        amax.to(at::ScalarType::Float).data_ptr<float>(), num_bits, is_unsigned, narrow_range);
  });

  return outputs;
}

template <typename T>
__global__ void
fake_tensor_quant_with_axis_cuda_kernel(const T *inputs, size_t n, T *outputs, const float *amax,
                                        int axis_size, int outer_size, int num_bits = 8,
                                        bool is_unsigned = false, bool narrow_range = true) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  float bound = bits_to_bound(num_bits, is_unsigned);
  float max_bound = bound;
  float min_bound = -(bound + !narrow_range);

  for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
    if (is_unsigned) {
      CUDA_KERNEL_ASSERT(inputs[idx] >= 0);
    }
    int axis_idx = (idx / outer_size) % axis_size;

    outputs[idx] =
        fake_tensor_quant_device((float)inputs[idx], amax[axis_idx], min_bound, max_bound);
  }
}

at::Tensor fake_tensor_quant_with_axis_cuda(at::Tensor inputs, at::Tensor amax, int axis,
                                            int num_bits = 8, bool is_unsigned = false,
                                            bool narrow_range = true) {
  auto outputs = torch::empty_like(inputs);
  size_t numel = inputs.numel();
  int axis_size = inputs.size(axis);

  int outer_size = inputs.stride(axis);

  auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(inputs.type().scalarType(), "fake_tensor_quant_cuda_with_axis", [&] {
    fake_tensor_quant_with_axis_cuda_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE, 0,
                                              stream>>>(
        inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<scalar_t>(),
        amax.to(at::ScalarType::Float).data_ptr<float>(), axis_size, outer_size, num_bits,
        is_unsigned, narrow_range);
  });
  return outputs;
}

__constant__ float NF4_LUT[16] = {-1.0000f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f,
                                  -0.0911f, 0.0000f,  0.0796f,  0.1609f,  0.2461f,  0.3379f,
                                  0.4407f,  0.5626f,  0.7230f,  1.0000f};

template <typename T>
__global__ void NF4_dequantize_kernel(const uint8_t *__restrict__ quantized_data,
                                      const T *__restrict__ scales,
                                      at::BFloat16 *__restrict__ output, int block_size) {
  const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // load scale
  const auto scale = at::BFloat16(scales[blockIdx.x]);

  // load the int8 bytes
  const int packed_value = quantized_data[byte_idx];
  const int first_half_idx = (packed_value >> 4) & 0x0F;
  const int second_half_idx = packed_value & 0x0F;

  // NF4 look up
  const at::BFloat16 first_half_value = NF4_LUT[first_half_idx];
  const at::BFloat16 second_half_value = NF4_LUT[second_half_idx];

  // de-quantize
  output[2 * byte_idx] = first_half_value * scale;
  output[2 * byte_idx + 1] = second_half_value * scale;
}

at::Tensor NF4_dequantize_cuda(torch::Tensor quantized_data, torch::Tensor scales, int block_size) {
  auto device = quantized_data.device();
  const int num_bytes = quantized_data.numel();
  const int num_scales = scales.numel();

  const auto options = quantized_data.options().dtype(at::kBFloat16).device(device);
  auto output = torch::empty({2 * num_bytes}, options);

  const int blocks = num_bytes / (block_size / 2);

  // checks before launching the kernel
  CUDA_KERNEL_ASSERT(quantized_data.is_cuda() && scales.is_cuda());
  CUDA_KERNEL_ASSERT(device.index() == scales.device().index() &&
                     device.index() == output.device().index());
  CUDA_KERNEL_ASSERT(quantized_data.dtype() == torch::kUInt8);
  CUDA_KERNEL_ASSERT(num_bytes * 2 == num_scales * block_size);

  cudaSetDevice(device.index());
  // each thread de-quantize a byte of quantized_data
  AT_DISPATCH_FLOATING_TYPES(scales.type().scalarType(), "NF4_dequantize", [&] {
    NF4_dequantize_kernel<<<blocks, block_size / 2>>>(quantized_data.data_ptr<uint8_t>(),
                                                      scales.data_ptr<scalar_t>(),
                                                      output.data_ptr<at::BFloat16>(), block_size);
  });

  return output;
}

__device__ int find_closest_index(float *LUT, float value, int lut_size) {
  float min_diff = fabs(LUT[0] - value);
  int min_index = 0;
  for (int i = 1; i < lut_size; ++i) {
    float diff = fabs(LUT[i] - value);
    if (diff < min_diff) {
      min_diff = diff;
      min_index = i;
    }
  }
  return min_index;
}

template <typename T>
__global__ void NF4_quantize_kernel(const T *__restrict__ input, const T *__restrict__ scales,
                                    uint8_t *__restrict__ output, int block_size) {
  const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load input and scale
  const auto first_input = input[byte_idx * 2];
  const auto second_input = input[byte_idx * 2 + 1];
  const auto scale = scales[blockIdx.x];

  // Scale input
  const auto scaled_first_input = first_input / scale;
  const auto scaled_second_input = second_input / scale;

  // Find the index
  const int first_index = find_closest_index(NF4_LUT, scaled_first_input, 16);
  const int second_index = find_closest_index(NF4_LUT, scaled_second_input, 16);

  // Pack the value
  const uint8_t packed_value = ((first_index & 0xF) << 4) | ((second_index & 0xF));

  // Store packed value in output tensor
  output[byte_idx] = packed_value;
}

torch::Tensor NF4_quantize_cuda(torch::Tensor input, torch::Tensor scales, int block_size) {
  auto device = input.device();
  const int numel = input.numel();
  const int num_scales = scales.numel();
  const int output_byte = numel / 2;

  auto options = input.options().dtype(torch::kUInt8).device(device);
  auto output = torch::empty({output_byte}, options);

  // checks before launching the kernel
  CUDA_KERNEL_ASSERT(input.is_cuda() && scales.is_cuda() && output.is_cuda());
  CUDA_KERNEL_ASSERT(device.index() == scales.device().index() &&
                     device.index() == output.device().index());
  CUDA_KERNEL_ASSERT(block_size % 2 == 0);
  CUDA_KERNEL_ASSERT(numel % block_size == 0);

  const int blocks = numel / block_size;
  cudaSetDevice(device.index());
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "NF4_quantize", [&] {
    NF4_quantize_kernel<<<blocks, block_size / 2>>>(input.data_ptr<scalar_t>(),
                                                    scales.data_ptr<scalar_t>(),
                                                    output.data_ptr<uint8_t>(), block_size);
  });
  return output;
}

template <typename T>
__global__ void INT4_dequantize_kernel(const uint8_t *__restrict__ quantized_data,
                                       const T *__restrict__ scales, T *__restrict__ output,
                                       int block_size) {
  const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const T scale_quant_maxbound = 7;

  // load scale
  const auto scale = scales[blockIdx.x];

  // load the scaled values and convert them to (-(scale_quant_maxbound+1), scale_quant_maxbound)
  const int packed_value = quantized_data[byte_idx];
  const T first_half_value = ((packed_value >> 4) & 0x0F) - (scale_quant_maxbound + 1);
  const T second_half_value = (packed_value & 0x0F) - (scale_quant_maxbound + 1);

  // de-quantize
  output[2 * byte_idx] = first_half_value / scale;
  output[2 * byte_idx + 1] = second_half_value / scale;
}

at::Tensor INT4_dequantize_cuda(torch::Tensor quantized_data, torch::Tensor scales,
                                int block_size) {
  auto device = quantized_data.device();
  const int num_bytes = quantized_data.numel();
  const int num_scales = scales.numel();

  const auto options = scales.options().device(device);
  auto output = torch::empty({2 * num_bytes}, options);

  const int blocks = num_bytes / (block_size / 2);

  // checks before launching the kernel
  CUDA_KERNEL_ASSERT(quantized_data.is_cuda() && scales.is_cuda() && output.is_cuda());
  CUDA_KERNEL_ASSERT(device.index() == scales.device().index() &&
                     device.index() == output.device().index());
  CUDA_KERNEL_ASSERT(quantized_data.dtype() == torch::kUInt8);
  CUDA_KERNEL_ASSERT(num_bytes * 2 == num_scales * block_size);

  // each thread de-quantize a byte of quantized_data
  cudaSetDevice(device.index());
  AT_DISPATCH_FLOATING_TYPES(scales.type().scalarType(), "INT4_dequantize", [&] {
    INT4_dequantize_kernel<<<blocks, block_size / 2>>>(quantized_data.data_ptr<uint8_t>(),
                                                       scales.data_ptr<scalar_t>(),
                                                       output.data_ptr<scalar_t>(), block_size);
  });

  return output;
}

template <typename T>
__global__ void INT4_quantize_kernel(const T *__restrict__ input, const T *__restrict__ scales,
                                     uint8_t *__restrict__ output, int block_size) {
  const int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const T scale_quant_maxbound = 7;

  // Load input and scale
  const auto first_input = input[byte_idx * 2];
  const auto second_input = input[byte_idx * 2 + 1];
  const auto scale = scales[blockIdx.x];

  // Scale input and clamp to (-(scale_quant_maxbound+1), scale_quant_maxbound)
  auto scaled_first_input = first_input * scale;
  scaled_first_input =
      max(-(scale_quant_maxbound + 1), min(scale_quant_maxbound, scaled_first_input));
  auto scaled_second_input = second_input * scale;
  scaled_second_input =
      max(-(scale_quant_maxbound + 1), min(scale_quant_maxbound, scaled_second_input));

  // int4 to uint4: 0 - 15
  const int uint4_first_input =
      static_cast<int>(roundf(scaled_first_input + (scale_quant_maxbound + 1)));
  const int uint4_second_input =
      static_cast<int>(roundf(scaled_second_input + (scale_quant_maxbound + 1)));

  // Pack the value
  const uint8_t packed_value = ((uint4_first_input & 0xF) << 4) | ((uint4_second_input & 0xF));

  // Store packed value in output tensor
  output[byte_idx] = packed_value;
}

torch::Tensor INT4_quantize_cuda(torch::Tensor input, torch::Tensor scales, int block_size) {
  auto device = input.device();
  const int numel = input.numel();
  const int num_scales = scales.numel();
  const int output_byte = numel / 2;

  auto options = input.options().dtype(torch::kUInt8);
  auto output = torch::empty({output_byte}, options);

  // checks before launching the kernel
  CUDA_KERNEL_ASSERT(input.is_cuda() && scales.is_cuda() && output.is_cuda());
  CUDA_KERNEL_ASSERT(device.index() == scales.device().index() &&
                     device.index() == output.device().index());
  CUDA_KERNEL_ASSERT(block_size % 2 == 0);
  CUDA_KERNEL_ASSERT(numel % block_size == 0);

  const int blocks = numel / block_size;
  cudaSetDevice(device.index());
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "INT4_quantize", [&] {
    INT4_quantize_kernel<<<blocks, block_size / 2>>>(input.data_ptr<scalar_t>(),
                                                     scales.data_ptr<scalar_t>(),
                                                     output.data_ptr<uint8_t>(), block_size);
  });
  return output;
}
