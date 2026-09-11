/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>
#include <limits>

namespace {

constexpr int kBlockSize = 256;
constexpr int kVectorSize = 8;
constexpr int kEntries = 512;
constexpr int kGroups = 16;
constexpr int kLocalScales = 16;
constexpr int kPayloadBytes = 74;
constexpr float kNativeMax = 166.625f;

template <typename scalar_t> __device__ __forceinline__ float load_float(const scalar_t *input) {
  return static_cast<float>(*input);
}

__device__ __forceinline__ float quant_error(float xnorm, float dot, float qnorm, float scale) {
  return fmaxf(fmaf(scale * scale, qnorm, fmaf(-2.0f * scale, dot, xnorm)), 0.0f);
}

__device__ __forceinline__ float even_parity_dot(const float *x, const float *q, bool odd_parity) {
  float dot = 0.0f;
  float weakest = FLT_MAX;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j) {
    const float term = fabsf(x[j]) * q[j];
    dot += term;
    weakest = fminf(weakest, term);
  }
  return odd_parity ? dot - 2.0f * weakest : dot;
}

template <typename scalar_t>
__global__ void find_scale(const scalar_t *input, int64_t num_blocks, int64_t *scale_bits) {
  const int64_t block = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= num_blocks)
    return;

  float amax = 0.0f;
  float sumsq = 0.0f;
  const scalar_t *values = input + block * kBlockSize;
#pragma unroll 1
  for (int i = 0; i < kBlockSize; ++i) {
    const float value = load_float(values + i);
    amax = fmaxf(amax, fabsf(value));
    sumsq = fmaf(value, value, sumsq);
  }
  if (amax == 0.0f) {
    scale_bits[block] = 0;
    return;
  }
  const float rms = sqrtf(sumsq / kBlockSize);
  const float peak_to_rms = rms > 0.0f ? amax / rms : 0.0f;
  const float anchor = fminf(0.92f, fmaxf(0.65f, 1.0f - 0.035f * peak_to_rms));
  const __half scale = __float2half_rn(fminf((amax / kNativeMax) * anchor, 65504.0f));
  scale_bits[block] = static_cast<int64_t>(__half_as_ushort(scale));
}

template <typename scalar_t>
__global__ void encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                       const int64_t *scale_bits, uint8_t *output) {
  __shared__ float shared_grid[kEntries * kVectorSize];
  __shared__ float grid_norm[kEntries];
  __shared__ float warp_best[8 * kLocalScales];
  __shared__ float group_error[kLocalScales];
  __shared__ unsigned long long warp_keys[8];
  __shared__ int selected_local;
  __shared__ uint8_t locals[kGroups];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int i = tid; i < kEntries * kVectorSize; i += blockDim.x)
    shared_grid[i] = grid[i];
  __syncthreads();
  for (int entry = tid; entry < kEntries; entry += blockDim.x) {
    float norm = 0.0f;
#pragma unroll
    for (int j = 0; j < kVectorSize; ++j) {
      const float q = shared_grid[entry * kVectorSize + j];
      norm = fmaf(q, q, norm);
    }
    grid_norm[entry] = norm;
  }
  __syncthreads();

  const int64_t block = blockIdx.x;
  if (block >= num_blocks)
    return;
  const scalar_t *source = input + block * kBlockSize;
  uint8_t *payload = output + block * kPayloadBytes;
  const uint16_t d_bits = static_cast<uint16_t>(scale_bits[block]);
  const float d = __half2float(__ushort_as_half(d_bits));
  if (tid == 0) {
    payload[0] = static_cast<uint8_t>(d_bits);
    payload[1] = static_cast<uint8_t>(d_bits >> 8);
  }

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kLocalScales)
      group_error[tid] = 0.0f;
    __syncthreads();

#pragma unroll
    for (int vector = 0; vector < 2; ++vector) {
      float x[kVectorSize];
      float xnorm = 0.0f;
      int negative_count = 0;
      const int offset = group * 16 + vector * 8;
#pragma unroll
      for (int j = 0; j < kVectorSize; ++j) {
        x[j] = load_float(source + offset + j);
        xnorm = fmaf(x[j], x[j], xnorm);
        negative_count += x[j] < 0.0f;
      }
      const bool odd_parity = (negative_count & 1) != 0;
      float local_best[kLocalScales];
#pragma unroll
      for (int local = 0; local < kLocalScales; ++local)
        local_best[local] = FLT_MAX;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float *q = shared_grid + entry * kVectorSize;
        const float dot = even_parity_dot(x, q, odd_parity);
#pragma unroll
        for (int local = 0; local < kLocalScales; ++local) {
          const float scale = d * (2 * local + 1) * 0.125f;
          local_best[local] =
              fminf(local_best[local], quant_error(xnorm, dot, grid_norm[entry], scale));
        }
      }
#pragma unroll
      for (int local = 0; local < kLocalScales; ++local) {
        float value = local_best[local];
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
          value = fminf(value, __shfl_down_sync(0xffffffff, value, delta));
        if (lane == 0)
          warp_best[warp * kLocalScales + local] = value;
      }
      __syncthreads();
      if (tid < kLocalScales) {
        float value = warp_best[tid];
#pragma unroll
        for (int w = 1; w < 8; ++w)
          value = fminf(value, warp_best[w * kLocalScales + tid]);
        group_error[tid] += value;
      }
      __syncthreads();
    }

    if (tid == 0) {
      selected_local = 0;
      float best = group_error[0];
#pragma unroll
      for (int local = 1; local < kLocalScales; ++local) {
        if (group_error[local] < best) {
          best = group_error[local];
          selected_local = local;
        }
      }
      locals[group] = static_cast<uint8_t>(selected_local);
    }
    __syncthreads();
    const float selected_scale = d * (2 * selected_local + 1) * 0.125f;

#pragma unroll
    for (int vector = 0; vector < 2; ++vector) {
      float x[kVectorSize];
      float xnorm = 0.0f;
      int negative_count = 0;
      const int offset = group * 16 + vector * 8;
#pragma unroll
      for (int j = 0; j < kVectorSize; ++j) {
        x[j] = load_float(source + offset + j);
        xnorm = fmaf(x[j], x[j], xnorm);
        negative_count += x[j] < 0.0f;
      }
      const bool odd_parity = (negative_count & 1) != 0;
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float error =
            quant_error(xnorm, even_parity_dot(x, shared_grid + entry * kVectorSize, odd_parity),
                        grid_norm[entry], selected_scale);
        const unsigned long long candidate =
            (static_cast<unsigned long long>(__float_as_uint(error)) << 32) |
            static_cast<unsigned long long>(entry);
        key = candidate < key ? candidate : key;
      }
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) {
        const auto other = __shfl_down_sync(0xffffffff, key, delta);
        key = other < key ? other : key;
      }
      if (lane == 0)
        warp_keys[warp] = key;
      __syncthreads();
      if (tid == 0) {
        key = warp_keys[0];
#pragma unroll
        for (int w = 1; w < 8; ++w)
          key = warp_keys[w] < key ? warp_keys[w] : key;
        const int entry = static_cast<int>(key & 0x1ff);
        const float *q = shared_grid + entry * kVectorSize;
        int flip_index = 0;
        float weakest = fabsf(x[0]) * q[0];
#pragma unroll
        for (int j = 1; j < kVectorSize; ++j) {
          const float term = fabsf(x[j]) * q[j];
          if (term < weakest) {
            weakest = term;
            flip_index = j;
          }
        }
        int sign_mask = 0;
#pragma unroll
        for (int j = 0; j < kVectorSize; ++j) {
          bool is_negative = x[j] < 0.0f;
          if (odd_parity && j == flip_index)
            is_negative = !is_negative;
          sign_mask |= static_cast<int>(is_negative) << j;
        }
        const uint16_t code = static_cast<uint16_t>(entry | ((sign_mask & 0x7f) << 9));
        const int code_offset = 2 + 2 * (group * 2 + vector);
        payload[code_offset] = static_cast<uint8_t>(code);
        payload[code_offset + 1] = static_cast<uint8_t>(code >> 8);
      }
      __syncthreads();
    }
  }

  if (tid < 8)
    payload[66 + tid] = locals[2 * tid] | (locals[2 * tid + 1] << 4);
}

} // namespace

at::Tensor iq2_xs_pack_cuda(at::Tensor input, at::Tensor grid) {
  TORCH_CHECK(input.is_contiguous() && grid.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(input.numel() > 0 && input.numel() % kBlockSize == 0,
              "input size must be a positive multiple of 256");
  TORCH_CHECK(grid.scalar_type() == at::kFloat && grid.numel() == kEntries * kVectorSize,
              "grid must be float32 [512, 8]");
  TORCH_CHECK(input.get_device() == grid.get_device(), "input and grid must share a device");
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t num_blocks = input.numel() / kBlockSize;
  TORCH_CHECK(num_blocks <= std::numeric_limits<int>::max(), "IQ2_XS CUDA grid is too large");
  auto scales = at::empty({num_blocks}, input.options().dtype(at::kLong));
  auto output = at::empty({num_blocks, kPayloadBytes}, input.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int scale_grid = static_cast<int>((num_blocks + 255) / 256);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "iq2_xs_pack", [&] {
        find_scale<scalar_t><<<scale_grid, 256, 0, stream>>>(input.data_ptr<scalar_t>(), num_blocks,
                                                             scales.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        encode<scalar_t><<<static_cast<int>(num_blocks), 256, 0, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, grid.data_ptr<float>(),
            scales.data_ptr<int64_t>(), output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}
