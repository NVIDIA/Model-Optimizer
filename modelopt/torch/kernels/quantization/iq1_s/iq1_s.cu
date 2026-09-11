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
constexpr int kEntries = 2048;
constexpr int kGroups = 8;
constexpr int kLocalScales = 8;
constexpr int kChoices = 16;
constexpr int kPayloadBytes = 50;
constexpr float kDelta = 0.125f;
constexpr float kNativeMax = 16.875f;

template <typename scalar_t> __device__ __forceinline__ float load_float(const scalar_t *input) {
  return static_cast<float>(*input);
}

__device__ __forceinline__ float quant_error(float xnorm, float xsum, const float *x,
                                              const float *q, float scale, float delta) {
  float dot = 0.0f;
  float qnorm = 0.0f;
  float qsum = 0.0f;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j) {
    dot = fmaf(x[j], q[j], dot);
    qnorm = fmaf(q[j], q[j], qnorm);
    qsum += q[j];
  }
  const float shifted_dot = dot + delta * xsum;
  const float shifted_norm = qnorm + 2.0f * delta * qsum + 8.0f * delta * delta;
  return fmaxf(fmaf(scale * scale, shifted_norm, fmaf(-2.0f * scale, shifted_dot, xnorm)),
               0.0f);
}

template <typename scalar_t>
__global__ void find_scale(const scalar_t *input, int64_t num_blocks, int64_t *scale_bits) {
  const int64_t block = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= num_blocks)
    return;

  float amax = 0.0f;
  const scalar_t *values = input + block * kBlockSize;
#pragma unroll 1
  for (int i = 0; i < kBlockSize; ++i)
    amax = fmaxf(amax, fabsf(load_float(values + i)));
  const __half scale = __float2half_rn(fminf((amax / kNativeMax) * 0.61f, 65504.0f));
  scale_bits[block] = static_cast<int64_t>(__half_as_ushort(scale));
}

template <typename scalar_t>
__global__ void encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                       const int64_t *scale_bits, uint8_t *output) {
  __shared__ float warp_best[8 * kChoices];
  __shared__ float group_error[kChoices];
  __shared__ unsigned long long warp_keys[8];
  __shared__ int selected_choice;
  __shared__ uint16_t selected_entries[4];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t block = blockIdx.x;
  if (block >= num_blocks)
    return;

  const scalar_t *source = input + block * kBlockSize;
  uint8_t *payload = output + block * kPayloadBytes;
  const uint16_t d_bits = static_cast<uint16_t>(scale_bits[block]);
  const float d = __half2float(__ushort_as_half(d_bits));
  if (d_bits == 0) {
    if (tid < kPayloadBytes)
      payload[tid] = 0;
    return;
  }
  if (tid == 0) {
    payload[0] = static_cast<uint8_t>(d_bits);
    payload[1] = static_cast<uint8_t>(d_bits >> 8);
  }

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kChoices)
      group_error[tid] = 0.0f;
    __syncthreads();

#pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
      float x[kVectorSize];
      float xnorm = 0.0f;
      float xsum = 0.0f;
      const int offset = group * 32 + vector * 8;
#pragma unroll
      for (int j = 0; j < kVectorSize; ++j) {
        x[j] = load_float(source + offset + j);
        xnorm = fmaf(x[j], x[j], xnorm);
        xsum += x[j];
      }
      float local_best[kChoices];
#pragma unroll
      for (int choice = 0; choice < kChoices; ++choice)
        local_best[choice] = FLT_MAX;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float *q = grid + entry * kVectorSize;
        float dot = 0.0f;
        float qnorm = 0.0f;
        float qsum = 0.0f;
#pragma unroll
        for (int j = 0; j < kVectorSize; ++j) {
          dot = fmaf(x[j], q[j], dot);
          qnorm = fmaf(q[j], q[j], qnorm);
          qsum += q[j];
        }
#pragma unroll
        for (int choice = 0; choice < kChoices; ++choice) {
          const int local = choice & 7;
          const float delta = choice < 8 ? kDelta : -kDelta;
          const float scale = d * (2 * local + 1);
          const float shifted_dot = dot + delta * xsum;
          const float shifted_norm = qnorm + 2.0f * delta * qsum + 8.0f * delta * delta;
          const float error = fmaxf(
              fmaf(scale * scale, shifted_norm, fmaf(-2.0f * scale, shifted_dot, xnorm)), 0.0f);
          local_best[choice] = fminf(local_best[choice], error);
        }
      }
#pragma unroll
      for (int choice = 0; choice < kChoices; ++choice) {
        float value = local_best[choice];
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1)
          value = fminf(value, __shfl_down_sync(0xffffffff, value, delta));
        if (lane == 0)
          warp_best[warp * kChoices + choice] = value;
      }
      __syncthreads();
      if (tid < kChoices) {
        float value = warp_best[tid];
#pragma unroll
        for (int w = 1; w < 8; ++w)
          value = fminf(value, warp_best[w * kChoices + tid]);
        group_error[tid] += value;
      }
      __syncthreads();
    }

    if (tid == 0) {
      selected_choice = 0;
      float best = group_error[0];
#pragma unroll
      for (int choice = 1; choice < kChoices; ++choice) {
        if (group_error[choice] < best) {
          best = group_error[choice];
          selected_choice = choice;
        }
      }
    }
    __syncthreads();
    const int selected_local = selected_choice & 7;
    const float selected_delta = selected_choice < 8 ? kDelta : -kDelta;
    const float selected_scale = d * (2 * selected_local + 1);

#pragma unroll
    for (int vector = 0; vector < 4; ++vector) {
      float x[kVectorSize];
      float xnorm = 0.0f;
      float xsum = 0.0f;
      const int offset = group * 32 + vector * 8;
#pragma unroll
      for (int j = 0; j < kVectorSize; ++j) {
        x[j] = load_float(source + offset + j);
        xnorm = fmaf(x[j], x[j], xnorm);
        xsum += x[j];
      }
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float error = quant_error(xnorm, xsum, x, grid + entry * kVectorSize,
                                        selected_scale, selected_delta);
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
        const uint16_t entry = static_cast<uint16_t>(key & 0x7ff);
        selected_entries[vector] = entry;
        payload[2 + group * 4 + vector] = static_cast<uint8_t>(entry);
      }
      __syncthreads();
    }

    if (tid == 0) {
      const uint16_t qh = static_cast<uint16_t>(
          ((selected_entries[0] >> 8) & 7) | (((selected_entries[1] >> 8) & 7) << 3) |
          (((selected_entries[2] >> 8) & 7) << 6) | (((selected_entries[3] >> 8) & 7) << 9) |
          (selected_local << 12) | ((selected_choice >> 3) << 15));
      payload[34 + 2 * group] = static_cast<uint8_t>(qh);
      payload[35 + 2 * group] = static_cast<uint8_t>(qh >> 8);
    }
    __syncthreads();
  }
}

} // namespace

at::Tensor iq1_s_pack_cuda(at::Tensor input, at::Tensor grid) {
  TORCH_CHECK(input.is_contiguous() && grid.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(input.numel() > 0 && input.numel() % kBlockSize == 0,
              "input size must be a positive multiple of 256");
  TORCH_CHECK(grid.scalar_type() == at::kFloat && grid.numel() == kEntries * kVectorSize,
              "grid must be float32 [2048, 8]");
  TORCH_CHECK(input.get_device() == grid.get_device(), "input and grid must share a device");
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t num_blocks = input.numel() / kBlockSize;
  TORCH_CHECK(num_blocks <= std::numeric_limits<int>::max(), "IQ1_S CUDA grid is too large");
  auto scales = at::empty({num_blocks}, input.options().dtype(at::kLong));
  auto output = at::empty({num_blocks, kPayloadBytes}, input.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int scale_grid = static_cast<int>((num_blocks + 255) / 256);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "iq1_s_pack", [&] {
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
