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

#include "common.cuh"

namespace {

using namespace modelopt::ggml;

// The IQ1_S packed payload layout and format constants below follow the GGML
// definition at:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
constexpr int kEntries = kIq1sEntries;
constexpr int kGroups = 8;
constexpr int kVectorsPerGroup = 4;
constexpr int kChoices = 16;
constexpr int kIndexOffset = kScaleBytes;
constexpr int kIndexBytes = kBlockSize / kVectorSize;
constexpr int kMetadataOffset = kIndexOffset + kIndexBytes;
constexpr int kPayloadBytes = kMetadataOffset + 2 * kGroups;
constexpr float kDelta = 0.125f;        // The metadata shift bit selects +1/8 or -1/8.
constexpr float kMaxLocalScale = 15.0f; // Largest multiplier: 2 * 7 + 1.
constexpr float kMaxShiftedMagnitude = 1.0f + kDelta;
constexpr float kNativeMax = kMaxLocalScale * kMaxShiftedMagnitude; // 16.875.
constexpr float kScaleAnchor = 0.61f;

static_assert(kEntries % kThreads == 0, "every thread must visit the same number of entries");
static_assert((kEntries & (kEntries - 1)) == 0, "the codebook index mask assumes a power of two");

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
  return clamped_quant_error(xnorm, shifted_dot, shifted_norm, scale);
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
  // Match the reference encoder's empirical predictor. The 0.61 anchor favors most values
  // instead of forcing the block's largest value to be exactly representable.
  const __half scale = __float2half_rn(fminf((amax / kNativeMax) * kScaleAnchor, 65504.0f));
  scale_bits[block] = static_cast<int64_t>(__half_as_ushort(scale));
}

template <typename scalar_t>
__global__ void encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                       const int64_t *scale_bits, uint8_t *output) {
  __shared__ float warp_best[kWarps * kChoices];
  __shared__ float group_error[kChoices];
  __shared__ unsigned long long warp_keys[kWarps];
  __shared__ int selected_choice;
  __shared__ uint16_t selected_entries[kVectorsPerGroup];

  const int tid = threadIdx.x;
  const int64_t block = blockIdx.x;
  if (block >= num_blocks)
    return;

  const scalar_t *source = input + block * kBlockSize;
  uint8_t *payload = output + block * kPayloadBytes;
  const uint16_t d_bits = static_cast<uint16_t>(scale_bits[block]);
  const float d = __half2float(__ushort_as_half(d_bits));
  if (!store_block_scale<kPayloadBytes>(payload, d_bits))
    return;

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kChoices)
      group_error[tid] = 0.0f;
    __syncthreads();

#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      float x[kVectorSize];
      float xnorm = 0.0f;
      float xsum = 0.0f;
      const int offset = group * (kVectorsPerGroup * kVectorSize) + vector * kVectorSize;
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
          local_best[choice] = fminf(local_best[choice],
                                     clamped_quant_error(xnorm, shifted_dot, shifted_norm, scale));
        }
      }
      block_min_accumulate<kChoices>(local_best, warp_best, group_error);
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
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      float x[kVectorSize];
      float xnorm = 0.0f;
      float xsum = 0.0f;
      const int offset = group * (kVectorsPerGroup * kVectorSize) + vector * kVectorSize;
#pragma unroll
      for (int j = 0; j < kVectorSize; ++j) {
        x[j] = load_float(source + offset + j);
        xnorm = fmaf(x[j], x[j], xnorm);
        xsum += x[j];
      }
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float error =
            quant_error(xnorm, xsum, x, grid + entry * kVectorSize, selected_scale, selected_delta);
        const unsigned long long candidate = error_key(error, entry);
        key = candidate < key ? candidate : key;
      }
      key = block_min_key(key, warp_keys);
      if (tid == 0) {
        const uint16_t entry = static_cast<uint16_t>(key & (kEntries - 1));
        selected_entries[vector] = entry;
        payload[kIndexOffset + group * kVectorsPerGroup + vector] = static_cast<uint8_t>(entry);
      }
    }

    if (tid == 0) {
      const uint16_t qh = static_cast<uint16_t>(
          ((selected_entries[0] >> 8) & 7) | (((selected_entries[1] >> 8) & 7) << 3) |
          (((selected_entries[2] >> 8) & 7) << 6) | (((selected_entries[3] >> 8) & 7) << 9) |
          (selected_local << 12) | ((selected_choice >> 3) << 15));
      payload[kMetadataOffset + 2 * group] = static_cast<uint8_t>(qh);
      payload[kMetadataOffset + 2 * group + 1] = static_cast<uint8_t>(qh >> 8);
    }
    __syncthreads();
  }
}

} // namespace

at::Tensor iq1_s_pack_cuda(at::Tensor input, at::Tensor grid) {
  TORCH_CHECK(input.is_contiguous() && grid.is_contiguous(), "inputs must be contiguous");
  check_pack_inputs("IQ1_S", input, grid, kEntries);
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t num_blocks = input.numel() / kBlockSize;
  auto scales = at::empty({num_blocks}, input.options().dtype(at::kLong));
  auto output = at::empty({num_blocks, kPayloadBytes}, input.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int scale_grid = static_cast<int>((num_blocks + kThreads - 1) / kThreads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "iq1_s_pack", [&] {
        find_scale<scalar_t><<<scale_grid, kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, scales.data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        encode<scalar_t><<<static_cast<int>(num_blocks), kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, grid.data_ptr<float>(),
            scales.data_ptr<int64_t>(), output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}
