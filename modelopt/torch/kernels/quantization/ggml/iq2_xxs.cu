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

// The IQ2_XXS packed payload layout and format constants below follow the GGML
// definition at:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
constexpr int kEntries = kIq2xxsEntries;
constexpr int kGroups = 8;          // one 4-bit local scale per 32 values
constexpr int kVectorsPerGroup = 4; // four 8-value codebook vectors per group
constexpr int kLocalScales = 16;
constexpr int kRecordBytes = 8; // four index bytes then one little-endian uint32
constexpr int kCodeOffset = kScaleBytes;
constexpr int kPayloadBytes = kCodeOffset + kGroups * kRecordBytes;
constexpr float kLocalScaleStep = 0.125f; // Encoded scale is d * (2 * ls + 1) / 8.

static_assert(kEntries % kThreads == 0, "every thread must visit the same number of entries");
static_assert((kEntries & (kEntries - 1)) == 0, "the codebook index mask assumes a power of two");
static_assert(kPayloadBytes == 66, "IQ2_XXS blocks are 66 bytes");

// Dot product of |x| against one codebook vector, under the format's even-parity sign rule.
// IQ2_XXS stores seven sign bits per vector and recovers the eighth from their parity, exactly as
// IQ2_XS does, so an odd sign pattern must flip the coordinate with the smallest |x| * q penalty.
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
__global__ void encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                       const __half *scales, uint8_t *output) {
  __shared__ float shared_grid[kEntries * kVectorSize];
  __shared__ float grid_norm[kEntries];
  __shared__ float warp_best[kWarps * kLocalScales];
  __shared__ float group_error[kLocalScales];
  __shared__ unsigned long long warp_keys[kWarps];
  __shared__ int selected_local;
  __shared__ uint8_t locals[kGroups];
  __shared__ uint8_t entry_bytes[kGroups * kVectorsPerGroup];
  __shared__ uint8_t sign_bits[kGroups * kVectorsPerGroup];

  const int tid = threadIdx.x;
  const int64_t block = blockIdx.x;
  if (block >= num_blocks)
    return;

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

  const scalar_t *source = input + block * kBlockSize;
  uint8_t *payload = output + block * kPayloadBytes;
  const __half d_half = scales[block];
  const uint16_t d_bits = __half_as_ushort(d_half);
  const float d = __half2float(d_half);
  if (!store_block_scale<kPayloadBytes>(payload, d_bits))
    return;

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kLocalScales)
      group_error[tid] = 0.0f;
    __syncthreads();

#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      float x[kVectorSize];
      float xnorm = 0.0f;
      int negative_count = 0;
      const int offset = group * (kVectorsPerGroup * kVectorSize) + vector * kVectorSize;
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
          const float scale = d * (2 * local + 1) * kLocalScaleStep;
          local_best[local] =
              fminf(local_best[local], clamped_quant_error(xnorm, dot, grid_norm[entry], scale));
        }
      }
      block_min_accumulate<kLocalScales>(local_best, warp_best, group_error);
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
    const float selected_scale = d * (2 * selected_local + 1) * kLocalScaleStep;

#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      float x[kVectorSize];
      float xnorm = 0.0f;
      int negative_count = 0;
      const int offset = group * (kVectorsPerGroup * kVectorSize) + vector * kVectorSize;
#pragma unroll
      for (int j = 0; j < kVectorSize; ++j) {
        x[j] = load_float(source + offset + j);
        xnorm = fmaf(x[j], x[j], xnorm);
        negative_count += x[j] < 0.0f;
      }
      const bool odd_parity = (negative_count & 1) != 0;
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float error = clamped_quant_error(
            xnorm, even_parity_dot(x, shared_grid + entry * kVectorSize, odd_parity),
            grid_norm[entry], selected_scale);
        const unsigned long long candidate = error_key(error, entry);
        key = candidate < key ? candidate : key;
      }
      key = block_min_key(key, warp_keys);
      if (tid == 0) {
        const int entry = static_cast<int>(key & (kEntries - 1));
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
        const int slot = group * kVectorsPerGroup + vector;
        entry_bytes[slot] = static_cast<uint8_t>(entry);
        sign_bits[slot] = static_cast<uint8_t>(sign_mask & 0x7f);
      }
      __syncthreads();
    }
  }

  // One 8-byte record per group: four index bytes, then a uint32 holding four 7-bit sign
  // indices in bits 0..27 and the 4-bit local scale in bits 28..31.
  if (tid < kGroups) {
    uint8_t *record = payload + kCodeOffset + tid * kRecordBytes;
    const int base = tid * kVectorsPerGroup;
#pragma unroll
    for (int j = 0; j < kVectorsPerGroup; ++j)
      record[j] = entry_bytes[base + j];
    const uint32_t aux = static_cast<uint32_t>(sign_bits[base]) |
                         (static_cast<uint32_t>(sign_bits[base + 1]) << 7) |
                         (static_cast<uint32_t>(sign_bits[base + 2]) << 14) |
                         (static_cast<uint32_t>(sign_bits[base + 3]) << 21) |
                         (static_cast<uint32_t>(locals[tid]) << 28);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      record[kVectorsPerGroup + j] = static_cast<uint8_t>(aux >> (8 * j));
  }
}

} // namespace

at::Tensor iq2_xxs_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  TORCH_CHECK(input.is_contiguous() && grid.is_contiguous() && scales.is_contiguous(),
              "inputs must be contiguous");
  check_pack_inputs("IQ2_XXS", input, grid, kEntries);
  const int64_t num_blocks = input.numel() / kBlockSize;
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == num_blocks,
              "scales must be float16 [numel / 256]");
  TORCH_CHECK(input.get_device() == scales.get_device(), "input and scales must share a device");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty({num_blocks, kPayloadBytes}, input.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "iq2_xxs_pack", [&] {
        encode<scalar_t><<<static_cast<int>(num_blocks), kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, grid.data_ptr<float>(),
            reinterpret_cast<const __half *>(scales.data_ptr<at::Half>()),
            output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}
