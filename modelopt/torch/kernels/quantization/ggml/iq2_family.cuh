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

// The IQ2 formats share one encoder. Every 8-value vector is matched against a grid of
// non-negative magnitudes held in shared memory, every group of vectors picks one of 16 local
// scales, and the signs come from the input. A format describes the rest with a Format type:
//
//   kEntries, kGroups, kVectorsPerGroup, kPayloadBytes  grid size and block layout
//   kParitySigns  stores seven sign bits and lets the eighth make the count of negatives even
//                 (IQ2_XS, IQ2_XXS), rather than storing all eight (IQ2_S)
//   store(payload, entries, signs, locals)  writes every vector's grid entry and 8-bit sign mask
//                 and every group's local scale; called by every thread once the search is done

#pragma once

#include "common.cuh"

namespace modelopt::ggml {

#ifdef __CUDACC__

constexpr int kIq2LocalScales = 16;
constexpr float kIq2LocalScaleStep = 0.125f; // Encoded scale is d * (2 * ls + 1) / 8.

// Dot product of |x| against one codebook vector under the even-parity sign rule: with an odd
// number of negatives, the coordinate with the smallest |x| * q penalty gets flipped.
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

// Dot product of |x| against one codebook vector when all eight signs are stored, so the best
// signs are simply the input's and the search compares magnitudes directly.
__device__ __forceinline__ float magnitude_dot(const float *x, const float *q) {
  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j)
    dot = fmaf(fabsf(x[j]), q[j], dot);
  return dot;
}

template <bool kParitySigns>
__device__ __forceinline__ float iq2_dot(const float *x, const float *q, bool odd_parity) {
  if constexpr (kParitySigns)
    return even_parity_dot(x, q, odd_parity);
  else
    return magnitude_dot(x, q);
}

// The input's sign mask, with the weakest coordinate flipped when the format stores parity and
// the count of negatives is odd.
template <bool kParitySigns>
__device__ __forceinline__ uint8_t iq2_sign_mask(const float *x, const float *q, bool odd_parity) {
  int flip_index = -1;
  if constexpr (kParitySigns) {
    if (odd_parity) {
      flip_index = 0;
      float weakest = fabsf(x[0]) * q[0];
#pragma unroll
      for (int j = 1; j < kVectorSize; ++j) {
        const float term = fabsf(x[j]) * q[j];
        if (term < weakest) {
          weakest = term;
          flip_index = j;
        }
      }
    }
  }
  int sign_mask = 0;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j)
    sign_mask |= static_cast<int>((x[j] < 0.0f) != (j == flip_index)) << j;
  return static_cast<uint8_t>(sign_mask);
}

// Loads one vector with its squared norm; returns whether it has an odd number of negatives.
template <typename scalar_t>
__device__ __forceinline__ bool load_signed_vector(const scalar_t *source, float (&x)[kVectorSize],
                                                   float &xnorm) {
  xnorm = 0.0f;
  int negative_count = 0;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j) {
    x[j] = load_float(source + j);
    xnorm = fmaf(x[j], x[j], xnorm);
    negative_count += x[j] < 0.0f;
  }
  return (negative_count & 1) != 0;
}

template <typename Format, typename scalar_t>
__global__ void iq2_encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                           const __half *scales, uint8_t *output) {
  constexpr int kEntries = Format::kEntries;
  constexpr int kGroups = Format::kGroups;
  constexpr int kVectorsPerGroup = Format::kVectorsPerGroup;
  constexpr bool kParitySigns = Format::kParitySigns;
  static_assert(kEntries % kThreads == 0, "every thread must visit the same number of entries");
  static_assert((kEntries & (kEntries - 1)) == 0, "the codebook index mask assumes a power of two");
  static_assert(kGroups * kVectorsPerGroup == kVectorsPerBlock, "groups must cover the block");

  __shared__ float shared_grid[kEntries * kVectorSize];
  __shared__ float grid_norm[kEntries];
  __shared__ float warp_best[kWarps * kIq2LocalScales];
  __shared__ float group_error[kIq2LocalScales];
  __shared__ unsigned long long warp_keys[kWarps];
  __shared__ int selected_local;
  __shared__ uint8_t locals[kGroups];
  __shared__ uint16_t entries[kVectorsPerBlock];
  __shared__ uint8_t signs[kVectorsPerBlock];

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
  uint8_t *payload = output + block * Format::kPayloadBytes;
  const __half d_half = scales[block];
  const uint16_t d_bits = __half_as_ushort(d_half);
  const float d = __half2float(d_half);
  if (!store_block_scale<Format::kPayloadBytes>(payload, d_bits))
    return;

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kIq2LocalScales)
      group_error[tid] = 0.0f;
    __syncthreads();

    // Score every local scale: each vector's best error under it, summed over the group.
#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      float x[kVectorSize];
      float xnorm;
      const bool odd_parity =
          load_signed_vector(source + (group * kVectorsPerGroup + vector) * kVectorSize, x, xnorm);
      float local_best[kIq2LocalScales];
#pragma unroll
      for (int local = 0; local < kIq2LocalScales; ++local)
        local_best[local] = FLT_MAX;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float dot = iq2_dot<kParitySigns>(x, shared_grid + entry * kVectorSize, odd_parity);
#pragma unroll
        for (int local = 0; local < kIq2LocalScales; ++local) {
          const float scale = d * (2 * local + 1) * kIq2LocalScaleStep;
          local_best[local] =
              fminf(local_best[local], clamped_quant_error(xnorm, dot, grid_norm[entry], scale));
        }
      }
      block_min_accumulate<kIq2LocalScales>(local_best, warp_best, group_error);
    }

    if (tid == 0) {
      selected_local = 0;
      float best = group_error[0];
#pragma unroll
      for (int local = 1; local < kIq2LocalScales; ++local) {
        if (group_error[local] < best) {
          best = group_error[local];
          selected_local = local;
        }
      }
      locals[group] = static_cast<uint8_t>(selected_local);
    }
    __syncthreads();
    const float selected_scale = d * (2 * selected_local + 1) * kIq2LocalScaleStep;

    // Under the chosen scale, each vector takes its best entry; ties go to the lowest index.
#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      const int slot = group * kVectorsPerGroup + vector;
      float x[kVectorSize];
      float xnorm;
      const bool odd_parity = load_signed_vector(source + slot * kVectorSize, x, xnorm);
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float error = clamped_quant_error(
            xnorm, iq2_dot<kParitySigns>(x, shared_grid + entry * kVectorSize, odd_parity),
            grid_norm[entry], selected_scale);
        const unsigned long long candidate = error_key(error, entry);
        key = candidate < key ? candidate : key;
      }
      key = block_min_key(key, warp_keys);
      if (tid == 0) {
        const int entry = static_cast<int>(key & (kEntries - 1));
        entries[slot] = static_cast<uint16_t>(entry);
        signs[slot] = iq2_sign_mask<kParitySigns>(x, shared_grid + entry * kVectorSize, odd_parity);
      }
    }
  }
  __syncthreads();
  Format::store(payload, entries, signs, locals);
}

// Packs one IQ2 format, once check_scaled_pack_inputs has accepted the arguments.
template <typename Format>
at::Tensor iq2_encode_blocks(const at::Tensor &input, const at::Tensor &grid,
                             const at::Tensor &scales) {
  const auto values = input.contiguous();
  const auto table = grid.contiguous();
  const auto block_scales = scales.contiguous();
  c10::cuda::CUDAGuard guard(values.device());
  const int64_t num_blocks = values.numel() / kBlockSize;
  at::Tensor output =
      at::empty({num_blocks, Format::kPayloadBytes}, values.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "iq2_pack", [&] {
        iq2_encode<Format, scalar_t><<<static_cast<int>(num_blocks), kThreads, 0, stream>>>(
            values.data_ptr<scalar_t>(), num_blocks, table.data_ptr<float>(),
            reinterpret_cast<const __half *>(block_scales.data_ptr<at::Half>()),
            output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

#endif // __CUDACC__

} // namespace modelopt::ggml
