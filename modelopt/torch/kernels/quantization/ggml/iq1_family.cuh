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

// The IQ1 formats share one encoder. Every 8-value vector is approximated by scale * (q + delta),
// with q from the 2048-entry ternary grid and delta = +/- 1/8; the grid is 64 KiB, past the 48 KiB
// static shared-memory limit, so it is read from global memory and left to the cache. Every group
// of vectors picks one of Format::kChoices options, with local scale choice % 8. A format
// describes the rest with a Format type:
//
//   kGroups, kVectorsPerGroup, kChoices, kPayloadBytes  block layout
//   kSharedShift  the choice also fixes the delta for the whole group, as choice / 8 (IQ1_S);
//                 otherwise each vector picks its own (IQ1_M)
//   begin(payload, d_bits)  false when the block is all zero and needs nothing more written
//   store(payload, picks, choices, d_bits)  writes every vector's pick -- its grid entry, with
//                 its own shift in bit kIq1EntryBits when the shift is per vector -- and every
//                 group's choice; called by every thread once the search is done

#pragma once

#include "common.cuh"

namespace modelopt::ggml {

#ifdef __CUDACC__

constexpr float kIq1Delta = 0.125f;
constexpr int kIq1EntryBits = 11; // 2048 entries; a per-vector shift sits just above them

__device__ __forceinline__ float iq1_delta(int shift) { return shift ? -kIq1Delta : kIq1Delta; }

template <typename Format, typename scalar_t>
__global__ void iq1_encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                           const __half *scales, uint8_t *output) {
  constexpr int kGroups = Format::kGroups;
  constexpr int kVectorsPerGroup = Format::kVectorsPerGroup;
  constexpr int kChoices = Format::kChoices;
  constexpr bool kSharedShift = Format::kSharedShift;
  static_assert(kIq1sEntries % kThreads == 0, "every thread must visit the same number of entries");
  static_assert(kIq1sEntries == 1 << kIq1EntryBits, "the shift bit sits just above the entry");
  static_assert(kGroups * kVectorsPerGroup == kVectorsPerBlock, "groups must cover the block");

  __shared__ float warp_best[kWarps * kChoices];
  __shared__ float group_error[kChoices];
  __shared__ unsigned long long warp_keys[kWarps];
  __shared__ int selected_choice;
  __shared__ uint8_t choices[kGroups];
  __shared__ uint16_t picks[kVectorsPerBlock];

  const int tid = threadIdx.x;
  const int64_t block = blockIdx.x;
  if (block >= num_blocks)
    return;

  const scalar_t *source = input + block * kBlockSize;
  uint8_t *payload = output + block * Format::kPayloadBytes;
  const __half d_half = scales[block];
  const uint16_t d_bits = __half_as_ushort(d_half);
  const float d = __half2float(d_half);
  if (!Format::begin(payload, d_bits))
    return;

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kChoices)
      group_error[tid] = 0.0f;
    __syncthreads();

    // Score every choice: each vector's best error under it, summed over the group. With a
    // per-vector shift, a vector takes the better of the two before the group chooses.
#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      float x[kVectorSize];
      float xnorm, xsum;
      load_vector(source + (group * kVectorsPerGroup + vector) * kVectorSize, x, xnorm, xsum);
      float local_best[kChoices];
#pragma unroll
      for (int choice = 0; choice < kChoices; ++choice)
        local_best[choice] = FLT_MAX;
      for (int entry = tid; entry < kIq1sEntries; entry += blockDim.x) {
        float dot, qnorm, qsum;
        grid_terms(x, grid + entry * kVectorSize, dot, qnorm, qsum);
#pragma unroll
        for (int choice = 0; choice < kChoices; ++choice) {
          const float scale = d * (2 * (choice & 7) + 1);
          if constexpr (kSharedShift) {
            local_best[choice] =
                fminf(local_best[choice],
                      shifted_error(xnorm, xsum, dot, qnorm, qsum, scale, iq1_delta(choice >> 3)));
          } else {
#pragma unroll
            for (int shift = 0; shift < 2; ++shift)
              local_best[choice] =
                  fminf(local_best[choice],
                        shifted_error(xnorm, xsum, dot, qnorm, qsum, scale, iq1_delta(shift)));
          }
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
      choices[group] = static_cast<uint8_t>(selected_choice);
    }
    __syncthreads();
    const float selected_scale = d * (2 * (selected_choice & 7) + 1);

    // Under the chosen option, each vector takes its best entry. A per-vector shift sits above
    // the entry index in the key, so a tie prefers the lower shift and then the lower entry.
#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      const int slot = group * kVectorsPerGroup + vector;
      float x[kVectorSize];
      float xnorm, xsum;
      load_vector(source + slot * kVectorSize, x, xnorm, xsum);
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kIq1sEntries; entry += blockDim.x) {
        float dot, qnorm, qsum;
        grid_terms(x, grid + entry * kVectorSize, dot, qnorm, qsum);
        if constexpr (kSharedShift) {
          const float error = shifted_error(xnorm, xsum, dot, qnorm, qsum, selected_scale,
                                            iq1_delta(selected_choice >> 3));
          const unsigned long long candidate = error_key(error, entry);
          key = candidate < key ? candidate : key;
        } else {
#pragma unroll
          for (int shift = 0; shift < 2; ++shift) {
            const float error =
                shifted_error(xnorm, xsum, dot, qnorm, qsum, selected_scale, iq1_delta(shift));
            const unsigned long long candidate = error_key(error, (shift << kIq1EntryBits) | entry);
            key = candidate < key ? candidate : key;
          }
        }
      }
      key = block_min_key(key, warp_keys);
      if (tid == 0)
        picks[slot] = static_cast<uint16_t>(key & ((1u << (kIq1EntryBits + 1)) - 1));
    }
  }
  __syncthreads();
  Format::store(payload, picks, choices, d_bits);
}

// Packs one IQ1 format from per-block FP16 scales, once its inputs have been validated.
template <typename Format>
at::Tensor iq1_encode_blocks(const at::Tensor &input, const at::Tensor &grid,
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
      at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "iq1_pack", [&] {
        iq1_encode<Format, scalar_t><<<static_cast<int>(num_blocks), kThreads, 0, stream>>>(
            values.data_ptr<scalar_t>(), num_blocks, table.data_ptr<float>(),
            reinterpret_cast<const __half *>(block_scales.data_ptr<at::Half>()),
            output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

#endif // __CUDACC__

} // namespace modelopt::ggml
