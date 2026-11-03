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
//
// A weighted encode scales each value's squared error by its column's importance w, so a vector's
// error under scale * (q + delta) is sum w x^2 - 2 scale sum w x (q + delta) +
// scale^2 sum w (q + delta)^2. With w = 1 every term is computed exactly as the unweighted one.

#pragma once

#include "common.cuh"

namespace modelopt::ggml {

#ifdef __CUDACC__

constexpr float kIq1Delta = 0.125f;
constexpr int kIq1EntryBits = 11; // 2048 entries; a per-vector shift sits just above them

__device__ __forceinline__ float iq1_delta(int shift) { return shift ? -kIq1Delta : kIq1Delta; }

// Loads vector slot's importance from the block's 256 weights, and replaces xnorm, xsum and
// weight_sum by sum w x^2, sum w x and sum w.
template <bool kWeighted>
__device__ __forceinline__ void
load_iq1_importance(const float *weights, int slot, const float (&x)[kVectorSize],
                    float (&w)[kVectorSize], float &xnorm, float &xsum, float &weight_sum) {
  if constexpr (kWeighted) {
    const float *source = weights + slot * kVectorSize;
    xnorm = 0.0f;
    xsum = 0.0f;
    weight_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kVectorSize; ++j) {
      w[j] = source[j];
      const float wx = w[j] * x[j];
      xnorm = fmaf(wx, x[j], xnorm);
      xsum += wx;
      weight_sum += w[j];
    }
  }
}

// grid_terms, weighted: sum w x q, sum w q^2 and sum w q.
template <bool kWeighted>
__device__ __forceinline__ void iq1_grid_terms(const float (&x)[kVectorSize],
                                               const float (&w)[kVectorSize], const float *q,
                                               float &dot, float &qnorm, float &qsum) {
  if constexpr (kWeighted) {
    dot = 0.0f;
    qnorm = 0.0f;
    qsum = 0.0f;
#pragma unroll
    for (int j = 0; j < kVectorSize; ++j) {
      const float wq = w[j] * q[j];
      dot = fmaf(x[j], wq, dot);
      qnorm = fmaf(wq, q[j], qnorm);
      qsum += wq;
    }
  } else {
    grid_terms(x, q, dot, qnorm, qsum);
  }
}

template <typename Format, bool kWeighted, typename scalar_t>
__global__ void iq1_encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                           const __half *scales, const float *importance, int64_t blocks_per_row,
                           uint8_t *output) {
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
  const float *weights = kWeighted ? importance + (block % blocks_per_row) * kBlockSize : nullptr;

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kChoices)
      group_error[tid] = 0.0f;
    __syncthreads();

    // Score every choice: each vector's best error under it, summed over the group. With a
    // per-vector shift, a vector takes the better of the two before the group chooses.
#pragma unroll
    for (int vector = 0; vector < kVectorsPerGroup; ++vector) {
      const int slot = group * kVectorsPerGroup + vector;
      float x[kVectorSize], w[kVectorSize];
      float xnorm, xsum, weight_sum = kVectorSize;
      load_vector(source + slot * kVectorSize, x, xnorm, xsum);
      load_iq1_importance<kWeighted>(weights, slot, x, w, xnorm, xsum, weight_sum);
      float local_best[kChoices];
#pragma unroll
      for (int choice = 0; choice < kChoices; ++choice)
        local_best[choice] = FLT_MAX;
      for (int entry = tid; entry < kIq1sEntries; entry += blockDim.x) {
        float dot, qnorm, qsum;
        iq1_grid_terms<kWeighted>(x, w, grid + entry * kVectorSize, dot, qnorm, qsum);
#pragma unroll
        for (int choice = 0; choice < kChoices; ++choice) {
          const float scale = d * (2 * (choice & 7) + 1);
          if constexpr (kSharedShift) {
            local_best[choice] =
                fminf(local_best[choice], shifted_error(xnorm, xsum, dot, qnorm, qsum, scale,
                                                        iq1_delta(choice >> 3), weight_sum));
          } else {
#pragma unroll
            for (int shift = 0; shift < 2; ++shift)
              local_best[choice] =
                  fminf(local_best[choice], shifted_error(xnorm, xsum, dot, qnorm, qsum, scale,
                                                          iq1_delta(shift), weight_sum));
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
      float x[kVectorSize], w[kVectorSize];
      float xnorm, xsum, weight_sum = kVectorSize;
      load_vector(source + slot * kVectorSize, x, xnorm, xsum);
      load_iq1_importance<kWeighted>(weights, slot, x, w, xnorm, xsum, weight_sum);
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kIq1sEntries; entry += blockDim.x) {
        float dot, qnorm, qsum;
        iq1_grid_terms<kWeighted>(x, w, grid + entry * kVectorSize, dot, qnorm, qsum);
        if constexpr (kSharedShift) {
          const float error = shifted_error(xnorm, xsum, dot, qnorm, qsum, selected_scale,
                                            iq1_delta(selected_choice >> 3), weight_sum);
          const unsigned long long candidate = error_key(error, entry);
          key = candidate < key ? candidate : key;
        } else {
#pragma unroll
          for (int shift = 0; shift < 2; ++shift) {
            const float error = shifted_error(xnorm, xsum, dot, qnorm, qsum, selected_scale,
                                              iq1_delta(shift), weight_sum);
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

// Packs one IQ1 format from per-block FP16 scales, once its inputs (and check_importance, if
// given) have been validated.
template <typename Format>
at::Tensor iq1_encode_blocks(const at::Tensor &input, const at::Tensor &grid,
                             const at::Tensor &scales,
                             const std::optional<at::Tensor> &importance) {
  const auto values = input.contiguous();
  const auto table = grid.contiguous();
  const auto block_scales = scales.contiguous();
  const auto weights = importance.has_value() ? importance->contiguous() : at::Tensor();
  c10::cuda::CUDAGuard guard(values.device());
  const int64_t num_blocks = values.numel() / kBlockSize;
  at::Tensor output =
      at::empty({num_blocks, Format::kPayloadBytes}, values.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "iq1_pack", [&] {
        const auto *half_scales =
            reinterpret_cast<const __half *>(block_scales.data_ptr<at::Half>());
        if (weights.defined()) {
          iq1_encode<Format, true, scalar_t><<<static_cast<int>(num_blocks), kThreads, 0, stream>>>(
              values.data_ptr<scalar_t>(), num_blocks, table.data_ptr<float>(), half_scales,
              weights.data_ptr<float>(), weights.size(0), output.data_ptr<uint8_t>());
        } else {
          iq1_encode<Format, false, scalar_t>
              <<<static_cast<int>(num_blocks), kThreads, 0, stream>>>(
                  values.data_ptr<scalar_t>(), num_blocks, table.data_ptr<float>(), half_scales,
                  nullptr, 1, output.data_ptr<uint8_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

#endif // __CUDACC__

} // namespace modelopt::ggml
