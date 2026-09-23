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

// The IQ1_M packed payload layout and format constants below follow the GGML
// definition at:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
constexpr int kEntries = kIq1sEntries; // IQ1_M shares the IQ1_S ternary grid
constexpr int kGroups = 16;            // one 3-bit local scale per 16 values
constexpr int kVectorsPerGroup = 2;
constexpr int kVectors = kGroups * kVectorsPerGroup;
constexpr int kLocalScales = 8;
constexpr int kSubBlocks = 8;
constexpr int kScaleWords = 4;
constexpr int kLowOffset = 0;                      // no leading block scale in this format
constexpr int kHighOffset = kLowOffset + kVectors; // 16 bytes, two vectors per byte
constexpr int kScaleWordOffset = kHighOffset + 2 * kSubBlocks;
constexpr int kPayloadBytes = kScaleWordOffset + 2 * kScaleWords;
constexpr float kDelta = 0.125f;
constexpr int kEntryBits = 11; // 2048 entries; the shift bit sits above them in the sort key

static_assert(kEntries % kThreads == 0, "every thread must visit the same number of entries");
static_assert(kPayloadBytes == 56, "IQ1_M blocks are 56 bytes");

// Squared error of approximating x by scale * (q + delta). The grid is ternary and the format
// offsets it by +/- 1/8, so the dot and norm are shifted the way IQ1_S shifts them.
__device__ __forceinline__ float shifted_error(float xnorm, float xsum, float dot, float qnorm,
                                               float qsum, float scale, float delta) {
  const float shifted_dot = dot + delta * xsum;
  const float shifted_norm = qnorm + 2.0f * delta * qsum + kVectorSize * delta * delta;
  return clamped_quant_error(xnorm, shifted_dot, shifted_norm, scale);
}

// Accumulates x . q, |q|^2 and sum(q) for one codebook vector read straight from global memory.
// The 2048-entry IQ1 grid is 64 KiB, so it does not fit in shared memory; IQ1_S reads it the
// same way and relies on the cache.
__device__ __forceinline__ void grid_terms(const float *x, const float *q, float &dot, float &qnorm,
                                           float &qsum) {
  dot = 0.0f;
  qnorm = 0.0f;
  qsum = 0.0f;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j) {
    dot = fmaf(x[j], q[j], dot);
    qnorm = fmaf(q[j], q[j], qnorm);
    qsum += q[j];
  }
}

template <typename scalar_t>
__global__ void encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                       const __half *scales, uint8_t *output) {
  __shared__ float warp_best[kWarps * kLocalScales];
  __shared__ float group_error[kLocalScales];
  __shared__ unsigned long long warp_keys[kWarps];
  __shared__ int selected_local;
  __shared__ uint8_t locals[kGroups];

  const int tid = threadIdx.x;
  const int64_t block = blockIdx.x;
  if (block >= num_blocks)
    return;

  const scalar_t *source = input + block * kBlockSize;
  uint8_t *payload = output + block * kPayloadBytes;
  const __half d_half = scales[block];
  const uint16_t d_bits = __half_as_ushort(d_half);
  const float d = __half2float(d_half);
  // IQ1_M has no leading scale field, so the shared helper does not apply: a zero block is all
  // zero bytes, which decodes to a zero scale and therefore zero values.
  if ((d_bits & 0x7FFF) == 0) {
    if (tid < kPayloadBytes)
      payload[tid] = 0;
    return;
  }
  if (tid < kHighOffset + 2 * kSubBlocks)
    payload[tid] = 0; // index-high and shift nibbles are OR-ed into below
  __syncthreads();

#pragma unroll 1
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kLocalScales)
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
      float local_best[kLocalScales];
#pragma unroll
      for (int local = 0; local < kLocalScales; ++local)
        local_best[local] = FLT_MAX;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        float dot, qnorm, qsum;
        grid_terms(x, grid + entry * kVectorSize, dot, qnorm, qsum);
        // The shift is free per vector here, unlike IQ1_S where it is shared, so take the
        // better of the two before the local scale is chosen.
#pragma unroll
        for (int shift = 0; shift < 2; ++shift) {
          const float delta = shift ? -kDelta : kDelta;
#pragma unroll
          for (int local = 0; local < kLocalScales; ++local) {
            const float scale = d * (2 * local + 1);
            local_best[local] = fminf(local_best[local],
                                      shifted_error(xnorm, xsum, dot, qnorm, qsum, scale, delta));
          }
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
        float dot, qnorm, qsum;
        grid_terms(x, grid + entry * kVectorSize, dot, qnorm, qsum);
#pragma unroll
        for (int shift = 0; shift < 2; ++shift) {
          const float delta = shift ? -kDelta : kDelta;
          const float error = shifted_error(xnorm, xsum, dot, qnorm, qsum, selected_scale, delta);
          // Shift occupies the bits above the entry index so that a tie prefers the lower
          // shift first and then the lower entry, matching the reference encoder.
          const unsigned long long candidate = error_key(error, (shift << kEntryBits) | entry);
          key = candidate < key ? candidate : key;
        }
      }
      key = block_min_key(key, warp_keys);
      if (tid == 0) {
        const int packed_index = static_cast<int>(key & 0xFFFFFFFFULL);
        const int entry = packed_index & ((1 << kEntryBits) - 1);
        const int shift = (packed_index >> kEntryBits) & 1;
        const int slot = group * kVectorsPerGroup + vector; // 0..31
        payload[kLowOffset + slot] = static_cast<uint8_t>(entry & 0xFF);
        // Two vectors share a qh byte: low nibble first, each holding three index-high bits
        // and one shift bit.
        const int nibble = (slot % 4) % 2;
        const int qh_index = 2 * (slot / 4) + (slot % 4) / 2;
        payload[kHighOffset + qh_index] |=
            static_cast<uint8_t>((((entry >> 8) & 0x7) | (shift << 3)) << (4 * nibble));
      }
      __syncthreads();
    }
  }

  // Four scale words: each carries two sub-blocks' 3-bit local scales in bits 0..11 and one
  // nibble of the FP16 block scale in bits 12..15.
  if (tid < kScaleWords) {
    uint32_t word = 0;
#pragma unroll
    for (int parity = 0; parity < 2; ++parity) {
      const int sub = 2 * tid + parity;
      const int base = 6 * parity;
      word |= static_cast<uint32_t>(locals[2 * sub]) << base;
      word |= static_cast<uint32_t>(locals[2 * sub + 1]) << (base + 3);
    }
    word |= static_cast<uint32_t>((d_bits >> (4 * tid)) & 0xF) << 12;
    payload[kScaleWordOffset + 2 * tid] = static_cast<uint8_t>(word);
    payload[kScaleWordOffset + 2 * tid + 1] = static_cast<uint8_t>(word >> 8);
  }
}

} // namespace

at::Tensor iq1_m_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  TORCH_CHECK(input.is_contiguous() && grid.is_contiguous() && scales.is_contiguous(),
              "inputs must be contiguous");
  check_pack_inputs("IQ1_M", input, grid, kEntries);
  const int64_t num_blocks = input.numel() / kBlockSize;
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == num_blocks,
              "scales must be float16 [numel / 256]");
  TORCH_CHECK(input.get_device() == scales.get_device(), "input and scales must share a device");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty({num_blocks, kPayloadBytes}, input.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "iq1_m_pack", [&] {
        encode<scalar_t><<<static_cast<int>(num_blocks), kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, grid.data_ptr<float>(),
            reinterpret_cast<const __half *>(scales.data_ptr<at::Half>()),
            output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}
