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

// The IQ2_S packed payload layout and format constants below follow the GGML
// definition at:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
constexpr int kEntries = kIq2sEntries;
constexpr int kGroups = 16; // one 4-bit local scale per 16 values
constexpr int kVectorsPerGroup = 2;
constexpr int kVectors = kGroups * kVectorsPerGroup; // 32 codebook vectors per block
constexpr int kLocalScales = 16;
constexpr int kSubBlocks = 8;
constexpr int kLowOffset = kScaleBytes;             // 32 low index bytes
constexpr int kSignOffset = kLowOffset + kVectors;  // 32 sign masks
constexpr int kHighOffset = kSignOffset + kVectors; // 8 bytes, two high bits per vector
constexpr int kLocalScaleOffset = kHighOffset + kSubBlocks;
constexpr int kPayloadBytes = kLocalScaleOffset + kGroups / 2;
constexpr float kLocalScaleStep = 0.125f; // Encoded scale is d * (2 * ls + 1) / 8.

static_assert(kEntries % kThreads == 0, "every thread must visit the same number of entries");
static_assert((kEntries & (kEntries - 1)) == 0, "the codebook index mask assumes a power of two");
static_assert(kPayloadBytes == 82, "IQ2_S blocks are 82 bytes");

// IQ2_S stores all eight sign bits per vector rather than seven plus a parity bit, so unlike
// IQ2_XS and IQ2_XXS the encoder never has to flip a coordinate: the best signs are simply the
// input's, and the search compares magnitudes directly.
__device__ __forceinline__ float magnitude_dot(const float *x, const float *q) {
  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j)
    dot = fmaf(fabsf(x[j]), q[j], dot);
  return dot;
}

template <typename scalar_t>
__global__ void encode(const scalar_t *input, int64_t num_blocks, const float *grid,
                       const __half *scales, uint8_t *output) {
  extern __shared__ float shared_grid[]; // kEntries * kVectorSize, then kEntries norms
  float *grid_norm = shared_grid + kEntries * kVectorSize;
  __shared__ float warp_best[kWarps * kLocalScales];
  __shared__ float group_error[kLocalScales];
  __shared__ unsigned long long warp_keys[kWarps];
  __shared__ int selected_local;
  __shared__ uint8_t locals[kGroups];

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
  // The high-bit bytes are OR-ed into below, so they must start clear.
  if (tid < kSubBlocks)
    payload[kHighOffset + tid] = 0;
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
      const int offset = group * (kVectorsPerGroup * kVectorSize) + vector * kVectorSize;
#pragma unroll
      for (int j = 0; j < kVectorSize; ++j) {
        x[j] = load_float(source + offset + j);
        xnorm = fmaf(x[j], x[j], xnorm);
      }
      float local_best[kLocalScales];
#pragma unroll
      for (int local = 0; local < kLocalScales; ++local)
        local_best[local] = FLT_MAX;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float dot = magnitude_dot(x, shared_grid + entry * kVectorSize);
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
      const int offset = group * (kVectorsPerGroup * kVectorSize) + vector * kVectorSize;
#pragma unroll
      for (int j = 0; j < kVectorSize; ++j) {
        x[j] = load_float(source + offset + j);
        xnorm = fmaf(x[j], x[j], xnorm);
      }
      unsigned long long key = ~0ULL;
      for (int entry = tid; entry < kEntries; entry += blockDim.x) {
        const float error =
            clamped_quant_error(xnorm, magnitude_dot(x, shared_grid + entry * kVectorSize),
                                grid_norm[entry], selected_scale);
        const unsigned long long candidate = error_key(error, entry);
        key = candidate < key ? candidate : key;
      }
      key = block_min_key(key, warp_keys);
      if (tid == 0) {
        const int entry = static_cast<int>(key & (kEntries - 1));
        int sign_mask = 0;
#pragma unroll
        for (int j = 0; j < kVectorSize; ++j)
          sign_mask |= static_cast<int>(x[j] < 0.0f) << j;
        const int slot = group * kVectorsPerGroup + vector; // 0..31, GGML's qs index
        payload[kLowOffset + slot] = static_cast<uint8_t>(entry & 0xFF);
        payload[kSignOffset + slot] = static_cast<uint8_t>(sign_mask);
        // Two high index bits per vector, packed four to a byte in vector order.
        payload[kHighOffset + slot / 4] |=
            static_cast<uint8_t>(((entry >> 8) & 0x3) << (2 * (slot % 4)));
      }
      __syncthreads();
    }
  }

  if (tid < kGroups / 2)
    payload[kLocalScaleOffset + tid] = locals[2 * tid] | (locals[2 * tid + 1] << 4);
}

} // namespace

at::Tensor iq2_s_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  TORCH_CHECK(input.is_contiguous() && grid.is_contiguous() && scales.is_contiguous(),
              "inputs must be contiguous");
  check_pack_inputs("IQ2_S", input, grid, kEntries);
  const int64_t num_blocks = input.numel() / kBlockSize;
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == num_blocks,
              "scales must be float16 [numel / 256]");
  TORCH_CHECK(input.get_device() == scales.get_device(), "input and scales must share a device");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty({num_blocks, kPayloadBytes}, input.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  // The 1024-entry codebook and its norms exceed the static shared limit, so they are dynamic.
  const size_t smem = static_cast<size_t>(kEntries) * (kVectorSize + 1) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "iq2_s_pack", [&] {
        auto kernel = encode<scalar_t>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(smem)));
        kernel<<<static_cast<int>(num_blocks), kThreads, smem, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, grid.data_ptr<float>(),
            reinterpret_cast<const __half *>(scales.data_ptr<at::Half>()),
            output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}
