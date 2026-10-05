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

#include "iq1_family.cuh"

namespace {

using namespace modelopt::ggml;

// The IQ1_S packed payload layout and format constants below follow the GGML
// definition at:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
struct Format {
  static constexpr int kEntries = kIq1sEntries;
  static constexpr int kGroups = 8;
  static constexpr int kVectorsPerGroup = 4;
  static constexpr int kChoices = 16; // shift * 8 + local, shared by the group
  static constexpr bool kSharedShift = true;
  static constexpr int kIndexOffset = kScaleBytes;
  static constexpr int kMetadataOffset = kIndexOffset + kVectorsPerBlock;
  static constexpr int kPayloadBytes = kMetadataOffset + 2 * kGroups;

  __device__ static bool begin(uint8_t *payload, uint16_t d_bits) {
    return store_block_scale<kPayloadBytes>(payload, d_bits);
  }

  // One index byte per vector, then one metadata word per group: three index-high bits per
  // vector, the 3-bit local scale in bits 12..14 and the shift in bit 15.
  __device__ static void store(uint8_t *payload, const uint16_t *picks, const uint8_t *choices,
                               uint16_t) {
    const int tid = threadIdx.x;
    if (tid < kVectorsPerBlock)
      payload[kIndexOffset + tid] = static_cast<uint8_t>(picks[tid]);
    if (tid < kGroups) {
      uint32_t qh = ((choices[tid] & 0x7u) << 12) | ((choices[tid] >> 3) << 15);
#pragma unroll
      for (int k = 0; k < kVectorsPerGroup; ++k)
        qh |= ((picks[tid * kVectorsPerGroup + k] >> 8) & 0x7u) << (3 * k);
      payload[kMetadataOffset + 2 * tid] = static_cast<uint8_t>(qh);
      payload[kMetadataOffset + 2 * tid + 1] = static_cast<uint8_t>(qh >> 8);
    }
  }

  // Vector v is index byte v plus three high bits from its group's metadata word, which also
  // holds the group's 3-bit local scale (bits 12..14) and delta sign (bit 15).
  __device__ static void decode(const uint8_t *block, int vector, const float *grid,
                                float (&values)[kVectorSize]) {
    const uint32_t qh = load_u16(block + kMetadataOffset + 2 * (vector / kVectorsPerGroup));
    const uint32_t entry =
        block[kIndexOffset + vector] | (((qh >> (3 * (vector % kVectorsPerGroup))) & 0x7) << 8);
    const float d = half_bits_to_float(load_u16(block + kScaleOffset));
    const float scale = __fmul_rn(d, static_cast<float>(2 * ((qh >> 12) & 0x7) + 1));
    shifted_scaled(grid + entry * kVectorSize, (qh & 0x8000) ? -kIq1Delta : kIq1Delta, scale,
                   values);
  }
};

static_assert(Format::kPayloadBytes == 50, "IQ1_S blocks are 50 bytes");

// IQ1_S derives its own block scale. The 0.61 anchor matches the reference encoder's empirical
// predictor: it favors most values instead of forcing the block's largest to be exactly
// representable.
constexpr float kMaxLocalScale = 15.0f;                           // 2 * 7 + 1
constexpr float kNativeMax = kMaxLocalScale * (1.0f + kIq1Delta); // 16.875
constexpr float kScaleAnchor = 0.61f;

template <typename scalar_t>
__global__ void find_scale(const scalar_t *input, int64_t num_blocks, __half *scales) {
  const int64_t block = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= num_blocks)
    return;

  float amax = 0.0f;
  const scalar_t *values = input + block * kBlockSize;
#pragma unroll 1
  for (int i = 0; i < kBlockSize; ++i)
    amax = fmaxf(amax, fabsf(load_float(values + i)));
  scales[block] = __float2half_rn(fminf((amax / kNativeMax) * kScaleAnchor, 65504.0f));
}

} // namespace

at::Tensor iq1_s_pack_cuda(at::Tensor input, at::Tensor grid) {
  check_pack_inputs("IQ1_S", input, grid, Format::kEntries);
  const auto values = input.contiguous();
  c10::cuda::CUDAGuard guard(values.device());
  const int64_t num_blocks = values.numel() / kBlockSize;
  auto scales = at::empty({num_blocks}, values.options().dtype(at::kHalf));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "iq1_s_scale", [&] {
        find_scale<scalar_t>
            <<<static_cast<int>((num_blocks + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
                values.data_ptr<scalar_t>(), num_blocks,
                reinterpret_cast<__half *>(scales.data_ptr<at::Half>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return iq1_encode_blocks<Format>(values, grid, scales);
}

at::Tensor iq1_s_unpack_cuda(at::Tensor packed, at::Tensor grid, at::ScalarType dtype) {
  return decode_blocks<Format>("IQ1_S", packed, grid, dtype);
}
