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

#include "iq2_family.cuh"

namespace {

using namespace modelopt::ggml;

// The IQ2_XXS packed payload layout and format constants below follow the GGML
// definition at:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
struct Format {
  static constexpr int kEntries = kIq2xxsEntries;
  static constexpr int kGroups = 8;          // one 4-bit local scale per 32 values
  static constexpr int kVectorsPerGroup = 4; // four 8-value codebook vectors per group
  static constexpr bool kParitySigns = true;
  static constexpr int kRecordBytes = 8; // four index bytes then one little-endian uint32
  static constexpr int kCodeOffset = kScaleBytes;
  static constexpr int kPayloadBytes = kCodeOffset + kGroups * kRecordBytes;

  // One 8-byte record per group: four index bytes, then a uint32 holding four 7-bit sign indices
  // in bits 0..27 and the 4-bit local scale in bits 28..31.
  __device__ static void store(uint8_t *payload, const uint16_t *entries, const uint8_t *signs,
                               const uint8_t *locals) {
    const int group = threadIdx.x;
    if (group >= kGroups)
      return;
    uint8_t *record = payload + kCodeOffset + group * kRecordBytes;
    uint32_t aux = static_cast<uint32_t>(locals[group]) << 28;
#pragma unroll
    for (int j = 0; j < kVectorsPerGroup; ++j) {
      record[j] = static_cast<uint8_t>(entries[group * kVectorsPerGroup + j]);
      aux |= (signs[group * kVectorsPerGroup + j] & 0x7Fu) << (7 * j);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
      record[kVectorsPerGroup + j] = static_cast<uint8_t>(aux >> (8 * j));
  }

  // Vector v is entry byte v % 4 of record v / 4, whose uint32 holds the four vectors' 7-bit sign
  // indices and, in its top nibble, the record's local scale.
  __device__ static void decode(const uint8_t *block, int vector, const float *grid,
                                float (&values)[kVectorSize]) {
    const uint8_t *record = block + kCodeOffset + kRecordBytes * (vector / kVectorsPerGroup);
    const uint32_t aux = load_u32(record + kVectorsPerGroup);
    const int slot = vector % kVectorsPerGroup;
    const float d = half_bits_to_float(load_u16(block + kScaleOffset));
    const float scale =
        __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, static_cast<float>(aux >> 28))), 0.25f);
    signed_scaled(grid + record[slot] * kVectorSize, with_parity_bit((aux >> (7 * slot)) & 0x7F),
                  scale, values);
  }
};

static_assert(Format::kPayloadBytes == 66, "IQ2_XXS blocks are 66 bytes");

} // namespace

at::Tensor iq2_xxs_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales,
                             std::optional<at::Tensor> importance) {
  check_scaled_pack_inputs("IQ2_XXS", input, grid, Format::kEntries, scales);
  if (importance.has_value())
    check_importance("IQ2_XXS", input, *importance);
  return iq2_encode_blocks<Format>(input, grid, scales, importance);
}

at::Tensor iq2_xxs_unpack_cuda(at::Tensor packed, at::Tensor grid, at::ScalarType dtype) {
  return decode_blocks<Format>("IQ2_XXS", packed, grid, dtype);
}
