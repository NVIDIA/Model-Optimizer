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

// The IQ1_M packed payload layout and format constants below follow the GGML
// definition at:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
struct Format {
  static constexpr int kEntries = kIq1sEntries; // IQ1_M shares the IQ1_S ternary grid
  static constexpr int kGroups = 16;            // one 3-bit local scale per 16 values
  static constexpr int kVectorsPerGroup = 2;
  static constexpr int kChoices = 8; // the local scale; each vector picks its own shift
  static constexpr bool kSharedShift = false;
  static constexpr int kSubBlocks = 8;
  static constexpr int kScaleWords = 4;
  static constexpr int kLowOffset = 0;                              // no leading block scale
  static constexpr int kHighOffset = kLowOffset + kVectorsPerBlock; // two vectors per byte
  static constexpr int kScaleWordOffset = kHighOffset + 2 * kSubBlocks;
  static constexpr int kPayloadBytes = kScaleWordOffset + 2 * kScaleWords;

  // IQ1_M has no leading scale field: a zero block is all zero bytes, which decodes to a zero
  // scale and therefore zero values.
  __device__ static bool begin(uint8_t *payload, uint16_t d_bits) {
    if ((d_bits & 0x7FFF) != 0)
      return true;
    if (threadIdx.x < kPayloadBytes)
      payload[threadIdx.x] = 0;
    return false;
  }

  // A low index byte per vector; a nibble per vector, low nibble first, holding three index-high
  // bits and the shift; then four scale words, each with two sub-blocks' 3-bit local scales in
  // bits 0..11 and one nibble of the FP16 block scale in bits 12..15.
  __device__ static void store(uint8_t *payload, const uint16_t *picks, const uint8_t *choices,
                               uint16_t d_bits) {
    const int tid = threadIdx.x;
    if (tid < kVectorsPerBlock)
      payload[kLowOffset + tid] = static_cast<uint8_t>(picks[tid]);
    if (tid < kVectorsPerBlock / 2) {
      uint32_t byte = 0;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint32_t pick = picks[2 * tid + half];
        byte |= (((pick >> 8) & 0x7u) | (((pick >> kIq1EntryBits) & 0x1u) << 3)) << (4 * half);
      }
      payload[kHighOffset + tid] = static_cast<uint8_t>(byte);
    }
    if (tid < kScaleWords) {
      uint32_t word = 0;
#pragma unroll
      for (int parity = 0; parity < 2; ++parity) {
        const int sub = 2 * tid + parity;
        const int base = 6 * parity;
        word |= static_cast<uint32_t>(choices[2 * sub]) << base;
        word |= static_cast<uint32_t>(choices[2 * sub + 1]) << (base + 3);
      }
      word |= static_cast<uint32_t>((d_bits >> (4 * tid)) & 0xF) << 12;
      payload[kScaleWordOffset + 2 * tid] = static_cast<uint8_t>(word);
      payload[kScaleWordOffset + 2 * tid + 1] = static_cast<uint8_t>(word >> 8);
    }
  }

  // Vector v is low byte v plus nibble v % 2 of qh byte v / 2: three high index bits and the delta
  // sign. Its 3-bit local scale is slot 2 * (sub-block % 2) + half of scale word sub-block / 2, and
  // d is reassembled from the four words' top nibbles.
  __device__ static void decode(const uint8_t *block, int vector, const float *grid,
                                float (&values)[kVectorSize]) {
    uint32_t words[kScaleWords];
#pragma unroll
    for (int word = 0; word < kScaleWords; ++word)
      words[word] = load_u16(block + kScaleWordOffset + 2 * word);
    const uint32_t d_bits = (words[0] >> 12) | ((words[1] >> 8) & 0x00F0) |
                            ((words[2] >> 4) & 0x0F00) | (words[3] & 0xF000);
    const uint32_t nibble = (block[kHighOffset + vector / 2] >> (4 * (vector % 2))) & 0xF;
    const uint32_t entry = block[kLowOffset + vector] | ((nibble & 0x7) << 8);
    const int sub = vector / 4;
    const uint32_t local = (words[sub / 2] >> (3 * (2 * (sub % 2) + (vector % 4) / 2))) & 0x7;
    const float scale = __fmul_rn(half_bits_to_float(d_bits), static_cast<float>(2 * local + 1));
    shifted_scaled(grid + entry * kVectorSize, (nibble & 0x8) ? -kIq1Delta : kIq1Delta, scale,
                   values);
  }
};

static_assert(Format::kPayloadBytes == 56, "IQ1_M blocks are 56 bytes");

} // namespace

at::Tensor iq1_m_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  check_scaled_pack_inputs("IQ1_M", input, grid, Format::kEntries, scales);
  return iq1_encode_blocks<Format>(input, grid, scales);
}

at::Tensor iq1_m_unpack_cuda(at::Tensor packed, at::Tensor grid, at::ScalarType dtype) {
  return decode_blocks<Format>("IQ1_M", packed, grid, dtype);
}
