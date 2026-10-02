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

// The IQ2_S packed payload layout and format constants below follow the GGML
// definition at:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
struct Format {
  static constexpr int kEntries = kIq2sEntries;
  static constexpr int kGroups = 16; // one 4-bit local scale per 16 values
  static constexpr int kVectorsPerGroup = 2;
  static constexpr bool kParitySigns = false; // all eight sign bits are stored
  static constexpr int kSubBlocks = 8;
  static constexpr int kLowOffset = kScaleBytes;                     // 32 low index bytes
  static constexpr int kSignOffset = kLowOffset + kVectorsPerBlock;  // 32 sign masks
  static constexpr int kHighOffset = kSignOffset + kVectorsPerBlock; // two high bits per vector
  static constexpr int kLocalScaleOffset = kHighOffset + kSubBlocks;
  static constexpr int kPayloadBytes = kLocalScaleOffset + kGroups / 2;

  // Low index byte and sign mask per vector, the two high index bits packed four vectors to a
  // byte, then the local scales as nibbles, two per byte.
  __device__ static void store(uint8_t *payload, const uint16_t *entries, const uint8_t *signs,
                               const uint8_t *locals) {
    const int tid = threadIdx.x;
    if (tid < kVectorsPerBlock) {
      payload[kLowOffset + tid] = static_cast<uint8_t>(entries[tid]);
      payload[kSignOffset + tid] = signs[tid];
    }
    if (tid < kSubBlocks) {
      uint32_t high = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k)
        high |= ((entries[4 * tid + k] >> 8) & 0x3u) << (2 * k);
      payload[kHighOffset + tid] = static_cast<uint8_t>(high);
    }
    if (tid < kGroups / 2)
      payload[kLocalScaleOffset + tid] = locals[2 * tid] | (locals[2 * tid + 1] << 4);
  }
};

static_assert(Format::kPayloadBytes == 82, "IQ2_S blocks are 82 bytes");

} // namespace

at::Tensor iq2_s_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  check_scaled_pack_inputs("IQ2_S", input, grid, Format::kEntries, scales);
  return iq2_encode_blocks<Format>(input, grid, scales);
}
