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

// The Q8_0 packed payload follows the GGML definition at:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
constexpr int kQ8BlockSize = 32;
constexpr int kPayloadBytes = kScaleBytes + kQ8BlockSize;
constexpr float kMaxQuant = 127.0f;

template <typename scalar_t>
__global__ void encode(const scalar_t *input, int64_t num_blocks, uint8_t *output) {
  const int lane = threadIdx.x;
  const int64_t block = blockIdx.x;
  if (block >= num_blocks)
    return;

  const float value = load_float(input + block * kQ8BlockSize + lane);
  float amax = fabsf(value);
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, delta));

  float d = 0.0f;
  uint8_t *payload = output + block * kPayloadBytes;
  if (lane == 0) {
    d = fminf(amax / kMaxQuant, 65504.0f);
    const uint16_t d_bits = __half_as_ushort(__float2half_rn(d));
    payload[0] = static_cast<uint8_t>(d_bits);
    payload[1] = static_cast<uint8_t>(d_bits >> 8);
  }
  d = __shfl_sync(0xffffffff, d, 0);
  const float rounded = d > 0.0f ? roundf(value / d) : 0.0f;
  const int quant = static_cast<int>(fminf(fmaxf(rounded, -kMaxQuant), kMaxQuant));
  payload[kScaleBytes + lane] = static_cast<uint8_t>(static_cast<int8_t>(quant));
}

} // namespace

at::Tensor q8_0_pack_cuda(at::Tensor input) {
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  modelopt::ggml::check_scalar_pack_input("Q8_0", input, kQ8BlockSize);
  const int64_t num_blocks = input.numel() / kQ8BlockSize;
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty({num_blocks, kPayloadBytes}, input.options().dtype(at::kByte));
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "q8_0_pack", [&] {
        encode<scalar_t><<<static_cast<int>(num_blocks), kQ8BlockSize, 0, stream>>>(
            input.data_ptr<scalar_t>(), num_blocks, output.data_ptr<uint8_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}
