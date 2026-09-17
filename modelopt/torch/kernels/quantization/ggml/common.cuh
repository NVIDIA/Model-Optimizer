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

#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include <cfloat>
#endif

namespace modelopt::ggml {

// Block geometry shared by every IQ format: 256 values are encoded as 8-element codebook vectors
// behind one fp16 block scale that occupies the first two payload bytes. These follow GGML's
// QK_K, its uint64 grid entry width, and the leading ggml_half of each block struct:
// https://github.com/ggml-org/llama.cpp/blob/9b05354ec6fb58b4e665e9a39ebc40285c015638/ggml/src/ggml-common.h
constexpr int kBlockSize = 256;
constexpr int kVectorSize = 8;
constexpr int kScaleOffset = 0;
constexpr int kScaleBytes = 2;

// Codebook sizes, from GGML's NGRID_IQ1S and the length of its iq2xs_grid table (see the link
// above). Defined here so the pybind wrappers that validate them and the kernels that index with
// them cannot drift apart.
constexpr int kIq1sEntries = 2048;
constexpr int kIq2xsEntries = 512;

// One CUDA block encodes one GGML block. The reductions below fold over exactly this many warps,
// and each kernel static_asserts that its codebook divides evenly among the threads.
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// Validates the packing contract every IQ format shares. Called from the pybind wrapper on the
// caller's tensors and again from the CUDA entry point on the materialized contiguous tensors, so
// the enforced rule and the message it reports are written once.
inline void check_pack_inputs(const char *format, const at::Tensor &input, const at::Tensor &grid,
                              int64_t entries) {
  const auto input_type = input.scalar_type();
  TORCH_CHECK(input_type == at::kFloat || input_type == at::kDouble || input_type == at::kHalf ||
                  input_type == at::kBFloat16,
              format, " packing supports float32, float64, float16, and bfloat16 inputs");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  TORCH_CHECK(input.dim() > 0 && input.size(-1) % kBlockSize == 0,
              "input's innermost dimension must be a multiple of ", kBlockSize,
              " so blocks do not straddle rows");
  TORCH_CHECK(grid.scalar_type() == at::kFloat && grid.dim() == 2 && grid.size(0) == entries &&
                  grid.size(1) == kVectorSize,
              "grid must be float32 [", entries, ", ", kVectorSize, "]");
  TORCH_CHECK(input.get_device() == grid.get_device(), "input and grid must share a device");
  TORCH_CHECK(input.numel() / kBlockSize <= std::numeric_limits<int>::max(), format,
              " CUDA grid is too large");
}

#ifdef __CUDACC__

// Reads one input element as float32. Non-finite elements are treated as zero, and finiteness is
// tested at the source precision so that a finite float64 such as 1e100 saturates at the float32
// maximum instead of overflowing to infinity and being dropped to zero.
template <typename scalar_t> __device__ __forceinline__ float load_float(const scalar_t *input) {
  if constexpr (sizeof(scalar_t) > sizeof(float)) {
    constexpr double kFloatMax = static_cast<double>(FLT_MAX);
    const double value = static_cast<double>(*input);
    if (!isfinite(value))
      return 0.0f;
    return static_cast<float>(fmin(fmax(value, -kFloatMax), kFloatMax));
  } else {
    const float value = static_cast<float>(*input);
    return isfinite(value) ? value : 0.0f;
  }
}

// Squared error of approximating x by scale * q, given |x|^2, x . q and |q|^2. The clamp keeps the
// result non-negative so that its bit pattern orders the same way the value does inside error_key.
__device__ __forceinline__ float clamped_quant_error(float xnorm, float dot, float qnorm,
                                                     float scale) {
  return fmaxf(fmaf(scale * scale, qnorm, fmaf(-2.0f * scale, dot, xnorm)), 0.0f);
}

// Orders candidates by error first and codebook index second, so the lowest index wins a tie --
// the rule the PyTorch reference encoder applies.
__device__ __forceinline__ unsigned long long error_key(float error, int entry) {
  return (static_cast<unsigned long long>(__float_as_uint(error)) << 32) |
         static_cast<unsigned long long>(entry);
}

// Adds the block-wide minimum of local[slot] to accum[slot] for every slot. scratch must hold
// kWarps * kSlots floats and accum kSlots floats. Barriers are internal, so every thread of the
// block must call this.
template <int kSlots>
__device__ __forceinline__ void block_min_accumulate(const float (&local)[kSlots], float *scratch,
                                                     float *accum) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
#pragma unroll
  for (int slot = 0; slot < kSlots; ++slot) {
    float value = local[slot];
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
      value = fminf(value, __shfl_down_sync(0xffffffff, value, delta));
    if (lane == 0)
      scratch[warp * kSlots + slot] = value;
  }
  __syncthreads();
  if (tid < kSlots) {
    float value = scratch[tid];
#pragma unroll
    for (int w = 1; w < kWarps; ++w)
      value = fminf(value, scratch[w * kSlots + tid]);
    accum[tid] += value;
  }
  __syncthreads();
}

// Block-wide minimum of key, valid on thread 0 only. scratch must hold kWarps entries. Barriers
// are internal -- including a trailing one, so scratch is free to reuse on return, matching
// block_min_accumulate above -- and every thread of the block must call this.
__device__ __forceinline__ unsigned long long block_min_key(unsigned long long key,
                                                            unsigned long long *scratch) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    const unsigned long long other = __shfl_down_sync(0xffffffff, key, delta);
    key = other < key ? other : key;
  }
  if (lane == 0)
    scratch[warp] = key;
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int w = 1; w < kWarps; ++w)
      key = scratch[w] < key ? scratch[w] : key;
  }
  __syncthreads();
  return key;
}

// Writes the fp16 block scale into the payload, or zeroes the whole payload when the block scale
// rounded to zero. Negative zero counts: it reconstructs every element as zero, so it takes the
// same branch instead of running a search whose candidates all score identically. Returns false
// once the payload is final and the caller should stop. The branch is uniform across the block, so
// returning on false is barrier-safe.
template <int kPayloadBytes>
__device__ __forceinline__ bool store_block_scale(uint8_t *payload, uint16_t d_bits) {
  if ((d_bits & 0x7FFF) == 0) {
    if (threadIdx.x < kPayloadBytes)
      payload[threadIdx.x] = 0;
    return false;
  }
  if (threadIdx.x == 0) {
    payload[kScaleOffset] = static_cast<uint8_t>(d_bits);
    payload[kScaleOffset + 1] = static_cast<uint8_t>(d_bits >> 8);
  }
  return true;
}

#endif // __CUDACC__

} // namespace modelopt::ggml
