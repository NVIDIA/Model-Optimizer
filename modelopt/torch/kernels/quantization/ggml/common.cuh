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
constexpr int kVectorsPerBlock = kBlockSize / kVectorSize;
constexpr int kScaleOffset = 0;
constexpr int kScaleBytes = 2;

// Codebook sizes, from GGML's NGRID_IQ1S and the length of its iq2xs_grid table (see the link
// above). Defined here so the pybind wrappers that validate them and the kernels that index with
// them cannot drift apart.
constexpr int kIq1sEntries = 2048;
constexpr int kIq2xsEntries = 512;
constexpr int kIq2xxsEntries = 256;
constexpr int kIq2sEntries = 1024;

// One CUDA block encodes one GGML block. The reductions below fold over exactly this many warps,
// and each kernel static_asserts that its codebook divides evenly among the threads.
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// Validates the scalar-input contract shared by the vector-codebook and scalar GGML packers.
inline void check_scalar_pack_input(const char *format, const at::Tensor &input,
                                    int64_t block_size) {
  const auto input_type = input.scalar_type();
  TORCH_CHECK(input_type == at::kFloat || input_type == at::kDouble || input_type == at::kHalf ||
                  input_type == at::kBFloat16,
              format, " packing supports float32, float64, float16, and bfloat16 inputs");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  TORCH_CHECK(input.dim() > 0 && input.size(-1) % block_size == 0,
              "input's innermost dimension must be a multiple of ", block_size,
              " so blocks do not straddle rows");
  TORCH_CHECK(input.numel() / block_size <= std::numeric_limits<int>::max(), format,
              " CUDA grid is too large");
}

// Validates the codebook contract every IQ format shares, once, at each format's CUDA entry point.
// A format that takes block scales passes them so their device is checked with the others', before
// any shape or dtype rule.
inline void check_pack_inputs(const char *format, const at::Tensor &input, const at::Tensor &grid,
                              int64_t entries, const at::Tensor *scales = nullptr) {
  TORCH_CHECK(input.is_cuda(), format, " packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), format, " packing requires a CUDA grid");
  TORCH_CHECK(scales == nullptr || scales->is_cuda(), format, " packing requires CUDA scales");
  check_scalar_pack_input(format, input, kBlockSize);
  TORCH_CHECK(grid.scalar_type() == at::kFloat && grid.dim() == 2 && grid.size(0) == entries &&
                  grid.size(1) == kVectorSize,
              "grid must be float32 [", entries, ", ", kVectorSize, "]");
  TORCH_CHECK(input.get_device() == grid.get_device(), "input and grid must share a device");
}

// Validates a format that takes one precomputed FP16 block scale per 256 values. The kernels copy
// these bits straight into the block scale field: a non-finite entry produces a payload that
// decodes to garbage, and a negative one inverts the sign of every decoded element while still
// packing cleanly -- GGML's own encoders assert a non-negative block scale. One fused reduction, so
// the synchronization is paid once per packed tensor, on an export path.
inline void check_scaled_pack_inputs(const char *format, const at::Tensor &input,
                                     const at::Tensor &grid, int64_t entries,
                                     const at::Tensor &scales) {
  check_pack_inputs(format, input, grid, entries, &scales);
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == input.numel() / kBlockSize,
              "scales must be float16 [numel / 256]");
  TORCH_CHECK((scales.isfinite() & (scales >= 0)).all().item<bool>(),
              "scales must be finite and non-negative");
  TORCH_CHECK(input.get_device() == scales.get_device(), "input and scales must share a device");
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

// The IQ1 kernels approximate each 8-value vector by scale * (q + delta), with q from the ternary
// grid and delta = +/- 1/8. The three helpers below are theirs.

// Loads one 8-value vector with its squared norm and its sum.
template <typename scalar_t>
__device__ __forceinline__ void load_vector(const scalar_t *source, float (&x)[kVectorSize],
                                            float &xnorm, float &xsum) {
  xnorm = 0.0f;
  xsum = 0.0f;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j) {
    x[j] = load_float(source + j);
    xnorm = fmaf(x[j], x[j], xnorm);
    xsum += x[j];
  }
}

// Accumulates x . q, |q|^2 and sum(q) for one codebook vector.
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

// Squared error of approximating x by scale * (q + delta): the offset shifts the dot and norm
// that grid_terms computed for q alone.
__device__ __forceinline__ float shifted_error(float xnorm, float xsum, float dot, float qnorm,
                                               float qsum, float scale, float delta) {
  const float shifted_dot = dot + delta * xsum;
  const float shifted_norm = qnorm + 2.0f * delta * qsum + 8.0f * delta * delta;
  return clamped_quant_error(xnorm, shifted_dot, shifted_norm, scale);
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

// Decoders run one thread per 8-value vector and follow the PyTorch decoders operation for
// operation. Every float operation is explicitly rounded so the compiler cannot fuse a multiply
// into an add, which keeps them bit-identical to that reference. A format's layout type supplies
// kPayloadBytes, kEntries and decode(block, vector, grid, values), and binds decode_blocks.

__device__ __forceinline__ uint32_t load_u16(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8);
}

__device__ __forceinline__ uint32_t load_u32(const uint8_t *bytes) {
  return load_u16(bytes) | (load_u16(bytes + 2) << 16);
}

__device__ __forceinline__ float half_bits_to_float(uint32_t bits) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// IQ2_XS and IQ2_XXS store seven sign bits; the eighth makes the count of negatives even.
__device__ __forceinline__ uint32_t with_parity_bit(uint32_t sign_index) {
  return sign_index | ((__popc(sign_index) & 1u) << 7);
}

// x * scale, with coordinate j negated where bit j of sign_mask is set.
__device__ __forceinline__ void signed_scaled(const float *q, uint32_t sign_mask, float scale,
                                              float (&values)[kVectorSize]) {
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j)
    values[j] = __fmul_rn((sign_mask >> j) & 1 ? -q[j] : q[j], scale);
}

// (q + delta) * scale, the IQ1 formats' shifted ternary grid.
__device__ __forceinline__ void shifted_scaled(const float *q, float delta, float scale,
                                               float (&values)[kVectorSize]) {
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j)
    values[j] = __fmul_rn(__fadd_rn(q[j], delta), scale);
}

template <typename Format, typename out_t>
__global__ void decode_vectors(const uint8_t *packed, int64_t num_vectors, const float *grid,
                               out_t *output) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_vectors)
    return;
  float values[kVectorSize];
  Format::decode(packed + (index / kVectorsPerBlock) * Format::kPayloadBytes,
                 static_cast<int>(index % kVectorsPerBlock), grid, values);
  out_t *out = output + index * kVectorSize;
#pragma unroll
  for (int j = 0; j < kVectorSize; ++j)
    out[j] = static_cast<out_t>(values[j]);
}

// Decodes uint8 [blocks, kPayloadBytes] into [blocks, 256] of dtype on the payload's device.
template <typename Format>
at::Tensor decode_blocks(const char *format, const at::Tensor &packed, const at::Tensor &grid,
                         at::ScalarType dtype) {
  constexpr int kPayloadBytes = Format::kPayloadBytes;
  constexpr int kEntries = Format::kEntries;
  TORCH_CHECK(packed.is_cuda() && grid.is_cuda(), format, " decoding requires CUDA tensors");
  TORCH_CHECK(packed.get_device() == grid.get_device(), "payload and grid must share a device");
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.dim() == 2 &&
                  packed.size(1) == kPayloadBytes,
              format, " payloads must be uint8 [blocks, ", kPayloadBytes, "]");
  TORCH_CHECK(grid.scalar_type() == at::kFloat && grid.dim() == 2 && grid.size(0) == kEntries &&
                  grid.size(1) == kVectorSize,
              format, " grid must be float32 [", kEntries, ", ", kVectorSize, "]");
  TORCH_CHECK(at::isFloatingType(dtype), format, " decodes to a floating-point dtype");
  c10::cuda::CUDAGuard guard(packed.device());
  const auto payload = packed.contiguous();
  const auto table = grid.contiguous();
  auto output = at::empty({packed.size(0), kBlockSize}, packed.options().dtype(dtype));
  const int64_t num_vectors = packed.size(0) * kVectorsPerBlock;
  if (num_vectors == 0)
    return output;
  const auto launch_blocks = static_cast<unsigned>((num_vectors + kThreads - 1) / kThreads);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "ggml_decode", [&] {
        decode_vectors<Format, scalar_t><<<launch_blocks, kThreads, 0, stream>>>(
            payload.data_ptr<uint8_t>(), num_vectors, table.data_ptr<float>(),
            output.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

#endif // __CUDACC__

} // namespace modelopt::ggml
