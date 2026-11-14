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

// The derived super-block scale of modelopt.torch.quantization.ggml.common: one CUDA block per
// 256-value GGML block and one thread per value. Each sub-block's lanes sort its values with their
// weights, prefix-sum them, and score every split into three runs between them; the block keeps its
// largest sub-block fit over the top local multiplier. Every float operation is explicitly rounded
// and taken in the PyTorch reference's order, so the two agree bit for bit.

#include "common.cuh"

#include <vector>

namespace {

using namespace modelopt::ggml;

constexpr int kMaxTriples = 2;
constexpr unsigned kFullMask = 0xffffffffu;

struct Levels {
  float value[kMaxTriples][3];
  int triples;
};

// Largest value over all 256 threads, on every thread.
__device__ __forceinline__ float block_max(float value, float *warp_maxes) {
  for (int offset = 16; offset > 0; offset /= 2)
    value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
  if ((threadIdx.x & 31) == 0)
    warp_maxes[threadIdx.x >> 5] = value;
  __syncthreads();
  for (int w = 0; w < kWarps; ++w)
    value = fmaxf(value, warp_maxes[w]);
  return value;
}

// The k of the power of two 2^k that brings largest into [0.5, 1), clamped so that 2^k and 2^-k
// stay normal, and 2^k built from its bits, as the reference's _unit_powers_of_two builds them.
__device__ __forceinline__ int unit_exponent(float largest) {
  int exponent;
  frexpf(largest, &exponent);
  return min(max(-exponent, -126), 126);
}

__device__ __forceinline__ float power_of_two(int k) { return __int_as_float((k + 127) << 23); }

// Sum over all 256 threads, adjacent pairs first, the order of the reference's pairwise sum.
__device__ __forceinline__ float pairwise_block_sum(float value, float *warp_sums) {
  for (int offset = 1; offset < 32; offset *= 2)
    value = __fadd_rn(value, __shfl_xor_sync(kFullMask, value, offset));
  if ((threadIdx.x & 31) == 0)
    warp_sums[threadIdx.x >> 5] = value;
  __syncthreads();
  float sums[kWarps];
#pragma unroll
  for (int w = 0; w < kWarps; ++w)
    sums[w] = warp_sums[w];
#pragma unroll
  for (int width = 1; width < kWarps; width *= 2)
#pragma unroll
    for (int w = 0; w < kWarps; w += 2 * width)
      sums[w] = __fadd_rn(sums[w], sums[w + width]);
  return sums[0];
}

// The (first, second) bounds of split index p, enumerated row-major over first <= second, as
// torch.triu_indices does; offset is second - first.
template <int kGroup> __device__ __forceinline__ void split_bounds(int p, int &first, int &second) {
  first = 0;
  int row_length = kGroup + 1;
  while (p >= row_length) {
    p -= row_length;
    ++first;
    --row_length;
  }
  second = first + p;
}

// Least-squares terms of the split: sum q.x and sum q.q under one level triple, from the runs'
// weight and weighted-value sums, summed in the reference's order.
__device__ __forceinline__ void split_terms(const float (&level)[3], const float (&w)[3],
                                            const float (&wx)[3], float &qx, float &qq) {
  qx = __fmul_rn(level[0], wx[0]);
  qq = __fmul_rn(__fmul_rn(level[0], level[0]), w[0]);
#pragma unroll
  for (int r = 1; r < 3; ++r) {
    qx = __fadd_rn(qx, __fmul_rn(level[r], wx[r]));
    qq = __fadd_rn(qq, __fmul_rn(__fmul_rn(level[r], level[r]), w[r]));
  }
}

__device__ __forceinline__ void run_sums(const float *cum_w, const float *cum_wx, int first,
                                         int second, int group, float (&w)[3], float (&wx)[3]) {
  const int bounds[4] = {0, first, second, group};
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    w[r] = __fsub_rn(cum_w[bounds[r + 1]], cum_w[bounds[r]]);
    wx[r] = __fsub_rn(cum_wx[bounds[r + 1]], cum_wx[bounds[r]]);
  }
}

template <typename scalar_t, int kGroup>
__global__ void __launch_bounds__(kThreads)
    fit_blocks(const scalar_t *input, const float *importance, int64_t importance_rows,
               Levels levels, bool magnitudes, float inverse_top, __half *scales) {
  constexpr int kSubBlocks = kBlockSize / kGroup;
  constexpr int kSplits = (kGroup + 1) * (kGroup + 2) / 2;
  __shared__ float warp_sums[kWarps];
  __shared__ float warp_maxes[kWarps];
  __shared__ float weight_maxes[kWarps];
  __shared__ float cum_w[kSubBlocks][kGroup + 1];
  __shared__ float cum_wx[kSubBlocks][kGroup + 1];
  __shared__ float fits[kSubBlocks];

  const int tid = threadIdx.x;
  const int64_t block = blockIdx.x;
  const int sub = tid / kGroup;
  const int lane = tid % kGroup;

  // llama.cpp's element weight sqrt(mean(x^2) + x^2), times the root of the column's importance.
  // The block scaled by 2^k bringing its largest magnitude into [0.5, 1), k clamped so 2^k and
  // 2^-k stay normal: exact, and the squares of values near the float32 maximum stay finite.
  const float raw = load_float(input + block * kBlockSize + tid);
  const int k = unit_exponent(block_max(fabsf(raw), warp_maxes));
  const float from_unit = power_of_two(-k);
  const float x = __fmul_rn(raw, power_of_two(k));
  const float square = __fmul_rn(x, x);
  const float mean =
      __fdiv_rn(pairwise_block_sum(square, warp_sums), static_cast<float>(kBlockSize));
  float weight = __fsqrt_rn(__fadd_rn(mean, square));
  if (importance != nullptr) {
    weight =
        __fmul_rn(weight, __fsqrt_rn(importance[(block % importance_rows) * kBlockSize + tid]));
    // Scaled likewise, so a statistic near the float32 maximum cannot overflow the scores: the fit
    // is the same whatever factor the block's weights share.
    weight = __fmul_rn(weight, power_of_two(unit_exponent(block_max(weight, weight_maxes))));
  }
  // Adding zero turns -0 into +0, so equal values compare equal and keep their order.
  float value = __fadd_rn(magnitudes ? fabsf(x) : x, 0.0f);

  // Bitonic sort of the sub-block's lanes by (value, original lane): a stable ascending sort.
  int index = lane;
#pragma unroll
  for (int size = 2; size <= kGroup; size *= 2) {
#pragma unroll
    for (int stride = size / 2; stride > 0; stride /= 2) {
      const float other_value = __shfl_xor_sync(kFullMask, value, stride);
      const float other_weight = __shfl_xor_sync(kFullMask, weight, stride);
      const int other_index = __shfl_xor_sync(kFullMask, index, stride);
      const bool other_first = other_value < value || (other_value == value && other_index < index);
      const bool keep_smaller = ((lane & stride) == 0) == ((lane & size) == 0);
      if (keep_smaller == other_first) {
        value = other_value;
        weight = other_weight;
        index = other_index;
      }
    }
  }

  // Inclusive log-step scans of w and w x along the sorted sub-block.
  float sum_w = weight;
  float sum_wx = __fmul_rn(weight, value);
#pragma unroll
  for (int shift = 1; shift < kGroup; shift *= 2) {
    const float up_w = __shfl_up_sync(kFullMask, sum_w, shift, kGroup);
    const float up_wx = __shfl_up_sync(kFullMask, sum_wx, shift, kGroup);
    if (lane >= shift) {
      sum_w = __fadd_rn(sum_w, up_w);
      sum_wx = __fadd_rn(sum_wx, up_wx);
    }
  }
  cum_w[sub][lane + 1] = sum_w;
  cum_wx[sub][lane + 1] = sum_wx;
  if (lane == 0) {
    cum_w[sub][0] = 0.0f;
    cum_wx[sub][0] = 0.0f;
  }
  __syncthreads();

  // Each lane scores every kGroup-th split under every level triple. The key orders by score, then
  // by the lowest triple and split, the reference's tie rule.
  unsigned long long best = 0;
  // (first, first + offset) walks the splits row by row; it is normalized only for a split that
  // exists, since past the last row the row length would reach zero.
  int first = 0;
  int offset = lane - kGroup;
  int row_length = kGroup + 1;
  for (int p = lane; p < kSplits; p += kGroup) {
    offset += kGroup;
    while (offset >= row_length) {
      offset -= row_length;
      ++first;
      --row_length;
    }
    float w[3], wx[3];
    run_sums(cum_w[sub], cum_wx[sub], first, first + offset, kGroup, w, wx);
    for (int t = 0; t < levels.triples; ++t) {
      float qx, qq;
      split_terms(levels.value[t], w, wx, qx, qq);
      const float score =
          qq > 0.0f && qx > 0.0f ? __fdiv_rn(__fmul_rn(qx, qx), fmaxf(qq, 1e-30f)) : 0.0f;
      const auto key = (static_cast<unsigned long long>(__float_as_uint(score)) << 32) |
                       static_cast<unsigned long long>(0xffffffffu - (t * kSplits + p));
      best = key > best ? key : best;
    }
  }
#pragma unroll
  for (int stride = kGroup / 2; stride > 0; stride /= 2) {
    const unsigned long long other = __shfl_xor_sync(kFullMask, best, stride);
    best = other > best ? other : best;
  }
  if (lane == 0) {
    const float score = __uint_as_float(static_cast<unsigned>(best >> 32));
    float fit = 0.0f;
    if (score > 0.0f) {
      const int choice = static_cast<int>(0xffffffffu - static_cast<unsigned>(best));
      int split_first, split_second;
      split_bounds<kGroup>(choice % kSplits, split_first, split_second);
      float w[3], wx[3];
      run_sums(cum_w[sub], cum_wx[sub], split_first, split_second, kGroup, w, wx);
      float qx, qq;
      split_terms(levels.value[choice / kSplits], w, wx, qx, qq);
      fit = __fdiv_rn(qx, qq);
    }
    fits[sub] = fit;
  }
  __syncthreads();
  if (tid == 0) {
    float largest = fits[0];
#pragma unroll
    for (int s = 1; s < kSubBlocks; ++s)
      largest = fmaxf(largest, fits[s]);
    scales[block] =
        __float2half_rn(fminf(__fmul_rn(__fmul_rn(largest, inverse_top), from_unit), 65504.0f));
  }
}

} // namespace

at::Tensor fit_scales_cuda(at::Tensor input, std::optional<at::Tensor> importance, int64_t group,
                           int64_t top, std::vector<double> levels, bool magnitudes) {
  TORCH_CHECK(input.is_cuda(), "the scale fit requires a CUDA input");
  check_scalar_pack_input("The scale fit", input, kBlockSize);
  if (importance.has_value())
    check_importance("The scale fit", input, *importance);
  TORCH_CHECK(group == 16 || group == 32, "sub-blocks must hold 16 or 32 values");
  TORCH_CHECK(top > 0, "the top local multiplier must be positive");
  TORCH_CHECK(!levels.empty() && levels.size() % 3 == 0 && levels.size() / 3 <= kMaxTriples,
              "levels must hold one to ", kMaxTriples, " ascending triples");
  Levels table{};
  table.triples = static_cast<int>(levels.size() / 3);
  for (int t = 0; t < table.triples; ++t)
    for (int r = 0; r < 3; ++r) {
      table.value[t][r] = static_cast<float>(levels[3 * t + r]);
      TORCH_CHECK(r == 0 || table.value[t][r] > table.value[t][r - 1],
                  "each level triple must be ascending");
    }
  const auto values = input.contiguous();
  const auto rows = importance.has_value() ? importance->contiguous() : at::Tensor();
  c10::cuda::CUDAGuard guard(values.device());
  const int64_t blocks = values.numel() / kBlockSize;
  at::Tensor scales = at::empty({blocks}, values.options().dtype(at::kHalf));
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const float *importance_ptr = rows.defined() ? rows.data_ptr<float>() : nullptr;
  const int64_t importance_rows = rows.defined() ? rows.size(0) : 1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, values.scalar_type(), "ggml_fit_scales", [&] {
        auto launch = [&](auto kernel) {
          kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
              values.data_ptr<scalar_t>(), importance_ptr, importance_rows, table, magnitudes,
              static_cast<float>(1.0 / static_cast<double>(top)),
              reinterpret_cast<__half *>(scales.data_ptr<at::Half>()));
        };
        if (group == 16)
          launch(fit_blocks<scalar_t, 16>);
        else
          launch(fit_blocks<scalar_t, 32>);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return scales;
}
