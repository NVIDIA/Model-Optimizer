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

// The Hessian-weighted choice search of modelopt.torch.quantization.ggml.gptq.search_choices, one
// CUDA block per row of a 256-column GGML block: given every choice's decoding error of the row,
// each group first takes the choice with the least error under its diagonal block of H, then the
// groups are revisited in turn, each taking the choice that most lowers e H e^T with the others
// held. The PyTorch reference issues a few small kernels per group and sweep; this runs it all in
// one launch. H must be symmetric: it is read by column so the threads of a warp read it coalesced.

#include "common.cuh"

namespace {

using namespace modelopt::ggml;

constexpr int kMaxChoices = 16;
constexpr int kMaxGroups = 16;

// Sum of value over each aligned run of width lanes (a power of two up to 32), on every lane.
__device__ __forceinline__ float segment_sum(float value, int width) {
  for (int offset = width / 2; offset > 0; offset /= 2)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  return value;
}

__global__ void search(const float *errors, const float *hessian, int64_t rows, int choices,
                       int groups, int sweeps, int64_t *picks) {
  __shared__ float error[kMaxChoices][kBlockSize];
  __shared__ float current[kBlockSize];
  __shared__ float gradient[kBlockSize];
  __shared__ float score[kMaxChoices][kMaxGroups];
  __shared__ int pick[kMaxGroups];

  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const int group_size = kBlockSize / groups;
  const int lane_in_group = tid % group_size;

  for (int c = 0; c < choices; ++c)
    error[c][tid] = errors[(c * rows + row) * kBlockSize + tid];
  __syncthreads();

  // Each group's error under each choice: sum over i, j in the group of e_i H_ij e_j. Thread tid
  // owns column tid, so a choice's groups are aligned runs of group_size threads.
  const int own_group = tid / group_size;
  const int group_start = own_group * group_size;
  for (int c = 0; c < choices; ++c) {
    float projected = 0.0f;
    for (int j = 0; j < group_size; ++j)
      projected =
          fmaf(hessian[(group_start + j) * kBlockSize + tid], error[c][group_start + j], projected);
    const float total = segment_sum(error[c][tid] * projected, group_size);
    if (lane_in_group == 0)
      score[c][own_group] = total;
  }
  __syncthreads();
  if (tid < groups) {
    int best = 0;
    for (int c = 1; c < choices; ++c)
      best = score[c][tid] < score[best][tid] ? c : best;
    pick[tid] = best;
  }
  __syncthreads();
  current[tid] = error[pick[own_group]][tid];
  __syncthreads();
  float grad = 0.0f;
  for (int i = 0; i < kBlockSize; ++i)
    grad = fmaf(current[i], hessian[i * kBlockSize + tid], grad);
  gradient[tid] = grad;
  __syncthreads();

  // Switching group g from its current errors to choice c's, delta, changes e H e^T by
  // delta . (2 gradient_g + H_gg delta). Work items are (choice, column of g) pairs, group_size
  // consecutive threads per choice, so the same segment sums apply.
  const int pairs = choices * group_size;
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    for (int g = 0; g < groups; ++g) {
      const int start = g * group_size;
      for (int base = 0; base < pairs; base += kThreads) {
        const int item = base + tid;
        const bool active = item < pairs;
        const int c = active ? item / group_size : 0;
        const int i = start + lane_in_group;
        float value = 0.0f;
        if (active) {
          float projected = 0.0f;
          for (int j = 0; j < group_size; ++j)
            projected = fmaf(hessian[(start + j) * kBlockSize + i],
                             error[c][start + j] - current[start + j], projected);
          value = (error[c][i] - current[i]) * (2.0f * gradient[i] + projected);
        }
        value = segment_sum(value, group_size);
        if (active && lane_in_group == 0)
          score[c][g] = value;
      }
      __syncthreads();
      if (tid == 0) {
        int best = 0;
        for (int c = 1; c < choices; ++c)
          best = score[c][g] < score[best][g] ? c : best;
        pick[g] = best;
      }
      __syncthreads();
      float update = 0.0f;
      for (int j = 0; j < group_size; ++j)
        update = fmaf(error[pick[g]][start + j] - current[start + j],
                      hessian[(start + j) * kBlockSize + tid], update);
      __syncthreads();
      gradient[tid] += update;
      if (tid >= start && tid < start + group_size)
        current[tid] = error[pick[g]][tid];
      __syncthreads();
    }
  }
  if (tid < groups)
    picks[row * groups + tid] = pick[tid];
}

} // namespace

at::Tensor search_choices_cuda(at::Tensor errors, at::Tensor hessian, int64_t groups,
                               int64_t sweeps) {
  TORCH_CHECK(errors.is_cuda() && errors.scalar_type() == at::kFloat && errors.dim() == 3 &&
                  errors.size(2) == kBlockSize,
              "errors must be a float32 CUDA [choices, rows, 256] tensor");
  TORCH_CHECK(errors.size(0) >= 1 && errors.size(0) <= kMaxChoices, "at most ", kMaxChoices,
              " choices are supported");
  TORCH_CHECK(hessian.is_cuda() && hessian.get_device() == errors.get_device() &&
                  hessian.scalar_type() == at::kFloat && hessian.dim() == 2 &&
                  hessian.size(0) == kBlockSize && hessian.size(1) == kBlockSize,
              "hessian must be a float32 [256, 256] tensor on the errors' device");
  TORCH_CHECK(groups >= kBlockSize / 32 && groups <= kMaxGroups && kBlockSize % groups == 0 &&
                  ((kBlockSize / groups) & (kBlockSize / groups - 1)) == 0,
              "groups must split 256 columns into power-of-two runs of at most 32");
  TORCH_CHECK(sweeps >= 0, "sweeps must be non-negative");
  const auto error_values = errors.contiguous();
  const auto hessian_values = hessian.contiguous();
  c10::cuda::CUDAGuard guard(error_values.device());
  const int64_t rows = error_values.size(1);
  at::Tensor picks = at::empty({rows, groups}, error_values.options().dtype(at::kLong));
  if (rows == 0)
    return picks;
  search<<<static_cast<int>(rows), kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      error_values.data_ptr<float>(), hessian_values.data_ptr<float>(), rows,
      static_cast<int>(error_values.size(0)), static_cast<int>(groups), static_cast<int>(sweeps),
      picks.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return picks;
}
