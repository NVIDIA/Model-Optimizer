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

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <limits>

at::Tensor iq1_s_pack_cuda(at::Tensor input, at::Tensor grid);

at::Tensor iq1_s_pack(at::Tensor input, at::Tensor grid) {
  TORCH_CHECK(input.is_cuda(), "IQ1_S packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ1_S packing requires a CUDA grid");
  const auto input_type = input.scalar_type();
  TORCH_CHECK(input_type == at::kFloat || input_type == at::kDouble || input_type == at::kHalf ||
                  input_type == at::kBFloat16,
              "IQ1_S packing supports float32, float64, float16, and bfloat16 inputs");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  TORCH_CHECK(input.dim() > 0 && input.size(-1) % 256 == 0,
              "input's innermost dimension must be a multiple of 256 so blocks do not straddle "
              "rows");
  TORCH_CHECK(grid.scalar_type() == at::kFloat && grid.dim() == 2 && grid.size(0) == 2048 &&
                  grid.size(1) == 8,
              "grid must be float32 [2048, 8]");
  TORCH_CHECK(input.get_device() == grid.get_device(), "input and grid must share a device");
  TORCH_CHECK(input.numel() / 256 <= std::numeric_limits<int>::max(),
              "IQ1_S CUDA grid is too large");
  return iq1_s_pack_cuda(input.contiguous(), grid.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack", &iq1_s_pack,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [2048, 8]. Returns uint8 "
             "[numel / 256, 50] on the input device. Non-finite input elements are treated as "
             "zero during packing.");
}
