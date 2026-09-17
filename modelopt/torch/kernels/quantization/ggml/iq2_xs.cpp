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

#include <torch/extension.h>

#include <limits>

at::Tensor iq2_xs_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);

at::Tensor iq2_xs_pack(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "IQ2_XS packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ2_XS packing requires a CUDA grid");
  TORCH_CHECK(scales.is_cuda(), "IQ2_XS packing requires CUDA scales");
  const auto input_type = input.scalar_type();
  TORCH_CHECK(input_type == at::kFloat || input_type == at::kDouble || input_type == at::kHalf ||
                  input_type == at::kBFloat16,
              "IQ2_XS packing supports float32, float64, float16, and bfloat16 inputs");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  TORCH_CHECK(input.dim() > 0 && input.size(-1) % 256 == 0,
              "input's innermost dimension must be a multiple of 256 so blocks do not straddle "
              "rows");
  TORCH_CHECK(grid.scalar_type() == at::kFloat && grid.dim() == 2 && grid.size(0) == 512 &&
                  grid.size(1) == 8,
              "grid must be float32 [512, 8]");
  const auto num_blocks = input.numel() / 256;
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == num_blocks,
              "scales must be float16 [numel / 256]");
  TORCH_CHECK(input.get_device() == grid.get_device() && input.get_device() == scales.get_device(),
              "input, grid, and scales must share a device");
  TORCH_CHECK(num_blocks <= std::numeric_limits<int>::max(), "IQ2_XS CUDA grid is too large");
  return iq2_xs_pack_cuda(input.contiguous(), grid.contiguous(), scales.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack", &iq2_xs_pack,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [512, 8], and scales must "
             "be float16 [numel / 256]. Returns uint8 [numel / 256, 74] on the input device. "
             "Non-finite input elements are treated as zero during packing.");
}
