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

at::Tensor iq2_xs_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);

at::Tensor iq2_xs_pack(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "IQ2_XS packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ2_XS packing requires a CUDA grid");
  TORCH_CHECK(scales.is_cuda(), "IQ2_XS packing requires CUDA scales");
  modelopt::ggml::check_pack_inputs("IQ2_XS", input, grid, modelopt::ggml::kIq2xsEntries);
  const auto num_blocks = input.numel() / modelopt::ggml::kBlockSize;
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == num_blocks,
              "scales must be float16 [numel / 256]");
  // The kernel copies these bits straight into the GGML block scale field, so a non-finite entry
  // would produce a payload that decodes to garbage. The synchronization this costs is paid once
  // per packed tensor, on an export path.
  TORCH_CHECK(scales.isfinite().all().item<bool>(), "scales must be finite");
  TORCH_CHECK(input.get_device() == scales.get_device(), "input and scales must share a device");
  return iq2_xs_pack_cuda(input.contiguous(), grid.contiguous(), scales.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack", &iq2_xs_pack,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [512, 8] holding "
             "non-negative codebook magnitudes, and scales must be finite float16 [numel / 256]. "
             "Returns uint8 [numel / 256, 74] on the input device. Non-finite input elements are "
             "treated as zero during packing, and finite elements outside the float32 range "
             "saturate.");
}
