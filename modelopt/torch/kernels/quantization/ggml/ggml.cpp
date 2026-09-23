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

// Every GGML IQ format shares common.cuh, the same CUDA version gate, and the same build flags,
// so they compile into one extension and bind here. Each format keeps its kernels in its own
// translation unit and exposes a single host entry point.

#include "common.cuh"

at::Tensor iq1_s_pack_cuda(at::Tensor input, at::Tensor grid);
at::Tensor iq2_xs_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);
at::Tensor iq2_xxs_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);
at::Tensor iq2_s_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);
at::Tensor iq1_m_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);

namespace {

at::Tensor iq1_s_pack(at::Tensor input, at::Tensor grid) {
  TORCH_CHECK(input.is_cuda(), "IQ1_S packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ1_S packing requires a CUDA grid");
  modelopt::ggml::check_pack_inputs("IQ1_S", input, grid, modelopt::ggml::kIq1sEntries);
  return iq1_s_pack_cuda(input.contiguous(), grid.contiguous());
}

at::Tensor iq2_xs_pack(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "IQ2_XS packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ2_XS packing requires a CUDA grid");
  TORCH_CHECK(scales.is_cuda(), "IQ2_XS packing requires CUDA scales");
  modelopt::ggml::check_pack_inputs("IQ2_XS", input, grid, modelopt::ggml::kIq2xsEntries);
  const auto num_blocks = input.numel() / modelopt::ggml::kBlockSize;
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == num_blocks,
              "scales must be float16 [numel / 256]");
  // The kernel copies these bits straight into the GGML block scale field. A non-finite entry
  // would produce a payload that decodes to garbage, and a negative one inverts the sign of every
  // decoded element while still packing cleanly -- GGML's own encoders assert a non-negative block
  // scale. One fused reduction, so the synchronization is paid once per packed tensor, on an
  // export path.
  TORCH_CHECK((scales.isfinite() & (scales >= 0)).all().item<bool>(),
              "scales must be finite and non-negative");
  TORCH_CHECK(input.get_device() == scales.get_device(), "input and scales must share a device");
  return iq2_xs_pack_cuda(input.contiguous(), grid.contiguous(), scales.contiguous());
}

at::Tensor iq2_xxs_pack(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "IQ2_XXS packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ2_XXS packing requires a CUDA grid");
  TORCH_CHECK(scales.is_cuda(), "IQ2_XXS packing requires CUDA scales");
  modelopt::ggml::check_pack_inputs("IQ2_XXS", input, grid, modelopt::ggml::kIq2xxsEntries);
  const auto num_blocks = input.numel() / modelopt::ggml::kBlockSize;
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == num_blocks,
              "scales must be float16 [numel / 256]");
  // Same rule as IQ2_XS: these bits become the block scale verbatim, and a negative or non-finite
  // one packs cleanly while decoding to garbage.
  TORCH_CHECK((scales.isfinite() & (scales >= 0)).all().item<bool>(),
              "scales must be finite and non-negative");
  TORCH_CHECK(input.get_device() == scales.get_device(), "input and scales must share a device");
  return iq2_xxs_pack_cuda(input.contiguous(), grid.contiguous(), scales.contiguous());
}

at::Tensor iq2_s_pack(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "IQ2_S packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ2_S packing requires a CUDA grid");
  TORCH_CHECK(scales.is_cuda(), "IQ2_S packing requires CUDA scales");
  modelopt::ggml::check_pack_inputs("IQ2_S", input, grid, modelopt::ggml::kIq2sEntries);
  const auto num_blocks = input.numel() / modelopt::ggml::kBlockSize;
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == num_blocks,
              "scales must be float16 [numel / 256]");
  TORCH_CHECK((scales.isfinite() & (scales >= 0)).all().item<bool>(),
              "scales must be finite and non-negative");
  TORCH_CHECK(input.get_device() == scales.get_device(), "input and scales must share a device");
  return iq2_s_pack_cuda(input.contiguous(), grid.contiguous(), scales.contiguous());
}

at::Tensor iq1_m_pack(at::Tensor input, at::Tensor grid, at::Tensor scales) {
  TORCH_CHECK(input.is_cuda(), "IQ1_M packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ1_M packing requires a CUDA grid");
  TORCH_CHECK(scales.is_cuda(), "IQ1_M packing requires CUDA scales");
  modelopt::ggml::check_pack_inputs("IQ1_M", input, grid, modelopt::ggml::kIq1sEntries);
  const auto num_blocks = input.numel() / modelopt::ggml::kBlockSize;
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.dim() == 1 &&
                  scales.numel() == num_blocks,
              "scales must be float16 [numel / 256]");
  TORCH_CHECK((scales.isfinite() & (scales >= 0)).all().item<bool>(),
              "scales must be finite and non-negative");
  TORCH_CHECK(input.get_device() == scales.get_device(), "input and scales must share a device");
  return iq1_m_pack_cuda(input.contiguous(), grid.contiguous(), scales.contiguous());
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("iq1_s_pack", &iq1_s_pack,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [2048, 8]. Returns uint8 "
             "[numel / 256, 50] on the input device. Non-finite input elements are treated as "
             "zero during packing, and finite elements outside the float32 range saturate.");
  module.def("iq2_xs_pack", &iq2_xs_pack,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [512, 8] holding "
             "non-negative codebook magnitudes, and scales must be finite non-negative float16 "
             "[numel / 256]. "
             "Returns uint8 [numel / 256, 74] on the input device. Non-finite input elements are "
             "treated as zero during packing, and finite elements outside the float32 range "
             "saturate.");
  module.def("iq2_xxs_pack", &iq2_xxs_pack,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [256, 8] holding "
             "non-negative codebook magnitudes, and scales must be finite non-negative float16 "
             "[numel / 256]. "
             "Returns uint8 [numel / 256, 66] on the input device. Non-finite input elements are "
             "treated as zero during packing, and finite elements outside the float32 range "
             "saturate.");
  module.def("iq2_s_pack", &iq2_s_pack,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [1024, 8] holding "
             "non-negative codebook magnitudes, and scales must be finite non-negative float16 "
             "[numel / 256]. "
             "Returns uint8 [numel / 256, 82] on the input device. Non-finite input elements are "
             "treated as zero during packing, and finite elements outside the float32 range "
             "saturate.");
  module.def("iq1_m_pack", &iq1_m_pack,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [2048, 8], and scales must "
             "be finite non-negative float16 [numel / 256]; IQ1_M carries that scale in the four "
             "packed scale words rather than a leading field. "
             "Returns uint8 [numel / 256, 56] on the input device. Non-finite input elements are "
             "treated as zero during packing, and finite elements outside the float32 range "
             "saturate.");
}
