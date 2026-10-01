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
// so they compile into one extension and bind here. Each IQ format keeps its block layout in its
// own translation unit, on the encoder its family shares (iq1_family.cuh or iq2_family.cuh), and
// exposes its packer.

#include "common.cuh"

at::Tensor iq1_s_pack_cuda(at::Tensor input, at::Tensor grid);
at::Tensor iq2_xs_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);
at::Tensor iq2_xxs_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);
at::Tensor iq2_s_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);
at::Tensor iq1_m_pack_cuda(at::Tensor input, at::Tensor grid, at::Tensor scales);
at::Tensor q8_0_pack_cuda(at::Tensor input);

namespace {

at::Tensor q8_0_pack(at::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Q8_0 packing requires a CUDA input");
  modelopt::ggml::check_scalar_pack_input("Q8_0", input, 32);
  return q8_0_pack_cuda(input.contiguous());
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("iq1_s_pack", &iq1_s_pack_cuda,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [2048, 8]. Returns uint8 "
             "[numel / 256, 50] on the input device. Non-finite input elements are treated as "
             "zero during packing, and finite elements outside the float32 range saturate.");
  module.def("iq2_xs_pack", &iq2_xs_pack_cuda,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [512, 8] holding "
             "non-negative codebook magnitudes, and scales must be finite non-negative float16 "
             "[numel / 256]. "
             "Returns uint8 [numel / 256, 74] on the input device. Non-finite input elements are "
             "treated as zero during packing, and finite elements outside the float32 range "
             "saturate.");
  module.def("iq2_xxs_pack", &iq2_xxs_pack_cuda,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [256, 8] holding "
             "non-negative codebook magnitudes, and scales must be finite non-negative float16 "
             "[numel / 256]. "
             "Returns uint8 [numel / 256, 66] on the input device. Non-finite input elements are "
             "treated as zero during packing, and finite elements outside the float32 range "
             "saturate.");
  module.def("iq2_s_pack", &iq2_s_pack_cuda,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [1024, 8] holding "
             "non-negative codebook magnitudes, and scales must be finite non-negative float16 "
             "[numel / 256]. "
             "Returns uint8 [numel / 256, 82] on the input device. Non-finite input elements are "
             "treated as zero during packing, and finite elements outside the float32 range "
             "saturate.");
  module.def("iq1_m_pack", &iq1_m_pack_cuda,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 256. The grid must be float32 [2048, 8], and scales must "
             "be finite non-negative float16 [numel / 256]; IQ1_M carries that scale in the four "
             "packed scale words rather than a leading field. "
             "Returns uint8 [numel / 256, 56] on the input device. Non-finite input elements are "
             "treated as zero during packing, and finite elements outside the float32 range "
             "saturate.");
  module.def("q8_0_pack", &q8_0_pack,
             "Pack a non-empty float32, float64, float16, or bfloat16 CUDA tensor whose innermost "
             "dimension is a multiple of 32. Returns uint8 [numel / 32, 34] on the input device. "
             "Non-finite input elements are treated as zero during packing, and finite elements "
             "outside the float32 range saturate.");
}
