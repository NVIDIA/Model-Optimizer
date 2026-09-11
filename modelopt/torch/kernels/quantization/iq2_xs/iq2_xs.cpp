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

at::Tensor iq2_xs_pack_cuda(at::Tensor input, at::Tensor grid);

at::Tensor iq2_xs_pack(at::Tensor input, at::Tensor grid) {
  TORCH_CHECK(input.is_cuda(), "IQ2_XS packing requires a CUDA input");
  TORCH_CHECK(grid.is_cuda(), "IQ2_XS packing requires a CUDA grid");
  return iq2_xs_pack_cuda(input.contiguous(), grid.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack", &iq2_xs_pack, "Pack a tensor into GGML IQ2_XS blocks");
}
