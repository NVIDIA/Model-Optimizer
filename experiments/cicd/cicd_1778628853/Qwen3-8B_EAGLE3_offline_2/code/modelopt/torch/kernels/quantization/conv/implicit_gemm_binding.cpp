/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

torch::Tensor conv3d_implicit_gemm_cuda(torch::Tensor x_pad, torch::Tensor w_flat,
                                        torch::Tensor bias, torch::Tensor act_amax, int N_batch,
                                        int Cin, int Dp, int Hp, int Wp, int Cout, int OD, int OH,
                                        int OW, int kD, int kH, int kW, int sd, int sh, int sw,
                                        int dd, int dh, int dw, int M, int K, bool quant_act,
                                        bool has_bias, int fp4_block_size);

torch::Tensor fp4_fake_quant_cuda(torch::Tensor x, torch::Tensor global_amax, int block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv3d_implicit_gemm_cuda", &conv3d_implicit_gemm_cuda,
        "Conv3D implicit GEMM with BF16 WMMA and optional FP4 quantization");
  m.def("fp4_fake_quant_cuda", &fp4_fake_quant_cuda,
        "Standalone FP4 fake quantization (blockwise, with FP8 scale quantization)");
}
