# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from modelopt.torch.quantization.ggml.iq1_s import dequantize_iq1_s, quantize_iq1_s


def test_iq1_s_cuda_pack_is_deterministic_and_decodable():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((8, 256), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed, shape = quantize_iq1_s(weight)
    packed_again, _ = quantize_iq1_s(weight)
    reconstructed = dequantize_iq1_s(packed, shape)

    assert packed.shape == (8, 1, 50)
    assert torch.equal(packed, packed_again)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.25


def test_iq1_s_cuda_zero_encoding_matches_ggml_block_layout():
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    packed, shape = quantize_iq1_s(weight)

    assert not packed.any()
    assert torch.equal(dequantize_iq1_s(packed, shape), weight)
