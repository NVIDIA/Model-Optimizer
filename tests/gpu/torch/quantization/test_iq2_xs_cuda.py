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

from modelopt.torch.quantization.ggml.iq2_xs import dequantize_iq2_xs, quantize_iq2_xs


def test_iq2_xs_cuda_pack_is_deterministic_and_decodable():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    weight = torch.randn((8, 512), generator=generator, device="cuda", dtype=torch.bfloat16)

    packed, shape = quantize_iq2_xs(weight)
    packed_again, _ = quantize_iq2_xs(weight)
    reconstructed = dequantize_iq2_xs(packed, shape)

    assert packed.shape == (8, 2, 74)
    assert torch.equal(packed, packed_again)
    normalized_mse = (
        reconstructed.float() - weight.float()
    ).square().mean() / weight.float().square().mean()
    assert normalized_mse < 0.1


def test_iq2_xs_cuda_zero_encoding_matches_ggml_block_layout():
    weight = torch.zeros((1, 256), device="cuda", dtype=torch.bfloat16)
    packed, shape = quantize_iq2_xs(weight)

    assert not packed.any()
    assert torch.equal(dequantize_iq2_xs(packed, shape), weight)
