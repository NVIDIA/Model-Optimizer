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
"""GPTQ for GGML block formats: the group update, and pinning a payload GPTQ chose."""

import pytest
import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.common import cache_packed_weight
from modelopt.torch.quantization.ggml.gptq import gptq_group_update
from modelopt.torch.quantization.utils.calib_utils import (
    compute_hessian_inverse,
    gptq_blockwise_update,
)


def _problem(rows=6, cols=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator)
    inputs = torch.randn(64, cols, generator=generator) @ torch.randn(
        cols, cols, generator=generator
    )
    hessian = inputs.T @ inputs / inputs.shape[0]
    return weight, hessian, compute_hessian_inverse(hessian, weight, 0.01)


def _round_to_tenths(weight):
    return torch.round(weight * 10) / 10


def _round_in_groups_of_4(group):
    scale = group.abs().amax(dim=-1, keepdim=True) / 2
    return torch.round(group / scale) * scale


def test_single_column_groups_match_columnwise_gptq():
    weight, _, h_inv = _problem()
    expected = weight.clone()
    gptq_blockwise_update(expected, h_inv, 8, _round_to_tenths)

    gptq_group_update(weight, h_inv, 8, 1, _round_to_tenths)

    torch.testing.assert_close(weight, expected)


def test_group_update_folds_each_group_error_through_its_inverse_hessian_block():
    weight, _, h_inv = _problem()
    expected = weight.clone()
    for start in range(0, expected.shape[1], 4):
        group, rest = slice(start, start + 4), slice(start + 4, None)
        qdq = _round_in_groups_of_4(expected[:, group])
        err = (expected[:, group] - qdq) @ torch.linalg.inv(h_inv[group, group])
        expected[:, group] = qdq
        expected[:, rest] -= err @ h_inv[group, rest]

    gptq_group_update(weight, h_inv, 8, 4, _round_in_groups_of_4)

    torch.testing.assert_close(weight, expected, rtol=1e-4, atol=1e-5)


def test_group_update_rejects_blocks_that_split_a_group():
    weight, _, h_inv = _problem()

    with pytest.raises(ValueError, match="multiples of the quantization group size"):
        gptq_group_update(weight, h_inv, 6, 4, _round_in_groups_of_4)


def _iq1_s_config(algorithm):
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": "iq1_s", "block_sizes": {-1: 256}, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": algorithm,
    }


def test_fake_quant_uses_a_payload_cached_for_the_whole_weight():
    model = torch.nn.Linear(512, 4, bias=False)
    inputs = torch.randn(3, 512)
    mtq.quantize(model, _iq1_s_config("max"), forward_loop=lambda m: m(inputs))
    iq1_s = GGML_FORMAT_REGISTRY["iq1_s"]
    other_packed, shape = iq1_s.quantize(torch.randn(4, 512))
    quantizer = model.weight_quantizer

    cache_packed_weight(quantizer, model.weight, "iq1_s", iq1_s.block_chunk_size, other_packed)

    other = iq1_s.dequantize(other_packed, shape, dtype=torch.float32)
    torch.testing.assert_close(quantizer(model.weight), other)
