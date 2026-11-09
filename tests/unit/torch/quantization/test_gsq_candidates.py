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

"""Native IQ candidates preserve independently captured llama.cpp payloads."""

import pytest
import torch
from _test_utils.torch.quantization.iq_llama_cpp_vectors import expected_values, packed_blocks

from modelopt.torch.quantization.ggml.gsq_candidates import build_candidates
from modelopt.torch.quantization.ggml.registry import IQ_FORMAT_REGISTRY


@pytest.mark.parametrize(
    ("name", "bundle_size"),
    [("iq1_s", 32), ("iq1_m", 16), ("iq2_xxs", 32), ("iq2_xs", 16), ("iq2_s", 16)],
)
def test_incumbent_preserves_external_payload_and_decoding(name, bundle_size):
    blocks = torch.from_numpy(packed_blocks(name))
    packed = blocks.reshape(2, -1, blocks.shape[-1])
    expected = torch.from_numpy(expected_values(name)).reshape(2, -1)

    candidates = build_candidates(name, expected, packed, max_candidates=1)
    selected = torch.zeros((2, packed.shape[1], 256 // bundle_size), dtype=torch.int64)

    assert torch.equal(candidates.pack(selected, candidates.base_scale), packed)
    reconstructed = candidates.coefficients[..., 0, :] * candidates.base_scale[..., None, None]
    assert torch.equal(reconstructed.reshape_as(expected), expected)


@pytest.mark.parametrize("name", ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"])
def test_alternatives_can_change_native_fields_and_improve_reconstruction(name):
    packed = torch.from_numpy(packed_blocks(name)[:1]).unsqueeze(0)
    target = -torch.from_numpy(expected_values(name)[:1]).reshape(1, 256)
    candidates = build_candidates(name, target, packed)
    assert candidates.coefficients.shape[-2] == 5
    assert candidates.valid_mask[..., 0].all()
    assert candidates.valid_mask[..., 1:].any()

    decoded = candidates.coefficients * candidates.base_scale[..., None, None, None]
    target_groups = target.reshape(*decoded.shape[:3], 1, decoded.shape[-1])
    errors = (
        (decoded - target_groups).square().sum(-1).masked_fill(~candidates.valid_mask, torch.inf)
    )
    selected = errors.argmin(-1)
    output = candidates.pack(selected, candidates.base_scale)
    actual = IQ_FORMAT_REGISTRY[name].dequantize(
        output, torch.tensor(target.shape), dtype=torch.float32
    )
    expected = decoded.gather(
        -2, selected[..., None, None].expand(*selected.shape, 1, decoded.shape[-1])
    ).reshape_as(target)

    assert torch.equal(actual, expected)
    assert (actual - target).square().sum() < errors[..., 0].sum()


@pytest.mark.parametrize("name", ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"])
def test_common_scale_uses_native_rounding_and_preserves_selected_fields(name):
    packed = torch.from_numpy(packed_blocks(name)[:1]).unsqueeze(0)
    expected = torch.from_numpy(expected_values(name)[:1]).reshape(1, 256)
    candidates = build_candidates(name, expected, packed, max_candidates=1)
    selected = torch.zeros(candidates.coefficients.shape[:3], dtype=torch.int64)

    output = candidates.pack(selected, candidates.base_scale * 2.00001)
    actual = IQ_FORMAT_REGISTRY[name].dequantize(
        output, torch.tensor(expected.shape), dtype=torch.float32
    )

    assert torch.equal(actual, expected * 2)


@pytest.mark.parametrize("name", ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"])
def test_zero_scale_keeps_one_candidate_and_canonical_zero_payload(name):
    packed = torch.zeros((1, 1, IQ_FORMAT_REGISTRY[name].block_bytes), dtype=torch.uint8)
    candidates = build_candidates(name, torch.zeros(1, 256), packed)
    selected = torch.zeros(candidates.coefficients.shape[:3], dtype=torch.int64)

    assert candidates.valid_mask[..., 0].all()
    assert not candidates.valid_mask[..., 1:].any()
    assert torch.isfinite(candidates.coefficients).all()
    assert torch.equal(candidates.pack(selected, torch.full((1, 1), 1e-12)), packed)
