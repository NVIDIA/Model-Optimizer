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

import numpy as np
import pytest

from modelopt.onnx.quantization.int4 import get_weight_scale


def _reference_weight_scale(weight: np.ndarray, block_size: int) -> np.ndarray:
    """Per-block-max scale using consecutive, non-overlapping row blocks."""
    ci, co = weight.shape
    n_blocks = -(-ci // block_size)  # ceil division, matches the padding in get_weight_scale
    padded = np.pad(weight, ((0, n_blocks * block_size - ci), (0, 0)))
    scale = np.zeros_like(padded)
    for b in range(n_blocks):
        block = padded[b * block_size : (b + 1) * block_size]
        amax = np.abs(block).max(axis=0, keepdims=True)
        scale[b * block_size : (b + 1) * block_size] = np.abs(block) / (
            amax + np.finfo(amax.dtype).tiny
        )
    return scale[:ci].mean(1)


@pytest.mark.parametrize(
    ("ci", "co"),
    [
        (16, 3),  # ci divisible by block_size
        (10, 3),  # ci not divisible by block_size, needs padding
    ],
)
def test_get_weight_scale_matches_reference(ci, co):
    rng = np.random.RandomState(0)
    weight = rng.randn(ci, co)
    block_size = 4

    scale = get_weight_scale(weight.copy(), block_size)

    assert scale.shape == (ci,)
    np.testing.assert_allclose(scale, _reference_weight_scale(weight, block_size))


def test_get_weight_scale_without_block_size_is_unchanged():
    rng = np.random.RandomState(2)
    weight = rng.randn(16, 3)

    scale = get_weight_scale(weight.copy(), block_size=None)

    amax = np.abs(weight).max(axis=0, keepdims=True)
    expected = (np.abs(weight) / (amax + np.finfo(amax.dtype).tiny)).mean(1)
    np.testing.assert_allclose(scale, expected)
