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

import onnx.numpy_helper as numpy_helper

from modelopt.onnx.quantization.int4 import get_weight_scale, get_x_w_mean_for_subgraph


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


def _reference_subgraph_w_mean(weight: np.ndarray, block_size: int) -> np.ndarray:
    """Per-block-max mean using consecutive, non-overlapping row blocks, 1e-6 damping."""
    ci, co = weight.shape
    n_blocks = -(-ci // block_size)
    padded = np.pad(weight, ((0, n_blocks * block_size - ci), (0, 0)))
    out = np.zeros_like(padded)
    for b in range(n_blocks):
        block = padded[b * block_size : (b + 1) * block_size]
        amax = np.abs(block).max(axis=0, keepdims=True)
        out[b * block_size : (b + 1) * block_size] = np.abs(block) / (amax + 1e-6)
    return out[:ci].mean(1)


@pytest.mark.parametrize(
    ("ci", "co_parts"),
    [
        (16, [3, 2]),  # ci divisible by block_size
        (10, [3, 2]),  # ci not divisible by block_size, needs padding
    ],
)
def test_get_x_w_mean_for_subgraph_matches_reference(ci, co_parts):
    """get_x_w_mean_for_subgraph must block rows the same way get_weight_scale does.

    Regression test: it reshaped the concatenated weight straight into
    (block_size, -1), which interleaves rows from different blocks instead of
    keeping each block's `block_size` consecutive rows together, and never
    padded a `ci` that doesn't divide evenly into `block_size`.
    """
    rng = np.random.RandomState(1)
    block_size = 4
    weight_parts = [rng.randn(ci, co) for co in co_parts]
    wa_pack = [
        (None, numpy_helper.from_array(w), False, 0, "") for w in weight_parts
    ]
    x = rng.randn(5, ci)

    x_m, w_m = get_x_w_mean_for_subgraph(
        wa_pack, list(range(len(wa_pack))), "unused_path", x, block_size
    )

    concatenated = np.concatenate(weight_parts, axis=1)
    expected_x_m = (np.abs(x).sum(axis=0) / x.shape[0]).reshape(-1)
    np.testing.assert_allclose(x_m, expected_x_m, rtol=1e-6)
    assert w_m.shape == (ci,)
    np.testing.assert_allclose(w_m, _reference_subgraph_w_mean(concatenated, block_size), rtol=1e-5)
