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

"""Unit tests for :mod:`modelopt.onnx.quantization.sensitivity.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from modelopt.onnx.quantization.sensitivity.metrics import (
    _flatten_per_sample,
    cos_dist,
    kl_div,
    mse,
)


class TestIdenticalInputs:
    """All three metrics collapse to (near-)zero on identical inputs."""

    @pytest.mark.parametrize("metric", [kl_div, mse, cos_dist])
    def test_identical_2d_inputs_score_zero(self, metric):
        x = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
        assert metric(x, x) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("metric", [kl_div, mse, cos_dist])
    def test_identical_4d_inputs_score_zero(self, metric):
        x = np.random.default_rng(0).standard_normal((2, 3, 4, 5)).astype(np.float32)
        assert metric(x, x) == pytest.approx(0.0, abs=1e-6)


class TestCosDistOrthogonal:
    """Orthogonal vectors produce ``cos_dist == 1`` (cosine similarity 0)."""

    def test_orthogonal_vectors(self):
        p = np.array([[1.0, 0.0]], dtype=np.float32)
        q = np.array([[0.0, 1.0]], dtype=np.float32)
        assert cos_dist(p, q) == pytest.approx(1.0, abs=1e-6)

    def test_anti_parallel_vectors(self):
        p = np.array([[1.0, 0.0]], dtype=np.float32)
        q = np.array([[-1.0, 0.0]], dtype=np.float32)
        # cos_sim = -1, so cos_dist = 1 - (-1) = 2.
        assert cos_dist(p, q) == pytest.approx(2.0, abs=1e-6)

    def test_both_zero_vectors_return_zero_distance(self):
        # A probe whose reference and quantized outputs are both all-zero (e.g. a hard-relu /
        # masked-out branch) should score as identical, not maximally sensitive.
        p = np.zeros((2, 4), dtype=np.float32)
        q = np.zeros((2, 4), dtype=np.float32)
        assert cos_dist(p, q) == pytest.approx(0.0, abs=1e-6)

    def test_asymmetric_zero_vectors_return_max_distance(self):
        # Different vectors (with one being zero) should score as orthogonal (distance 1)
        p = np.array([[1.0, 0.0]], dtype=np.float32)
        q = np.array([[0.0, 0.0]], dtype=np.float32)
        assert cos_dist(p, q) == pytest.approx(1.0, abs=1e-6)
        assert cos_dist(q, p) == pytest.approx(1.0, abs=1e-6)


class TestScaleSensitivity:
    """``mse`` scales with input magnitude; ``cos_dist`` does not; ``kl_div`` is invariant on
    softmax outputs regardless of scale."""

    def test_mse_grows_with_magnitude(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((4, 8)).astype(np.float32)
        perturbed = base + 0.1 * rng.standard_normal((4, 8)).astype(np.float32)
        small = mse(base, perturbed)
        big = mse(10.0 * base, 10.0 * perturbed)
        assert big > 50.0 * small, (
            f"MSE should scale ~100x on 10x-larger inputs; got small={small} big={big}"
        )

    def test_cos_dist_is_scale_invariant(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((4, 8)).astype(np.float32)
        perturbed = base + 0.1 * rng.standard_normal((4, 8)).astype(np.float32)
        small = cos_dist(base, perturbed)
        big = cos_dist(10.0 * base, 10.0 * perturbed)
        assert small == pytest.approx(big, rel=1e-4)


class TestFlattenPerSample:
    """``_flatten_per_sample`` reshapes any-rank tensor to ``(num_samples, num_features)``."""

    def test_1d_treated_as_single_sample(self):
        x = np.arange(4, dtype=np.float32)
        assert _flatten_per_sample(x).shape == (4, 1)

    def test_2d_passes_through(self):
        x = np.zeros((3, 5), dtype=np.float32)
        assert _flatten_per_sample(x).shape == (3, 5)

    def test_4d_collapses_feature_dims(self):
        x = np.zeros((2, 3, 4, 5), dtype=np.float32)
        assert _flatten_per_sample(x).shape == (2, 60)

    def test_0d_becomes_1x1(self):
        x = np.float32(3.14)
        assert _flatten_per_sample(x).shape == (1, 1)
