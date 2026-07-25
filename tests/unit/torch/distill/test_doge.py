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

import pytest

from modelopt.torch.distill.doge import (
    DoGEWeightUpdater,
    apply_source_max_blend_weights,
    apply_source_min_blend_weights,
    normalize_data_path_weights,
    resolve_source_max_blend_weights,
    resolve_source_min_blend_weights,
    sample_data_path_by_weight,
)


def test_normalize_data_path_weights():
    weights = normalize_data_path_weights(["3", "/data/a", "1", "/data/b"])

    assert weights == {"/data/a": 0.75, "/data/b": 0.25}


def test_normalize_data_path_weights_rejects_duplicate_paths():
    with pytest.raises(ValueError, match="duplicate dataset path"):
        normalize_data_path_weights(["2", "/data/a", "1", "/data/a"])


def test_sample_data_path_by_weight_is_deterministic():
    weights = {"/data/a": 0.25, "/data/b": 0.75}

    assert sample_data_path_by_weight(
        weights, iteration=7, seed=1234
    ) == sample_data_path_by_weight(weights, iteration=7, seed=1234)


def test_sample_data_path_by_weight_respects_zero_weight():
    weights = {"/data/a": 1.0, "/data/b": 0.0}

    assert {sample_data_path_by_weight(weights, iteration=i, seed=1234) for i in range(20)} == {
        "/data/a"
    }


def test_resolve_source_min_blend_weights_accepts_unique_suffix():
    source_paths = ["/data/wiki", "/data/nemotron/math", "/data/nemotron/stem"]

    weights = resolve_source_min_blend_weights({"nemotron/math": 0.2}, source_paths)

    assert weights == {"/data/nemotron/math": 0.2}


def test_resolve_source_max_blend_weights_accepts_unique_suffix():
    source_paths = ["/data/wiki", "/data/nemotron/math", "/data/nemotron/stem"]

    weights = resolve_source_max_blend_weights({"nemotron/stem": 0.0}, source_paths)

    assert weights == {"/data/nemotron/stem": 0.0}


def test_apply_source_min_blend_weights_rescales_remaining_sources():
    weights = apply_source_min_blend_weights(
        {"wiki": 0.8, "math": 0.05, "stem": 0.15},
        {"math": 0.1},
    )

    assert weights["math"] == 0.1
    assert weights["wiki"] == pytest.approx(0.7578947368)
    assert weights["stem"] == pytest.approx(0.1421052632)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_apply_source_max_blend_weights_rescales_remaining_sources():
    weights = apply_source_max_blend_weights(
        {"wiki": 0.2, "math": 0.5, "stem": 0.3},
        {"stem": 0.0},
    )

    assert weights["stem"] == 0.0
    assert weights["wiki"] == pytest.approx(0.2857142857)
    assert weights["math"] == pytest.approx(0.7142857143)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_doge_weight_updater_increases_aligned_source():
    updater = DoGEWeightUpdater(meta_lr=1.0)

    weights = updater.update(
        {"wiki": 0.5, "nemotron": 0.5},
        {"wiki": 1.0, "nemotron": -1.0},
    )

    assert round(weights["wiki"], 4) == 0.8808
    assert round(weights["nemotron"], 4) == 0.1192
