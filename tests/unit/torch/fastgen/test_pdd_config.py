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

"""Validation and loading tests for framework-neutral PDD configuration."""

from __future__ import annotations

import pytest

from modelopt.torch.fastgen import PDDConfig, load_pdd_config


def test_default_pdd_config_is_canonical():
    config = PDDConfig()

    assert config.guidance_scale is None
    assert config.grid_size == 128
    assert config.grid_max_t == 0.999
    assert config.flow_shift == 5.0
    assert config.block_size_min == 4
    assert config.block_size_max == 64
    assert config.teacher_integrator == "euler"
    assert config.inference_blocks == (32, 32, 32, 32)
    assert config.data_free is False


def test_pdd_config_accepts_supported_schedule():
    config = PDDConfig(
        inference_blocks=[64, 64],
        teacher_integrator="midpoint",
        data_free=True,
    )

    assert config.inference_blocks == (64, 64)
    assert config.teacher_integrator == "midpoint"
    assert config.data_free is True


@pytest.mark.parametrize("value", [True, 1])
def test_pdd_config_rejects_non_float_grid_max_t_before_coercion(value):
    with pytest.raises(ValueError, match="grid_max_t must be a float"):
        PDDConfig(grid_max_t=value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"grid_size": 0}, "grid_size must be > 0"),
        ({"grid_max_t": 0.0}, "0 < grid_max_t <= 1"),
        ({"grid_max_t": 1.0001}, "0 < grid_max_t <= 1"),
        ({"grid_max_t": float("nan")}, "0 < grid_max_t <= 1"),
        ({"flow_shift": 0.5}, "flow_shift must be finite and >= 1"),
        ({"block_size_min": 0}, "0 < block_size_min"),
        ({"block_size_max": 129}, "block_size_max <= grid_size"),
        ({"grid_size": 130}, "must be divisible"),
        ({"inference_blocks": []}, "at least one block"),
        ({"inference_blocks": [32, 32, 32]}, "must sum to grid_size"),
    ],
)
def test_pdd_config_rejects_invalid_grid_and_block_boundaries(overrides, message):
    with pytest.raises(ValueError, match=message):
        PDDConfig(**overrides)


def test_pdd_config_accepts_inference_partition_outside_training_block_support():
    config = PDDConfig(inference_blocks=[1, 127])

    assert config.inference_blocks == (1, 127)


@pytest.mark.parametrize("mapping_assignment", [False, True])
def test_rejected_assignment_leaves_pdd_config_unchanged(mapping_assignment):
    config = PDDConfig()

    with pytest.raises(ValueError, match="must sum to grid_size"):
        if mapping_assignment:
            config["grid_size"] = 64
        else:
            config.grid_size = 64

    assert config.grid_size == 128
    assert config.inference_blocks == (32, 32, 32, 32)


@pytest.mark.parametrize(
    "overrides", [{"teacher_integrator": "heun"}, {"teacher_integrator": "rk4"}]
)
def test_pdd_config_locks_algorithm_modes(overrides):
    with pytest.raises(ValueError):
        PDDConfig(**overrides)


def test_pdd_config_loads_filesystem_yaml_with_optional_suffix(tmp_path):
    config_path = tmp_path / "pdd.yaml"
    config_path.write_text(
        "inference_blocks: [64, 64]\nteacher_integrator: midpoint\n",
        encoding="utf-8",
    )

    loaded = load_pdd_config(config_path.with_suffix(""))
    from_class = PDDConfig.from_yaml(config_path)

    assert loaded == from_class
    assert loaded.inference_blocks == (64, 64)
    assert loaded.teacher_integrator == "midpoint"
