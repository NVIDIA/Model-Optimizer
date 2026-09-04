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

from modelopt.torch.fastgen import PDDConfig, SampleTimestepConfig, load_pdd_config


def test_default_pdd_config_is_canonical_and_lists_are_independent():
    first = PDDConfig()
    second = PDDConfig()

    assert first.pred_type == "flow"
    assert first.student_sample_type == "ode"
    assert first.student_sample_steps == 4
    assert first.grid_size == 128
    assert first.grid_max_t == 0.999
    assert first.flow_shift == 5.0
    assert first.block_size_min == 4
    assert first.block_size_max == 64
    assert first.teacher_integrator == "euler"
    assert first.inference_blocks == [32, 32, 32, 32]
    assert first.data_free is False
    assert first.inference_blocks is not second.inference_blocks

    first.inference_blocks[0] = 16
    assert second.inference_blocks == [32, 32, 32, 32]


def test_pdd_config_accepts_supported_schedule_and_adapter_time_scale():
    config = PDDConfig(
        inference_blocks=[64, 64],
        student_sample_steps=2,
        teacher_integrator="midpoint",
        num_train_timesteps=1000,
        data_free=True,
    )

    assert config.inference_blocks == [64, 64]
    assert config.teacher_integrator == "midpoint"
    assert config.num_train_timesteps == 1000
    assert config.data_free is True


def test_rejected_attribute_assignment_leaves_pdd_config_unchanged():
    config = PDDConfig()

    with pytest.raises(ValueError, match="student_sample_steps must equal"):
        config.student_sample_steps = 3

    assert config.student_sample_steps == 4
    assert config.inference_blocks == [32, 32, 32, 32]


def test_rejected_mapping_assignment_leaves_pdd_config_unchanged():
    config = PDDConfig()

    with pytest.raises(ValueError, match="student_sample_steps must equal"):
        config["inference_blocks"] = [64, 64]

    assert config.student_sample_steps == 4
    assert config.inference_blocks == [32, 32, 32, 32]


@pytest.mark.parametrize("value", [True, 1])
def test_pdd_config_rejects_non_float_grid_max_t_before_coercion(value):
    with pytest.raises(ValueError, match="grid_max_t must be a float"):
        PDDConfig(grid_max_t=value)

    config = PDDConfig()
    with pytest.raises(ValueError, match="grid_max_t must be a float"):
        config.grid_max_t = value
    assert config.grid_max_t == 0.999


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
        (
            {"inference_blocks": [64, 64], "student_sample_steps": 4},
            "student_sample_steps must equal",
        ),
    ],
)
def test_pdd_config_rejects_invalid_grid_and_block_boundaries(overrides, message):
    with pytest.raises(ValueError, match=message):
        PDDConfig(**overrides)


def test_pdd_config_accepts_inference_partition_outside_training_block_support():
    config = PDDConfig(inference_blocks=[1, 127], student_sample_steps=2)

    assert config.inference_blocks == [1, 127]


@pytest.mark.parametrize(
    "overrides",
    [
        {"pred_type": "x0"},
        {"student_sample_type": "sde"},
        {"teacher_integrator": "heun"},
        {"teacher_integrator": "rk4"},
    ],
)
def test_pdd_config_locks_algorithm_modes(overrides):
    with pytest.raises(ValueError):
        PDDConfig(**overrides)


def test_pdd_config_rejects_nondefault_sample_timestep_config():
    with pytest.raises(ValueError, match="sample_t_cfg is unused by PDD"):
        PDDConfig(sample_t_cfg=SampleTimestepConfig(shift=6.0))


def test_pdd_config_loads_filesystem_yaml_with_optional_suffix(tmp_path):
    config_path = tmp_path / "pdd.yaml"
    config_path.write_text(
        "inference_blocks: [64, 64]\nstudent_sample_steps: 2\nteacher_integrator: midpoint\n",
        encoding="utf-8",
    )

    loaded = load_pdd_config(config_path.with_suffix(""))
    from_class = PDDConfig.from_yaml(config_path)

    assert loaded == from_class
    assert loaded.inference_blocks == [64, 64]
    assert loaded.teacher_integrator == "midpoint"
