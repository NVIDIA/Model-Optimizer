# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for sweep controller."""

import json
import tempfile

from modelopt.torch.recipes.experiment import SweepConfig, SweepController


def test_dry_run(sweep_examples_dir):
    config = SweepConfig.from_yaml(sweep_examples_dir / "sweep_demo.yaml")
    controller = SweepController(config)
    output = controller.dry_run()
    assert "Big Pareto Sweep" in output
    assert "Phase 1:" in output
    assert "Phase 4:" in output
    assert "16 jobs validated" in output


def test_generate_jobs(sweep_examples_dir):
    config = SweepConfig.from_yaml(sweep_examples_dir / "sweep_demo.yaml")
    controller = SweepController(config)
    jobs = controller.generate_jobs()
    assert len(jobs) == 16
    # Check first job
    assert jobs[0].job_id == 1
    assert "pyt_ckpt_path" in jobs[0].ptq_config["hf_ptq"]


def test_export_jobs(sweep_examples_dir):
    config = SweepConfig.from_yaml(sweep_examples_dir / "sweep_demo.yaml")
    controller = SweepController(config)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    controller.export_jobs(output_path)

    with open(output_path) as f:
        data = json.load(f)

    assert len(data) == 16
    assert data[0]["job_id"] == 1


def test_verbose_dry_run(sweep_examples_dir):
    config = SweepConfig.from_yaml(sweep_examples_dir / "sweep_demo.yaml")
    controller = SweepController(config)
    output = controller.dry_run(verbose=True)
    assert "_resolved_quantize_config:" in output
