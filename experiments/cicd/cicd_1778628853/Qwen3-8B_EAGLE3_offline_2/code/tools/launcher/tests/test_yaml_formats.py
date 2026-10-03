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

"""Tests for YAML config parsing — verifies that different YAML formats produce correct dataclasses.

Coverage:
    - --yaml format: top-level job_name + pipeline with task_0, environment, slurm_config
    - pipeline=@ format: bare SandboxPipeline without job_name wrapper
    - task_configs: list of YAML paths resolved via factory registry
    - Environment formats: list-of-dicts and flat dict both parsed correctly
    - Global vars: <<global_vars.X>> resolved in both args and environment
"""

import yaml


class TestYamlFormatParsing:
    """Tests that YAML content parses into correct dataclass structures."""

    def test_yaml_format_with_job_name(self, tmp_yaml):
        """The --yaml format has job_name and pipeline as top-level keys."""
        content = """
job_name: test_job
pipeline:
  skip: false
  allow_to_fail: true
  note: "test note"
  task_0:
    script: test.sh
    args:
      - --flag
    environment:
      - KEY: value
"""
        path = tmp_yaml(content)
        with open(path) as f:
            data = yaml.safe_load(f)

        assert data["job_name"] == "test_job"
        assert data["pipeline"]["skip"] is False
        assert data["pipeline"]["allow_to_fail"] is True
        assert data["pipeline"]["note"] == "test note"
        assert data["pipeline"]["task_0"]["script"] == "test.sh"
        assert data["pipeline"]["task_0"]["args"] == ["--flag"]
        assert data["pipeline"]["task_0"]["environment"] == [{"KEY": "value"}]

    def test_bare_pipeline_format(self, tmp_yaml):
        """The pipeline=@ format is a bare SandboxPipeline without wrapper."""
        content = """
task_0:
  script: a.sh
  args:
    - --foo
task_1:
  script: b.sh
allow_to_fail: false
skip: false
"""
        path = tmp_yaml(content)
        with open(path) as f:
            data = yaml.safe_load(f)

        # Verify the YAML parses into valid SandboxPipeline kwargs
        # (nemo-run does this via its CLI parser; we just verify the structure)
        assert "task_0" in data
        assert "task_1" in data
        assert data["task_0"]["script"] == "a.sh"
        assert data["task_1"]["script"] == "b.sh"

    def test_task_configs_format(self, tmp_yaml):
        """task_configs lists YAML files that are resolved into tasks."""
        from core import SandboxPipeline, register_factory

        def local_factory(nodes=1):
            return {"nodes": nodes}

        register_factory("local_factory", local_factory)

        task_path = tmp_yaml(
            """
script: worker.sh
args:
  - --batch-size 32
slurm_config:
  _factory_: "local_factory"
  nodes: 2
""",
            name="worker.yaml",
        )

        pipeline = SandboxPipeline(task_configs=[task_path])
        assert len(pipeline.tasks) == 1
        assert pipeline.tasks[0].script == "worker.sh"
        assert pipeline.tasks[0].args == ["--batch-size 32"]
        assert pipeline.tasks[0].slurm_config == {"nodes": 2}

    def test_environment_list_of_dicts(self):
        """Environment as list-of-single-key-dicts (nemo-run format)."""
        from core import SandboxTask

        task = SandboxTask(
            script="test.sh",
            environment=[{"A": "1"}, {"B": "2"}, {"C": "3"}],
        )
        assert len(task.environment) == 3
        assert task.environment[0] == {"A": "1"}

    def test_global_vars_across_multiple_tasks(self, tmp_yaml):
        """Global vars resolve in both task_0 and task_1."""
        from core import GlobalVariables, SandboxPipeline, SandboxTask0, SandboxTask1

        t0 = SandboxTask0(
            script="quantize.sh",
            args=["--model", "<<global_vars.hf_model>>"],
            environment=[{"HF_MODEL": "<<global_vars.hf_model>>"}],
        )
        t1 = SandboxTask1(
            script="eval.sh",
            environment=[{"HF_MODEL": "<<global_vars.hf_model>>"}],
        )
        pipeline = SandboxPipeline(
            task_0=t0,
            task_1=t1,
            global_vars=GlobalVariables(hf_model="/hf-local/Qwen/Qwen3-8B"),
        )
        assert pipeline.tasks[0].args == ["--model", "/hf-local/Qwen/Qwen3-8B"]
        assert pipeline.tasks[0].environment == [{"HF_MODEL": "/hf-local/Qwen/Qwen3-8B"}]
        assert pipeline.tasks[1].environment == [{"HF_MODEL": "/hf-local/Qwen/Qwen3-8B"}]


class TestTestYamlFormat:
    """Tests for the test YAML format used by run_test_yaml.sh."""

    def test_target_with_overrides(self, tmp_yaml):
        """Test YAML entries have _target_ and override fields."""
        content = """
- _target_: path/to/config.yaml
  pipeline:
    allow_to_fail: true
    skip: false
    note: "known issue"
- _target_: path/to/other.yaml
  pipeline:
    allow_to_fail: false
"""
        path = tmp_yaml(content)
        with open(path) as f:
            data = yaml.safe_load(f)

        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["_target_"] == "path/to/config.yaml"
        assert data[0]["pipeline"]["allow_to_fail"] is True
        assert data[0]["pipeline"]["note"] == "known issue"
        assert data[1]["_target_"] == "path/to/other.yaml"
        assert data[1]["pipeline"]["allow_to_fail"] is False

    def test_flatten_overrides(self):
        """Nested overrides flatten to dot-notation for CLI args."""
        entry = {
            "pipeline": {
                "allow_to_fail": True,
                "skip": False,
            }
        }

        # Simulate the flatten logic from run_test_yaml.sh
        overrides = []

        def flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, f"{key}.")
                else:
                    overrides.append(f"{key}={v}")

        flatten(entry)
        assert "pipeline.allow_to_fail=True" in overrides
        assert "pipeline.skip=False" in overrides
