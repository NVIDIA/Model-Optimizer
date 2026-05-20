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

# ruff: noqa: D102
"""Tests for launcher/core.py — shared dataclasses, factory registry, and utilities.

Coverage:
    - SandboxTask: dataclass fields and defaults, skip flag
    - SandboxPipeline: task slot collection, task_configs resolution, global_vars interpolation
    - Factory registry: register_factory, lookup in create_task_from_yaml
    - set_slurm_config_type: patches SandboxTask annotation
    - get_default_env: returns correct env dicts for a given experiment title
    - report_versions: runs without error on a git repo
"""

import os
from dataclasses import dataclass

from common.megatron_lm.quantize.task import MegatronLMQuantizeConfig, MegatronLMQuantizeTask
from core import (
    _FACTORY_REGISTRY,
    GlobalVariables,
    SandboxPipeline,
    SandboxTask,
    SandboxTask0,
    SandboxTask1,
    create_task_from_yaml,
    get_default_env,
    register_factory,
    report_versions,
    set_slurm_config_type,
)


class TestSandboxTask:
    """Tests for the SandboxTask dataclass."""

    def test_defaults(self):
        task = SandboxTask()
        assert task.script is None
        assert task.slurm_config is None
        assert task.args is None
        assert task.environment is None
        assert task.skip is False

    def test_with_values(self):
        task = SandboxTask(
            script="test.sh",
            args=["--foo", "bar"],
            environment=[{"KEY": "val"}],
            skip=True,
        )
        assert task.script == "test.sh"
        assert task.args == ["--foo", "bar"]
        assert task.environment == [{"KEY": "val"}]
        assert task.skip is True


class TestSandboxPipeline:
    """Tests for SandboxPipeline task collection and global_vars interpolation."""

    def test_task_slots_collected(self):
        t0 = SandboxTask0(script="a.sh")
        t1 = SandboxTask1(script="b.sh")
        pipeline = SandboxPipeline(task_0=t0, task_1=t1)
        assert len(pipeline.tasks) == 2
        assert pipeline.tasks[0].script == "a.sh"
        assert pipeline.tasks[1].script == "b.sh"

    def test_empty_pipeline(self):
        pipeline = SandboxPipeline()
        assert pipeline.tasks == []

    def test_global_vars_interpolation_in_environment(self):
        t0 = SandboxTask0(
            script="test.sh",
            environment=[{"MODEL": "<<global_vars.hf_model>>"}],
        )
        pipeline = SandboxPipeline(
            task_0=t0,
            global_vars=GlobalVariables(hf_model="/hf-local/Qwen/Qwen3-8B"),
        )
        assert pipeline.tasks[0].environment == [{"MODEL": "/hf-local/Qwen/Qwen3-8B"}]

    def test_global_vars_interpolation_in_args(self):
        t0 = SandboxTask0(
            script="test.sh",
            args=["--model", "<<global_vars.hf_model>>"],
        )
        pipeline = SandboxPipeline(
            task_0=t0,
            global_vars=GlobalVariables(hf_model="/models/llama"),
        )
        assert pipeline.tasks[0].args == ["--model", "/models/llama"]

    def test_global_vars_unresolved_passthrough(self):
        t0 = SandboxTask0(
            script="test.sh",
            args=["<<global_vars.nonexistent>>"],
        )
        pipeline = SandboxPipeline(
            task_0=t0,
            global_vars=GlobalVariables(hf_model="/models/llama"),
        )
        # Unresolved references are left as-is
        assert pipeline.tasks[0].args == ["<<global_vars.nonexistent>>"]

    def test_skip_and_allow_to_fail(self):
        pipeline = SandboxPipeline(skip=True, allow_to_fail=True, note="test note")
        assert pipeline.skip is True
        assert pipeline.allow_to_fail is True
        assert pipeline.note == "test note"


class TestFactoryRegistry:
    """Tests for register_factory and its use in create_task_from_yaml."""

    def test_register_and_lookup(self, tmp_yaml):
        # Register a mock factory
        def mock_factory(nodes=1, **kwargs):
            return {"nodes": nodes, "factory": "mock"}

        register_factory("mock_factory", mock_factory)
        assert "mock_factory" in _FACTORY_REGISTRY
        assert _FACTORY_REGISTRY["mock_factory"] is mock_factory

    def test_create_task_from_yaml_uses_registry(self, tmp_yaml):
        def test_factory(nodes=1):
            return {"nodes": nodes}

        register_factory("test_factory", test_factory)

        yaml_content = """
script: test.sh
args:
  - --flag
slurm_config:
  _factory_: "test_factory"
  nodes: 2
"""
        path = tmp_yaml(yaml_content)
        task = create_task_from_yaml(path, factory_lookup={"test_factory": test_factory})
        assert task.script == "test.sh"
        assert task.args == ["--flag"]
        assert task.slurm_config == {"nodes": 2}

    def test_task_configs_resolved_via_registry(self, tmp_yaml):
        def dummy_factory(nodes=1):
            return {"nodes": nodes}

        register_factory("dummy_factory", dummy_factory)

        task_yaml = tmp_yaml(
            """
script: hello.sh
slurm_config:
  _factory_: "dummy_factory"
  nodes: 3
""",
            name="task.yaml",
        )
        pipeline = SandboxPipeline(task_configs=[task_yaml])
        assert len(pipeline.tasks) == 1
        assert pipeline.tasks[0].script == "hello.sh"
        assert pipeline.tasks[0].slurm_config == {"nodes": 3}


class TestSetSlurmConfigType:
    """Tests for set_slurm_config_type annotation patching."""

    def test_patches_annotation(self):
        @dataclass
        class MockSlurmConfig:
            host: str = "test"

        set_slurm_config_type(MockSlurmConfig)
        assert SandboxTask.__annotations__["slurm_config"] is MockSlurmConfig
        assert SandboxTask.__dataclass_fields__["slurm_config"].type is MockSlurmConfig


class TestGetDefaultEnv:
    """Tests for get_default_env utility."""

    def test_default_title(self):
        slurm_env, local_env = get_default_env()
        assert slurm_env["TRITON_CACHE_DIR"] == "/cicd/triton-cache"
        assert slurm_env["HF_HOME"] == "/cicd/hf-cache"
        assert slurm_env["MLM_SKIP_INSTALL"] == "1"
        assert "LAUNCH_SCRIPT" in slurm_env
        assert local_env["TRITON_CACHE_DIR"] == "/cicd/triton-cache"
        assert "LAUNCH_SCRIPT" not in local_env

    def test_custom_title(self):
        slurm_env, local_env = get_default_env("modelopt")
        assert slurm_env["TRITON_CACHE_DIR"] == "/modelopt/triton-cache"
        assert slurm_env["HF_HOME"] == "/modelopt/hf-cache"
        assert local_env["HF_HOME"] == "/modelopt/hf-cache"


class TestReportVersions:
    """Tests for report_versions git info utility."""

    def test_runs_on_repo(self, capsys):
        # Should not raise — runs git on the current repo
        report_versions(os.getcwd())
        captured = capsys.readouterr()
        assert "Version Report" in captured.out

    def test_runs_on_nonexistent_dir(self, capsys):
        # Should handle gracefully — "unknown" for non-git dirs
        report_versions("/tmp/nonexistent_dir_12345")
        captured = capsys.readouterr()
        assert "Version Report" in captured.out
        assert "unknown" in captured.out


class TestMaterializeFromConfig:
    """Regression coverage for typed-task config materialization.

    The original bug: MegatronLMQuantizeTask derived `script`/`args`/`environment` in
    __post_init__. Under nemo_run/Fiddle YAML loading, __post_init__ runs before nested
    fields like `config` are populated, so the derivation produced None and never re-ran.
    Result: every typed-config task in the launcher silently ran with an empty
    QUANT_CFG env var and the downstream M-LM script defaulted to FP8_DEFAULT_CFG.

    These tests exercise both call paths: direct Python construction (which never
    triggered the bug) and the Fiddle-style "construct then assign" pattern (which did).
    """

    def _env_dict(self, task):
        return {k: v for d in task.environment for k, v in d.items()}

    def test_direct_construction_materializes_environment(self):
        t0 = MegatronLMQuantizeTask(
            config=MegatronLMQuantizeConfig(quant_cfg="NVFP4_DEFAULT_CFG", mmlu_lower_bound=0.68),
        )
        t1 = MegatronLMQuantizeTask(
            config=MegatronLMQuantizeConfig(quant_cfg="FP8_DEFAULT_CFG", mmlu_lower_bound=0.76),
        )
        pipeline = SandboxPipeline(task_0=t0, task_1=t1)

        env_0 = self._env_dict(pipeline.tasks[0])
        env_1 = self._env_dict(pipeline.tasks[1])
        assert env_0["QUANT_CFG"] == "NVFP4_DEFAULT_CFG"
        assert env_0["MMLU_LOWER_BOUND"] == "0.68"
        assert env_1["QUANT_CFG"] == "FP8_DEFAULT_CFG"
        assert env_1["MMLU_LOWER_BOUND"] == "0.76"

    def test_post_init_assignment_still_materializes(self):
        """Simulates Fiddle's two-phase build: dataclass __init__ with defaults, then field assignment."""
        t0 = MegatronLMQuantizeTask()
        assert t0.environment is None  # __post_init__ ran with config=None — early-exited
        t0.config = MegatronLMQuantizeConfig(quant_cfg="NVFP4_DEFAULT_CFG")

        t1 = MegatronLMQuantizeTask()
        t1.config = MegatronLMQuantizeConfig(quant_cfg="FP8_DEFAULT_CFG")

        # SandboxPipeline.__post_init__ calls materialize_from_config() on every task,
        # which is what makes the typed config take effect when Fiddle has populated
        # `config` post-construction.
        pipeline = SandboxPipeline(task_0=t0, task_1=t1)

        env_0 = self._env_dict(pipeline.tasks[0])
        env_1 = self._env_dict(pipeline.tasks[1])
        assert env_0["QUANT_CFG"] == "NVFP4_DEFAULT_CFG"
        assert env_1["QUANT_CFG"] == "FP8_DEFAULT_CFG"

    def test_no_config_leaves_fields_untouched(self):
        """A plain SandboxTask (no typed config) survives pipeline construction unchanged."""
        t0 = SandboxTask0(script="x.sh", environment=[{"K": "v"}])
        pipeline = SandboxPipeline(task_0=t0)
        assert pipeline.tasks[0].script == "x.sh"
        assert pipeline.tasks[0].environment == [{"K": "v"}]

    def test_global_vars_resolve_inside_materialized_env(self):
        """Global_vars interpolation runs after materialize_from_config.

        Confirms that <<global_vars.X>> placed inside a typed-config field still gets
        resolved on the materialized environment list.
        """
        t0 = MegatronLMQuantizeTask(
            config=MegatronLMQuantizeConfig(
                model="<<global_vars.hf_model>>",
                quant_cfg="NVFP4_DEFAULT_CFG",
                hf_local="",
            ),
        )
        pipeline = SandboxPipeline(
            task_0=t0,
            global_vars=GlobalVariables(hf_model="Qwen/Qwen3-8B"),
        )
        env_0 = self._env_dict(pipeline.tasks[0])
        assert env_0["MLM_MODEL_CFG"] == "Qwen/Qwen3-8B"
        assert env_0["HF_MODEL_CKPT"] == "Qwen/Qwen3-8B"
