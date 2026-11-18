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
"""Extended tests for launcher/core.py — edge cases and remaining coverage gaps.

Coverage:
    - create_task_from_yaml: error cases (missing factory, bad YAML)
    - SandboxPipeline: dict environment (not list), task_configs with registry fallback
    - _git_info: direct tests for success and failure
    - run_jobs: environment merging (list vs dict), test_level filtering, pipeline skip,
      detach flag, version report
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestCreateTaskFromYamlErrors:
    """Error handling in create_task_from_yaml."""

    def test_missing_factory_raises(self, tmp_yaml):
        from core import create_task_from_yaml

        yaml_content = """
script: test.sh
slurm_config:
  _factory_: "nonexistent_factory"
  nodes: 1
"""
        path = tmp_yaml(yaml_content)
        with pytest.raises(KeyError):
            create_task_from_yaml(path, factory_lookup={})

    def test_missing_slurm_config_raises(self, tmp_yaml):
        from core import create_task_from_yaml

        yaml_content = """
script: test.sh
"""
        path = tmp_yaml(yaml_content)
        with pytest.raises((KeyError, TypeError)):
            create_task_from_yaml(path, factory_lookup={})

    def test_environment_preserved(self, tmp_yaml):
        from core import create_task_from_yaml

        def factory(nodes=1):
            return {"nodes": nodes}

        yaml_content = """
script: test.sh
environment:
  - KEY1: val1
  - KEY2: val2
slurm_config:
  _factory_: "f"
  nodes: 1
"""
        path = tmp_yaml(yaml_content)
        task = create_task_from_yaml(path, factory_lookup={"f": factory})
        assert task.environment == [{"KEY1": "val1"}, {"KEY2": "val2"}]


class TestSandboxPipelineExtended:
    """Extended SandboxPipeline tests."""

    def test_dict_environment_interpolation(self):
        """Global vars resolve in dict-format environment (not list)."""
        from core import GlobalVariables, SandboxPipeline, SandboxTask0

        t0 = SandboxTask0(
            script="test.sh",
            environment={"MODEL": "<<global_vars.hf_model>>", "STATIC": "value"},
        )
        pipeline = SandboxPipeline(
            task_0=t0,
            global_vars=GlobalVariables(hf_model="/hf-local/model"),
        )
        assert pipeline.tasks[0].environment == {
            "MODEL": "/hf-local/model",
            "STATIC": "value",
        }

    def test_tasks_list_directly(self):
        """Pipeline can receive tasks as a list directly."""
        from core import SandboxPipeline, SandboxTask

        tasks = [
            SandboxTask(script="a.sh"),
            SandboxTask(script="b.sh"),
            SandboxTask(script="c.sh"),
        ]
        pipeline = SandboxPipeline(tasks=tasks)
        assert len(pipeline.tasks) == 3
        assert pipeline.tasks[2].script == "c.sh"

    def test_no_global_vars_no_error(self):
        """Pipeline without global_vars doesn't crash on interpolation."""
        from core import SandboxPipeline, SandboxTask0

        t0 = SandboxTask0(
            script="test.sh",
            args=["<<global_vars.hf_model>>"],
        )
        pipeline = SandboxPipeline(task_0=t0)
        # No interpolation happens — args kept as-is
        assert pipeline.tasks[0].args == ["<<global_vars.hf_model>>"]


class TestGitInfo:
    """Direct tests for _git_info helper."""

    def test_valid_git_repo(self):
        from core import _git_info

        commit, branch = _git_info(os.getcwd())
        assert commit != "unknown"
        assert branch != "unknown"
        assert len(commit) >= 7  # short hash

    def test_nonexistent_directory(self):
        from core import _git_info

        commit, branch = _git_info("/tmp/nonexistent_xyz_12345")
        assert commit == "unknown"
        assert branch == "unknown"

    def test_non_git_directory(self):
        from core import _git_info

        # Use /tmp which is outside any git repo
        commit, branch = _git_info("/tmp")
        # /tmp may or may not be inside a git worktree depending on the system
        # Just verify it returns strings without crashing
        assert isinstance(commit, str)
        assert isinstance(branch, str)


class TestRunJobsExtended:
    """Extended run_jobs tests for env merging, test_level, and detach."""

    @patch("core.run.Experiment")
    @patch("core.build_docker_executor")
    def test_environment_list_merged_to_env(self, mock_docker, mock_exp, tmp_path):
        """List-of-dicts environment is merged into task_env."""
        from core import SandboxPipeline, SandboxTask0, get_default_env, run_jobs

        mock_exp_inst = MagicMock()
        mock_exp_inst._id = "exp_env"
        mock_exp_inst.__enter__ = MagicMock(return_value=mock_exp_inst)
        mock_exp_inst.__exit__ = MagicMock(return_value=False)
        mock_exp.return_value = mock_exp_inst
        mock_docker.return_value = MagicMock()

        slurm_env, local_env = get_default_env()

        t0 = SandboxTask0(
            script="test.sh",
            slurm_config=MagicMock(),
            environment=[{"A": "1"}, {"B": "2"}],
        )
        pipeline = SandboxPipeline(task_0=t0)

        with patch("core.run.Script") as mock_script:
            run_jobs(
                job_table={"job": pipeline},
                hf_local="/tmp/hf",
                user="u",
                identity=None,
                job_dir=str(tmp_path),
                packager=MagicMock(),
                default_slurm_env=slurm_env,
                default_local_env=local_env,
                base_dir=str(tmp_path),
            )
            # Script called with merged env
            call_kwargs = mock_script.call_args[1]
            assert "A" in call_kwargs["env"]
            assert "B" in call_kwargs["env"]
            assert call_kwargs["env"]["A"] == "1"

    @patch("core.run.Experiment")
    @patch("core.build_docker_executor")
    def test_none_env_values_converted_to_empty_string(self, mock_docker, mock_exp, tmp_path):
        from core import SandboxPipeline, SandboxTask0, get_default_env, run_jobs

        mock_exp_inst = MagicMock()
        mock_exp_inst._id = "exp_none"
        mock_exp_inst.__enter__ = MagicMock(return_value=mock_exp_inst)
        mock_exp_inst.__exit__ = MagicMock(return_value=False)
        mock_exp.return_value = mock_exp_inst
        mock_docker.return_value = MagicMock()

        slurm_env, local_env = get_default_env()

        t0 = SandboxTask0(
            script="test.sh",
            slurm_config=MagicMock(),
            environment=[{"KEY": None}],
        )
        pipeline = SandboxPipeline(task_0=t0)

        with patch("core.run.Script") as mock_script:
            run_jobs(
                job_table={"job": pipeline},
                hf_local="/tmp/hf",
                user="u",
                identity=None,
                job_dir=str(tmp_path),
                packager=MagicMock(),
                default_slurm_env=slurm_env,
                default_local_env=local_env,
                base_dir=str(tmp_path),
            )
            env = mock_script.call_args[1]["env"]
            assert env["KEY"] == ""

    @patch("core.run.Experiment")
    @patch("core.build_docker_executor")
    def test_test_level_filters_pipeline(self, mock_docker, mock_exp, tmp_path):
        """Pipelines with test_level > current are skipped."""
        from core import SandboxPipeline, SandboxTask0, get_default_env, run_jobs

        mock_exp_inst = MagicMock()
        mock_exp_inst._id = "exp_lvl"
        mock_exp_inst.__enter__ = MagicMock(return_value=mock_exp_inst)
        mock_exp_inst.__exit__ = MagicMock(return_value=False)
        mock_exp.return_value = mock_exp_inst
        mock_docker.return_value = MagicMock()

        slurm_env, local_env = get_default_env()

        t0 = SandboxTask0(script="test.sh", slurm_config=MagicMock())
        pipeline = SandboxPipeline(task_0=t0, test_level=2)

        run_jobs(
            job_table={"job": pipeline},
            hf_local="/tmp/hf",
            user="u",
            identity=None,
            job_dir=str(tmp_path),
            packager=MagicMock(),
            default_slurm_env=slurm_env,
            default_local_env=local_env,
            test_level=0,  # lower than pipeline's test_level=2
            base_dir=str(tmp_path),
        )

        # Experiment should not be created for skipped pipelines
        mock_exp.assert_not_called()

    @patch("core.run.Experiment")
    @patch("core.build_docker_executor")
    def test_skipped_pipeline_not_run(self, mock_docker, mock_exp, tmp_path):
        from core import SandboxPipeline, SandboxTask0, get_default_env, run_jobs

        slurm_env, local_env = get_default_env()

        t0 = SandboxTask0(script="test.sh", slurm_config=MagicMock())
        pipeline = SandboxPipeline(task_0=t0, skip=True)

        run_jobs(
            job_table={"job": pipeline},
            hf_local="/tmp/hf",
            user="u",
            identity=None,
            job_dir=str(tmp_path),
            packager=MagicMock(),
            default_slurm_env=slurm_env,
            default_local_env=local_env,
            base_dir=str(tmp_path),
        )

        mock_exp.assert_not_called()

    @patch("core.run.Experiment")
    @patch("core.build_docker_executor")
    def test_detach_flag_passed_to_experiment(self, mock_docker, mock_exp, tmp_path):
        from core import SandboxPipeline, SandboxTask0, get_default_env, run_jobs

        mock_exp_inst = MagicMock()
        mock_exp_inst._id = "exp_detach"
        mock_exp_inst.__enter__ = MagicMock(return_value=mock_exp_inst)
        mock_exp_inst.__exit__ = MagicMock(return_value=False)
        mock_exp.return_value = mock_exp_inst
        mock_docker.return_value = MagicMock()

        slurm_env, local_env = get_default_env()

        t0 = SandboxTask0(script="test.sh", slurm_config=MagicMock())
        pipeline = SandboxPipeline(task_0=t0)

        run_jobs(
            job_table={"job": pipeline},
            hf_local="/tmp/hf",
            user="u",
            identity=None,
            job_dir=str(tmp_path),
            packager=MagicMock(),
            default_slurm_env=slurm_env,
            default_local_env=local_env,
            detach=True,
            base_dir=str(tmp_path),
        )

        mock_exp_inst.run.assert_called_once_with(detach=True)

    @patch("core.run.Experiment")
    @patch("core.build_docker_executor")
    def test_version_report_called(self, mock_docker, mock_exp, tmp_path, capsys):
        from core import SandboxPipeline, SandboxTask0, get_default_env, run_jobs

        mock_exp_inst = MagicMock()
        mock_exp_inst._id = "exp_ver"
        mock_exp_inst.__enter__ = MagicMock(return_value=mock_exp_inst)
        mock_exp_inst.__exit__ = MagicMock(return_value=False)
        mock_exp.return_value = mock_exp_inst
        mock_docker.return_value = MagicMock()

        slurm_env, local_env = get_default_env()

        t0 = SandboxTask0(script="test.sh", slurm_config=MagicMock())
        pipeline = SandboxPipeline(task_0=t0)

        run_jobs(
            job_table={"job": pipeline},
            hf_local="/tmp/hf",
            user="u",
            identity=None,
            job_dir=str(tmp_path),
            packager=MagicMock(),
            default_slurm_env=slurm_env,
            default_local_env=local_env,
            base_dir=str(tmp_path),
        )

        captured = capsys.readouterr()
        assert "Version Report" in captured.out
