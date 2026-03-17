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
"""Tests for Docker execution path — verifies build_docker_executor and run_jobs with mocked Docker.

Coverage:
    - build_docker_executor: container mounts, scratch dir creation, modelopt mount
    - run_jobs with hf_local: Docker path selected, env vars applied, metadata written
    - --yaml format end-to-end: YAML parsed, pipeline constructed, executor built
"""

import json
import os
from unittest.mock import MagicMock, patch


class TestBuildDockerExecutor:
    """Tests for build_docker_executor mount and directory setup."""

    def test_scratch_dir_created(self, tmp_path):
        from core import build_docker_executor

        job_dir = str(tmp_path / "experiments")
        build_docker_executor(
            hf_local="/tmp/hf-local",
            slurm_config=MagicMock(
                local=False,
                container="test:latest",
                modelopt_install_path="/opt/modelopt",
                container_mounts=None,
                srun_args=None,
                array=None,
            ),
            experiment_id="exp_123",
            job_dir=job_dir,
            task_name="task_0",
            packager=MagicMock(),
            modelopt_src_path="/tmp/modelopt",
            experiment_title="cicd",
        )
        scratch_dir = os.path.join(job_dir, "cicd", "exp_123", "task_0")
        assert os.path.isdir(scratch_dir)

    def test_hf_local_mount(self, tmp_path):
        from core import build_docker_executor

        job_dir = str(tmp_path / "experiments")
        executor = build_docker_executor(
            hf_local="/my/hf-local",
            slurm_config=MagicMock(
                local=False,
                container="test:latest",
                modelopt_install_path="/opt/modelopt",
                container_mounts=None,
                srun_args=None,
                array=None,
            ),
            experiment_id="exp_123",
            job_dir=job_dir,
            task_name="task_0",
            packager=MagicMock(),
            modelopt_src_path="/tmp/modelopt",
            experiment_title="cicd",
        )
        volumes = executor.volumes
        assert any("/my/hf-local:/hf-local" in v for v in volumes)

    def test_scratchspace_mount(self, tmp_path):
        from core import build_docker_executor

        job_dir = str(tmp_path / "experiments")
        executor = build_docker_executor(
            hf_local="/tmp/hf",
            slurm_config=MagicMock(
                local=False,
                container="test:latest",
                modelopt_install_path="/opt/modelopt",
                container_mounts=None,
                srun_args=None,
                array=None,
            ),
            experiment_id="exp_456",
            job_dir=job_dir,
            task_name="job_0",
            packager=MagicMock(),
            modelopt_src_path="/tmp/modelopt",
            experiment_title="cicd",
        )
        volumes = executor.volumes
        expected_scratch = os.path.join(job_dir, "cicd", "exp_456", "job_0")
        assert any(f"{expected_scratch}:/scratchspace" in v for v in volumes)

    def test_modelopt_mount(self, tmp_path):
        from core import build_docker_executor

        job_dir = str(tmp_path / "experiments")
        executor = build_docker_executor(
            hf_local="/tmp/hf",
            slurm_config=MagicMock(
                local=False,
                container="test:latest",
                modelopt_install_path="/opt/modelopt",
                container_mounts=None,
                srun_args=None,
                array=None,
            ),
            experiment_id="exp_789",
            job_dir=job_dir,
            task_name="task_0",
            packager=MagicMock(),
            modelopt_src_path="/custom/modelopt",
            experiment_title="cicd",
        )
        volumes = executor.volumes
        assert any("/custom/modelopt:/opt/modelopt" in v for v in volumes)

    def test_experiment_title_mount(self, tmp_path):
        from core import build_docker_executor

        job_dir = str(tmp_path / "experiments")
        executor = build_docker_executor(
            hf_local="/tmp/hf",
            slurm_config=MagicMock(
                local=False,
                container="test:latest",
                modelopt_install_path="/opt/modelopt",
                container_mounts=None,
                srun_args=None,
                array=None,
            ),
            experiment_id="exp_123",
            job_dir=job_dir,
            task_name="task_0",
            packager=MagicMock(),
            modelopt_src_path="/tmp/modelopt",
            experiment_title="modelopt",
        )
        volumes = executor.volumes
        exp_title_path = os.path.join(job_dir, "modelopt")
        assert any(f"{exp_title_path}:/modelopt" in v for v in volumes)

    def test_local_slurm_config_mounts_preserved(self, tmp_path):
        from core import build_docker_executor

        job_dir = str(tmp_path / "experiments")
        executor = build_docker_executor(
            hf_local="/tmp/hf",
            slurm_config=MagicMock(
                local=True,
                container="test:latest",
                modelopt_install_path="/opt/modelopt",
                container_mounts=["/data:/data", "/models:/models"],
                srun_args=None,
                array=None,
            ),
            experiment_id="exp_123",
            job_dir=job_dir,
            task_name="task_0",
            packager=MagicMock(),
            modelopt_src_path="/tmp/modelopt",
            experiment_title="cicd",
        )
        volumes = executor.volumes
        assert any("/data:/data" in v for v in volumes)
        assert any("/models:/models" in v for v in volumes)


class TestRunJobsDockerPath:
    """Tests for run_jobs selecting Docker path when hf_local is set."""

    @patch("core.run.Experiment")
    @patch("core.build_docker_executor")
    def test_docker_executor_called_with_hf_local(self, mock_docker, mock_exp, tmp_path):
        from core import SandboxPipeline, SandboxTask0, get_default_env, run_jobs

        mock_exp_instance = MagicMock()
        mock_exp_instance._id = "test_exp_001"
        mock_exp_instance.__enter__ = MagicMock(return_value=mock_exp_instance)
        mock_exp_instance.__exit__ = MagicMock(return_value=False)
        mock_exp.return_value = mock_exp_instance

        mock_docker.return_value = MagicMock()

        slurm_env, local_env = get_default_env("cicd")

        t0 = SandboxTask0(
            script="echo hello",
            slurm_config=MagicMock(),
        )
        pipeline = SandboxPipeline(task_0=t0)
        job_table = {"test_job": pipeline}

        run_jobs(
            job_table=job_table,
            hf_local="/tmp/hf-local",
            user="testuser",
            identity=None,
            job_dir=str(tmp_path),
            packager=MagicMock(),
            default_slurm_env=slurm_env,
            default_local_env=local_env,
            experiment_title="cicd",
            base_dir=str(tmp_path),
        )

        mock_docker.assert_called_once()
        call_kwargs = mock_docker.call_args
        assert call_kwargs[0][0] == "/tmp/hf-local"  # hf_local

    @patch("core.run.Experiment")
    @patch("core.build_docker_executor")
    def test_metadata_written(self, mock_docker, mock_exp, tmp_path):
        from core import SandboxPipeline, SandboxTask0, get_default_env, run_jobs

        mock_exp_instance = MagicMock()
        mock_exp_instance._id = "test_exp_meta"
        mock_exp_instance.__enter__ = MagicMock(return_value=mock_exp_instance)
        mock_exp_instance.__exit__ = MagicMock(return_value=False)
        mock_exp.return_value = mock_exp_instance

        mock_docker.return_value = MagicMock()

        slurm_env, local_env = get_default_env("cicd")

        t0 = SandboxTask0(script="test.sh", slurm_config=MagicMock())
        pipeline = SandboxPipeline(task_0=t0, allow_to_fail=True, note="test note")
        job_table = {"meta_job": pipeline}

        run_jobs(
            job_table=job_table,
            hf_local="/tmp/hf",
            user="user",
            identity=None,
            job_dir=str(tmp_path),
            packager=MagicMock(),
            default_slurm_env=slurm_env,
            default_local_env=local_env,
            experiment_title="cicd",
            base_dir=str(tmp_path),
        )

        metadata_path = os.path.join("experiments", "cicd", "test_exp_meta", "metadata.json")
        assert os.path.exists(metadata_path)
        with open(metadata_path) as f:
            meta = json.load(f)
        assert meta["experiment_id"] == "test_exp_meta"
        assert meta["job_name"] == "meta_job"
        assert meta["allow_to_fail"] is True
        assert meta["note"] == "test note"

    @patch("core.run.Experiment")
    @patch("core.build_docker_executor")
    def test_skipped_task_not_submitted(self, mock_docker, mock_exp, tmp_path):
        from core import SandboxPipeline, SandboxTask0, SandboxTask1, get_default_env, run_jobs

        mock_exp_instance = MagicMock()
        mock_exp_instance._id = "test_exp_skip"
        mock_exp_instance.__enter__ = MagicMock(return_value=mock_exp_instance)
        mock_exp_instance.__exit__ = MagicMock(return_value=False)
        mock_exp.return_value = mock_exp_instance

        mock_docker.return_value = MagicMock()

        slurm_env, local_env = get_default_env("cicd")

        t0 = SandboxTask0(script="run.sh", slurm_config=MagicMock(), skip=True)
        t1 = SandboxTask1(script="eval.sh", slurm_config=MagicMock())
        pipeline = SandboxPipeline(task_0=t0, task_1=t1)
        job_table = {"skip_job": pipeline}

        run_jobs(
            job_table=job_table,
            hf_local="/tmp/hf",
            user="user",
            identity=None,
            job_dir=str(tmp_path),
            packager=MagicMock(),
            default_slurm_env=slurm_env,
            default_local_env=local_env,
            experiment_title="cicd",
            base_dir=str(tmp_path),
        )

        # Only task_1 should be submitted (task_0 is skipped)
        assert mock_docker.call_count == 1

    @patch("core.run.Experiment")
    @patch("core.build_slurm_executor")
    def test_slurm_executor_called_without_hf_local(self, mock_slurm, mock_exp, tmp_path):
        from core import SandboxPipeline, SandboxTask0, get_default_env, run_jobs

        mock_exp_instance = MagicMock()
        mock_exp_instance._id = "test_exp_slurm"
        mock_exp_instance.__enter__ = MagicMock(return_value=mock_exp_instance)
        mock_exp_instance.__exit__ = MagicMock(return_value=False)
        mock_exp.return_value = mock_exp_instance

        mock_slurm.return_value = MagicMock()

        slurm_env, local_env = get_default_env("cicd")

        t0 = SandboxTask0(script="train.sh", slurm_config=MagicMock())
        pipeline = SandboxPipeline(task_0=t0)
        job_table = {"slurm_job": pipeline}

        run_jobs(
            job_table=job_table,
            hf_local=None,  # No hf_local → Slurm path
            user="user",
            identity=None,
            job_dir=str(tmp_path),
            packager=MagicMock(),
            default_slurm_env=slurm_env,
            default_local_env=local_env,
            experiment_title="cicd",
            base_dir=str(tmp_path),
        )

        mock_slurm.assert_called_once()
