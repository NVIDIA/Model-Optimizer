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
"""Tests for build_slurm_executor — container mounts, scratch paths, executor params.

Note: actual SSH tunnel and sbatch submission are not tested (require live infra).
We mock run.SSHTunnel and run.SlurmExecutor to verify the arguments passed.
"""

from unittest.mock import MagicMock, patch

import pytest
from core import ControlMasterSSHTunnel, build_slurm_executor


class TestBuildSlurmExecutor:
    """Tests for build_slurm_executor mount construction and executor params."""

    @patch("core.run.SlurmExecutor")
    @patch("core.ControlMasterSSHTunnel")
    def test_controlmaster_env_selects_controlmaster_tunnel(
        self, mock_tunnel, mock_executor, monkeypatch
    ):
        """MFA clusters reuse OpenSSH ControlMaster instead of Paramiko SSHTunnel."""
        monkeypatch.setenv("MODELOPT_LAUNCHER_SSH_CONTROL_PATH", "/tmp/ptyche.sock")
        mock_tunnel.return_value = MagicMock()

        slurm_config = MagicMock(
            requeue=False,
            host="login-ptyche.nvidia.com",
            port=22,
            account="test_account",
            partition="batch",
            container="ubuntu:24.04",
            modelopt_install_path="/opt/modelopt",
            container_mounts=[],
            srun_args=[],
            nodes=1,
            ntasks_per_node=1,
            gpus_per_node=None,
            array=None,
        )

        build_slurm_executor(
            user="chenhany-mfa",
            identity=None,
            slurm_config=slurm_config,
            experiment_id="exp_001",
            job_dir="/lustre/experiments",
            task_name="job_0",
            packager=MagicMock(),
        )

        mock_tunnel.assert_called_once()
        assert mock_tunnel.call_args.kwargs["host"] == "login-ptyche.nvidia.com"
        assert mock_tunnel.call_args.kwargs["user"] == "chenhany-mfa"
        mock_executor.assert_called_once()


class TestControlMasterSSHTunnel:
    """Tests for the OpenSSH ControlMaster-backed tunnel shim."""

    def test_connect_reuses_live_controlmaster(self, monkeypatch, tmp_path):
        sock = tmp_path / "cm.sock"
        sock.touch()
        monkeypatch.setenv("MODELOPT_LAUNCHER_SSH_CONTROL_PATH", str(sock))

        def fake_run(argv, **kwargs):
            assert argv[:5] == ["ssh", "-p", "22", "-O", "check"]
            assert f"ControlPath={sock}" in argv
            return MagicMock(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("core.subprocess.run", fake_run)

        tunnel = ControlMasterSSHTunnel(
            host="login-ptyche.nvidia.com",
            user="chenhany-mfa",
            job_dir="/tmp/job",
        )
        tunnel.connect()

        assert tunnel.session.is_connected is True
        assert tunnel.session.host == "login-ptyche.nvidia.com"

    def test_connect_requires_reauth_when_controlmaster_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MODELOPT_LAUNCHER_SSH_CONTROL_PATH", str(tmp_path / "missing.sock"))
        monkeypatch.setenv("MODELOPT_LAUNCHER_SSH_RECONNECT_COMMAND", "ssh ptyche")

        tunnel = ControlMasterSSHTunnel(
            host="login-ptyche.nvidia.com",
            user="chenhany-mfa",
            job_dir="/tmp/job",
        )

        with pytest.raises(RuntimeError) as exc_info:
            tunnel.connect()
        assert "mfa_reauth_required" in str(exc_info.value)
        assert "ssh ptyche" in str(exc_info.value)

    @patch("core.run.SlurmExecutor")
    @patch("core.run.SSHTunnel")
    def test_installed_mode_skips_modelopt_source_mounts(self, mock_tunnel, mock_executor):
        mock_tunnel.return_value = MagicMock()

        slurm_config = MagicMock(
            requeue=False,
            host="test-host",
            port=22,
            account="test_account",
            partition="batch",
            container="nvcr.io/test:latest",
            modelopt_install_path="/opt/modelopt",
            container_mounts=["/hf-local:/hf-local"],
            srun_args=["--no-container-mount-home"],
            nodes=1,
            ntasks_per_node=4,
            gpus_per_node=4,
            array=None,
        )

        build_slurm_executor(
            user="testuser",
            identity=None,
            slurm_config=slurm_config,
            experiment_id="exp_001",
            job_dir="/lustre/experiments",
            task_name="job_0",
            packager=MagicMock(),
            experiment_title="cicd",
        )

        # Check SlurmExecutor was called
        mock_executor.assert_called_once()
        call_kwargs = mock_executor.call_args[1]

        # Installed package mode packages only launcher examples/common. Do not
        # mount ModelOpt source overlays that were not packaged into remote code.
        mounts = call_kwargs["container_mounts"]
        assert any("/scratchspace" in m for m in mounts)
        assert any("/cicd" in m for m in mounts)
        assert not any("/opt/modelopt" in m for m in mounts)
        assert not any("modelopt_recipes" in m for m in mounts)
        # Original mount preserved
        assert any("/hf-local:/hf-local" in m for m in mounts)

    @patch("core.run.SlurmExecutor")
    @patch("core.run.SSHTunnel")
    def test_source_mode_adds_modelopt_source_mounts(self, mock_tunnel, mock_executor):
        mock_tunnel.return_value = MagicMock()

        slurm_config = MagicMock(
            requeue=False,
            host="test-host",
            port=22,
            account="test_account",
            partition="batch",
            container="nvcr.io/test:latest",
            modelopt_install_path="/opt/modelopt",
            container_mounts=[],
            srun_args=["--no-container-mount-home"],
            nodes=1,
            ntasks_per_node=4,
            gpus_per_node=4,
            array=None,
        )

        build_slurm_executor(
            user="testuser",
            identity=None,
            slurm_config=slurm_config,
            experiment_id="exp_001",
            job_dir="/lustre/experiments",
            task_name="job_0",
            packager=MagicMock(),
            modelopt_src_path="/checkout/modelopt",
            experiment_title="cicd",
        )

        mounts = mock_executor.call_args[1]["container_mounts"]
        assert any(
            "/lustre/experiments/cicd/exp_001/job_0/code/modules/Model-Optimizer/modelopt:"
            "/opt/modelopt" in m
            for m in mounts
        )
        assert any(
            "/lustre/experiments/cicd/exp_001/job_0/code/modules/Model-Optimizer/"
            "modelopt_recipes:/opt/modelopt_recipes" in m
            for m in mounts
        )

    @patch("core.run.SlurmExecutor")
    @patch("core.run.SSHTunnel")
    def test_scratch_path_uses_experiment_title(self, mock_tunnel, mock_executor):
        mock_tunnel.return_value = MagicMock()

        slurm_config = MagicMock(
            requeue=False,
            host="host",
            port=22,
            account="acct",
            partition="batch",
            container="img",
            modelopt_install_path="/opt/mo",
            container_mounts=[],
            srun_args=[],
            nodes=1,
            ntasks_per_node=1,
            gpus_per_node=1,
            array=None,
        )

        build_slurm_executor(
            user="u",
            identity=None,
            slurm_config=slurm_config,
            experiment_id="exp_xyz",
            job_dir="/data",
            task_name="task_0",
            packager=MagicMock(),
            experiment_title="modelopt",
        )

        mounts = mock_executor.call_args[1]["container_mounts"]
        assert any("/data/modelopt/exp_xyz:/scratchspace" in m for m in mounts)
        assert any("/data/modelopt:/modelopt" in m for m in mounts)

    @patch("core.run.SlurmExecutor")
    @patch("core.run.SSHTunnel")
    def test_tunnel_created_with_correct_params(self, mock_tunnel, mock_executor):
        mock_tunnel.return_value = MagicMock()

        slurm_config = MagicMock(
            requeue=False,
            host="login.cluster.com",
            port=30022,
            account="acct",
            partition="batch",
            container="img",
            modelopt_install_path="/opt/mo",
            container_mounts=[],
            srun_args=[],
            nodes=1,
            ntasks_per_node=1,
            gpus_per_node=1,
            array=None,
        )

        build_slurm_executor(
            user="myuser",
            identity="/home/.ssh/id_rsa",
            slurm_config=slurm_config,
            experiment_id="exp_1",
            job_dir="/job",
            task_name="t0",
            packager=MagicMock(),
        )

        mock_tunnel.assert_called_once()
        tunnel_kwargs = mock_tunnel.call_args[1]
        assert tunnel_kwargs["host"] == "login.cluster.com"
        assert tunnel_kwargs["user"] == "myuser"
        assert tunnel_kwargs["port"] == 30022
        assert tunnel_kwargs["identity"] == "/home/.ssh/id_rsa"
        assert tunnel_kwargs["job_dir"] == "/job"

    @patch("core.run.SlurmExecutor")
    @patch("core.run.SSHTunnel")
    def test_executor_params(self, mock_tunnel, mock_executor):
        mock_tunnel.return_value = MagicMock()

        slurm_config = MagicMock(
            requeue=False,
            host="h",
            port=22,
            account="my_acct",
            partition="gpu",
            container="nvcr.io/img:v1",
            modelopt_install_path="/opt/mo",
            container_mounts=[],
            srun_args=["--mpi=pmix"],
            nodes=2,
            ntasks_per_node=8,
            gpus_per_node=8,
            array="0-3",
            time="04:00:00",
            mem="128G",
        )

        packager = MagicMock()
        build_slurm_executor(
            user="u",
            identity=None,
            slurm_config=slurm_config,
            experiment_id="e1",
            job_dir="/j",
            task_name="t0",
            packager=packager,
        )

        kw = mock_executor.call_args[1]
        assert kw["account"] == "my_acct"
        assert kw["partition"] == "gpu"
        assert kw["nodes"] == 2
        assert kw["ntasks_per_node"] == 8
        assert kw["gpus_per_node"] == 8
        assert kw["container_image"] == "nvcr.io/img:v1"
        assert kw["srun_args"] == ["--mpi=pmix"]
        assert kw["array"] == "0-3"
        assert kw["packager"] is packager
        assert kw["time"] == "04:00:00"
        assert kw["mem"] == "128G"
        assert kw["retries"] == 0

    @patch("core.run.SlurmExecutor")
    @patch("core.run.SSHTunnel")
    def test_default_mem_remains_all_node_memory(self, mock_tunnel, mock_executor):
        mock_tunnel.return_value = MagicMock()

        for mem in ("missing", "", None):
            slurm_config = MagicMock(
                requeue=False,
                host="h",
                port=22,
                account="a",
                partition="b",
                container="c",
                modelopt_install_path="/m",
                container_mounts=[],
                srun_args=[],
                nodes=1,
                ntasks_per_node=1,
                gpus_per_node=1,
                array=None,
            )
            if mem == "missing":
                del slurm_config.mem
            else:
                slurm_config.mem = mem

            build_slurm_executor(
                user="u",
                identity=None,
                slurm_config=slurm_config,
                experiment_id="e",
                job_dir="/j",
                task_name="t",
                packager=MagicMock(),
            )

            assert mock_executor.call_args[1]["mem"] == "0"
            mock_executor.reset_mock()

    @patch("core.run.SlurmExecutor")
    @patch("core.run.SSHTunnel")
    def test_none_container_mounts_handled(self, mock_tunnel, mock_executor):
        mock_tunnel.return_value = MagicMock()

        slurm_config = MagicMock(
            requeue=False,
            host="h",
            port=22,
            account="a",
            partition="b",
            container="c",
            modelopt_install_path="/m",
            container_mounts=None,
            srun_args=None,
            nodes=1,
            ntasks_per_node=1,
            gpus_per_node=1,
            array=None,
        )

        build_slurm_executor(
            user="u",
            identity=None,
            slurm_config=slurm_config,
            experiment_id="e",
            job_dir="/j",
            task_name="t",
            packager=MagicMock(),
        )

        # Should not crash; installed mode still includes scratch + title mounts.
        mounts = mock_executor.call_args[1]["container_mounts"]
        assert "/j/cicd/e:/scratchspace" in mounts
        assert "/j/cicd:/cicd" in mounts

    @patch("core.run.SlurmExecutor")
    @patch("core.run.SSHTunnel")
    def test_requeue_sets_param_and_bumps_retries(self, mock_tunnel, mock_executor):
        mock_tunnel.return_value = MagicMock()
        executor = mock_executor.return_value
        executor.retries = 0
        executor.additional_parameters = {}

        slurm_config = MagicMock(
            requeue=True,
            host="h",
            port=22,
            account="a",
            partition="b",
            container="c",
            modelopt_install_path="/m",
            container_mounts=[],
            srun_args=[],
            nodes=1,
            ntasks_per_node=1,
            gpus_per_node=1,
            array=None,
        )

        build_slurm_executor(
            user="u",
            identity=None,
            slurm_config=slurm_config,
            experiment_id="e",
            job_dir="/j",
            task_name="t",
            packager=MagicMock(),
        )

        # requeue=True flags the additional parameter and bumps retries above 0 so
        # nemo-run's sbatch wrapper actually issues `scontrol requeue` on preemption.
        assert executor.additional_parameters["requeue"] is True
        assert executor.retries == 3
