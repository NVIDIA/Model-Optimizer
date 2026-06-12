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

"""Unit tests for modelopt-mcp's bridge module — subprocess + filesystem interactions mocked."""

from __future__ import annotations

import subprocess

import pytest

# Skip the whole module if mcp / pydantic aren't installed (the [mcp]
# extra is opt-in).
pytest.importorskip("mcp")
pytest.importorskip("pydantic")

from modelopt_mcp import bridge

# ---------------------------------------------------------------------------
# list_examples
# ---------------------------------------------------------------------------


def test_list_examples_returns_structured_metadata(tmp_path, monkeypatch):
    """Drop two YAMLs into a fake examples dir and verify metadata extraction (model, description) and path shape."""
    examples = tmp_path / "examples"
    (examples / "Qwen").mkdir(parents=True)
    (examples / "Qwen" / "ptq.yaml").write_text(
        "job_name: qwen-ptq\nmodel: Qwen/Qwen3-8B\ndescription: PTQ test\n"
    )
    (examples / "moonshotai").mkdir(parents=True)
    (examples / "moonshotai" / "train.yaml").write_text(
        "job_name: kimi-train\nbase_model: moonshotai/Kimi-K2\n"
    )
    monkeypatch.setenv("MODELOPT_LAUNCHER_EXAMPLES_DIR", str(examples))

    result = bridge.list_examples_impl()
    assert result["ok"] is True
    assert result["count"] == 2
    by_model = {e["model"]: e for e in result["examples"]}
    assert "Qwen/Qwen3-8B" in by_model
    assert by_model["Qwen/Qwen3-8B"]["description"] == "PTQ test"
    assert "moonshotai/Kimi-K2" in by_model


def test_list_examples_missing_dir(monkeypatch, tmp_path):
    """When examples dir can't be located, return a structured failure — no exception."""
    monkeypatch.setenv("MODELOPT_LAUNCHER_EXAMPLES_DIR", str(tmp_path / "ghost"))
    result = bridge.list_examples_impl()
    assert result["ok"] is False
    assert result["reason"] == "examples_dir_not_found"


def test_list_examples_tolerates_malformed_yaml(tmp_path, monkeypatch):
    """A single malformed YAML doesn't crash list_examples — it lands with model=None."""
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "good.yaml").write_text("job_name: g\nmodel: ok\n")
    (examples / "bad.yaml").write_text("not: [unbalanced\n")
    monkeypatch.setenv("MODELOPT_LAUNCHER_EXAMPLES_DIR", str(examples))

    result = bridge.list_examples_impl()
    assert result["ok"] is True
    assert result["count"] == 2
    by_path = {e["path"]: e for e in result["examples"]}
    assert any("bad.yaml" in p for p in by_path)
    bad = next(e for e in result["examples"] if "bad.yaml" in e["path"])
    assert bad["model"] is None


# ---------------------------------------------------------------------------
# verify_docker_setup
# ---------------------------------------------------------------------------


def test_verify_docker_daemon_unavailable(monkeypatch):
    """When `docker info` exits non-zero, verify returns docker_daemon_unavailable."""

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("MODELOPT_MCP_SKIP_GPU_CHECK", "1")

    result = bridge.verify_docker_setup_impl()
    assert result["ok"] is False
    assert result["reason"] == "docker_daemon_unavailable"


def test_verify_docker_daemon_not_installed(monkeypatch):
    """When `docker` is not on PATH, verify returns docker_not_installed."""

    def fake_run(argv, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bridge.verify_docker_setup_impl()
    assert result["ok"] is False
    assert result["reason"] == "docker_not_installed"


def test_verify_docker_skip_gpu_when_env_set(monkeypatch):
    """MODELOPT_MCP_SKIP_GPU_CHECK lets CI hosts without GPUs report ok after the daemon check passes."""

    def fake_run(argv, **kwargs):
        # Daemon check passes; GPU check is skipped — so only one call.
        assert argv[:2] == ["docker", "info"], f"only `docker info` should run; got {argv}"
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("MODELOPT_MCP_SKIP_GPU_CHECK", "1")
    result = bridge.verify_docker_setup_impl()
    assert result["ok"] is True
    assert result["gpu_check_skipped"] is True


def test_verify_docker_gpu_unavailable(monkeypatch):
    """GPU passthrough container exits non-zero → gpu_unavailable + install-toolkit pointer."""
    call_count = {"n": 0}

    def fake_run(argv, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Daemon check: ok
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="",
                stderr="",
            )
        # GPU check: failed
        return subprocess.CompletedProcess(
            args=argv,
            returncode=125,
            stdout="",
            stderr='could not select device driver "" with capabilities: [[gpu]]',
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.delenv("MODELOPT_MCP_SKIP_GPU_CHECK", raising=False)
    result = bridge.verify_docker_setup_impl()
    assert result["ok"] is False
    assert result["reason"] == "gpu_unavailable"
    assert "NVIDIA Container Toolkit" in result["diagnostic"]


# ---------------------------------------------------------------------------
# verify_slurm_setup
# ---------------------------------------------------------------------------


def test_verify_slurm_ssh_success(monkeypatch):
    """Mocked ssh probe returns whoami + hostname; verify returns ok."""

    def fake_run(argv, **kwargs):
        assert argv[0] == "ssh"
        assert "BatchMode=yes" in argv
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="chenhany\ncluster-login-01\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bridge.verify_slurm_setup_impl(
        cluster_host="cw-dfw-cs-001-login-01.nvidia.com",
        cluster_user="chenhany",
    )
    assert result["ok"] is True
    assert result["whoami"] == "chenhany"
    assert result["remote_hostname"] == "cluster-login-01"


def test_verify_slurm_auth_failed(monkeypatch):
    """Ssh -o BatchMode=yes exit 255 → ssh_auth_failed with diagnostic."""

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=255,
            stdout="",
            stderr="Permission denied (publickey).",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bridge.verify_slurm_setup_impl(
        cluster_host="ghost-cluster.nvidia.com",
    )
    assert result["ok"] is False
    assert result["reason"] == "ssh_auth_failed"


# ---------------------------------------------------------------------------
# submit_job mode resolution
# ---------------------------------------------------------------------------


def test_submit_job_rejects_no_executor():
    """Neither hf_local nor cluster_host → no_executor_specified."""
    result = bridge.submit_job_impl(
        yaml_path="examples/test.yaml",
        hf_local=None,
        cluster_host=None,
        cluster_user=None,
        identity=None,
        job_dir=None,
        job_name=None,
        extra_overrides=None,
        skip_verify=True,
    )
    assert result["ok"] is False
    assert result["reason"] == "no_executor_specified"


def test_submit_job_rejects_both_executors():
    """Both hf_local AND cluster_host → ambiguous_executor."""
    result = bridge.submit_job_impl(
        yaml_path="examples/test.yaml",
        hf_local="/tmp/hf",
        cluster_host="cluster.nvidia.com",
        cluster_user=None,
        identity=None,
        job_dir=None,
        job_name=None,
        extra_overrides=None,
        skip_verify=True,
    )
    assert result["ok"] is False
    assert result["reason"] == "ambiguous_executor"


def test_submit_job_yaml_not_found(monkeypatch, tmp_path):
    """yaml_path that doesn't resolve to an existing file → yaml_not_found."""
    monkeypatch.setenv("MODELOPT_LAUNCHER_EXAMPLES_DIR", str(tmp_path))
    result = bridge.submit_job_impl(
        yaml_path="does/not/exist.yaml",
        hf_local="/tmp/hf",
        cluster_host=None,
        cluster_user=None,
        identity=None,
        job_dir=None,
        job_name=None,
        extra_overrides=None,
        skip_verify=True,
    )
    assert result["ok"] is False
    assert result["reason"] == "yaml_not_found"


# ---------------------------------------------------------------------------
# job_status / job_logs — filesystem-based
# ---------------------------------------------------------------------------


def test_job_status_done_success(tmp_path, monkeypatch):
    """_DONE marker + all task statuses succeeded → status='done'."""
    exp = tmp_path / "experiments" / "exp_1781000000"
    exp.mkdir(parents=True)
    (exp / "_DONE").touch()
    (exp / "status_task_0.out").write_text("succeeded\n")
    (exp / "status_task_1.out").write_text("succeeded\n")
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))

    result = bridge.job_status_impl("exp_1781000000")
    assert result["ok"] is True
    assert result["status"] == "done"
    assert result["task_statuses"] == {"task_0": "succeeded", "task_1": "succeeded"}


def test_job_status_failed_task(tmp_path, monkeypatch):
    """_DONE marker + at least one task status contains 'fail' → status='failed'."""
    exp = tmp_path / "experiments" / "exp_1781000001"
    exp.mkdir(parents=True)
    (exp / "_DONE").touch()
    (exp / "status_task_0.out").write_text("succeeded\n")
    (exp / "status_task_1.out").write_text("failed (rc=1)\n")
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))

    result = bridge.job_status_impl("exp_1781000001")
    assert result["ok"] is True
    assert result["status"] == "failed"
    assert "failed" in result["task_statuses"]["task_1"]


def test_job_status_running(tmp_path, monkeypatch):
    """No _DONE marker → running."""
    exp = tmp_path / "experiments" / "exp_1781000002"
    exp.mkdir(parents=True)
    (exp / "status_task_0.out").write_text("running\n")
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))

    result = bridge.job_status_impl("exp_1781000002")
    assert result["ok"] is True
    assert result["status"] == "running"
    assert result["has_done_marker"] is False


def test_job_status_unknown_id(tmp_path, monkeypatch):
    """No experiment dir matching the id → experiment_dir_not_found."""
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))
    result = bridge.job_status_impl("does_not_exist")
    assert result["ok"] is False
    assert result["reason"] == "experiment_dir_not_found"


def test_job_logs_all_tasks(tmp_path, monkeypatch):
    """task=None returns logs for every log_*.out under the experiment dir."""
    exp = tmp_path / "experiments" / "exp_1781000003"
    exp.mkdir(parents=True)
    (exp / "log_task_0.out").write_text("hello\nworld\n")
    (exp / "log_task_1.out").write_text("done\n")
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))

    result = bridge.job_logs_impl("exp_1781000003", task=None, tail=None)
    assert result["ok"] is True
    assert set(result["logs"].keys()) == {"task_0", "task_1"}
    assert "hello" in result["logs"]["task_0"]


def test_job_logs_with_tail(tmp_path, monkeypatch):
    """tail=N returns only the last N lines per task."""
    exp = tmp_path / "experiments" / "exp_1781000004"
    exp.mkdir(parents=True)
    (exp / "log_task_0.out").write_text("line1\nline2\nline3\nline4\n")
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))

    result = bridge.job_logs_impl("exp_1781000004", task="task_0", tail=2)
    assert result["ok"] is True
    body = result["logs"]["task_0"]
    assert body.splitlines() == ["line3", "line4"]


def test_job_logs_missing_task(tmp_path, monkeypatch):
    """Requested task name has no log file → task_log_not_found."""
    exp = tmp_path / "experiments" / "exp_1781000005"
    exp.mkdir(parents=True)
    (exp / "log_task_0.out").write_text("only task 0\n")
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))

    result = bridge.job_logs_impl("exp_1781000005", task="task_99", tail=None)
    assert result["ok"] is False
    assert result["reason"] == "task_log_not_found"
