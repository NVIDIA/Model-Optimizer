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
# submit_job dry-run branch
# ---------------------------------------------------------------------------


def test_submit_job_dry_run_yaml_validates(monkeypatch, tmp_path):
    """dry_run=True with a valid YAML → ok+validated, no cluster contact."""
    yaml_dir = tmp_path / "examples" / "fam" / "model"
    yaml_dir.mkdir(parents=True)
    yaml_path = yaml_dir / "config.yaml"
    yaml_path.write_text("job_name: t\npipeline: []\n")
    monkeypatch.setenv("MODELOPT_LAUNCHER_EXAMPLES_DIR", str(tmp_path / "examples"))

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="Dry-run OK — YAML compiles\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = bridge.submit_job_impl(
        yaml_path="fam/model/config.yaml",
        hf_local=None,
        cluster_host=None,
        cluster_user=None,
        identity=None,
        job_dir=None,
        job_name=None,
        extra_overrides=None,
        skip_verify=True,
        dry_run=True,
    )
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["validated"] is True
    assert "experiment_id" not in result  # dry-run produces no experiment
    assert "--dryrun" in captured["argv"]  # launcher CLI spells it as one word
    assert "--yes" not in captured["argv"]  # dry-run skips --yes


def test_submit_job_dry_run_yaml_invalid(monkeypatch, tmp_path):
    """dry_run=True + launcher rejects YAML → ok=True but validated=False."""
    yaml_dir = tmp_path / "examples"
    yaml_dir.mkdir()
    yaml_path = yaml_dir / "bad.yaml"
    yaml_path.write_text("not: [unbalanced\n")
    monkeypatch.setenv("MODELOPT_LAUNCHER_EXAMPLES_DIR", str(yaml_dir))

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=1,
            stdout="",
            stderr="yaml.YAMLError: while parsing a flow sequence\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = bridge.submit_job_impl(
        yaml_path="bad.yaml",
        hf_local=None,
        cluster_host=None,
        cluster_user=None,
        identity=None,
        job_dir=None,
        job_name=None,
        extra_overrides=None,
        skip_verify=True,
        dry_run=True,
    )
    assert result["ok"] is True  # The TOOL ran cleanly
    assert result["dry_run"] is True
    assert result["validated"] is False  # ...but the YAML failed validation
    assert result["exit_code"] == 1
    assert "yaml.YAMLError" in result["stderr_tail"]


def test_submit_job_dry_run_yaml_not_found(monkeypatch, tmp_path):
    """dry_run=True + missing yaml → yaml_not_found with dry_run flag set."""
    monkeypatch.setenv("MODELOPT_LAUNCHER_EXAMPLES_DIR", str(tmp_path))
    result = bridge.submit_job_impl(
        yaml_path="missing.yaml",
        hf_local=None,
        cluster_host=None,
        cluster_user=None,
        identity=None,
        job_dir=None,
        job_name=None,
        extra_overrides=None,
        skip_verify=True,
        dry_run=True,
    )
    assert result["ok"] is False
    assert result["dry_run"] is True
    assert result["reason"] == "yaml_not_found"


def test_submit_job_dry_run_skips_verify(monkeypatch, tmp_path):
    """dry_run=True bypasses verify_setup even when skip_verify=False."""
    yaml_dir = tmp_path / "examples"
    yaml_dir.mkdir()
    (yaml_dir / "ok.yaml").write_text("job_name: ok\npipeline: []\n")
    monkeypatch.setenv("MODELOPT_LAUNCHER_EXAMPLES_DIR", str(yaml_dir))

    verify_called = {"n": 0}
    real_verify = bridge.verify_slurm_setup_impl

    def fake_verify(**kwargs):
        verify_called["n"] += 1
        return real_verify(**kwargs)

    monkeypatch.setattr(bridge, "verify_slurm_setup_impl", fake_verify)
    monkeypatch.setattr(bridge, "verify_docker_setup_impl", lambda: {"ok": True})

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="dry-run ok\n",
            stderr="",
        ),
    )

    bridge.submit_job_impl(
        yaml_path="ok.yaml",
        hf_local=None,
        cluster_host="cluster.example.com",
        cluster_user="alice",
        identity=None,
        job_dir=None,
        job_name=None,
        extra_overrides=None,
        skip_verify=False,  # would normally trigger verify
        dry_run=True,
    )
    assert verify_called["n"] == 0  # verify_setup never invoked in dry-run


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


# ---------------------------------------------------------------------------
# wait_for_experiment
# ---------------------------------------------------------------------------


def test_wait_for_experiment_returns_terminal_immediately(tmp_path, monkeypatch):
    """If the experiment is already terminal, return without polling."""
    exp = tmp_path / "experiments" / "exp_already_done"
    exp.mkdir(parents=True)
    (exp / "_DONE").touch()
    (exp / "status_task_0.out").write_text("succeeded\n")
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))

    result = bridge.wait_for_experiment_impl(
        "exp_already_done",
        timeout_sec=10,
        poll_interval_sec=1,
    )
    assert result["ok"] is True
    assert result["status"] == "done"
    assert result["waited_seconds"] < 1  # didn't actually wait


def test_wait_for_experiment_polls_until_done(tmp_path, monkeypatch):
    """Spin through running → done."""
    exp = tmp_path / "experiments" / "exp_in_flight"
    exp.mkdir(parents=True)
    (exp / "status_task_0.out").write_text("running\n")
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))

    # Flip the marker after 2 polls via a counter side-effect
    call_count = {"n": 0}
    real_status = bridge.job_status_impl

    def fake_status(experiment_id):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            (exp / "_DONE").touch()
        return real_status(experiment_id)

    monkeypatch.setattr(bridge, "job_status_impl", fake_status)
    result = bridge.wait_for_experiment_impl(
        "exp_in_flight",
        timeout_sec=10,
        poll_interval_sec=0,
    )
    assert result["ok"] is True
    assert result["status"] == "done"
    assert call_count["n"] >= 2


def test_wait_for_experiment_timeout(tmp_path, monkeypatch):
    """Never reaches terminal → wait_timeout with last_status."""
    exp = tmp_path / "experiments" / "exp_stuck"
    exp.mkdir(parents=True)
    (exp / "status_task_0.out").write_text("running\n")
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))

    result = bridge.wait_for_experiment_impl(
        "exp_stuck",
        timeout_sec=1,
        poll_interval_sec=0,
    )
    assert result["ok"] is False
    assert result["reason"] == "wait_timeout"
    assert result["last_status"]["status"] == "running"


def test_wait_for_experiment_passes_through_dir_not_found(tmp_path, monkeypatch):
    """If the experiment dir doesn't exist, don't spin to timeout."""
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))
    result = bridge.wait_for_experiment_impl(
        "does_not_exist",
        timeout_sec=60,
        poll_interval_sec=0,
    )
    assert result["ok"] is False
    assert result["reason"] == "experiment_dir_not_found"
    # Bail out fast — not the full timeout
    assert result["waited_seconds"] < 1


# ---------------------------------------------------------------------------
# provision_passwordless_ssh_dry_run
# ---------------------------------------------------------------------------


def test_provision_ssh_no_key_emits_keygen(tmp_path, monkeypatch):
    """Missing private key → keygen command, step='keygen_required'."""
    fake_home = tmp_path / "home"
    (fake_home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("IDENTITY", raising=False)

    result = bridge.provision_passwordless_ssh_dry_run_impl(
        cluster_host="cw-dfw.example.com",
        cluster_user="alice",
        identity=None,
    )
    assert result["ok"] is True
    assert result["step"] == "keygen_required"
    assert "ssh-keygen" in result["commands"][0]
    assert "ed25519" in result["commands"][0]
    assert result["identity_path"].endswith(".ssh/id_ed25519")


def test_provision_ssh_key_present_emits_copy_id(tmp_path, monkeypatch):
    """Key + pubkey present → ssh-copy-id command + pubkey content."""
    fake_home = tmp_path / "home"
    ssh = fake_home / ".ssh"
    ssh.mkdir(parents=True)
    (ssh / "id_ed25519").write_text("PRIVKEY")
    pubkey_content = "ssh-ed25519 AAAAC3NzaC... alice@host"
    (ssh / "id_ed25519.pub").write_text(pubkey_content + "\n")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("IDENTITY", raising=False)

    result = bridge.provision_passwordless_ssh_dry_run_impl(
        cluster_host="cw-dfw.example.com",
        cluster_user="alice",
        identity=None,
    )
    assert result["ok"] is True
    assert result["step"] == "ssh_copy_id_required"
    assert "ssh-copy-id" in result["commands"][0]
    assert "alice@cw-dfw.example.com" in result["commands"][0]
    assert result["pubkey"] == pubkey_content
    assert "verify_setup" in result["next_check"]


def test_provision_ssh_priv_without_pub_surfaces_failure(tmp_path, monkeypatch):
    """Private key but no .pub → pubkey_missing with recovery hint."""
    fake_home = tmp_path / "home"
    ssh = fake_home / ".ssh"
    ssh.mkdir(parents=True)
    (ssh / "id_ed25519").write_text("PRIVKEY")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("IDENTITY", raising=False)

    result = bridge.provision_passwordless_ssh_dry_run_impl(
        cluster_host="cw-dfw.example.com",
        cluster_user="alice",
        identity=None,
    )
    assert result["ok"] is False
    assert result["reason"] == "pubkey_missing"
    assert "ssh-keygen" in result["diagnostic"]


def test_provision_ssh_explicit_identity_overrides_default(tmp_path, monkeypatch):
    """Explicit identity arg wins over $IDENTITY and ~/.ssh/id_ed25519."""
    explicit = tmp_path / "custom_key"
    explicit.write_text("CUSTOM")
    (tmp_path / "custom_key.pub").write_text("ssh-ed25519 AAAA alice\n")
    monkeypatch.setenv("IDENTITY", "/wrong/path")  # should be ignored

    result = bridge.provision_passwordless_ssh_dry_run_impl(
        cluster_host="cw-dfw.example.com",
        cluster_user="alice",
        identity=str(explicit),
    )
    assert result["ok"] is True
    assert result["identity_path"] == str(explicit)


# ---------------------------------------------------------------------------
# read_cluster_artifact — logs mode (subprocess mocked)
# ---------------------------------------------------------------------------


def test_read_cluster_artifact_logs_mode_ok(monkeypatch):
    """path=None → wraps `nemo experiment logs <id> <job_idx>`."""
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout="line 1\nline 2\nline 3\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bridge.read_cluster_artifact_impl(
        experiment_id="cicd_42",
        path=None,
        job_idx=0,
    )
    assert result["ok"] is True
    assert result["mode"] == "logs"
    assert "line 1" in result["content"]
    # Verify the wrapped command
    assert "nemo" in captured["argv"]
    assert "experiment" in captured["argv"]
    assert "logs" in captured["argv"]
    assert "cicd_42" in captured["argv"]


def test_read_cluster_artifact_logs_mode_subprocess_failed(monkeypatch):
    """Nemo cli non-zero → structured logs_fetch_failed."""

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=1,
            stdout="",
            stderr="experiment cicd_42 not found\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bridge.read_cluster_artifact_impl(
        experiment_id="cicd_42",
        path=None,
        job_idx=0,
    )
    assert result["ok"] is False
    assert result["reason"] == "logs_fetch_failed"
    assert result["exit_code"] == 1


def test_read_cluster_artifact_logs_mode_timeout(monkeypatch):
    """Hanging tunnel → structured logs_fetch_timeout, no exception."""

    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=60)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bridge.read_cluster_artifact_impl(
        experiment_id="cicd_42",
        path=None,
        job_idx=0,
    )
    assert result["ok"] is False
    assert result["reason"] == "logs_fetch_timeout"


# ---------------------------------------------------------------------------
# open_draft_pr — subprocess mocked
# ---------------------------------------------------------------------------


def test_open_draft_pr_happy_path(monkeypatch, tmp_path):
    """Git push ok → gh pr create ok → returns parsed pr_url."""
    (tmp_path / ".git").mkdir()  # pretend it's a git repo

    call_log = []

    def fake_run(argv, **kwargs):
        call_log.append(argv[0])
        if argv[0] == "git":
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="",
                stderr="",
            )
        # gh pr create — return a typical PR URL on stdout
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=(
                "Warning: 1 uncommitted change\n"
                "https://github.com/NVIDIA/Model-Optimizer/pull/9999\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bridge.open_draft_pr_impl(
        target_repo="NVIDIA/Model-Optimizer",
        title="test pr",
        body="body text",
        base_branch="main",
        cwd=str(tmp_path),
    )
    assert result["ok"] is True
    assert result["pr_url"] == "https://github.com/NVIDIA/Model-Optimizer/pull/9999"
    assert call_log == ["git", "gh"]


def test_open_draft_pr_not_a_git_repo(tmp_path):
    """Cwd without .git → structured not_a_git_repo failure, no subprocess."""
    result = bridge.open_draft_pr_impl(
        target_repo="NVIDIA/Model-Optimizer",
        title="x",
        body="x",
        base_branch="main",
        cwd=str(tmp_path),
    )
    assert result["ok"] is False
    assert result["reason"] == "not_a_git_repo"


def test_open_draft_pr_git_push_failed(monkeypatch, tmp_path):
    """Git push non-zero → structured git_push_failed."""
    (tmp_path / ".git").mkdir()

    def fake_run(argv, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(
                args=argv,
                returncode=128,
                stdout="",
                stderr="rejected: non-fast-forward\n",
            )
        pytest.fail(f"gh should not be called when git push fails; got {argv}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bridge.open_draft_pr_impl(
        target_repo="NVIDIA/Model-Optimizer",
        title="x",
        body="x",
        base_branch="main",
        cwd=str(tmp_path),
    )
    assert result["ok"] is False
    assert result["reason"] == "git_push_failed"


def test_open_draft_pr_gh_failed_but_branch_pushed(monkeypatch, tmp_path):
    """Gh pr create non-zero → reports branch_pushed=True so the operator can retry just the PR-open step."""
    (tmp_path / ".git").mkdir()

    def fake_run(argv, **kwargs):
        if argv[0] == "git":
            return subprocess.CompletedProcess(
                args=argv,
                returncode=0,
                stdout="",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=argv,
            returncode=1,
            stdout="",
            stderr="resource not found: NVIDIA/Model-Optimizer\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bridge.open_draft_pr_impl(
        target_repo="NVIDIA/Model-Optimizer",
        title="x",
        body="x",
        base_branch="main",
        cwd=str(tmp_path),
    )
    assert result["ok"] is False
    assert result["reason"] == "gh_pr_create_failed"
    assert result["branch_pushed"] is True
