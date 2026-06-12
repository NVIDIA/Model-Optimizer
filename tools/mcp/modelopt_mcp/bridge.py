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

"""Thin Python bridge between the MCP tool layer and the launcher's ``core.py`` orchestrator.

Responsibilities, in order of how the MCP tools call in:

1. **List**: enumerate launcher example YAMLs at
   ``tools/launcher/examples/`` with metadata extracted from each YAML's
   top-level fields.
2. **Verify**: probe a target executor (docker or slurm) is reachable.
3. **Submit**: invoke the launcher's ``core.run_jobs`` for a single
   launcher-format YAML. Returns immediately with the experiment id
   (Docker mode spawns a background thread; Slurm mode uses
   ``detach=True``).
4. **Status / Logs**: read nemo_run's experiment dir directly.

This module deliberately doesn't expose anything the MCP tools don't
need — keeps the surface area auditable.
"""

from __future__ import annotations

import os
import shlex
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path

import yaml

# Locate the bundled launcher examples relative to THIS package's install
# path. Works for both editable installs (../launcher/examples/) and
# uvx-from-git installs (the launcher is a sibling site-packages install).
_THIS_DIR = Path(__file__).resolve().parent


def _find_launcher_examples_dir() -> Path | None:
    """Resolve the launcher examples directory.

    Strategy (in order):
    1. ``MODELOPT_LAUNCHER_EXAMPLES_DIR`` env override — for tests + ad-hoc
       relocations.
    2. ``../../launcher/examples/`` from this file — the in-repo layout
       when running from a Model-Optimizer clone (this is the dev mode
       AND the uvx-from-git mode, since uvx checks out the whole repo).
    3. Site-packages install: walk back through the modelopt_launcher
       package to find its examples/ — fallback for the case where the
       launcher was pip-installed standalone.

    Returns None if no candidate exists; callers surface that as a
    structured failure rather than blowing up.
    """
    env = os.environ.get("MODELOPT_LAUNCHER_EXAMPLES_DIR")
    if env:
        p = Path(env)
        return p if p.exists() else None

    # In-repo: this file is at tools/mcp/modelopt_mcp/bridge.py;
    # examples are at tools/launcher/examples/.
    candidate = _THIS_DIR.parent.parent / "launcher" / "examples"
    if candidate.exists():
        return candidate

    # Site-packages fallback: the modelopt-launcher package may carry
    # its examples next to its core.py.
    try:
        import modelopt_launcher

        pkg_dir = Path(modelopt_launcher.__file__).resolve().parent
        candidate = pkg_dir / "examples"
        if candidate.exists():
            return candidate
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------
# list_examples
# ---------------------------------------------------------------------------


@dataclass
class ExampleEntry:
    """One bundled launcher example YAML."""

    path: str  # repo-relative path (from launcher/examples/)
    abs_path: str  # absolute path on disk
    model: str | None  # extracted from job_name / task fields
    description: str | None  # first comment block or top-level field


def list_examples_impl() -> dict:
    """Enumerate all .yaml files under tools/launcher/examples/.

    Returns ``{"ok": True, "examples": [...]}`` with one entry per YAML.
    Each entry carries a best-effort ``model`` + ``description`` parsed
    from the YAML — useful for the LLM to pick a relevant example
    without reading every file.
    """
    examples_dir = _find_launcher_examples_dir()
    if examples_dir is None:
        return {
            "ok": False,
            "reason": "examples_dir_not_found",
            "diagnostic": (
                "Could not locate tools/launcher/examples/. Set "
                "MODELOPT_LAUNCHER_EXAMPLES_DIR or run from inside a "
                "Model-Optimizer checkout."
            ),
        }

    entries: list[dict] = []
    for path in sorted(examples_dir.rglob("*.yaml")):
        rel = path.relative_to(examples_dir.parent)  # launcher/examples/...
        entry = ExampleEntry(
            path=str(rel),
            abs_path=str(path),
            model=None,
            description=None,
        )
        # Best-effort parse — examples occasionally have non-standard
        # shapes, don't crash list_examples on a single bad YAML.
        try:
            with open(path) as f:
                doc = yaml.safe_load(f) or {}
            if isinstance(doc, dict):
                # job_name is the most common identifier; "model" may
                # also be a top-level field in some examples.
                entry.model = doc.get("model") or doc.get("base_model") or doc.get("job_name")
                entry.description = doc.get("description")
        except (yaml.YAMLError, OSError):
            pass
        entries.append(
            {
                "path": entry.path,
                "abs_path": entry.abs_path,
                "model": entry.model,
                "description": entry.description,
            }
        )

    return {
        "ok": True,
        "examples_dir": str(examples_dir),
        "count": len(entries),
        "examples": entries,
    }


# ---------------------------------------------------------------------------
# verify_setup
# ---------------------------------------------------------------------------


def verify_docker_setup_impl() -> dict:
    """Probe local Docker daemon + GPU access.

    Two checks:
    1. ``docker info`` exits 0 → daemon is up
    2. ``docker run --rm --gpus all <small-image> nvidia-smi`` exits 0 →
       GPU passthrough works (skipped if NO_GPU_CHECK env is set, for
       CPU-only test environments)
    """
    # Daemon check. Bandit B603/B607 are false positives here: we're
    # invoking the docker CLI by name with a fixed argv list, no
    # shell-interpretation, no untrusted input.
    try:
        proc = subprocess.run(  # nosec B603 B607
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "executor": "docker",
            "reason": "docker_not_installed",
            "diagnostic": "`docker` binary not on PATH.",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "executor": "docker",
            "reason": "docker_daemon_timeout",
            "diagnostic": "`docker info` did not respond within 10s.",
        }
    if proc.returncode != 0:
        return {
            "ok": False,
            "executor": "docker",
            "reason": "docker_daemon_unavailable",
            "diagnostic": (
                f"`docker info` exit={proc.returncode}. stderr: {proc.stderr.strip()[-400:]}"
            ),
        }

    # GPU check (opt-out for CI runners without GPU)
    if os.environ.get("MODELOPT_MCP_SKIP_GPU_CHECK"):
        return {
            "ok": True,
            "executor": "docker",
            "daemon_ok": True,
            "gpu_check_skipped": True,
        }

    # Same B603/B607 false-positive shape as the daemon check above —
    # fixed argv, no shell interpolation, no untrusted input.
    try:
        gpu = subprocess.run(  # nosec B603 B607
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:12.0-base-ubuntu22.04",
                "nvidia-smi",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "executor": "docker",
            "daemon_ok": True,
            "reason": "gpu_check_timeout",
            "diagnostic": "GPU smoketest container did not return in 60s.",
        }
    if gpu.returncode != 0:
        return {
            "ok": False,
            "executor": "docker",
            "daemon_ok": True,
            "reason": "gpu_unavailable",
            "diagnostic": (
                "Docker daemon is up but `--gpus all` + nvidia-smi "
                f"failed. exit={gpu.returncode}. Likely missing the "
                "NVIDIA Container Toolkit; install per "
                "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html. "
                f"stderr: {gpu.stderr.strip()[-400:]}"
            ),
        }
    return {
        "ok": True,
        "executor": "docker",
        "daemon_ok": True,
        "gpu_ok": True,
    }


def verify_slurm_setup_impl(
    cluster_host: str,
    cluster_user: str | None = None,
    identity: str | None = None,
) -> dict:
    """Probe passwordless SSH to a Slurm cluster login node.

    Uses ``ssh -o BatchMode=yes`` (refuses to prompt for password) +
    a 5s connect timeout. Failure means either the cluster is
    unreachable from this host OR key-auth is broken — both are
    actionable diagnostics for the user.
    """
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=5",
    ]
    if identity:
        argv += ["-i", identity]
    target = f"{cluster_user}@{cluster_host}" if cluster_user else cluster_host
    argv += [target, "whoami && hostname"]

    # B603/B607 false positive — `ssh` invoked by name with a controlled
    # argv (BatchMode, ConnectTimeout, identity path, target). No shell.
    try:
        proc = subprocess.run(  # nosec B603 B607
            argv,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "executor": "slurm",
            "cluster_host": cluster_host,
            "reason": "ssh_timeout",
            "diagnostic": (
                f"ssh to {cluster_host} did not respond within 15s. "
                f"Cluster login node unreachable from this host."
            ),
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "executor": "slurm",
            "reason": "ssh_not_installed",
            "diagnostic": "`ssh` binary not on PATH.",
        }
    if proc.returncode != 0:
        return {
            "ok": False,
            "executor": "slurm",
            "cluster_host": cluster_host,
            "cluster_user": cluster_user,
            "identity": identity,
            "reason": "ssh_auth_failed",
            "diagnostic": (
                "ssh -o BatchMode=yes failed — key-auth isn't working "
                "(no password prompts in this mode). Check that the "
                "right identity is loaded into ssh-agent and the "
                "cluster has the public key in ~/.ssh/authorized_keys. "
                f"exit={proc.returncode}. stderr: "
                f"{proc.stderr.strip()[-400:]}"
            ),
        }
    lines = (proc.stdout or "").strip().splitlines()
    return {
        "ok": True,
        "executor": "slurm",
        "cluster_host": cluster_host,
        "cluster_user": cluster_user,
        "whoami": lines[0] if lines else "",
        "remote_hostname": lines[1] if len(lines) > 1 else "",
    }


# ---------------------------------------------------------------------------
# submit_job
# ---------------------------------------------------------------------------


def _normalize_yaml_path(yaml_path: str) -> Path:
    """Resolve a launcher YAML path to an absolute Path.

    Lookup order:
    1. Absolute path — use as-is
    2. Relative to ``MODELOPT_LAUNCHER_EXAMPLES_DIR`` (or its parent)
    3. Relative to cwd

    The double-fallback lets the agent pass either ``examples/Qwen/.../X.yaml``
    or just the absolute path.
    """
    p = Path(yaml_path)
    if p.is_absolute():
        return p
    # Look under examples dir
    examples_dir = _find_launcher_examples_dir()
    if examples_dir is not None:
        candidate = examples_dir / yaml_path
        if candidate.exists():
            return candidate
        candidate = examples_dir.parent / yaml_path
        if candidate.exists():
            return candidate
    # cwd fallback
    return (Path.cwd() / yaml_path).resolve()


def submit_job_impl(
    *,
    yaml_path: str,
    hf_local: str | None,
    cluster_host: str | None,
    cluster_user: str | None,
    identity: str | None,
    job_dir: str | None,
    job_name: str | None,
    extra_overrides: dict[str, str] | None,
    skip_verify: bool,
) -> dict:
    """Submit a launcher YAML.

    Mode is determined by mutually-exclusive args:
      * ``hf_local`` set → Docker (local GPU)
      * ``cluster_host`` set → Slurm (remote SSH)
      * Neither set → error
      * Both set → error

    The actual orchestration is delegated to the launcher's
    ``core.run_jobs``. We don't re-implement nemo_run integration here —
    that lives upstream.
    """
    # ---- Mode resolution -------------------------------------------
    if hf_local and cluster_host:
        return {
            "ok": False,
            "reason": "ambiguous_executor",
            "diagnostic": (
                "Both hf_local (Docker mode) and cluster_host (Slurm "
                "mode) were provided — these are mutually exclusive. "
                "Pass exactly one."
            ),
        }
    if not hf_local and not cluster_host:
        return {
            "ok": False,
            "reason": "no_executor_specified",
            "diagnostic": (
                "Must pass either hf_local=<path> for local Docker mode "
                "or cluster_host=<hostname> for remote Slurm mode."
            ),
        }
    executor = "docker" if hf_local else "slurm"

    # ---- Pre-flight verification -----------------------------------
    if not skip_verify:
        if executor == "docker":
            check = verify_docker_setup_impl()
        else:
            check = verify_slurm_setup_impl(
                cluster_host=cluster_host or "",
                cluster_user=cluster_user,
                identity=identity,
            )
        if not check.get("ok"):
            return {
                "ok": False,
                "reason": "verify_setup_failed",
                "executor": executor,
                "diagnostic": (
                    f"Skipping submission — verify_setup returned "
                    f"ok=false with reason={check.get('reason')!r}. Fix "
                    f"the underlying issue, then retry."
                ),
                "verify_result": check,
            }

    # ---- Resolve the YAML path ------------------------------------
    abs_yaml = _normalize_yaml_path(yaml_path)
    if not abs_yaml.exists():
        return {
            "ok": False,
            "reason": "yaml_not_found",
            "yaml_path": yaml_path,
            "resolved_path": str(abs_yaml),
            "diagnostic": (
                f"YAML not found at {abs_yaml}. Pass a path under "
                f"tools/launcher/examples/ (relative), an absolute path, "
                f"or one of the examples returned by list_examples."
            ),
        }

    # ---- Dispatch to the launcher ---------------------------------
    # Subprocess `uv run launch.py --yaml <abs_yaml> --yes ...` rather
    # than calling core.run_jobs directly in-process. Why subprocess:
    # launch.py's run.cli.entrypoint integration handles arg parsing,
    # NEMORUN_HOME defaulting, and signal handling in ways that are
    # painful to replicate. Phase 2 may move to direct in-process
    # invocation once we've audited those edge cases.
    argv = ["uv", "run", "launch.py", "--yaml", str(abs_yaml), "--yes"]
    if hf_local:
        argv.append(f"hf_local={shlex.quote(hf_local)}")
    else:
        # Slurm mode — pass cluster config knobs as nemo-run overrides.
        argv.append(f"cluster_host={shlex.quote(cluster_host or '')}")
        if cluster_user:
            argv.append(f"user={shlex.quote(cluster_user)}")
        if identity:
            argv.append(f"identity={shlex.quote(identity)}")
        argv.append("detach=true")
    if job_dir:
        argv.append(f"job_dir={shlex.quote(job_dir)}")
    if job_name:
        argv.append(f"job_name={shlex.quote(job_name)}")
    for k, v in (extra_overrides or {}).items():
        argv.append(f"{k}={shlex.quote(str(v))}")

    # Run from the launcher dir so it picks up its own ./core.py etc.
    launcher_dir = _THIS_DIR.parent.parent / "launcher"
    if not launcher_dir.exists():
        return {
            "ok": False,
            "reason": "launcher_dir_not_found",
            "diagnostic": (
                f"Expected tools/launcher/ at {launcher_dir} but it "
                f"doesn't exist. modelopt-mcp must be installed from a "
                f"Model-Optimizer clone or via uvx-from-git."
            ),
        }

    if executor == "docker":
        # Docker mode: spawn in background so we don't block the MCP
        # call. The subprocess writes its experiment dir + status into
        # NEMORUN_HOME; we'll read it back via job_status.
        # B603 false positive — `argv` is a list built by this module
        # from typed parameters, no shell interpretation.
        proc = subprocess.Popen(  # nosec B603
            argv,
            cwd=str(launcher_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Detached: don't wait. The caller polls job_status by
        # experiment_id (derived from job_name or auto-named).
        # Generate a best-effort experiment_id from job_name + timestamp.
        # nemo_run names experiments deterministically as
        # `<title>_<job_name>_<timestamp>`; if the caller didn't provide
        # job_name we can't predict the id ahead of time.
        return {
            "ok": True,
            "executor": "docker",
            "pid": proc.pid,
            "argv": argv,
            "experiment_id": None,  # Phase 2: tail the subprocess output
            # until nemo_run logs the id, then return it
            "spike_note": (
                "Docker mode launched in background. Phase 1: the "
                "experiment_id is None — Phase 2 tails the subprocess "
                "output to capture the id. For now, list experiments "
                "via nemo_run's CLI or check NEMORUN_HOME."
            ),
        }

    # Slurm mode: synchronous call (slurm.py exits quickly after sbatch
    # with detach=true). Capture stdout to parse experiment_id.
    # B603 false positive — same as above; controlled argv list.
    try:
        proc = subprocess.run(  # nosec B603
            argv,
            cwd=str(launcher_dir),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "executor": "slurm",
            "reason": "submission_timeout",
            "diagnostic": (
                "launch.py submission did not return within 5 minutes. "
                f"Partial stdout: "
                f"{(e.stdout or b'').decode(errors='replace')[-400:]}"
            ),
            "argv": argv,
        }

    # `proc` here is the CompletedProcess from subprocess.run with
    # text=True, but mypy's narrowing widens across the Docker-branch
    # Popen assignment above. Coerce explicitly.
    stdout_tail = str(proc.stdout or "")[-2000:]
    stderr_tail = str(proc.stderr or "")[-2000:]

    if proc.returncode != 0:
        return {
            "ok": False,
            "executor": "slurm",
            "reason": "launch_py_failed",
            "exit_code": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "diagnostic": (
                f"launch.py exited with code {proc.returncode}. Common "
                f"causes: SSH publickey rejection, malformed YAML, "
                f"NEMORUN_HOME unset. Inspect stderr_tail."
            ),
            "argv": argv,
        }

    # Best-effort experiment_id parse
    import re as _re

    experiment_id = None
    experiment_dir = None
    slurm_job_id = None
    for m in _re.finditer(
        r"experiment[_\s-]+([a-zA-Z0-9_]+_\d{10,})",
        stdout_tail,
        _re.IGNORECASE,
    ):
        experiment_id = m.group(1)
        break
    for m in _re.finditer(
        r"(/lustre/[^\s]+|/home/[^\s]+)/experiments/[^\s]+",
        stdout_tail,
    ):
        experiment_dir = m.group(0)
        break
    for m in _re.finditer(r"Submitted batch job (\d+)", stdout_tail):
        slurm_job_id = m.group(1)
        break

    return {
        "ok": True,
        "executor": "slurm",
        "experiment_id": experiment_id,
        "experiment_dir": experiment_dir,
        "slurm_job_id": slurm_job_id,
        "exit_code": 0,
        "stdout_tail": stdout_tail,
        "argv": argv,
    }


# ---------------------------------------------------------------------------
# job_status / job_logs — filesystem-based
# ---------------------------------------------------------------------------


def _resolve_experiment_dir(experiment_id: str) -> Path | None:
    """Map an experiment_id to its on-disk directory.

    nemo_run lays experiments out under ``$NEMORUN_HOME/experiments/<id>/``
    by default; ``NEMORUN_HOME`` falls back to cwd. We also check
    ``./experiments/<id>`` directly and ``./local_experiments/<id>``
    (the Docker-mode fallback path).
    """
    candidates = []
    nemorun_home = os.environ.get("NEMORUN_HOME")
    if nemorun_home:
        candidates.append(Path(nemorun_home) / "experiments" / experiment_id)
    candidates.append(Path.cwd() / "experiments" / experiment_id)
    candidates.append(Path.cwd() / "local_experiments" / experiment_id)
    for c in candidates:
        if c.exists():
            return c
    return None


def job_status_impl(experiment_id: str) -> dict:
    """Read filesystem-based status from a nemo_run experiment dir.

    Status resolution:
      * ``_DONE`` file present + no ``status_*.out`` with ``failed`` →
        ``done``
      * ``_DONE`` present + any ``status_*.out`` contains ``failed`` →
        ``failed``
      * No ``_DONE`` + experiment dir exists → ``running``
      * Experiment dir missing → ``unknown`` (with reason)

    Per-task statuses (``status_<task_name>.out``) are also surfaced so
    multi-task pipelines can be inspected.
    """
    exp_dir = _resolve_experiment_dir(experiment_id)
    if exp_dir is None:
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "reason": "experiment_dir_not_found",
            "diagnostic": (
                "Searched NEMORUN_HOME/experiments/, ./experiments/, "
                "./local_experiments/ — no match. Either the id is "
                "wrong or NEMORUN_HOME isn't set the same as it was "
                "at submit time."
            ),
        }

    done_marker = exp_dir / "_DONE"
    task_statuses: dict[str, str] = {}
    any_failed = False
    for status_file in sorted(exp_dir.glob("status_*.out")):
        # Convention: filename is `status_<task_name>.out`, contents
        # are a single word ("succeeded" or "failed").
        task_name = status_file.stem.removeprefix("status_")
        body = status_file.read_text(encoding="utf-8", errors="replace").strip()
        task_statuses[task_name] = body
        if "fail" in body.lower():
            any_failed = True

    if done_marker.exists():
        overall = "failed" if any_failed else "done"
    else:
        overall = "running"

    return {
        "ok": True,
        "experiment_id": experiment_id,
        "experiment_dir": str(exp_dir),
        "status": overall,
        "task_statuses": task_statuses,
        "has_done_marker": done_marker.exists(),
    }


def job_logs_impl(
    experiment_id: str,
    task: str | None,
    tail: int | None,
) -> dict:
    """Read ``log_<task>.out`` files from the experiment dir.

    If ``task`` is None, returns logs for ALL tasks.
    If ``tail`` is set, returns only the last N lines per task.
    """
    exp_dir = _resolve_experiment_dir(experiment_id)
    if exp_dir is None:
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "reason": "experiment_dir_not_found",
        }

    if task is not None:
        log_files = list(exp_dir.glob(f"log_{task}.out"))
        if not log_files:
            return {
                "ok": False,
                "experiment_id": experiment_id,
                "reason": "task_log_not_found",
                "diagnostic": (
                    f"No log_{task}.out under {exp_dir}. Available logs: "
                    f"{[p.name for p in exp_dir.glob('log_*.out')]}"
                ),
            }
    else:
        log_files = sorted(exp_dir.glob("log_*.out"))

    logs: dict[str, str] = {}
    for log_file in log_files:
        task_name = log_file.stem.removeprefix("log_")
        body = log_file.read_text(encoding="utf-8", errors="replace")
        if tail is not None:
            body = "\n".join(body.splitlines()[-tail:])
        logs[task_name] = body

    return {
        "ok": True,
        "experiment_id": experiment_id,
        "experiment_dir": str(exp_dir),
        "logs": logs,
    }
