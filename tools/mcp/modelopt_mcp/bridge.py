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
import re
import subprocess  # nosec B404
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# Locate the bundled launcher examples relative to THIS package's install
# path. Works for both editable installs (../launcher/examples/) and
# uvx-from-git installs (the launcher is a sibling site-packages install).
_THIS_DIR = Path(__file__).resolve().parent

# Canonical task-status failure tokens — matched against the FIRST word
# of each ``status_<task>.out`` file by ``job_status_impl``.
_STATUS_FAILURE_WORDS: frozenset[str] = frozenset(
    {"failed", "error", "errored", "cancelled", "canceled"}
)


def _find_launcher_dir() -> Path | None:
    """Resolve the modelopt launcher's directory.

    Tries, in order:

    1. ``$MODELOPT_LAUNCHER_DIR`` env override — the deterministic
       path agents/operators can set when the package layout doesn't
       match the in-repo expectation.
    2. ``_THIS_DIR.parent.parent / "launcher"`` — the in-repo layout
       (``tools/mcp/modelopt_mcp/bridge.py`` → ``tools/launcher/``).
       Works in dev installs (``pip install -e tools/mcp``) and in
       direct ``Model-Optimizer`` clones.
    3. Walk up from ``os.getcwd()`` looking for
       ``modules/Model-Optimizer/tools/launcher/`` (intern-agent
       workspace layout) or ``tools/launcher/`` (direct
       Model-Optimizer checkout) at each ancestor. Stops at the
       filesystem root.

    Returns ``None`` if no candidate resolves to an existing dir.
    Callers surface that as a structured ``launcher_dir_not_found``
    failure with the searched paths in the diagnostic.

    Empirically: when modelopt-mcp is installed via ``uv tool install``
    (intern-agent's CI install pattern, MR !226), ``_THIS_DIR`` lives
    inside ``~/.local/share/uv/tools/modelopt-mcp/lib/.../site-packages/``
    and step 2's parent-walk doesn't find the launcher. Step 3 (cwd
    walk-up) handles that case — the agent's CWD is always inside
    its cloned nmm-sandbox workspace where ``modules/Model-Optimizer/
    tools/launcher/`` does exist.
    """
    env = os.environ.get("MODELOPT_LAUNCHER_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # In-repo layout (dev install / direct clone)
    candidate = _THIS_DIR.parent.parent / "launcher"
    if candidate.exists():
        return candidate

    # cwd walk-up (uv-tool-install + agent workspace layout)
    cwd = Path.cwd().resolve()
    for ancestor in (cwd, *cwd.parents):
        for rel in ("modules/Model-Optimizer/tools/launcher", "tools/launcher"):
            candidate = ancestor / rel
            if candidate.exists():
                return candidate

    return None


def _launcher_dir_not_found_response(*, dry_run: bool = False) -> dict:
    """Structured failure when ``_find_launcher_dir()`` returns None.

    Centralized so the five callsites that need the launcher dir
    return a consistent diagnostic listing the searched paths.
    """
    env_path = os.environ.get("MODELOPT_LAUNCHER_DIR") or "(unset)"
    in_repo = _THIS_DIR.parent.parent / "launcher"
    resp: dict = {
        "ok": False,
        "reason": "launcher_dir_not_found",
        "diagnostic": (
            "Could not locate tools/launcher/. Searched:\n"
            f"  1. $MODELOPT_LAUNCHER_DIR={env_path}\n"
            f"  2. in-repo layout: {in_repo} (exists={in_repo.exists()})\n"
            f"  3. cwd walk-up from {Path.cwd().resolve()} looking for "
            "modules/Model-Optimizer/tools/launcher or tools/launcher\n"
            "Fix: set $MODELOPT_LAUNCHER_DIR to the absolute path of your "
            "Model-Optimizer checkout's tools/launcher/, or run modelopt-mcp "
            "from inside such a checkout."
        ),
    }
    if dry_run:
        resp["dry_run"] = True
    return resp


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
        # Derive a model identifier from the path layout first
        # (`examples/<family>/<model>/<task>.yaml`). The launcher's
        # bundled examples don't carry top-level `model` / `description`
        # fields — only `job_name` — so path-derivation gives the LLM
        # useful routing metadata even when the YAML body says nothing.
        parts = rel.parts  # ('examples', <family>, <model>, <file>) typically
        path_model = f"{parts[1]}/{parts[2]}" if len(parts) >= 4 else None
        entry = ExampleEntry(
            path=str(rel),
            abs_path=str(path),
            model=path_model,
            description=None,
        )
        # Best-effort YAML body parse — prefer body-supplied fields over
        # the path-derived defaults when present. Don't crash on a
        # malformed YAML.
        try:
            with open(path) as f:
                doc = yaml.safe_load(f) or {}
            if isinstance(doc, dict):
                body_model = doc.get("model") or doc.get("base_model") or doc.get("job_name")
                if body_model:
                    entry.model = body_model
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

    # GPU passthrough probe. Earlier versions of this code ran
    # `docker run --gpus all nvidia/cuda:12.0-base nvidia-smi`, which
    # pulled a ~150 MB CUDA image on first invocation and blew past
    # the 60s timeout on healthy hosts that simply hadn't cached it
    # yet (a real PR review finding — see
    # https://github.com/NVIDIA/Model-Optimizer/pull/1701).
    #
    # Replacement: ask the Docker daemon directly whether the NVIDIA
    # runtime is registered via `docker info --format '{{json .}}'`.
    # No image pull, no container run; the daemon already knows
    # whether the NVIDIA Container Toolkit registered "nvidia" as a
    # runtime when nvidia-ctk runtime configure was last invoked.
    # B603/B607 same false-positive shape as daemon check.
    try:
        gpu = subprocess.run(  # nosec B603 B607
            ["docker", "info", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "executor": "docker",
            "daemon_ok": True,
            "reason": "gpu_check_timeout",
            "diagnostic": "`docker info --format` did not return in 10s.",
        }
    runtimes: list[str] = []
    if gpu.returncode == 0 and gpu.stdout.strip():
        try:
            import json as _json

            info = _json.loads(gpu.stdout)
            runtimes = list((info.get("Runtimes") or {}).keys())
        except (ValueError, AttributeError):
            runtimes = []
    if "nvidia" not in runtimes:
        return {
            "ok": False,
            "executor": "docker",
            "daemon_ok": True,
            "reason": "gpu_unavailable",
            "diagnostic": (
                "Docker daemon is up but the `nvidia` runtime is not "
                "registered. Install the NVIDIA Container Toolkit + "
                "register the runtime: "
                "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html. "
                f"Registered runtimes: {runtimes!r}."
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
    dry_run: bool = False,
) -> dict:
    """Submit a launcher YAML.

    Mode is determined by mutually-exclusive args:
      * ``hf_local`` set → Docker (local GPU)
      * ``cluster_host`` set → Slurm (remote SSH)
      * Neither set → error (unless ``dry_run=True``)
      * Both set → error

    When ``dry_run=True``, the launcher is invoked with ``--dryrun`` —
    the YAML is parsed and validated but no cluster contact / no
    container spawn / no sbatch happens. ``hf_local`` and
    ``cluster_host`` are optional in dry-run mode (pass one to validate
    that the YAML's executor-specific config compiles for the intended
    target; omit both to validate just the YAML shape). ``verify_setup``
    is skipped automatically — there's nothing to talk to.

    The actual orchestration is delegated to the launcher's
    ``core.run_jobs``. We don't re-implement nemo_run integration here —
    that lives upstream.
    """
    # ---- Dry-run branch (no cluster contact) -----------------------
    if dry_run:
        return _submit_job_dry_run(
            yaml_path=yaml_path,
            hf_local=hf_local,
            cluster_host=cluster_host,
            cluster_user=cluster_user,
            identity=identity,
            job_dir=job_dir,
            job_name=job_name,
            extra_overrides=extra_overrides,
        )

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
    # Build argv WITHOUT shell-quoting values — subprocess.run/Popen with a
    # list never goes through a shell, so quoting bakes literal quote chars
    # into the values that nemo-run's CLI parser sees. Verbatim values
    # carry spaces / special chars safely.
    argv = ["uv", "run", "launch.py", "--yaml", str(abs_yaml), "--yes"]
    if hf_local:
        argv.append(f"hf_local={hf_local}")
    else:
        # Slurm mode — `launch.py`'s entrypoint does not accept a
        # `cluster_host` arg (see tools/launcher/launch.py:82). The host
        # is sourced via the SLURM_HOST env var, consumed by
        # `slurm_factory(host=os.environ.get("SLURM_HOST", ""))` in
        # tools/launcher/slurm_config.py. Propagate via env, not argv.
        if cluster_user:
            argv.append(f"user={cluster_user}")
        if identity:
            argv.append(f"identity={identity}")
        argv.append("detach=true")
    if job_dir:
        argv.append(f"job_dir={job_dir}")
    if job_name:
        argv.append(f"job_name={job_name}")
    for k, v in (extra_overrides or {}).items():
        argv.append(f"{k}={v}")

    # Run from the launcher dir so it picks up its own ./core.py etc.
    launcher_dir = _find_launcher_dir()
    if launcher_dir is None:
        return _launcher_dir_not_found_response()

    # Propagate env so submit-side and status-side agree on NEMORUN_HOME.
    # Without this, `launch.py` defaults NEMORUN_HOME to its own cwd
    # (tools/launcher/), but `_resolve_experiment_dir` later checks the
    # MCP server's cwd — different paths, so job_status would return
    # experiment_dir_not_found for jobs that actually succeeded.
    child_env = os.environ.copy()
    child_env.setdefault("NEMORUN_HOME", os.getcwd())
    if executor == "slurm":
        # Required for slurm_factory's host default. Verify_setup ran
        # against this same host above (when verify_setup=True), so the
        # value is known good.
        child_env["SLURM_HOST"] = cluster_host or ""

    if executor == "docker":
        # Docker mode: spawn detached. Discard stdout/stderr to /dev/null —
        # leaving them as Popen.PIPE without a reader fills the kernel's
        # ~64 KB pipe buffer and BLOCKS the launcher's next write(), which
        # would hang long-running PTQ jobs forever while the MCP server
        # appears to have "succeeded".
        # `start_new_session=True` detaches from the MCP server's process
        # group so an MCP server restart / SIGINT doesn't SIGHUP the
        # in-flight launcher.
        # B603 false positive — argv is a controlled list built above.
        proc = subprocess.Popen(  # nosec B603
            argv,
            cwd=str(launcher_dir),
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {
            "ok": True,
            "executor": "docker",
            "pid": proc.pid,
            "argv": argv,
            "nemorun_home": child_env["NEMORUN_HOME"],
            "experiment_id": None,  # Phase 2: tail launcher's output
            # via a side-channel log file to capture nemo_run's id
            "spike_note": (
                "Docker mode launched detached. Phase 1: experiment_id "
                "is None — list under $NEMORUN_HOME/experiments/ or use "
                "Phase 2's tail-based id capture."
            ),
        }

    # Slurm mode: synchronous call (launch.py exits quickly after sbatch
    # with detach=true). Capture stdout to parse experiment_id.
    # B603 false positive — argv is a controlled list built above.
    try:
        proc = subprocess.run(  # nosec B603
            argv,
            cwd=str(launcher_dir),
            env=child_env,
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

    # Best-effort experiment_id + dir + slurm_job_id parse. nemo_run's
    # output format may shift across versions; on parse miss, fields
    # come back None and the caller still gets stdout_tail to inspect
    # by hand.
    experiment_id = None
    experiment_dir = None
    slurm_job_id = None
    # nemo_run prints "Entering Experiment <title>_<id> with id: <id>" —
    # match the trailing id directly so we don't have to encode the
    # title prefix or hard-code timestamp width.
    m = re.search(
        r"experiment[_\s-]+id[:\s]+(\S+)",
        stdout_tail,
        re.IGNORECASE,
    )
    if m:
        experiment_id = m.group(1)
    else:
        # Fallback for older nemo_run output that lacked the explicit
        # "id:" label. Accepts any path-safe id token following the
        # word "experiment" — not just timestamp-style.
        m = re.search(
            r"experiment[_\s-]+([A-Za-z0-9_-]+)",
            stdout_tail,
            re.IGNORECASE,
        )
        if m:
            experiment_id = m.group(1)
    # Match any path containing `/experiments/<id>/` — don't anchor on
    # cluster-specific filesystem roots (NVIDIA's /lustre, partner
    # clusters' /scratch / /work / /data / /p / ...).
    m = re.search(r"(\S+/experiments/[^\s/]+)", stdout_tail)
    if m:
        experiment_dir = m.group(1)
    m = re.search(r"Submitted batch job (\d+)", stdout_tail)
    if m:
        slurm_job_id = m.group(1)

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


def _submit_job_dry_run(
    *,
    yaml_path: str,
    hf_local: str | None,
    cluster_host: str | None,
    cluster_user: str | None,
    identity: str | None,
    job_dir: str | None,
    job_name: str | None,
    extra_overrides: dict[str, str] | None,
) -> dict:
    """Validate a launcher YAML by running ``launch.py --dryrun``.

    No cluster contact, no container spawn, no sbatch. Used by
    verify-task workflow stages (deployment_support,
    hidden_state_dump_support, mlm_eval, ...) that just need to confirm
    a YAML compiles before declaring support is ready.

    Returns ``{ok, dry_run: True, validated: bool, diagnostic?: str,
    exit_code: int|None, stdout_tail: str, stderr_tail: str,
    argv: list[str]}``. Never returns ``experiment_id`` or ``pid`` —
    there's nothing to track. ``diagnostic`` is present only on the
    failure / timeout branches (the validated-success branch omits
    it since there's nothing to diagnose).
    """
    # Same path resolution as the live submit, so dry-run and live use
    # exactly the same YAML.
    abs_yaml = _normalize_yaml_path(yaml_path)
    if not abs_yaml.exists():
        return {
            "ok": False,
            "dry_run": True,
            "reason": "yaml_not_found",
            "yaml_path": yaml_path,
            "resolved_path": str(abs_yaml),
            "diagnostic": (
                f"YAML not found at {abs_yaml}. Pass a path under "
                f"tools/launcher/examples/ (relative), an absolute path, "
                f"or one of the examples returned by list_examples."
            ),
        }

    # Build argv — launch.py supports --dryrun as a flag that prevents
    # actual submission while still exercising the YAML loader, factory
    # resolution, and arg parser. Same argv shape as live submit minus
    # `--yes` pairs with `--dryrun` in every launcher CLI example (see
    # `tools/launcher/CLAUDE.md:28` and `:93`, plus `tools/launcher/docs/
    # contributing.md:24`). Without it, nemo_run's `run.cli.entrypoint`
    # blocks on its confirmation prompt — and since we're capturing
    # stdout (no TTY), the prompt would hang until the 60-second
    # timeout fires.
    argv = ["uv", "run", "launch.py", "--yaml", str(abs_yaml), "--dryrun", "--yes"]
    if hf_local:
        argv.append(f"hf_local={hf_local}")
    if cluster_user:
        argv.append(f"user={cluster_user}")
    if identity:
        argv.append(f"identity={identity}")
    if job_dir:
        argv.append(f"job_dir={job_dir}")
    if job_name:
        argv.append(f"job_name={job_name}")
    for k, v in (extra_overrides or {}).items():
        argv.append(f"{k}={v}")

    launcher_dir = _find_launcher_dir()
    if launcher_dir is None:
        return _launcher_dir_not_found_response(dry_run=True)

    # Propagate env so the launcher's factory resolution matches what
    # the live submit would see (mainly: SLURM_HOST for slurm-factory
    # default when cluster_host is set).
    child_env = os.environ.copy()
    child_env.setdefault("NEMORUN_HOME", os.getcwd())
    if cluster_host:
        child_env["SLURM_HOST"] = cluster_host

    # Dry-run is fast (no network, no container) — 60s timeout is
    # generous. Same subprocess invocation shape as the live-submit
    # branch above (line 590): list-form argv, no shell, inherited
    # env. ``argv`` members are string-literal constants
    # ("uv", "run", "launch.py", "--yaml", "--dryrun"), validated
    # filesystem paths (``str(abs_yaml)``, ``str(launcher_dir)``), or
    # key=value override strings sourced from typed MCP-tool args.
    # B603 false-positive matches the precedent in this module's
    # `submit_job_impl` (Popen at line 563 + run at line 590), the
    # verify probes (line 197 + 251), and the SSH probe (line 326).
    try:
        proc = subprocess.run(  # nosec B603
            argv,
            cwd=str(launcher_dir),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "dry_run": True,
            "reason": "dry_run_timeout",
            "exit_code": None,
            "stdout_tail": (e.stdout or b"").decode(errors="replace")[-2000:] if e.stdout else "",
            "stderr_tail": (e.stderr or b"").decode(errors="replace")[-2000:] if e.stderr else "",
            "diagnostic": (
                "launch.py --dryrun did not return within 60 seconds. "
                "This usually means a YAML import / factory resolution "
                "hung."
            ),
            "argv": argv,
        }

    stdout_tail = str(proc.stdout or "")[-2000:]
    stderr_tail = str(proc.stderr or "")[-2000:]

    if proc.returncode != 0:
        return {
            "ok": True,  # The tool itself ran cleanly
            "dry_run": True,
            "validated": False,  # ...but the YAML failed validation
            "exit_code": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "diagnostic": (
                f"launch.py --dryrun rejected the YAML (exit code "
                f"{proc.returncode}). Common reasons: invalid YAML "
                f"syntax, missing required fields, factory function "
                f"not registered, or a referenced file (HF model path, "
                f"container tag) doesn't exist. See stderr_tail for the "
                f"specific error."
            ),
            "argv": argv,
        }

    # Success branch returns the same field set as the failure branch
    # (plus diagnostic-free since there's nothing to diagnose) so the
    # caller can read stderr_tail / exit_code uniformly.
    return {
        "ok": True,
        "dry_run": True,
        "validated": True,
        "exit_code": 0,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "argv": argv,
    }


# ---------------------------------------------------------------------------
# job_status / job_logs — filesystem-based
# ---------------------------------------------------------------------------


def _resolve_experiment_dir(experiment_id: str) -> Path | None:
    """Map an experiment_id to its on-disk directory.

    nemo_run lays experiments out under ``$NEMORUN_HOME/experiments/<id>/``
    by default; ``NEMORUN_HOME`` falls back to cwd. We check several
    candidate roots in order:

    1. ``$NEMORUN_HOME/experiments/`` — what submit_job_impl pins via env.
    2. cwd's ``experiments/`` + ``local_experiments/`` — for operators
       running the MCP server from their own checkout.
    3. The launcher's own ``experiments/`` directory — belt-and-braces
       for the case where the operator didn't set NEMORUN_HOME at all
       AND the MCP server's cwd differs from where launch.py ran.
    """
    candidates = []
    nemorun_home = os.environ.get("NEMORUN_HOME")
    if nemorun_home:
        candidates.append(Path(nemorun_home) / "experiments" / experiment_id)
    candidates.append(Path.cwd() / "experiments" / experiment_id)
    candidates.append(Path.cwd() / "local_experiments" / experiment_id)
    # The launcher's own experiments dir — submit_job_impl uses
    # cwd=str(launcher_dir) for the subprocess, so when NEMORUN_HOME is
    # unset, launch.py defaults to launcher_dir/experiments/. If the
    # launcher dir can't be resolved (uv-tool-install without an
    # override + the agent's cwd doesn't see a launcher checkout),
    # we skip this fallback rather than crashing — the env-vs-cwd
    # candidates above still cover the common cases.
    launcher_dir = _find_launcher_dir()
    if launcher_dir is not None:
        candidates.append(launcher_dir / "experiments" / experiment_id)
        candidates.append(launcher_dir / "local_experiments" / experiment_id)
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
        task_name = status_file.stem.removeprefix("status_")
        body = status_file.read_text(encoding="utf-8", errors="replace").strip()
        task_statuses[task_name] = body
        # Anchor on the FIRST word of the status file. Anchoring this way
        # (instead of `in body.lower()`) avoids substring false-positives
        # like "succeeded after retry; previous attempt failed" — the
        # canonical convention is a single word but the runner has been
        # observed to append context (e.g. "failed (rc=1)").
        first_word = (body.split() or [""])[0].lower()
        if first_word in _STATUS_FAILURE_WORDS:
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


# ---------------------------------------------------------------------------
# wait_for_experiment — closes the polling loop the agent would write by hand
# ---------------------------------------------------------------------------


def wait_for_experiment_impl(
    experiment_id: str,
    timeout_sec: int,
    poll_interval_sec: int,
) -> dict:
    """Block until ``experiment_id`` reaches a terminal status or the timeout elapses.

    Returns the same dict shape as ``job_status_impl`` plus a
    ``waited_seconds`` field. On timeout, returns
    ``{ok: False, reason: "wait_timeout", last_status: <last poll>}``
    instead of raising — same structured-failure convention as the
    other tools.

    The poll uses ``job_status_impl`` directly (no subprocess shell-out),
    so the resolution rules for finding the experiment dir match
    exactly. If the dir never shows up,
    ``job_status_impl`` returns ``experiment_dir_not_found`` and we
    pass that through immediately rather than spinning to the timeout.
    """
    started = time.monotonic()
    while True:
        status = job_status_impl(experiment_id)
        if not status.get("ok"):
            # Pass through job_status's structured failure (e.g.
            # experiment_dir_not_found) — no point spinning when the
            # dir doesn't exist.
            return {**status, "waited_seconds": time.monotonic() - started}
        if status["status"] in ("done", "failed"):
            return {**status, "waited_seconds": time.monotonic() - started}
        if time.monotonic() - started > timeout_sec:
            return {
                "ok": False,
                "experiment_id": experiment_id,
                "reason": "wait_timeout",
                "diagnostic": (
                    f"Experiment {experiment_id!r} still "
                    f"{status['status']!r} after {timeout_sec}s. "
                    f"Last status: {status}."
                ),
                "last_status": status,
                "waited_seconds": time.monotonic() - started,
            }
        time.sleep(poll_interval_sec)


# ---------------------------------------------------------------------------
# provision_passwordless_ssh_dry_run — operator UX helper
# ---------------------------------------------------------------------------


def provision_passwordless_ssh_dry_run_impl(
    cluster_host: str,
    cluster_user: str | None,
    identity: str | None,
) -> dict:
    """Emit the exact commands to set up passwordless SSH — does NOT execute them.

    Strategy:

    1. Resolve which identity file to inspect (explicit arg → env →
       ``~/.ssh/id_ed25519`` default).
    2. If the private key file is missing, emit a ``ssh-keygen`` command
       and stop. Operator runs it, then re-invokes this tool.
    3. If the private key exists but the public key (``<identity>.pub``)
       is missing, that's an unusual state — surface it as a failure
       with a recovery hint.
    4. If both exist, read the public key + emit the exact
       ``ssh-copy-id`` command pointing at the named cluster. Note
       that this is the only step that needs a password — the operator
       runs it, types the cluster password once, and from then on
       passwordless SSH works.

    Returns ``{ok, step, commands, next_check, ...}``. ``commands`` is
    a list of shell strings the operator should run in order.
    ``next_check`` always recommends calling
    ``verify_setup(executor="slurm", ...)`` after the operator
    runs the commands.
    """
    # Resolve identity
    resolved_identity = (
        identity or os.environ.get("IDENTITY") or str(Path.home() / ".ssh" / "id_ed25519")
    )
    identity_path = Path(resolved_identity).expanduser()
    pubkey_path = Path(f"{identity_path}.pub")
    target = f"{cluster_user}@{cluster_host}" if cluster_user else cluster_host

    next_check_hint = (
        f"After running the commands above, call "
        f"verify_setup(executor='slurm', cluster_host={cluster_host!r}"
        + (f", cluster_user={cluster_user!r}" if cluster_user else "")
        + (f", identity={resolved_identity!r}" if identity else "")
        + ") to confirm key-auth now works."
    )

    if not identity_path.exists():
        # Step 1: keygen needed
        keygen_cmd = (
            f"ssh-keygen -t ed25519 -N '' -f {identity_path} "
            f'-C "{os.environ.get("USER", "user")}@$(hostname)"'
        )
        return {
            "ok": True,
            "step": "keygen_required",
            "identity_path": str(identity_path),
            "commands": [
                keygen_cmd,
                # After keygen, the operator should re-invoke this tool
                # — they'll hit the next branch and get the ssh-copy-id
                # command.
            ],
            "diagnostic": (
                f"No SSH private key at {identity_path}. Run the "
                f"command above, then call this tool again — it will "
                f"emit the ssh-copy-id step to authorize the new key "
                f"on the cluster."
            ),
            "next_step": (
                f"Re-invoke provision_passwordless_ssh_dry_run("
                f"cluster_host={cluster_host!r}"
                + (f", cluster_user={cluster_user!r}" if cluster_user else "")
                + ")"
            ),
        }

    if not pubkey_path.exists():
        return {
            "ok": False,
            "step": "pubkey_missing",
            "identity_path": str(identity_path),
            "pubkey_path": str(pubkey_path),
            "reason": "pubkey_missing",
            "diagnostic": (
                f"Private key exists at {identity_path} but the matching "
                f"public key at {pubkey_path} is missing. This is "
                f"unusual — typically both are produced together by "
                f"ssh-keygen. Recover by regenerating the keypair "
                f"(move {identity_path} aside first if you don't want "
                f"to lose it):\n"
                f"  mv {identity_path} {identity_path}.bak\n"
                f"  ssh-keygen -t ed25519 -N '' -f {identity_path}"
            ),
        }

    # Step 2: ssh-copy-id needed
    pubkey_content = pubkey_path.read_text(encoding="utf-8").strip()
    copy_cmd = f"ssh-copy-id -i {pubkey_path} {target}"
    return {
        "ok": True,
        "step": "ssh_copy_id_required",
        "identity_path": str(identity_path),
        "pubkey_path": str(pubkey_path),
        "pubkey": pubkey_content,
        "commands": [copy_cmd],
        "diagnostic": (
            f"Public key exists at {pubkey_path}. Run the command above "
            f"to authorize it on {target}. ssh-copy-id will prompt for "
            f"the cluster password ONCE — that's the only place a "
            f"password is required. After it succeeds, passwordless "
            f"key-auth is set up."
        ),
        "next_check": next_check_hint,
    }


# ---------------------------------------------------------------------------
# read_cluster_artifact — wraps nemo_run's tunnel
# ---------------------------------------------------------------------------


def read_cluster_artifact_impl(
    experiment_id: str,
    path: str | None,
    job_idx: int,
) -> dict:
    """Read an artifact from a remote experiment via nemo_run's tunnel.

    nemo_run already knows how to talk to the cluster — the executor
    metadata is stored alongside the experiment locally. This tool
    delegates so we don't reinvent the SSH path.

    Two modes:

    * ``path=None`` + ``job_idx=N`` → fetch the job's log via
      ``nemo experiment logs <id> <N>``.
    * ``path="<rel>"`` → relative path inside the experiment dir; we
      use the experiment's executor tunnel to ``cat`` it.

    The tool returns ``{ok, content, ...}`` with the file content as a
    text string (8 KB max — same as the launcher's log_excerpt cap).
    """
    if not path:
        # Mode 1: fetch log via `nemo experiment logs`. Subprocess
        # because the CLI handles tunnel auth and remote-path
        # resolution.
        argv = [
            "uv",
            "run",
            "nemo",
            "experiment",
            "logs",
            experiment_id,
            str(job_idx),
        ]
        launcher_dir = _find_launcher_dir()
        cwd = str(launcher_dir) if launcher_dir is not None else None
        try:
            proc = subprocess.run(  # nosec B603 B607
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "experiment_id": experiment_id,
                "job_idx": job_idx,
                "reason": "logs_fetch_timeout",
                "diagnostic": (
                    f"`nemo experiment logs {experiment_id} {job_idx}` "
                    f"did not return within 60s — tunnel may be slow "
                    f"or unreachable."
                ),
            }
        if proc.returncode != 0:
            return {
                "ok": False,
                "experiment_id": experiment_id,
                "job_idx": job_idx,
                "reason": "logs_fetch_failed",
                "exit_code": proc.returncode,
                "diagnostic": (
                    f"`nemo experiment logs` exited with code "
                    f"{proc.returncode}. stderr: {proc.stderr.strip()[-400:]}"
                ),
            }
        content = (proc.stdout or "")[-8192:]
        return {
            "ok": True,
            "experiment_id": experiment_id,
            "job_idx": job_idx,
            "mode": "logs",
            "content": content,
            "bytes": len(content),
        }

    # Mode 2: arbitrary path via the experiment's tunnel. nemo_run's
    # Experiment loads the executor + tunnel from disk; we rsync the
    # named relative path into a local tmp dir, then read it back.
    try:
        from nemo_run.run.experiment import Experiment
    except ImportError:
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "reason": "nemo_run_not_installed",
            "diagnostic": (
                "nemo_run is not importable from the MCP server's "
                "environment. The arbitrary-path mode requires "
                "nemo_run's tunnel + executor metadata. Install "
                "nemo_run (>=0.8) or use path=None + job_idx=N to "
                "fall back to the `nemo experiment logs` CLI path."
            ),
        }
    try:
        exp = Experiment.from_id(experiment_id)
    except (FileNotFoundError, ValueError) as e:
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "reason": "experiment_not_loadable",
            "diagnostic": (
                f"nemo_run could not load experiment {experiment_id!r}: "
                f"{e}. Check that NEMORUN_HOME points at the same dir "
                f"the submission used."
            ),
        }

    # The executor exposes a tunnel; tunnel.run() runs a remote shell
    # command. Use `cat` for small files; rsync for large ones is a
    # Phase-2.1 follow-up.
    try:
        task = exp.tasks[job_idx]
        executor = task.executor
        tunnel = getattr(executor, "tunnel", None)
    except (IndexError, AttributeError) as e:
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "reason": "no_tunnel",
            "diagnostic": (
                f"Cannot reach the experiment's tunnel: {e}. Local-mode "
                f"experiments don't have a remote tunnel; use "
                f"`read_local_artifact` (Phase 2 follow-up) or read "
                f"the file directly via job_logs / job_status."
            ),
        }
    if tunnel is None:
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "reason": "no_tunnel",
            "diagnostic": "Experiment executor has no tunnel attribute.",
        }

    # Resolve the remote path. The experiment dir on the cluster is
    # exposed via tunnel.job_dir or executor.job_dir; for the launcher's
    # SlurmExecutor, this is set at submit time.
    remote_dir = getattr(executor, "job_dir", None) or getattr(tunnel, "job_dir", None)
    if not remote_dir:
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "reason": "no_remote_job_dir",
            "diagnostic": (
                "Executor / tunnel metadata didn't carry a remote "
                "job_dir. Pass the full remote path as `path` instead "
                "of a relative one."
            ),
        }
    remote_path = path if path.startswith("/") else f"{remote_dir}/{experiment_id}/{path}"

    try:
        result = tunnel.run(f"cat {remote_path}", warn=True)
    except Exception as e:
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "remote_path": remote_path,
            "reason": "tunnel_run_failed",
            "diagnostic": f"tunnel.run failed: {type(e).__name__}: {e}",
        }
    stdout = getattr(result, "stdout", "") or ""
    return {
        "ok": True,
        "experiment_id": experiment_id,
        "remote_path": remote_path,
        "mode": "arbitrary_path",
        "content": stdout[-8192:],
        "bytes": len(stdout),
    }


# ---------------------------------------------------------------------------
# open_draft_pr — wraps `gh pr create --draft`
# ---------------------------------------------------------------------------


def open_draft_pr_impl(
    target_repo: str,
    title: str,
    body: str,
    base_branch: str,
    cwd: str | None,
) -> dict:
    """Open a draft PR on the named target repo from the caller's current branch.

    Preconditions enforced by the agent (NOT this tool):
    * The agent's working tree at ``cwd`` is at the branch it wants to
      PR (created + committed).
    * The branch carries a DCO ``Signed-off-by:`` trailer where the
      target repo requires one (e.g. NVIDIA repos).

    Steps:
    1. ``git push -u origin HEAD`` to publish the branch.
    2. ``gh pr create --draft --title ... --body ... --base ...`` to
       open the draft PR against the target_repo.
    3. Parse the PR URL out of gh's stdout and return it.
    """
    cwd_path = Path(cwd or os.getcwd())
    if not (cwd_path / ".git").exists():
        return {
            "ok": False,
            "reason": "not_a_git_repo",
            "cwd": str(cwd_path),
            "diagnostic": (
                f"{cwd_path} does not look like a git repo (no .git "
                f"directory). Pass an explicit cwd= pointing at the "
                f"checkout where the branch + commit live."
            ),
        }

    # Step 1: push
    try:
        push = subprocess.run(  # nosec B603 B607
            ["git", "push", "-u", "origin", "HEAD"],
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "reason": "git_not_installed",
            "diagnostic": "`git` not on PATH.",
        }
    if push.returncode != 0:
        return {
            "ok": False,
            "reason": "git_push_failed",
            "exit_code": push.returncode,
            "diagnostic": (
                f"git push failed (exit={push.returncode}). stderr: {push.stderr.strip()[-400:]}"
            ),
        }

    # Step 2: gh pr create
    try:
        gh = subprocess.run(  # nosec B603 B607
            [
                "gh",
                "pr",
                "create",
                "--repo",
                target_repo,
                "--draft",
                "--title",
                title,
                "--body",
                body,
                "--base",
                base_branch,
            ],
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "reason": "gh_not_installed",
            "diagnostic": "`gh` (GitHub CLI) not on PATH.",
            "branch_pushed": True,  # branch is published; only PR-open failed
        }
    if gh.returncode != 0:
        return {
            "ok": False,
            "reason": "gh_pr_create_failed",
            "exit_code": gh.returncode,
            "diagnostic": (
                f"gh pr create failed (exit={gh.returncode}). stderr: {gh.stderr.strip()[-400:]}"
            ),
            "branch_pushed": True,
        }

    # Parse the PR URL out of gh's stdout (last URL on stdout is the
    # newly-created PR).
    url_match = re.search(
        r"https://github\.com/[^\s]+/pull/\d+",
        gh.stdout or "",
    )
    pr_url = url_match.group(0) if url_match else None
    return {
        "ok": True,
        "target_repo": target_repo,
        "title": title,
        "base_branch": base_branch,
        "pr_url": pr_url,
        "stdout_tail": (gh.stdout or "")[-400:],
    }
