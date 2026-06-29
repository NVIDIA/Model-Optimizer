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

import hashlib
import os
import re
import shutil
import subprocess  # nosec B404 - fixed-argv CLI probes are required; shell=True is not used.
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# Locate the bundled launcher examples relative to THIS package's install
# path. Works for both editable installs (../launcher/examples/) and
# uvx-from-git installs (the launcher is a sibling site-packages install).
_THIS_DIR = Path(__file__).resolve().parent

_DEFAULT_SOURCE_REPO = "https://github.com/NVIDIA/Model-Optimizer.git"
_DEFAULT_SOURCE_REF = "main"

# Canonical task-status failure tokens — matched against the FIRST word
# of each ``status_<task>.out`` file by ``job_status_impl``.
_STATUS_FAILURE_WORDS: frozenset[str] = frozenset(
    {"failed", "error", "errored", "cancelled", "canceled"}
)

_SAFE_EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_LAUNCHER_ERROR_RE = re.compile(
    r"(?:^|\n)(?:Unexpected error:|Error processing argument )",
    re.IGNORECASE,
)

_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_SAFE_PATH_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class SourceCheckout:
    """A managed Model-Optimizer checkout used for one launcher invocation."""

    repo: str
    ref: str
    resolved_sha: str
    root: Path

    @property
    def launcher_dir(self) -> Path:
        """Return the launcher package directory inside this checkout."""
        return self.root / "tools" / "launcher"

    @property
    def examples_dir(self) -> Path:
        """Return the launcher examples directory inside this checkout."""
        return self.launcher_dir / "examples"


def _launcher_reported_error(stdout: str, stderr: str) -> bool:
    """Return True when launcher text contains a fatal error despite exit 0."""
    return bool(_LAUNCHER_ERROR_RE.search(f"{stdout}\n{stderr}"))


def _parse_launcher_submission(text: str) -> tuple[str | None, str | None, str | None]:
    """Best-effort parse of launcher/nemo_run submission output."""
    experiment_id = None
    experiment_dir = None
    slurm_job_id = None

    # nemo_run prints "Experiment Status for <id>" and often also the
    # reconstructable form `Experiment.from_id("<id>")`.
    m = re.search(r'Experiment\.from_id\("([^"]+)"\)', text)
    if m:
        experiment_id = m.group(1)
    else:
        m = re.search(
            r"Experiment Status for\s+(\S+)",
            text,
            re.IGNORECASE,
        )
        if m:
            experiment_id = m.group(1)
    if not experiment_id:
        m = re.search(
            r"experiment[_\s-]+id[:\s]+(\S+)",
            text,
            re.IGNORECASE,
        )
        if m:
            experiment_id = m.group(1)
    if not experiment_id:
        # Fallback for older nemo_run output that lacked the explicit
        # "id:" label. Accepts any path-safe id token following the
        # word "experiment" — not just timestamp-style.
        m = re.search(
            r"experiment[_\s-]+([A-Za-z0-9_-]+)",
            text,
            re.IGNORECASE,
        )
        if m and m.group(1).lower() not in {"status", "dir", "directory"}:
            experiment_id = m.group(1)

    # Match any path containing `/experiments/<id>/` — don't anchor on
    # cluster-specific filesystem roots (NVIDIA's /lustre, partner
    # clusters' /scratch / /work / /data / /p / ...).
    m = re.search(r"(?:experiment_dir[:=]\s*|(?<!\S))(\S+/experiments/[^\s/]+)", text)
    if m:
        experiment_dir = m.group(1)
    m = re.search(r"Submitted batch job (\d+)", text)
    if m:
        slurm_job_id = m.group(1)
    else:
        m = re.search(r"Job id:\s*(\d+)", text, re.IGNORECASE)
        if m:
            slurm_job_id = m.group(1)

    return experiment_id, experiment_dir, slurm_job_id


def _docker_experiment_id_capture_timeout() -> float:
    """Return how long Docker submit should tail launcher output for an id."""
    raw = os.environ.get("MODELOPT_MCP_DOCKER_ID_TIMEOUT_SEC", "10")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 10.0


def _tail_docker_launch_log(log_path: Path, proc: subprocess.Popen) -> tuple[str | None, str]:
    """Tail detached Docker launcher output for an early experiment id."""
    deadline = time.monotonic() + _docker_experiment_id_capture_timeout()
    text = ""
    while True:
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            text = ""
        complete_text = text if text.endswith(("\n", "\r")) else text.rsplit("\n", 1)[0]
        experiment_id, _, _ = _parse_launcher_submission(complete_text)
        if experiment_id:
            return experiment_id, _tail(text, 2000)
        if time.monotonic() >= deadline or proc.poll() is not None:
            experiment_id, _, _ = _parse_launcher_submission(text)
            if experiment_id:
                return experiment_id, _tail(text, 2000)
            return None, _tail(text, 2000)
        time.sleep(0.2)


def _validate_experiment_id(experiment_id: str) -> dict | None:
    """Reject experiment ids that could escape path joins or alter glob matching."""
    if _SAFE_EXPERIMENT_ID_RE.fullmatch(experiment_id):
        return None
    return {
        "ok": False,
        "experiment_id": experiment_id,
        "reason": "invalid_experiment_id",
        "diagnostic": (
            "experiment_id must be a single path-safe token containing "
            "only letters, numbers, underscores, and hyphens."
        ),
    }


def _find_launcher_examples_dir() -> Path | None:
    """Resolve the launcher examples directory.

    Strategy (in order):
    1. ``MODELOPT_LAUNCHER_EXAMPLES_DIR`` env override — for tests + ad-hoc
       relocations.
    2. ``import modelopt_launcher`` — works whether the launcher is
       installed via pip/uvx or in editable dev mode; ``PACKAGE_DIR``
       points at ``tools/launcher/``, which contains ``examples/``.

    Returns None if no candidate exists; callers surface that as a
    structured failure rather than blowing up.
    """
    env = os.environ.get("MODELOPT_LAUNCHER_EXAMPLES_DIR")
    if env:
        p = Path(env)
        return p if p.exists() else None

    try:
        import modelopt_launcher

        candidate = Path(modelopt_launcher.PACKAGE_DIR) / "examples"
        if candidate.exists():
            return candidate
    except ImportError:
        pass
    return None


def _find_launcher_package_dir() -> Path | None:
    """Resolve the installed launcher's package directory."""
    try:
        import modelopt_launcher

        candidate = Path(modelopt_launcher.PACKAGE_DIR)
        if candidate.exists():
            return candidate
    except ImportError:
        pass
    return None


def _launcher_not_installed(argv: list[str]) -> dict:
    """Structured failure when the ``modelopt-launcher`` binary is not on PATH."""
    if argv and argv[0] == _uv_binary():
        return {
            "ok": False,
            "reason": "uv_not_installed",
            "diagnostic": (
                "`uv` was not found on PATH. Managed Model-Optimizer source "
                "checkouts use `uv run --project <checkout>/tools/launcher "
                "modelopt-launcher ...`. Install uv or set "
                "MODELOPT_MCP_DISABLE_MANAGED_SOURCE=1 to use the installed "
                "`modelopt-launcher` entrypoint directly."
            ),
            "argv": argv,
        }
    return {
        "ok": False,
        "reason": "launcher_not_installed",
        "diagnostic": (
            "`modelopt-launcher` was not found on PATH. "
            "Install it with `pip install modelopt-launcher` or "
            "`uv tool install modelopt-launcher` and retry."
        ),
        "argv": argv,
    }


# ---------------------------------------------------------------------------
# Managed Model-Optimizer source checkouts
# ---------------------------------------------------------------------------


def _uv_binary() -> str:
    """Return the uv executable used for managed-source launcher runs."""
    return os.environ.get("MODELOPT_MCP_UV", "uv")


def _source_cache_root() -> Path:
    """Return the root directory for MCP-managed source checkouts."""
    env = os.environ.get("MODELOPT_MCP_SOURCE_CACHE")
    if env:
        return Path(env).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "modelopt-mcp" / "sources"


def _source_disabled() -> bool:
    """Return True when callers explicitly opt out of managed source checkouts."""
    return os.environ.get("MODELOPT_MCP_DISABLE_MANAGED_SOURCE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _sanitize_path_token(value: str, *, fallback: str) -> str:
    """Make a short, filesystem-safe display token."""
    token = _SAFE_PATH_TOKEN_RE.sub("-", value.strip()).strip(".-")
    return (token or fallback)[:48]


def _tail(text: str | None, limit: int = 1200) -> str:
    """Return a short tail suitable for structured diagnostics."""
    return str(text or "")[-limit:]


def _git_failure(
    *,
    reason: str,
    diagnostic: str,
    argv: list[str],
    proc: subprocess.CompletedProcess | None = None,
) -> dict:
    """Return a structured managed-source git failure."""
    result = {
        "ok": False,
        "reason": reason,
        "diagnostic": diagnostic,
        "argv": argv,
    }
    if proc is not None:
        result.update(
            {
                "exit_code": proc.returncode,
                "stdout_tail": _tail(proc.stdout),
                "stderr_tail": _tail(proc.stderr),
            }
        )
    return result


def _run_git(argv: list[str], *, cwd: Path | None = None, timeout: int = 300) -> dict:
    """Run a fixed git argv list and return either proc or a structured failure."""
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed git argv list; no shell.
            argv,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return _git_failure(
            reason="git_not_installed",
            diagnostic="`git` was not found on PATH; cannot prepare the managed source checkout.",
            argv=argv,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "reason": "git_timeout",
            "diagnostic": f"`{' '.join(argv)}` did not finish within {timeout}s.",
            "argv": argv,
            "stdout_tail": (e.stdout or b"").decode(errors="replace")[-1200:]
            if isinstance(e.stdout, bytes)
            else _tail(e.stdout),
            "stderr_tail": (e.stderr or b"").decode(errors="replace")[-1200:]
            if isinstance(e.stderr, bytes)
            else _tail(e.stderr),
        }
    if proc.returncode != 0:
        return _git_failure(
            reason="git_failed",
            diagnostic=f"`{' '.join(argv)}` failed while preparing the managed source checkout.",
            argv=argv,
            proc=proc,
        )
    return {"ok": True, "proc": proc}


def _resolve_source_ref(repo: str, ref: str) -> dict:
    """Resolve a branch/tag/ref to a commit SHA without mutating local state."""
    if _GIT_SHA_RE.fullmatch(ref):
        return {"ok": True, "resolved_sha": ref.lower()}

    patterns = [ref]
    if not ref.startswith("refs/"):
        patterns.extend([f"refs/heads/{ref}", f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"])

    argv = ["git", "ls-remote", repo, *patterns]
    result = _run_git(argv, timeout=60)
    if not result.get("ok"):
        return {
            **result,
            "reason": "source_ref_resolve_failed",
            "diagnostic": (
                f"Could not resolve Model-Optimizer source ref {ref!r} from {repo}. "
                "Check the branch/tag/SHA and network credentials."
            ),
        }

    lines = [line.split() for line in result["proc"].stdout.splitlines() if line.strip()]
    by_name = {name: sha for sha, name, *_ in lines if len(sha) == 40}
    for name in (
        f"refs/heads/{ref}",
        f"refs/tags/{ref}^{{}}",
        f"refs/tags/{ref}",
        ref,
    ):
        sha = by_name.get(name)
        if sha:
            return {"ok": True, "resolved_sha": sha}
    for sha, *_ in lines:
        if len(sha) == 40:
            return {"ok": True, "resolved_sha": sha}

    return {
        "ok": False,
        "reason": "source_ref_not_found",
        "diagnostic": f"Model-Optimizer source ref {ref!r} was not found in {repo}.",
        "argv": argv,
        "stdout_tail": result["proc"].stdout[-1200:],
        "stderr_tail": result["proc"].stderr[-1200:],
    }


def _checkout_path(repo: str, ref: str, resolved_sha: str) -> Path:
    """Return the immutable checkout path for a resolved source ref."""
    repo_hash = hashlib.sha256(repo.encode()).hexdigest()[:12]
    ref_token = _sanitize_path_token(ref, fallback="ref")
    sha_token = resolved_sha[:12]
    return _source_cache_root() / repo_hash / f"{ref_token}-{sha_token}"


def _checkout_ready(path: Path, resolved_sha: str) -> bool:
    """Return True when a managed checkout already exists at the requested SHA."""
    if not (path / ".git").exists() or not (path / "tools" / "launcher" / "launch.py").exists():
        return False
    result = _run_git(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=30)
    return bool(result.get("ok") and result["proc"].stdout.strip().startswith(resolved_sha))


def _materialize_checkout(repo: str, ref: str, resolved_sha: str, path: Path) -> dict:
    """Clone Model-Optimizer and initialize submodules for a resolved ref."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = (
        parent / f".tmp-{_sanitize_path_token(ref, fallback='ref')}-{os.getpid()}-{time.time_ns()}"
    )
    if tmp.exists():
        shutil.rmtree(tmp)

    clone = ["git", "clone", "--no-checkout", "--filter=blob:none", repo, str(tmp)]
    fetch_refs = [resolved_sha] if _GIT_SHA_RE.fullmatch(ref) else [ref, resolved_sha]
    post_fetch_steps = [
        ["git", "-C", str(tmp), "checkout", "--detach", "FETCH_HEAD"],
        ["git", "-C", str(tmp), "submodule", "sync", "--recursive"],
        ["git", "-C", str(tmp), "submodule", "update", "--init", "--recursive", "--depth=1"],
    ]
    try:
        result = _run_git(clone)
        if not result.get("ok"):
            return _source_checkout_failure(result, repo, ref, resolved_sha, path)

        fetch_result = None
        for fetch_ref in fetch_refs:
            fetch = ["git", "-C", str(tmp), "fetch", "--depth=1", "origin", fetch_ref]
            fetch_result = _run_git(fetch)
            if fetch_result.get("ok"):
                break
        if fetch_result is None or not fetch_result.get("ok"):
            return _source_checkout_failure(fetch_result or {}, repo, ref, resolved_sha, path)

        for argv in post_fetch_steps:
            result = _run_git(argv)
            if not result.get("ok"):
                return _source_checkout_failure(result, repo, ref, resolved_sha, path)
        if path.exists():
            if _checkout_ready(path, resolved_sha):
                shutil.rmtree(tmp)
            else:
                shutil.rmtree(path)
                tmp.rename(path)
        else:
            tmp.rename(path)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)

    return {"ok": True}


def _source_checkout_failure(
    result: dict,
    repo: str,
    ref: str,
    resolved_sha: str,
    path: Path,
) -> dict:
    """Attach source provenance to a failed checkout step."""
    return {
        **result,
        "ok": False,
        "reason": "source_checkout_failed",
        "diagnostic": (
            "Failed to prepare the managed Model-Optimizer source checkout. "
            "The launcher was not run."
        ),
        "source_repo": repo,
        "source_ref": ref,
        "source_sha": resolved_sha,
        "source_root": str(path),
    }


def _ensure_source_checkout(
    source_ref: str | None = None,
    source_repo: str | None = None,
) -> dict:
    """Return a managed source checkout, or None when explicitly disabled."""
    if _source_disabled():
        return {"ok": True, "checkout": None}

    repo = source_repo or os.environ.get("MODELOPT_MCP_SOURCE_REPO") or _DEFAULT_SOURCE_REPO
    ref = source_ref or os.environ.get("MODELOPT_MCP_SOURCE_REF") or _DEFAULT_SOURCE_REF

    resolved = _resolve_source_ref(repo, ref)
    if not resolved.get("ok"):
        return {**resolved, "source_repo": repo, "source_ref": ref}

    resolved_sha = resolved["resolved_sha"]
    path = _checkout_path(repo, ref, resolved_sha)
    if not _checkout_ready(path, resolved_sha):
        materialized = _materialize_checkout(repo, ref, resolved_sha, path)
        if not materialized.get("ok"):
            return materialized

    checkout = SourceCheckout(repo=repo, ref=ref, resolved_sha=resolved_sha, root=path)
    return {"ok": True, "checkout": checkout}


def _source_result_fields(checkout: SourceCheckout | None) -> dict:
    """Return source provenance fields for tool results."""
    if checkout is None:
        return {}
    return {
        "source_repo": checkout.repo,
        "source_ref": checkout.ref,
        "source_sha": checkout.resolved_sha,
        "source_root": str(checkout.root),
    }


def _launcher_argv(abs_yaml: Path, checkout: SourceCheckout | None, *flags: str) -> list[str]:
    """Build the launcher argv for installed or managed-source execution."""
    if checkout is None:
        return ["modelopt-launcher", "--yaml", str(abs_yaml), *flags]
    return [
        _uv_binary(),
        "run",
        "--reinstall-package",
        "modelopt-launcher",
        "--project",
        str(checkout.launcher_dir),
        "modelopt-launcher",
        "--yaml",
        str(abs_yaml),
        *flags,
    ]


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
        proc = subprocess.run(  # nosec B603 B607 - fixed docker CLI argv; no shell.
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
        gpu = subprocess.run(  # nosec B603 B607 - fixed docker CLI argv; no shell.
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
    control_socket: str | None = None,
    reconnect_command: str | None = None,
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
    if control_socket:
        expanded_socket = os.path.expanduser(control_socket)
        check_target = f"{cluster_user}@{cluster_host}" if cluster_user else cluster_host
        try:
            check = subprocess.run(  # nosec B603 B607 - fixed ssh argv; no shell.
                [
                    "ssh",
                    "-O",
                    "check",
                    "-o",
                    f"ControlPath={expanded_socket}",
                    check_target,
                ],
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
                    f"ssh ControlMaster check for {cluster_host} did not respond within 15s. "
                    "The control socket may be wedged; reconnect and retry."
                ),
            }
        except FileNotFoundError:
            return {
                "ok": False,
                "executor": "slurm",
                "reason": "ssh_not_installed",
                "diagnostic": "`ssh` binary not on PATH.",
            }
        if check.returncode != 0:
            return {
                "ok": False,
                "executor": "slurm",
                "cluster_host": cluster_host,
                "cluster_user": cluster_user,
                "reason": "mfa_reauth_required",
                "control_socket": control_socket,
                "reconnect_command": reconnect_command,
                "diagnostic": (
                    "OpenSSH ControlMaster socket is absent or expired. "
                    f"Run `{reconnect_command or 'ssh <cluster>'}`, keep it "
                    "connected, then retry."
                ),
            }
        argv += [
            "-o",
            f"ControlPath={expanded_socket}",
            "-o",
            "ControlMaster=no",
        ]
    if identity:
        argv += ["-i", identity]
    target = f"{cluster_user}@{cluster_host}" if cluster_user else cluster_host
    argv += [target, "whoami && hostname"]

    # B603/B607 false positive — `ssh` invoked by name with a controlled
    # argv (BatchMode, ConnectTimeout, identity path, target). No shell.
    try:
        proc = subprocess.run(  # nosec B603 - fixed ssh CLI argv; no shell.
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
        "control_socket": control_socket,
        "whoami": lines[0] if lines else "",
        "remote_hostname": lines[1] if len(lines) > 1 else "",
    }


# ---------------------------------------------------------------------------
# submit_job
# ---------------------------------------------------------------------------


def _normalize_yaml_path(yaml_path: str, *, examples_dir: Path | None = None) -> Path:
    """Resolve a launcher YAML path to an absolute Path.

    Lookup order:
    1. Absolute path — use as-is
    2. Relative to the managed checkout's examples dir, when present
    3. Relative to ``MODELOPT_LAUNCHER_EXAMPLES_DIR`` (or its parent)
    4. Relative to cwd

    The double-fallback lets the agent pass either ``examples/Qwen/.../X.yaml``
    or just the absolute path.
    """
    p = Path(yaml_path)
    if p.is_absolute():
        return p
    # Look under a managed checkout first, then the installed examples dir.
    examples_dirs: list[Path] = []
    if examples_dir is not None:
        examples_dirs.append(examples_dir)
    installed_examples_dir = _find_launcher_examples_dir()
    if installed_examples_dir is not None and installed_examples_dir not in examples_dirs:
        examples_dirs.append(installed_examples_dir)
    for root in examples_dirs:
        candidate = root / yaml_path
        if candidate.exists():
            return candidate
        candidate = root.parent / yaml_path
        if candidate.exists():
            return candidate
    # cwd fallback
    return (Path.cwd() / yaml_path).resolve()


def _launcher_overrides(
    *,
    executor: str,
    extra_overrides: dict[str, str] | None,
    account: str | None,
    partition: str | None,
    container: str | None,
    gpus_per_node: int | None,
    ntasks_per_node: int | None,
) -> dict[str, str]:
    """Build launcher CLI overrides shared by live submit and dry-run."""
    overrides = dict(extra_overrides or {})
    if executor == "slurm":
        slurm_prefix = "pipeline.task_0.slurm_config."
        if account:
            overrides.setdefault(f"{slurm_prefix}account", account)
        if partition:
            overrides.setdefault(f"{slurm_prefix}partition", partition)
        if container:
            overrides.setdefault(f"{slurm_prefix}container", container)
        if gpus_per_node is not None:
            overrides.setdefault(f"{slurm_prefix}gpus_per_node", str(gpus_per_node))
        if ntasks_per_node is not None:
            overrides.setdefault(f"{slurm_prefix}ntasks_per_node", str(ntasks_per_node))
    return overrides


def _apply_launcher_env(
    env: dict[str, str],
    *,
    checkout: SourceCheckout | None,
    executor: str,
    cluster_host: str | None,
    account: str | None,
    partition: str | None,
    control_socket: str | None,
    reconnect_command: str | None,
) -> None:
    """Apply launcher env shared by live submit and dry-run."""
    env.setdefault("NEMORUN_HOME", os.getcwd())
    if checkout is not None:
        env["MODELOPT_MCP_SOURCE_ROOT"] = str(checkout.root)
        env["MODELOPT_MCP_SOURCE_REF"] = checkout.ref
        env["MODELOPT_MCP_SOURCE_SHA"] = checkout.resolved_sha
    if executor == "slurm":
        env["SLURM_HOST"] = cluster_host or ""
        if account:
            env["SLURM_ACCOUNT"] = account
        if partition:
            env["SLURM_PARTITION"] = partition
        if control_socket:
            env["MODELOPT_LAUNCHER_SSH_CONTROL_PATH"] = os.path.expanduser(control_socket)
        if reconnect_command:
            env["MODELOPT_LAUNCHER_SSH_RECONNECT_COMMAND"] = reconnect_command


def submit_job_impl(
    *,
    yaml_path: str,
    hf_local: str | None = None,
    cluster_host: str | None = None,
    cluster_user: str | None = None,
    identity: str | None = None,
    job_dir: str | None = None,
    job_name: str | None = None,
    extra_overrides: dict[str, str] | None = None,
    skip_verify: bool = False,
    account: str | None = None,
    partition: str | None = None,
    container: str | None = None,
    gpus_per_node: int | None = None,
    ntasks_per_node: int | None = None,
    control_socket: str | None = None,
    reconnect_command: str | None = None,
    gpu_type: str | None = None,
    mfa: bool = False,
    ssh_alias: str | None = None,
    dry_run: bool = False,
    source_ref: str | None = None,
    source_repo: str | None = None,
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
    reconnect_command = reconnect_command or (f"ssh {ssh_alias}" if ssh_alias else None)
    if mfa and cluster_host and not control_socket:
        return {
            "ok": False,
            "reason": "mfa_control_socket_required",
            "executor": "slurm",
            "cluster_host": cluster_host,
            "ssh_alias": ssh_alias,
            "diagnostic": (
                "mfa=True requires control_socket so the launcher can reuse "
                "an authenticated OpenSSH ControlMaster session."
            ),
        }

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
            account=account,
            partition=partition,
            container=container,
            gpus_per_node=gpus_per_node,
            ntasks_per_node=ntasks_per_node,
            control_socket=control_socket,
            reconnect_command=reconnect_command,
            source_ref=source_ref,
            source_repo=source_repo,
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
                control_socket=control_socket,
                reconnect_command=reconnect_command,
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

    # ---- Resolve source + YAML path -------------------------------
    source = _ensure_source_checkout(source_ref=source_ref, source_repo=source_repo)
    if not source.get("ok"):
        return source
    checkout: SourceCheckout | None = source["checkout"]

    abs_yaml = _normalize_yaml_path(
        yaml_path,
        examples_dir=checkout.examples_dir if checkout else None,
    )
    if not abs_yaml.exists():
        return {
            "ok": False,
            "reason": "yaml_not_found",
            "yaml_path": yaml_path,
            "resolved_path": str(abs_yaml),
            **_source_result_fields(checkout),
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
    argv = _launcher_argv(abs_yaml, checkout, "--yes")
    if hf_local:
        argv.append(f"hf_local={hf_local}")
    else:
        # Slurm mode — the launcher entrypoint does not accept a
        # `cluster_host` arg. The host is sourced via the SLURM_HOST env
        # var, consumed by slurm_factory in slurm_config.py.
        # Propagate via env, not argv.
        if cluster_user:
            argv.append(f"user={cluster_user}")
        if identity:
            argv.append(f"identity={identity}")
        argv.append("detach=true")
    if job_dir:
        argv.append(f"job_dir={job_dir}")
    if job_name:
        argv.append(f"job_name={job_name}")

    for k, v in _launcher_overrides(
        executor=executor,
        extra_overrides=extra_overrides,
        account=account,
        partition=partition,
        container=container,
        gpus_per_node=gpus_per_node,
        ntasks_per_node=ntasks_per_node,
    ).items():
        argv.append(f"{k}={v}")

    # Propagate env so submit-side and status-side agree on NEMORUN_HOME.
    # Without this, `launch.py` defaults NEMORUN_HOME to its own cwd
    # (tools/launcher/), but `_resolve_experiment_dir` later checks the
    # MCP server's cwd — different paths, so job_status would return
    # experiment_dir_not_found for jobs that actually succeeded.
    child_env = os.environ.copy()
    _apply_launcher_env(
        child_env,
        checkout=checkout,
        executor=executor,
        cluster_host=cluster_host,
        account=account,
        partition=partition,
        control_socket=control_socket,
        reconnect_command=reconnect_command,
    )

    if executor == "docker":
        # Docker mode: spawn detached. Redirect stdout/stderr to a side-channel
        # log file, then tail it briefly for nemo_run's experiment id. This
        # avoids PIPE deadlock while still giving callers the id needed for
        # job_status/job_logs polling.
        # `start_new_session=True` detaches from the MCP server's process
        # group so an MCP server restart / SIGINT doesn't SIGHUP the
        # in-flight launcher.
        # B603 false positive — argv is a controlled list built above.
        log_dir = Path(child_env["NEMORUN_HOME"]) / ".modelopt-mcp" / "docker-submit-logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = tempfile.NamedTemporaryFile(
                prefix="submit-",
                suffix=".log",
                dir=log_dir,
                delete=False,
                mode="w+b",
            )
        except OSError as e:
            return {
                "ok": False,
                "executor": "docker",
                "reason": "docker_submit_log_unavailable",
                "diagnostic": f"Unable to create Docker submit log under {log_dir}: {e}",
                "argv": argv,
                **_source_result_fields(checkout),
            }
        log_path = Path(log_file.name)
        try:
            proc = subprocess.Popen(  # nosec B603 - fixed launcher argv list; no shell.
                argv,
                env=child_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except FileNotFoundError:
            log_file.close()
            log_path.unlink(missing_ok=True)
            return _launcher_not_installed(argv)
        finally:
            log_file.close()

        experiment_id, stdout_tail = _tail_docker_launch_log(log_path, proc)
        return {
            "ok": True,
            "executor": "docker",
            "pid": proc.pid,
            "argv": argv,
            "nemorun_home": child_env["NEMORUN_HOME"],
            "experiment_id": experiment_id,
            "stdout_log": str(log_path),
            "stdout_tail": stdout_tail,
            **_source_result_fields(checkout),
            "diagnostic": (
                "Docker mode launched detached and experiment_id was captured from launcher output."
                if experiment_id
                else (
                    "Docker mode launched detached, but no experiment_id was "
                    "captured before the short output-tail timeout. Inspect "
                    "stdout_log or retry with MODELOPT_MCP_DOCKER_ID_TIMEOUT_SEC "
                    "set higher."
                )
            ),
        }

    # Slurm mode: synchronous call (launch.py exits quickly after sbatch
    # with detach=true). Capture stdout to parse experiment_id.
    # B603 false positive — argv is a controlled list built above.
    try:
        proc = subprocess.run(  # nosec B603 - fixed launcher argv list; no shell.
            argv,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except FileNotFoundError:
        return _launcher_not_installed(argv)
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
            **_source_result_fields(checkout),
        }

    # `proc` here is the CompletedProcess from subprocess.run with
    # text=True, but mypy's narrowing widens across the Docker-branch
    # Popen assignment above. Coerce explicitly.
    stdout_tail = str(proc.stdout or "")[-2000:]
    stderr_tail = str(proc.stderr or "")[-2000:]

    if proc.returncode != 0 or _launcher_reported_error(stdout_tail, stderr_tail):
        return {
            "ok": False,
            "executor": "slurm",
            "reason": "launch_py_failed",
            "exit_code": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "diagnostic": (
                "launch.py failed or printed a fatal launcher error. "
                "Common causes: SSH publickey rejection, malformed YAML, "
                "factory parsing failure, or NEMORUN_HOME unset. Inspect "
                "stdout_tail/stderr_tail."
            ),
            "argv": argv,
            **_source_result_fields(checkout),
        }

    experiment_id, experiment_dir, slurm_job_id = _parse_launcher_submission(stdout_tail)

    if not experiment_id:
        return {
            "ok": False,
            "executor": "slurm",
            "reason": "launch_result_unparsed",
            "exit_code": 0,
            "slurm_job_id": slurm_job_id,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "diagnostic": (
                "launch.py exited 0 but did not report an experiment_id "
                "that callers can use for job_status/job_logs polling. "
                "Treating this as failed even if a Slurm job id was parsed."
            ),
            "argv": argv,
            **_source_result_fields(checkout),
        }

    return {
        "ok": True,
        "executor": "slurm",
        "experiment_id": experiment_id,
        "experiment_dir": experiment_dir,
        "slurm_job_id": slurm_job_id,
        "exit_code": 0,
        "stdout_tail": stdout_tail,
        "argv": argv,
        **_source_result_fields(checkout),
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
    account: str | None,
    partition: str | None,
    container: str | None,
    gpus_per_node: int | None,
    ntasks_per_node: int | None,
    control_socket: str | None,
    reconnect_command: str | None,
    source_ref: str | None,
    source_repo: str | None,
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
    # Same source + path resolution as the live submit, so dry-run and live
    # use exactly the same launcher checkout and YAML.
    source = _ensure_source_checkout(source_ref=source_ref, source_repo=source_repo)
    if not source.get("ok"):
        return {**source, "dry_run": True}
    checkout: SourceCheckout | None = source["checkout"]

    abs_yaml = _normalize_yaml_path(
        yaml_path,
        examples_dir=checkout.examples_dir if checkout else None,
    )
    if not abs_yaml.exists():
        return {
            "ok": False,
            "dry_run": True,
            "reason": "yaml_not_found",
            "yaml_path": yaml_path,
            "resolved_path": str(abs_yaml),
            **_source_result_fields(checkout),
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
    argv = _launcher_argv(abs_yaml, checkout, "--dryrun", "--yes")
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
    executor = "docker" if hf_local else "slurm" if cluster_host else "dryrun"
    for k, v in _launcher_overrides(
        executor=executor,
        extra_overrides=extra_overrides,
        account=account,
        partition=partition,
        container=container,
        gpus_per_node=gpus_per_node,
        ntasks_per_node=ntasks_per_node,
    ).items():
        argv.append(f"{k}={v}")

    # Propagate env so the launcher's factory resolution matches what
    # the live submit would see (mainly: SLURM_HOST for slurm-factory
    # default when cluster_host is set).
    child_env = os.environ.copy()
    _apply_launcher_env(
        child_env,
        checkout=checkout,
        executor=executor,
        cluster_host=cluster_host,
        account=account,
        partition=partition,
        control_socket=control_socket,
        reconnect_command=reconnect_command,
    )

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
        proc = subprocess.run(  # nosec B603 - fixed dry-run launcher argv list; no shell.
            argv,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return {**_launcher_not_installed(argv), "dry_run": True}
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
            **_source_result_fields(checkout),
        }

    stdout_tail = str(proc.stdout or "")[-2000:]
    stderr_tail = str(proc.stderr or "")[-2000:]

    if proc.returncode != 0 or _launcher_reported_error(stdout_tail, stderr_tail):
        return {
            "ok": True,  # The tool itself ran cleanly
            "dry_run": True,
            "validated": False,  # ...but the YAML failed validation
            "exit_code": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "diagnostic": (
                "launch.py --dryrun rejected the YAML or printed a fatal "
                "launcher error. Common reasons: invalid YAML syntax, "
                "missing required fields, factory function not registered, "
                "factory parsing failure, or a referenced file (HF model "
                "path, container tag) doesn't exist. See stdout_tail/"
                "stderr_tail for the specific error."
            ),
            "argv": argv,
            **_source_result_fields(checkout),
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
        **_source_result_fields(checkout),
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
    for root in _experiment_search_roots():
        direct = root / experiment_id
        if direct.exists():
            return direct
        for nested in root.glob(f"*/{experiment_id}"):
            if nested.exists():
                return nested
    return None


def _experiment_search_roots() -> list[Path]:
    """Return experiment roots searched by status/log tools."""
    roots = []
    nemorun_home = os.environ.get("NEMORUN_HOME")
    if nemorun_home:
        roots.append(Path(nemorun_home) / "experiments")
    roots.append(Path.cwd() / "experiments")
    roots.append(Path.cwd() / "local_experiments")
    launcher_dir = _find_launcher_package_dir()
    if launcher_dir is not None:
        roots.append(launcher_dir / "experiments")
    return roots


def _experiment_not_found_diagnostic() -> str:
    """Describe all experiment roots used by _resolve_experiment_dir."""
    roots = ", ".join(str(root) for root in _experiment_search_roots())
    return (
        f"Searched experiment roots: {roots}. Either the id is wrong or "
        "NEMORUN_HOME isn't set the same as it was at submit time."
    )


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
    invalid = _validate_experiment_id(experiment_id)
    if invalid:
        return invalid

    exp_dir = _resolve_experiment_dir(experiment_id)
    if exp_dir is None:
        return {
            "ok": False,
            "experiment_id": experiment_id,
            "reason": "experiment_dir_not_found",
            "diagnostic": _experiment_not_found_diagnostic(),
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
    invalid = _validate_experiment_id(experiment_id)
    if invalid:
        return invalid

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
        try:
            proc = subprocess.run(  # nosec B603 B607 - fixed nemo CLI argv; no shell.
                argv,
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
        push = subprocess.run(  # nosec B603 B607 - fixed git CLI argv; no shell.
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
        gh = subprocess.run(  # nosec B603 B607 - fixed gh CLI argv; no shell.
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
