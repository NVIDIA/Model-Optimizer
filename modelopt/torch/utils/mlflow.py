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

"""Record a script run on an MLflow tracking server.

Lets an example script upload its invocation, configuration, log and outputs so the run can
be reproduced from its MLflow entry alone. ``mlflow`` is an optional dependency, imported
only once tracking is actually enabled.
"""

import argparse
import contextlib
import getpass
import json
import logging
import os
import re
import shlex
import shutil
import socket
import sys
import tempfile
import time
import traceback
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

import modelopt
from modelopt.torch.utils.logging import TeeStream

__all__ = [
    "EXPERIMENT_JSON",
    "TRACKING_URI_ENV",
    "MlflowRunLogger",
    "Tool",
    "add_mlflow_args",
    "command_text",
    "current_user",
    "default_experiment_name",
    "default_run_name",
    "describe_run",
    "drop_experiment_json",
    "log_active_run_experiment_json",
    "mask_tracking_uri",
    "masked_args",
    "resolve_mlflow_args",
    "resolve_tracking_uri",
    "resolved_recipe_texts",
    "run_tags",
    "split_tracking_credentials",
    "tracked_run",
    "validate_tracking_uri",
]

# MLflow experiment names are stored in a VARCHAR(256) column by the SQL-backed stores. The
# per-component cap stops one pathological component from crowding out the others; the name
# cap is what actually keeps the result storable.
_MAX_COMPONENT_LEN = 100
_MAX_NAME_LEN = 250
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

# Anything uploaded or printed passes through _redact first: a tracking URI may carry
# ``user:token@`` and a caller's own flags may carry a secret.
_SECRET_NAME = re.compile(r"token|api[-_]?key|password|passwd|secret|credential", re.IGNORECASE)
_URI_USERINFO = re.compile(r"(?<=://)[^/\s@]+(?=@)")
_MASK = "***"

# Provenance pointer a checkpoint carries, dotted like the other sidecars a quantization
# drops next to the weights so a checkpoint loader ignores it and it does not look like part
# of the model.
EXPERIMENT_JSON = ".experiment.json"

# MLflow's own variable, so a shell that already exports it opts in without a flag. Public
# because the vLLM example republishes the resolved URI under it for its worker processes.
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"


def _experiment_json(
    tracking_uri: str, experiment_name: str, info: Any, run_name: str | None = None
) -> dict[str, str]:
    """The provenance record's fields, read off the run the server returned.

    Shared by the two writers -- a run this process opened, and one it merely found.
    """
    uri = _redact(tracking_uri).rstrip("/")
    experiment_id = str(info.experiment_id)
    run_id = str(info.run_id)
    return {
        "tracking_uri": uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "run_name": getattr(info, "run_name", None) or run_name or "",
        "run_url": f"{uri}/#/experiments/{experiment_id}/runs/{run_id}",
    }


def _stat_key(path: Path) -> tuple[int, int] | None:
    """Identity of a file's contents-in-time, or ``None`` when it does not exist."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _redact(value: Any) -> Any:
    """Mask credentials embedded in a URI, leaving non-strings untouched."""
    return _URI_USERINFO.sub(_MASK, value) if isinstance(value, str) else value


def _redact_argv(argv: list[str]) -> list[str]:
    """Mask the value of any ``--*token*`` style option, and credentials in any URI."""
    redacted: list[str] = []
    mask_next = False
    for token in argv:
        if mask_next:
            # Unconditionally, since a secret may itself start with "-"; an option there
            # instead would mean the caller passed no value, which argparse rejects anyway.
            redacted.append(_MASK)
        elif token.startswith("-") and _SECRET_NAME.search(token):
            option, sep, _ = token.partition("=")
            redacted.append(option + sep + _MASK if sep else option)
        else:
            redacted.append(_redact(token))
        mask_next = (
            token.startswith("-") and _SECRET_NAME.search(token) is not None and "=" not in token
        )
    return redacted


def validate_tracking_uri(uri: str) -> str:
    """Validate an MLflow tracking URI and return it without a trailing slash.

    Only ``http(s)`` servers are accepted; MLflow's local ``file:`` / ``sqlite:`` backends
    are not a useful destination for a shared record of a run.

    Raises:
        ValueError: If *uri* is empty, has no host, or is not an http(s) URL.
    """
    if not uri:
        raise ValueError(
            "MLflow tracking URI is empty; pass one explicitly or set MLFLOW_TRACKING_URI."
        )
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        message = f"MLflow tracking URI must be http(s), got {uri!r}."
        if not parsed.scheme:
            # Only a bare host is plausibly a forgotten scheme; suggesting https://sqlite:///...
            # for a URI that already has one would be nonsense.
            message += f" Did you mean https://{uri.lstrip('/')}?"
        raise ValueError(message)
    if not parsed.netloc:
        raise ValueError(f"MLflow tracking URI {uri!r} has no host.")
    return uri.rstrip("/")


def default_experiment_name(tool: str, model: str, variant: str, user: str | None = None) -> str:
    """Build an experiment name of the form ``<user>/<tool>/<model>-<variant>``.

    Only the basename of *model* is used, so a local checkpoint directory and an
    ``org/name`` Hugging Face id collapse to the same readable name; *variant* is whatever
    distinguishes this run of *tool* on *model*, such as a recipe name or a quantization
    format. Each component is reduced to ``[A-Za-z0-9._-]`` so the ``/`` separators stay
    meaningful, and *user* defaults to the current user.

    Example:
        >>> default_experiment_name("hf_ptq", "/models/Qwen3-0.6B/", "nvfp4", user="alice")
        'alice/hf_ptq/Qwen3-0.6B-nvfp4'
    """
    owner = user if user is not None else current_user()
    name = (
        f"{_sanitize(owner)}/{_sanitize(tool)}/{_sanitize(Path(model).name)}-{_sanitize(variant)}"
    )
    return name[:_MAX_NAME_LEN]


def default_run_name() -> str:
    """The UTC start time as ``YYYYmmdd-HHMMSS``, which is what the flags document.

    Used by :class:`MlflowRunLogger` and by a caller handing the name to something else that
    opens the run, so both honour the documented default rather than MLflow's random one.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def current_user() -> str:
    """Return the current username, or ``"unknown"`` if the uid has no passwd entry."""
    try:
        return getpass.getuser()
    except OSError:  # container without a passwd entry for the uid
        return "unknown"


def _sanitize(component: str) -> str:
    """Reduce one experiment-name component to ``[A-Za-z0-9._-]``."""
    cleaned = _UNSAFE_CHARS.sub("_", component).strip("._-")
    return cleaned[:_MAX_COMPONENT_LEN] or "unknown"


def _git_sha() -> str:
    """Short commit of the ModelOpt source, or ``"unknown"`` outside a git checkout.

    Read out of ``.git`` rather than by shelling out to ``git``, which keeps the library
    free of subprocess use. Handles worktrees, where ``.git`` is a file pointing at the
    real git directory and refs live in the main checkout alongside it.
    """
    try:
        git_path = Path(__file__).resolve().parents[3] / ".git"
        if git_path.is_file():
            git_dir = Path(git_path.read_text().split("gitdir:", 1)[1].strip())
        else:
            git_dir = git_path
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head[:9]  # detached HEAD
        ref = head.removeprefix("ref: ")
        # A worktree keeps HEAD locally but shares refs with the checkout named by commondir.
        bases = [git_dir]
        commondir = git_dir / "commondir"
        if commondir.is_file():
            bases.append((git_dir / commondir.read_text().strip()).resolve())
        for base in bases:
            if (base / ref).is_file():
                return (base / ref).read_text().strip()[:9]
            packed = base / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    sha, _, name = line.partition(" ")
                    if name.strip() == ref:
                        return sha[:9]
    except (OSError, IndexError):
        pass
    return "unknown"


def command_text(argv: list[str] | None = None) -> str:
    """The invocation, as a copy-pasteable line, with credentials masked.

    *argv* defaults to this process's own ``sys.argv``. Pass another process's argv when the
    run is opened somewhere the user never typed a command -- a worker subprocess, whose own
    ``sys.argv`` is an implementation detail rather than a reproducible invocation.
    """
    lines = [shlex.join([sys.executable, *_redact_argv(sys.argv if argv is None else argv)])]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        lines += [
            "",
            f"# Launched under torchrun with WORLD_SIZE={world_size}, "
            f"LOCAL_WORLD_SIZE={os.environ.get('LOCAL_WORLD_SIZE', '?')}. The torchrun "
            "wrapper is not part of sys.argv and is therefore not shown above.",
        ]
    return "\n".join(lines) + "\n"


def _ask(what: str, callback: Callable[[], Any], default: Any) -> Any:
    """Read what a run reports about itself, never at the cost of the caller's exception."""
    try:
        return callback()
    except Exception as e:
        print(f"[mlflow] WARNING: could not read this run's {what}: {e}")
        return default


@contextmanager
def _closing_run(finish: Callable[[str], None]) -> Iterator[None]:
    """Run the block, then *finish* the run with the status the block earned."""
    status = "FAILED"
    try:
        yield
        status = "FINISHED"
    except SystemExit as e:
        # A script that ends by exiting -- Megatron-Bridge does, from inside its training
        # loop -- finished if it exited cleanly.
        status = "FINISHED" if e.code in (0, None) else "FAILED"
        raise
    finally:
        finish(status)


class MlflowRunLogger:
    """Record one script invocation as an MLflow run.

    :meth:`start` opens the run *before* the expensive work begins, so a bad URI, a missing
    token or an unreachable server fails there rather than after hours; it also
    uploads the invocation and any configuration passed to it, which keeps a crashed run
    useful. :meth:`finish` uploads the captured log plus any outputs and closes the run.
    Everything is a no-op when ``enabled`` is false, so callers need no branching.

    While the run is open, ``stdout``/``stderr`` are teed to a file that is uploaded as
    ``logs/<script>.log``. Logging handlers that libraries bound to ``sys.stderr`` at import
    time are re-pointed at the tee for the duration and handed back afterwards.

    Failures after the run is open are reported as warnings and never raised: losing a
    tracking server must not turn a successful job into a failed one.

    Note:
        ``command.txt`` masks ``--*token*``-style option values and credentials embedded in
        a URI, but the captured log is whatever the script printed, so a secret echoed to
        stdout still reaches the server. Prefer passing credentials via the environment.

    *tracking_uri* must already be validated (see :func:`validate_tracking_uri`),
    *experiment_name* is created if absent, *run_name* defaults to the UTC start time
    ``YYYYmmdd-HHMMSS``, and ``enabled=False`` makes every method a no-op -- which is how
    callers skip non-main ranks or an absent flag. ``required=False`` additionally downgrades
    a failure to open the run into a warning: use it when tracking was inferred from the
    environment rather than asked for, so an uninstalled client or an unreachable server
    cannot take the job down with it.

    Example:
        >>> logger = MlflowRunLogger(uri, "alice/hf_ptq/Qwen3-0.6B-nvfp4")
        >>> logger.start(params={"model": ckpt}, texts={"config.yaml": config_yaml})
        >>> status = "FAILED"
        >>> try:
        ...     quantize_and_export()
        ...     status = "FINISHED"
        ... finally:
        ...     logger.finish(status, files={"summary/report.txt": report_path})
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None = None,
        enabled: bool = True,
        required: bool = True,
    ):
        """Configure the run without contacting the server; see the class docstring."""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.enabled = enabled
        self.required = required
        self._mlflow: Any = None
        self._run: Any = None
        self._log_path: Path | None = None
        self._saved_streams: tuple | None = None
        self._tees: tuple | None = None
        self._file_stats: dict[str, tuple[int, int] | None] = {}
        self._start_time = 0.0
        # The status a co-owner closed this run with, if one did; see _reattach.
        self._closed_as: str | None = None

    @property
    def run_url(self) -> str:
        """Link to this run in the MLflow UI, or ``""`` before the run is open."""
        return self.run_info.get("run_url", "")

    @property
    def run_info(self) -> dict[str, str]:
        """Identity of this run on the server, or ``{}`` before it is open.

        Enough for a consumer holding only this run's outputs to find it again: ``run_id``
        is MLflow's own identifier for the run, a uuid4 hex, unique across experiments.
        Every field is read back off the run the server returned rather than off what was
        requested, so a run MLflow resolved differently is reported as it really is.
        """
        if self._run is None:
            return {}
        return _experiment_json(
            self.tracking_uri, self.experiment_name, self._run.info, self.run_name
        )

    def start(
        self,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        texts: dict[str, str] | None = None,
        files: Mapping[str, Path | str] | None = None,
    ) -> None:
        """Open the run: capture output, verify the server, upload the inputs.

        *params* are searchable; *tags* merge over the defaults (user, hostname, ModelOpt
        version and commit); *texts* maps artifact path to content, uploaded here rather
        than at the end so it survives a crash. *files* names the outputs the run is
        expected to produce, so :meth:`finish` can tell them from files that were already
        there -- pass the same mapping to both.

        Opening the run is the readiness check: it is MLflow's own first request, so it
        honours the client's TLS and retry configuration rather than second-guessing it.
        Set ``MLFLOW_HTTP_REQUEST_MAX_RETRIES`` to shorten the wait on a dead host.

        Raises:
            ImportError: If ``mlflow`` is not installed and ``required``.
            Exception: Whatever MLflow raises for an unusable server, if ``required``.
        """
        if not self.enabled or self._run is not None:
            return
        self._start_time = time.time()
        # Keyed through Path on both sides: a caller may pass strings, and "./out/x" and
        # "out/x" are the same file but not the same string.
        self._file_stats = {str(p): _stat_key(p) for p in map(Path, (files or {}).values())}
        self._start_capture()
        try:
            self._open_run()
            self._log_inputs(params, tags, texts)
        except Exception as e:
            # start_run() may already have succeeded, and the caller gets an exception
            # rather than a logger to call finish() on, so close the run here.
            self._abort_run()
            self._stop_capture()
            if self.required:
                raise
            self.enabled = False
            print(f"[mlflow] WARNING: tracking disabled, continuing without it ({e})")

    @contextmanager
    def track(
        self,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        texts: dict[str, str] | None = None,
        files: Mapping[str, Path | str] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> Iterator["MlflowRunLogger"]:
        """Open the run for the duration of the block, closing it with the right status.

        Mirrors ``mlflow.start_run()``. *files* and *metrics* are uploaded when the block
        exits; naming the paths upfront is fine because only files this run actually wrote
        are uploaded (see :meth:`finish`).

        Example:
            >>> with logger.track(params={"model": ckpt}, files={"summary.txt": report}):
            ...     quantize_and_export()
        """
        self.start(params=params, tags=tags, texts=texts, files=files)
        with _closing_run(lambda status: self.finish(status, files=files, metrics=metrics)):
            yield self

    def log_text(self, artifact_path: str, text: str) -> None:
        """Upload *text* as an artifact while the run is open, best-effort.

        For a value that is only settled midway through the run and is worth having even if
        the run later crashes -- the quantization config a calibration is about to apply,
        say. :meth:`start` and :meth:`finish` cover everything known at the two ends.
        """
        if not self.enabled or self._run is None:
            return
        try:
            if self._reattach():
                self._log_texts({artifact_path: text})
        except Exception as e:
            print(f"[mlflow] WARNING: could not upload {artifact_path}: {e}")

    def log_experiment_json(self, checkpoint_dir: Path | str | None = None) -> None:
        """Record which MLflow run produced a checkpoint, on the server and in the checkpoint.

        Tags point from a run to the checkpoint it wrote; this is the reverse, so a checkpoint
        found on disk can be traced back to the run that produced it without searching the
        server. The artifact goes up for any run that opened, so a failure is traceable from
        the server side too.

        *checkpoint_dir* also writes the JSON there as :data:`EXPERIMENT_JSON`. Pass it only
        once the checkpoint is really on disk, since the file claims authorship of the weights
        sitting next to it: an output directory existing proves nothing, as it may hold a
        checkpoint from an earlier attempt whose weights this run never touched.

        After the checkpoint is written, the pointer beside it is this run's or absent --
        never a previous run's. So a run that never opened *removes* the pointer rather than
        leaving one: tracking can disable itself mid-flight (an unreachable server or an
        uninstalled client, which a URI inherited from the environment tolerates by design),
        and the caller's untracked cleanup was skipped because tracking looked configured.
        """
        info = self.run_info
        if not info:
            if checkpoint_dir is not None:
                drop_experiment_json(checkpoint_dir)
            return
        text = json.dumps(info, indent=2) + "\n"
        self.log_text(EXPERIMENT_JSON.removeprefix("."), text)
        if checkpoint_dir is None:
            return
        target = Path(checkpoint_dir) / EXPERIMENT_JSON
        try:
            target.write_text(text)
        except OSError as e:
            print(f"[mlflow] WARNING: could not write {target}: {e}")

    def _abort_run(self) -> None:
        """End a run that failed before :meth:`start` returned, so it is not left RUNNING."""
        if self._run is None:
            return
        try:
            self._mlflow.end_run(status="FAILED")
        except Exception as e:
            print(f"[mlflow] WARNING: could not close the interrupted run: {e}")
        self._run = None

    def finish(
        self,
        status: str,
        texts: dict[str, str] | None = None,
        files: Mapping[str, Path | str] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Upload the run's outputs and close it with *status*, e.g. ``"FINISHED"``.

        *texts* and *files* both map artifact path to content, from memory and from disk
        respectively. A *files* entry is skipped when its file is absent, or was last
        modified before the run started -- so callers can list optional outputs, and a run
        that produced none of them does not upload a previous run's leftovers.
        *metrics* merges over the default ``total_time_s``.
        """
        if not self.enabled or self._run is None:
            self._stop_capture()
            return
        if status != "FINISHED":
            self._note_active_exception()
        ours = False
        try:
            ours = self._reattach()
            if ours:
                self._log_outputs(texts, files, metrics)
        except Exception as e:
            print(f"[mlflow] WARNING: could not upload run outputs: {e}")
        self._stop_capture()
        # A co-owner's status is kept only when it reports trouble this block cannot see --
        # Megatron-Bridge ends the run as KILLED on SIGTERM -- so a clean close elsewhere
        # never masks a failure here, and a non-terminal status never reaches end_run.
        if self._closed_as in ("FAILED", "KILLED"):
            status = self._closed_as
        try:
            if ours:
                self._mlflow.end_run(status=status)
            else:
                # Another run owns the fluent slot, so close this one by id rather than leave
                # it RUNNING; MLflow's atexit only terminates whatever is active.
                from mlflow.tracking import MlflowClient

                MlflowClient().set_terminated(self._run.info.run_id, status=status)
            print(f"[mlflow] {status}: {self.run_url}")
        except Exception as e:
            print(f"[mlflow] WARNING: could not close the run: {e}")

    def _reattach(self) -> bool:
        """Make this run the fluent API's target again, reporting whether it is.

        A co-owner can end the run first -- Megatron-Bridge does, as ``KILLED``, on SIGTERM --
        and a fluent call with none active opens one, so this run's log would land there. The
        status it was closed with is remembered here, since re-attaching sets it ``RUNNING``.
        A *different* run being active is reported rather than uploaded through.
        """
        active = self._mlflow.active_run()
        if active is not None:
            if str(active.info.run_id) == str(self._run.info.run_id):
                return True
            print(
                f"[mlflow] WARNING: run {active.info.run_id} is active instead of this one, "
                f"so {self.run_url} keeps neither its outputs nor a final status."
            )
            return False
        self._closed_as = str(self._mlflow.get_run(self._run.info.run_id).info.status)
        self._mlflow.start_run(run_id=self._run.info.run_id)
        return True

    def _note_active_exception(self) -> None:
        """Append the exception being handled to the captured log.

        :meth:`finish` runs from the caller's ``finally``, which is *before* the interpreter
        prints the traceback to ``sys.stderr`` -- no longer teed by then -- so the log would
        otherwise stop at the last line the script printed. Written to the file only, so the
        console still shows the traceback exactly once.
        """
        if sys.exc_info()[0] is None or self._saved_streams is None:
            return
        sink = self._saved_streams[2]
        if not sink.closed:
            sink.write("\n" + traceback.format_exc())

    def _open_run(self) -> None:
        try:
            import mlflow  # optional dependency: only needed once tracking is enabled
        except ImportError as e:
            raise ImportError(
                "MLflow tracking requires the 'mlflow' package: pip install nvidia-modelopt[mlflow]"
            ) from e

        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        # Settled here rather than passed straight through, so run_info reports the name the
        # run actually carries.
        self.run_name = self.run_name or default_run_name()
        self._run = mlflow.start_run(run_name=self.run_name)
        print(f"[mlflow] experiment: {self.experiment_name}\n[mlflow] run: {self.run_url}")

    def _log_inputs(self, params, tags, texts) -> None:
        if params:
            self._mlflow.log_params(
                {k: _MASK if _SECRET_NAME.search(k) else _redact(v) for k, v in params.items()}
            )
        self._mlflow.set_tags(
            {
                "user": current_user(),
                "hostname": socket.gethostname(),
                "modelopt_version": modelopt.__version__,
                "git_sha": _git_sha(),
                **(tags or {}),
            }
        )
        # The version is a tag as well, for searching; the artifact travels with the run.
        self._log_texts(
            {
                "command.txt": command_text(),
                "version.txt": f"{modelopt.__version__}\n",
                **(texts or {}),
            }
        )

    def _log_outputs(self, texts, files, metrics) -> None:
        elapsed = time.time() - self._start_time
        self._mlflow.log_metrics({"total_time_s": elapsed, **(metrics or {})})
        self._log_texts(texts)
        sys.stdout.flush()
        sys.stderr.flush()
        if self._log_path is not None:
            self._log_file(f"logs/{self._log_path.name}", self._log_path)
        for artifact_path, local in (files or {}).items():
            path = Path(local)
            if not path.is_file():
                continue
            # Only what this run produced: an export directory is commonly reused across
            # attempts, so a run that crashes early would otherwise upload the previous
            # run's summary as its own. Compared against the stat taken when the run opened
            # rather than against the wall clock, whose resolution outruns the filesystem's.
            if str(path) in self._file_stats and self._file_stats[str(path)] == _stat_key(path):
                continue
            self._log_file(artifact_path, path)

    def _log_texts(self, texts) -> None:
        for artifact_path, text in (texts or {}).items():
            self._mlflow.log_text(text, artifact_path)

    def _log_file(self, artifact_path: str, local: Path) -> None:
        """Upload *local* to *artifact_path*, staging a copy when it must be renamed."""
        target = PurePosixPath(artifact_path)
        directory = str(target.parent) if str(target.parent) != "." else None
        if local.name == target.name:
            self._mlflow.log_artifact(str(local), artifact_path=directory)
            return
        # log_artifact keeps the local basename, so rename via a staged copy rather than
        # reading the file into memory -- these can be hundreds of MB.
        with tempfile.TemporaryDirectory() as staging:
            staged = Path(staging) / target.name
            shutil.copy2(local, staged)
            self._mlflow.log_artifact(str(staged), artifact_path=directory)

    def _start_capture(self) -> None:
        script = Path(sys.argv[0]).stem or "run"
        self._log_path = Path(tempfile.mkdtemp(prefix="modelopt-mlflow-")) / f"{script}.log"
        sink = open(self._log_path, "w", buffering=1, encoding="utf-8")
        stdout, stderr = sys.stdout, sys.stderr
        self._saved_streams = (stdout, stderr, sink)
        self._tees = (TeeStream(stdout, sink), TeeStream(stderr, sink))
        sys.stdout, sys.stderr = self._tees
        self._repoint_handlers({stdout: self._tees[0], stderr: self._tees[1]})
        print(f"[mlflow] capturing this run's log to {self._log_path}")

    @staticmethod
    def _repoint_handlers(replacements: dict) -> None:
        """Move already-configured logging handlers from one stream to another.

        transformers and huggingface_hub bind ``sys.stderr`` into a ``StreamHandler`` when
        they are imported, long before the capture starts; without this their warnings reach
        the console but never the log. Scanning again on the way out -- rather than replaying
        a list captured on the way in -- also hands back handlers a library bound *during*
        the run, so nothing is left pointing at the tee once its file is closed.
        """
        loggers = [logging.getLogger(), *list(logging.Logger.manager.loggerDict.values())]
        for logger in loggers:
            for handler in list(getattr(logger, "handlers", [])):
                if not isinstance(handler, logging.StreamHandler):
                    continue
                if handler.stream not in replacements:
                    continue
                # logging._StderrHandler exposes ``stream`` as a read-only property that
                # already resolves to whatever sys.stderr currently is, so it follows the tee
                # on its own and cannot -- and must not -- be repointed.
                with contextlib.suppress(AttributeError):
                    handler.setStream(replacements[handler.stream])

    def _stop_capture(self) -> None:
        if self._saved_streams is None:
            return
        stdout, stderr, _ = self._saved_streams
        if self._tees is not None:
            self._repoint_handlers({self._tees[0]: stdout, self._tees[1]: stderr})
            self._tees = None
        sys.stdout, sys.stderr, sink = self._saved_streams
        sink.close()
        self._saved_streams = None
        if self._log_path is not None:
            shutil.rmtree(self._log_path.parent, ignore_errors=True)
            self._log_path = None


# The CLI surface below is shared by the example scripts that offer tracking, so a run is
# configured the same way and named by the same convention whichever script opened it.


@dataclass(frozen=True)
class Tool:
    """What distinguishes one script's tracking from another's.

    A script declares one of these and the functions below do the rest. Each callable reads
    the arguments naming this run's input, output and what it consumed -- *source* is what
    the next run in a chain joins on, *variant* names what this run did. *settles_pointer* is
    false when something other than :func:`tracked_run` writes the provenance pointer.
    """

    name: str
    tracks: str
    variant_help: str
    variant: Callable[[argparse.Namespace], str]
    model: Callable[[argparse.Namespace], str]
    checkpoint: Callable[[argparse.Namespace], str | None] = field(default=lambda args: None)
    source: Callable[[argparse.Namespace], str] | None = None
    texts: Callable[[argparse.Namespace], dict[str, str]] = field(default=lambda args: {})
    outputs: Callable[[argparse.Namespace], dict[str, Path]] = field(default=lambda args: {})
    # Read on the way *out*, so it can report something the run computed -- a pruning score,
    # say -- which the script stashes on its own namespace.
    metrics: Callable[[argparse.Namespace], dict[str, float]] = field(default=lambda args: {})
    non_params: frozenset[str] = frozenset()
    settles_pointer: bool = True


# The tracking settings describe the destination rather than the work, so they are never
# params; a script adds its own bookkeeping through ``Tool.non_params``.
_NEVER_PARAMS = frozenset({"mlflow", "mlflow_experiment", "mlflow_required", "mlflow_run_name"})


_ENV_HELP = (
    f"MLflow's own ${TRACKING_URI_ENV} enables tracking without this flag, which overrides "
    "it. A URI taken from the environment is best-effort: if it is unusable the run warns and "
    "continues untracked."
)

_TRACKS_HELP = (
    "Track this run on an MLflow server (e.g. https://<your-mlflow-server>/), uploading the "
    "command, the resolved configuration, the run log and the run's summaries."
)


def add_mlflow_args(parser: argparse.ArgumentParser, tool: Tool) -> None:
    """Add ``--mlflow``, ``--mlflow_experiment`` and ``--mlflow_run_name`` to *parser*.

    The help text comes from *tool*: its ``tracks`` describes what this script uploads and its
    ``variant_help`` says what the experiment name's variant is derived from. Pair with
    :func:`resolve_mlflow_args`.

    The multi-word flags are registered under both the underscored and the dashed spelling:
    vLLM's ``FlexibleArgumentParser`` rewrites every ``--foo_bar`` on the command line to
    ``--foo-bar`` before matching, so the dashed spelling has to exist for the flag to be
    reachable there at all, and a user moving between the example scripts should not have to
    remember which spelling each one took.
    """
    parser.add_argument("--mlflow", default=None, help=f"{tool.tracks} {_ENV_HELP}")
    parser.add_argument(
        "--mlflow_experiment",
        "--mlflow-experiment",
        default=None,
        help=(
            f"MLflow experiment name. Default: "
            f"$USER/{tool.name}/<model basename>-<{tool.variant_help}>."
        ),
    )
    parser.add_argument(
        "--mlflow_run_name",
        "--mlflow-run-name",
        default=None,
        help="MLflow run name. Default: the UTC start time as YYYYmmdd-HHMMSS.",
    )


def split_tracking_credentials(uri: str) -> str | None:
    """Move any ``user:token@`` out of *uri* into MLflow's own credential variables.

    For a caller that hands the URI to something which *records* it -- Megatron-Bridge logs
    its resolved config as params and writes it into the checkpoint. Masking is not an option
    there, since the value is also what authenticates. Returns ``None`` when the credential
    cannot be moved, so the caller can decline to record it. Variables the caller already
    exported win.
    """
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        # Fail closed. Without a scheme urlparse puts a "user:tok@host" in .path, where the
        # userinfo below never sees it and the URI would be handed back with the credential
        # still in it -- the one outcome this function exists to prevent. Callers reach this
        # through validate_tracking_uri, which rejects the same URIs, but the precondition is
        # not the caller's to remember.
        return None
    userinfo, separator, host = parsed.netloc.rpartition("@")
    if not separator:
        return uri
    username, colon, password = userinfo.partition(":")
    if not (username and colon and password):
        # Half a credential cannot be moved: MLflow sends basic auth only when both variables
        # are set, while requests authenticates off the URI it is given -- so the URI keeps
        # working where it is passed through, and only a caller that *records* it is stuck.
        return None
    # Percent-decoded, because userinfo in a URI is percent-encoded and the variables hold
    # the credential itself: a token containing "/" *must* be written "%2F" in the URI, and
    # requests -- which is what authenticates when the credential is left in the URI -- has
    # already decoded it there, so copying it across verbatim would authenticate differently.
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", unquote(username))
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", unquote(password))
    return parsed._replace(netloc=host).geturl()


def mask_tracking_uri(uri: str | None) -> str | None:
    """Mask any ``user:token@`` a tracking URI carries, for printing.

    Credentials in the URI are a supported form, so everything this module prints or uploads
    masks them -- ``command.txt``, the logged params, :attr:`MlflowRunLogger.run_url`. A
    caller that prints the URI itself (a script echoing its parsed arguments, say) has to do
    the same, or the secret reaches a console log that is routinely archived.
    """
    return _redact(uri)


def resolve_tracking_uri(
    uri: str | None, parser: argparse.ArgumentParser
) -> tuple[str | None, bool]:
    """Settle the tracking URI from ``--mlflow`` and the environment.

    Returns ``(uri or None, required)``, where *required* records that the flag was passed.
    Only the flag is a deliberate request, so only the flag is fatal when the URI is unusable:
    the environment variable is commonly exported for unrelated tooling and must not fail a
    job that would otherwise have worked.
    """
    # An empty value is not a deliberate request: ``--mlflow "$UNSET_VAR"`` is a wrapper
    # script whose variable did not resolve, so it neither names a server nor should take a
    # job down. It falls back like an absent flag -- but loudly, since the caller did ask
    # for tracking and what it gets is whatever the environment names, or nothing.
    required = bool(uri)
    if not uri:
        if uri is not None:
            warnings.warn(
                f"--mlflow was given an empty value; falling back to ${TRACKING_URI_ENV} "
                "if it is set, and running untracked otherwise."
            )
        uri = os.environ.get(TRACKING_URI_ENV) or None
        if uri is None:
            return None, required
    try:
        return validate_tracking_uri(uri), required
    except ValueError as e:
        if required:
            parser.error(f"--mlflow: {e}")  # exits
        warnings.warn(f"Ignoring ${TRACKING_URI_ENV}, continuing untracked: {e}")
        return None, required


def resolve_mlflow_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, tool: Tool
) -> None:
    """Settle where tracking is configured from, and name the experiment, in place.

    Sets ``args.mlflow`` to the validated URI or ``None``, ``args.mlflow_required`` to whether
    the flag asked for it, and defaults ``args.mlflow_experiment`` from *tool*. Pair with
    :func:`add_mlflow_args`.
    """
    args.mlflow, args.mlflow_required = resolve_tracking_uri(args.mlflow, parser)
    if args.mlflow:
        args.mlflow_experiment = args.mlflow_experiment or default_experiment_name(
            tool.name, tool.model(args), tool.variant(args)
        )


def log_active_run_experiment_json(checkpoint_dir: Path | str) -> bool:
    """Record MLflow's *currently active* run as the producer of a checkpoint.

    For a caller whose run is owned by something else -- Megatron-Bridge opens it for a
    training job, on its last rank. Call it from the rank that owns the run, once the
    checkpoint is on disk. With no run to name, any pointer already there is *removed*: the
    weights are new, so a previous run's pointer would misname their author. Returns whether
    a run was found, so a caller that asked for tracking can tell that from an untracked job.
    """
    run = None
    try:
        import mlflow

        # last_active_run() covers a run mlflow's atexit has already closed, which is what
        # a caller running from its own atexit or shutdown path sees.
        run = mlflow.active_run() or mlflow.last_active_run()
    except Exception:
        # mlflow absent, or unusable: an untracked run reaches here too, and quietly.
        pass
    if run is None:
        # Same invariant as MlflowRunLogger.log_experiment_json: after a save the pointer
        # beside the checkpoint is this run's or absent, never a previous run's.
        drop_experiment_json(checkpoint_dir)
        return False
    try:
        # The only field that needs the server. Everything else -- the ids, the run name, and
        # the URL built from them -- is on run.info already, and run_id is what anything
        # resolves the run by, so a blip here costs a display name rather than the pointer.
        experiment_name = mlflow.get_experiment(str(run.info.experiment_id)).name
    except Exception as e:
        print(f"[mlflow] WARNING: could not read the run's experiment name: {e}")
        experiment_name = ""
    try:
        info = _experiment_json(mlflow.get_tracking_uri(), experiment_name, run.info)
        text = json.dumps(info, indent=2) + "\n"
    except Exception as e:
        # Same invariant as the branch above: the weights are new, so a pointer naming an
        # earlier run is worse than none at all.
        print(f"[mlflow] WARNING: could not read the active run: {e}")
        drop_experiment_json(checkpoint_dir)
        return False

    # The file beside the weights first: it is the record that travels with the checkpoint,
    # and it must not be lost to an upload that fails. A failure here is reported on its own;
    # the return value answers "was there a run to name", which is what the callers ask.
    try:
        (Path(checkpoint_dir) / EXPERIMENT_JSON).write_text(text)
    except OSError as e:
        print(f"[mlflow] WARNING: could not write {Path(checkpoint_dir) / EXPERIMENT_JSON}: {e}")

    try:
        # Through the client, not the fluent ``mlflow.log_text``: the fluent one resolves its
        # target with ``_get_or_start_run()``, which on the closed-run branch above opens a
        # second run, and its ``run_id`` argument postdates this project's mlflow floor.
        from mlflow.tracking import MlflowClient

        MlflowClient().log_text(info["run_id"], text, EXPERIMENT_JSON.removeprefix("."))
    except Exception as e:
        print(f"[mlflow] WARNING: could not upload {EXPERIMENT_JSON.removeprefix('.')}: {e}")
    return True


def drop_experiment_json(checkpoint_dir: Path | str) -> None:
    """Remove a provenance pointer an untracked export would otherwise inherit.

    A fresh checkpoint written into a reused output directory would keep the previous run's
    pointer, and one produced from a tracked source checkpoint could be handed that source's
    pointer. Either way the file would name a run that did not produce these weights. Call it
    only for a completed export; a failed run leaves whatever checkpoint was already there,
    pointer included.
    """
    stale = Path(checkpoint_dir) / EXPERIMENT_JSON
    try:
        stale.unlink(missing_ok=True)
    except OSError as e:
        print(f"[mlflow] WARNING: could not remove stale {stale}: {e}")


def masked_args(args: argparse.Namespace, attr: str = "mlflow") -> argparse.Namespace:
    """A copy of *args* whose tracking URI cannot leak credentials into a printed namespace.

    For a script that echoes its parsed arguments: a ``user:token@`` in the URI is a supported
    form that this module masks wherever it prints or uploads one, and a job log is routinely
    archived. Uploaded artifacts are unaffected -- :func:`command_text` redacts, and the URI
    is not worth logging as a param.
    """
    return argparse.Namespace(**{**vars(args), attr: mask_tracking_uri(getattr(args, attr, None))})


def resolved_recipe_texts(recipe: str | None) -> dict[str, str]:
    r"""``{artifact path: content}`` for *recipe*, or ``{}`` when the run used none.

    The resolved recipe, not the source file: a recipe may be a directory or use ``$import``\ s,
    and only the resolved form stands alone.
    """
    if not recipe:
        return {}
    # Lazy import: modelopt.recipe imports modelopt.torch.quantization, which imports this
    # package at top level (circular), as with the other upward imports in modelopt/torch/utils.
    from modelopt.recipe import load_recipe

    resolved = load_recipe(recipe).model_dump(mode="json")
    return {"recipe/resolved_recipe.yaml": yaml.safe_dump(resolved, sort_keys=False)}


def run_tags(args: argparse.Namespace, tool: Tool) -> dict[str, str]:
    """This run's join keys, shared with whatever is later done with what it produced.

    ``checkpoint_path`` is the checkpoint the run *writes*, because that is what an export or
    an evaluation is later pointed at (NEL takes ``deployment.checkpoint_path``), and
    ``source_checkpoint_path`` is what it consumed, so a chain of runs joins on the pair: a
    distillation's source is the checkpoint it continues from, not the model that was
    quantized. Both are resolved, since a relative path is useless as a join key -- except a
    source that names no directory, such as a Hub ``org/name`` id.
    """
    source = tool.source(args) if tool.source else tool.model(args)
    checkpoint = tool.checkpoint(args)
    tags = {
        "model": Path(tool.model(args)).name,
        "source_checkpoint_path": (
            str(Path(source).resolve()) if os.path.exists(source) else str(source)
        ),
    }
    # Omitted rather than empty when the run writes no checkpoint: a search for runs that
    # produced one should not match it.
    if checkpoint is not None:
        tags["checkpoint_path"] = str(Path(checkpoint).resolve())
    return tags


def describe_run(args: argparse.Namespace, tool: Tool, world_size: int = 1) -> dict:
    """The keyword arguments :meth:`MlflowRunLogger.track` takes, for this run.

    Every command-line argument becomes a searchable param, so a flag added later is tracked
    without touching this. *world_size* is recorded separately because the parallelism flags
    say how a run was laid out but not how many processes it took.
    """
    skip = _NEVER_PARAMS | tool.non_params
    params = {k: v for k, v in vars(args).items() if k not in skip}
    params["world_size"] = world_size
    return {
        "params": params,
        "tags": run_tags(args, tool),
        "texts": tool.texts(args),
        "files": tool.outputs(args),
    }


@contextmanager
def tracked_run(
    args: argparse.Namespace,
    tool: Tool,
    is_main: bool,
    exported: Callable[[], bool],
    world_size: int = 1,
) -> Iterator[MlflowRunLogger]:
    """Track one invocation of *tool* for the duration of the block.

    Inert unless ``--mlflow`` settled a URI and this is the rank that records it, so the
    caller needs no branching. *is_main* is that rank, and also gates the writes every rank
    would otherwise race on; *exported* is read on the way out, once the run knows whether it
    wrote the checkpoint its pointer would claim.

    Example:
        >>> with tracked_run(args, HF_PTQ, is_main, lambda: args.exported, world_size):
        ...     quantize_and_export(args)
    """
    logger = MlflowRunLogger(
        args.mlflow or "",
        args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        enabled=bool(args.mlflow) and is_main,
        required=args.mlflow_required,
    )
    # None when the run writes no checkpoint at all -- a pruning run that only scores, say,
    # or a script that points each of several checkpoints at the run itself -- so there is
    # nothing to point at and nothing that could inherit a stale pointer.
    path = None
    if tool.settles_pointer and (checkpoint := tool.checkpoint(args)) is not None:
        path = Path(checkpoint)
    if not logger.enabled:
        # Gathering the inputs re-reads the recipe, so keep it off the untracked path.
        try:
            yield logger
        finally:
            if path is not None and is_main and _ask("exported flag", exported, False):
                drop_experiment_json(path)
        return

    described = describe_run(args, tool, world_size)
    logger.start(**described)

    def close(status: str) -> None:
        # Only a completed export may claim the checkpoint the pointer sits next to: the
        # directory usually exists before the weights do.
        wrote_it = path is not None and _ask("exported flag", exported, False)
        logger.log_experiment_json(path if wrote_it else None)
        logger.finish(
            status,
            files=described["files"],
            metrics=_ask("metrics", lambda: tool.metrics(args), {}),
        )

    with _closing_run(close):
        yield logger
