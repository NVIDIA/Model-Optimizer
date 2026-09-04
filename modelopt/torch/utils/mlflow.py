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

import contextlib
import getpass
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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import modelopt
from modelopt.torch.utils.logging import TeeStream

__all__ = [
    "MlflowRunLogger",
    "command_text",
    "current_user",
    "default_experiment_name",
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

    @property
    def run_url(self) -> str:
        """Link to this run in the MLflow UI, or ``""`` before the run is open."""
        if self._run is None:
            return ""
        info = self._run.info
        uri = _redact(self.tracking_uri)
        return f"{uri}/#/experiments/{info.experiment_id}/runs/{info.run_id}"

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
        status = "FAILED"
        try:
            yield self
            status = "FINISHED"
        finally:
            self.finish(status, files=files, metrics=metrics)

    def log_text(self, artifact_path: str, text: str) -> None:
        """Upload *text* as an artifact while the run is open, best-effort.

        For a value that is only settled midway through the run and is worth having even if
        the run later crashes -- the quantization config a calibration is about to apply,
        say. :meth:`start` and :meth:`finish` cover everything known at the two ends.
        """
        if not self.enabled or self._run is None:
            return
        try:
            self._log_texts({artifact_path: text})
        except Exception as e:
            print(f"[mlflow] WARNING: could not upload {artifact_path}: {e}")

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
        try:
            self._log_outputs(texts, files, metrics)
        except Exception as e:
            print(f"[mlflow] WARNING: could not upload run outputs: {e}")
        self._stop_capture()
        try:
            self._mlflow.end_run(status=status)
            print(f"[mlflow] {status}: {self.run_url}")
        except Exception as e:
            print(f"[mlflow] WARNING: could not close the run: {e}")

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
        self._run = mlflow.start_run(
            run_name=self.run_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        )
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
