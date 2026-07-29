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
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import modelopt

__all__ = [
    "MlflowRunLogger",
    "TeeStream",
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


def _redact(value: Any) -> Any:
    """Mask credentials embedded in a URI, leaving non-strings untouched."""
    return _URI_USERINFO.sub(_MASK, value) if isinstance(value, str) else value


def _redact_argv(argv: list[str]) -> list[str]:
    """Mask the value of any ``--*token*`` style option, and credentials in any URI."""
    redacted: list[str] = []
    mask_next = False
    for token in argv:
        if mask_next and not token.startswith("-"):
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
    free of subprocess use.
    """
    git_dir = Path(__file__).resolve().parents[3] / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head[:9]  # detached HEAD
        ref = head.removeprefix("ref: ")
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text().strip()[:9]
        for line in (git_dir / "packed-refs").read_text().splitlines():
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha[:9]
    except OSError:
        pass
    return "unknown"


def _command_text() -> str:
    """The invocation, as a copy-pasteable line."""
    lines = [shlex.join([sys.executable, *_redact_argv(sys.argv)])]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        lines += [
            "",
            f"# Launched under torchrun with WORLD_SIZE={world_size}, "
            f"LOCAL_WORLD_SIZE={os.environ.get('LOCAL_WORLD_SIZE', '?')}. The torchrun "
            "wrapper is not part of sys.argv and is therefore not shown above.",
        ]
    return "\n".join(lines) + "\n"


class TeeStream:
    """Mirror a text stream to *sink* while passing writes through to *stream*.

    Scripts that report progress with bare ``print()`` have no log file; wrapping
    ``sys.stdout``/``sys.stderr`` in this is what produces one. Attribute access falls
    through to the wrapped stream so ``isatty()`` keeps progress bars behaving. Native
    (C-level) writes go straight to the real file descriptor and are *not* captured.
    """

    def __init__(self, stream, sink):
        """Wrap *stream*, mirroring everything written to it into the open file *sink*."""
        self._stream = stream
        self._sink = sink

    def write(self, data: str) -> int:
        """Write to both the original stream and the sink."""
        self._stream.write(data)
        if not self._sink.closed:
            self._sink.write(data)
        return len(data)

    def flush(self) -> None:
        """Flush both the original stream and the sink."""
        self._stream.flush()
        if not self._sink.closed:
            self._sink.flush()

    def __getattr__(self, name):
        # Guard the wrapped attributes themselves: __getattr__ runs whenever they are absent
        # (during unpickling, or on a copy), and delegating then would recurse forever.
        if name in ("_stream", "_sink"):
            raise AttributeError(name)
        return getattr(self._stream, name)


class MlflowRunLogger:
    """Record one script invocation as an MLflow run.

    :meth:`start` verifies the server and opens the run *before* the expensive work begins,
    so a bad URI or a missing token fails in seconds rather than after hours; it also
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
    callers skip non-main ranks or an absent flag.

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
    ):
        """Configure the run without contacting the server; see the class docstring."""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.enabled = enabled
        self._mlflow: Any = None
        self._run: Any = None
        self._log_path: Path | None = None
        self._saved_streams: tuple | None = None
        self._redirected_handlers: list[tuple[logging.StreamHandler, Any]] = []
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
    ) -> None:
        """Open the run: capture output, verify the server, upload the inputs.

        *params* are searchable; *tags* merge over the defaults (user, hostname, ModelOpt
        version and commit); *texts* maps artifact path to content, uploaded here rather
        than at the end so it survives a crash.

        Raises:
            ImportError: If ``mlflow`` is not installed.
            ConnectionError: If the tracking server is unreachable.
        """
        if not self.enabled or self._run is not None:
            return
        self._start_time = time.time()
        self._start_capture()
        try:
            self._open_run()
            self._log_inputs(params, tags, texts)
        except Exception:
            # start_run() may already have succeeded, and the caller gets an exception
            # rather than a logger to call finish() on, so close the run here.
            self._abort_run()
            self._stop_capture()
            raise

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
        respectively; a *files* entry whose file is absent is skipped, so callers can list
        optional outputs. *metrics* merges over the default ``total_time_s``.
        """
        if not self.enabled or self._run is None:
            self._stop_capture()
            return
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

    def _open_run(self) -> None:
        try:
            import mlflow  # optional dependency: only needed once tracking is enabled
        except ImportError as e:
            raise ImportError(
                "MLflow tracking requires the 'mlflow' package: pip install nvidia-modelopt[mlflow]"
            ) from e

        self._check_reachable()
        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(
            run_name=self.run_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        )
        print(f"[mlflow] experiment: {self.experiment_name}\n[mlflow] run: {self.run_url}")

    def _check_reachable(self) -> None:
        """Fail fast on an unreachable host.

        Any HTTP response -- including 401 -- means the host is up, so authorization is
        left to the first real API call, which reports it precisely.
        """
        import requests

        try:
            requests.get(f"{self.tracking_uri}/health", timeout=10)
        except requests.RequestException as e:
            raise ConnectionError(f"MLflow server {self.tracking_uri} is unreachable: {e}") from e

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
                "command.txt": _command_text(),
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
            if Path(local).is_file():
                self._log_file(artifact_path, Path(local))

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
        sys.stdout, sys.stderr = TeeStream(stdout, sink), TeeStream(stderr, sink)
        self._redirect_log_handlers({stdout: sys.stdout, stderr: sys.stderr})
        print(f"[mlflow] capturing this run's log to {self._log_path}")

    def _redirect_log_handlers(self, replacements: dict) -> None:
        """Point already-configured logging handlers at the tee.

        Libraries such as transformers and huggingface_hub bind ``sys.stderr`` into a
        ``StreamHandler`` when they are imported, which happens long before the capture
        starts; without this their warnings reach the console but never the log.
        """
        self._redirected_handlers = []
        loggers = [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]
        for logger in loggers:
            for handler in getattr(logger, "handlers", []):
                if isinstance(handler, logging.StreamHandler) and handler.stream in replacements:
                    self._redirected_handlers.append((handler, handler.stream))
                    handler.setStream(replacements[handler.stream])

    def _stop_capture(self) -> None:
        # Only handlers seen at start are handed back; any registered since keeps writing to the
        # tee, which is why TeeStream tolerates a closed sink rather than raising on it.
        if self._saved_streams is None:
            return
        for handler, stream in self._redirected_handlers:
            handler.setStream(stream)
        self._redirected_handlers = []
        sys.stdout, sys.stderr, sink = self._saved_streams
        sink.close()
        self._saved_streams = None
        if self._log_path is not None:
            shutil.rmtree(self._log_path.parent, ignore_errors=True)
            self._log_path = None
