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

"""MLflow tracking for ``hf_ptq.py``, enabled by ``--mlflow <tracking-uri>``.

Uploads the invocation, the resolved recipe, the run log and the quantization
summaries to an MLflow tracking server so a PTQ run can be reproduced from its
MLflow entry alone. Everything here is a no-op unless ``--mlflow`` is given.
"""

import getpass
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

import modelopt
from modelopt.recipe import load_recipe

__all__ = ["MlflowRunLogger", "TeeStream", "default_experiment_name", "validate_tracking_uri"]

# MLflow experiment names are stored in a VARCHAR(256) column by the SQL-backed stores.
_MAX_COMPONENT_LEN = 100
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def validate_tracking_uri(uri: str) -> str:
    """Validate an MLflow tracking URI and return it without a trailing slash.

    Only ``http(s)`` servers are accepted; MLflow's local ``file:`` / ``sqlite:``
    backends are not a useful destination for a shared PTQ record.

    Raises:
        ValueError: if *uri* is empty, has no host, or is not an http(s) URL.
    """
    if not uri:
        raise ValueError(
            "--mlflow requires a tracking URI (e.g. https://<your-mlflow-server>/); "
            "pass one explicitly or set MLFLOW_TRACKING_URI."
        )
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"--mlflow expects an http(s) tracking URI, got {uri!r}. "
            "Did you mean https://" + uri.lstrip("/") + "?"
        )
    if not parsed.netloc:
        raise ValueError(f"--mlflow tracking URI {uri!r} has no host.")
    return uri.rstrip("/")


def default_experiment_name(args) -> str:
    """Build the default experiment name, ``<user>/hf_ptq/<model>-<recipe or qformat>``.

    The model component is the checkpoint's basename, so both a local directory and a
    ``org/name`` Hugging Face id collapse to the same readable name.
    """
    model = Path(args.pyt_ckpt_path).name
    variant = Path(args.recipe).stem if args.recipe else args.qformat
    return f"{_sanitize(_user())}/hf_ptq/{_sanitize(model)}-{_sanitize(variant)}"


def _sanitize(component: str) -> str:
    """Reduce one experiment-name component to ``[A-Za-z0-9._-]``."""
    cleaned = _UNSAFE_CHARS.sub("_", component).strip("._-")
    return cleaned[:_MAX_COMPONENT_LEN] or "unknown"


def _user() -> str:
    try:
        return getpass.getuser()
    except OSError:  # container without a passwd entry for the uid
        return "unknown"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _command_text() -> str:
    """The invocation, as a copy-pasteable line."""
    lines = [shlex.join([sys.executable, *sys.argv])]
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

    ``hf_ptq.py`` reports progress with bare ``print()`` and has no log file; wrapping
    ``sys.stdout``/``sys.stderr`` in this is what produces one. Attribute access falls
    through to the wrapped stream so ``isatty()`` keeps progress bars behaving. Native
    (C-level) writes go straight to the real file descriptor and are *not* captured.
    """

    def __init__(self, stream, sink):
        self._stream = stream
        self._sink = sink

    def write(self, data: str) -> int:
        self._stream.write(data)
        if not self._sink.closed:
            self._sink.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        if not self._sink.closed:
            self._sink.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class MlflowRunLogger:
    """Records one ``hf_ptq.py`` invocation as an MLflow run.

    Disabled -- every method a no-op -- unless ``--mlflow`` was passed and this is the
    main rank. :meth:`start` validates the server and opens the run *before* the model
    loads, so a bad URI or a missing token fails in seconds rather than after hours of
    calibration; :meth:`finish` uploads the run outputs and closes the run.

    Example:
        >>> logger = MlflowRunLogger(args)
        >>> logger.start()
        >>> try:
        ...     quantize_and_export()
        ...     status = "FINISHED"
        ... finally:
        ...     logger.finish(status)
    """

    def __init__(self, args):
        self.args = args
        self.enabled = bool(args.mlflow) and args.dist_state.is_main
        self._run: Any = None
        self._log_dir: Path | None = None
        self._saved_streams: tuple | None = None
        self._redirected_handlers: list[tuple[logging.StreamHandler, Any]] = []
        self._start_time = 0.0

    def start(self) -> None:
        """Open the run: capture stdout, verify the server, log the inputs."""
        if not self.enabled:
            return
        self._start_time = time.time()
        self._start_capture()
        try:
            self._open_run()
            self._log_inputs()
        except Exception:
            self._stop_capture()
            raise

    def finish(self, status: str) -> None:
        """Upload the run outputs and close the run with *status*.

        Never raises: an MLflow outage must not turn a successful quantization into a
        failed exit.
        """
        if not self.enabled or self._run is None:
            self._stop_capture()
            return
        try:
            self._log_outputs()
        except Exception as e:
            print(f"[mlflow] WARNING: could not upload run outputs: {e}")
        self._stop_capture()
        try:
            import mlflow

            mlflow.end_run(status=status)
            print(f"[mlflow] {status}: {self._run_url()}")
        except Exception as e:
            print(f"[mlflow] WARNING: could not close the run: {e}")

    def _open_run(self) -> None:
        # Optional dependency: only examples using --mlflow need it installed.
        try:
            import mlflow
        except ImportError as e:
            raise ImportError("--mlflow requires the 'mlflow' package: pip install mlflow") from e

        self._check_reachable()
        mlflow.set_tracking_uri(self.args.mlflow)
        mlflow.set_experiment(self.args.mlflow_experiment)
        run_name = self.args.mlflow_run_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self._run = mlflow.start_run(run_name=run_name)
        print(f"[mlflow] experiment: {self.args.mlflow_experiment}")
        print(f"[mlflow] run: {self._run_url()}")

    def _check_reachable(self) -> None:
        """Fail fast on an unreachable host.

        Any HTTP response -- including 401 -- means the host is up, so authorization is
        left to the first real API call, which reports it precisely.
        """
        import requests

        try:
            requests.get(f"{self.args.mlflow}/health", timeout=10)
        except requests.RequestException as e:
            raise ConnectionError(f"MLflow server {self.args.mlflow} is unreachable: {e}") from e

    def _log_inputs(self) -> None:
        """Log what the run was given -- upfront, so a crash still leaves a usable record."""
        import mlflow

        args = self.args
        mlflow.log_params(
            {
                "model": args.pyt_ckpt_path,
                "qformat": args.qformat,
                "kv_cache_qformat": args.kv_cache_qformat,
                "recipe": args.recipe or "",
                "calib_size": args.calib_size,
                "calib_seq": args.calib_seq,
                "batch_size": args.batch_size,
                "sparsity_fmt": args.sparsity_fmt,
                "export_path": args.export_path,
                "world_size": args.dist_state.world_size,
            }
        )
        mlflow.set_tags(
            {
                "user": _user(),
                "hostname": socket.gethostname(),
                "modelopt_version": modelopt.__version__,
                "git_sha": _git_sha(),
            }
        )
        mlflow.log_text(_command_text(), "command.txt")
        if args.recipe:
            # The resolved recipe, not the source file: a recipe may be a directory or
            # use $imports, and only the resolved form is self-contained.
            resolved = load_recipe(args.recipe).model_dump(mode="json")
            mlflow.log_text(
                yaml.safe_dump(resolved, sort_keys=False), "recipe/resolved_recipe.yaml"
            )

    def _log_outputs(self) -> None:
        """Log what the run produced: the captured log and the quantization summaries."""
        import mlflow

        mlflow.log_metric("total_time_s", time.time() - self._start_time)
        sys.stdout.flush()
        sys.stderr.flush()
        if self._log_dir is not None:
            mlflow.log_artifact(str(self._log_dir / "hf_ptq.log"), artifact_path="logs")
        export_path = Path(self.args.export_path)
        # Written by print_quant_summary / save_expert_token_count_table; hidden names are
        # awkward to browse in the MLflow UI, so they are uploaded without the leading dot.
        for source, artifact in (
            (".quant_summary.txt", "summary/quant_summary.txt"),
            (".moe.html", "summary/moe.html"),
        ):
            if (export_path / source).is_file():
                text = (export_path / source).read_text(encoding="utf-8", errors="replace")
                mlflow.log_text(text, artifact)

    def _start_capture(self) -> None:
        self._log_dir = Path(tempfile.mkdtemp(prefix="hf_ptq-mlflow-"))
        sink = open(self._log_dir / "hf_ptq.log", "w", buffering=1, encoding="utf-8")
        original_stdout, original_stderr = sys.stdout, sys.stderr
        self._saved_streams = (original_stdout, original_stderr, sink)
        sys.stdout = TeeStream(original_stdout, sink)
        sys.stderr = TeeStream(original_stderr, sink)
        self._redirect_log_handlers(
            {original_stdout: sys.stdout, original_stderr: sys.stderr},
        )
        print(f"[mlflow] capturing this run's log to {self._log_dir / 'hf_ptq.log'}")

    def _redirect_log_handlers(self, replacements: dict) -> None:
        """Point already-configured logging handlers at the tee.

        transformers and huggingface_hub bind ``sys.stderr`` into a ``StreamHandler`` when
        they are imported, which happens long before the capture starts; without this their
        warnings -- rate limits, deprecations -- reach the console but never the log.
        """
        self._redirected_handlers = []
        loggers = [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]
        for logger in loggers:
            for handler in getattr(logger, "handlers", []):
                if isinstance(handler, logging.StreamHandler) and handler.stream in replacements:
                    self._redirected_handlers.append((handler, handler.stream))
                    handler.setStream(replacements[handler.stream])

    def _stop_capture(self) -> None:
        if self._saved_streams is None:
            return
        for handler, stream in self._redirected_handlers:
            handler.setStream(stream)
        self._redirected_handlers = []
        sys.stdout, sys.stderr, sink = self._saved_streams
        sink.close()
        self._saved_streams = None
        if self._log_dir is not None:
            shutil.rmtree(self._log_dir, ignore_errors=True)
            self._log_dir = None

    def _run_url(self) -> str:
        info = self._run.info
        return f"{self.args.mlflow}/#/experiments/{info.experiment_id}/runs/{info.run_id}"
