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

"""MLflow tracking for the vLLM fake-quant server, mirroring ``examples/hf_ptq``.

The quantization this example performs happens inside the vLLM **worker** process, not in
``vllm_serve_fakequant.py``: the launcher is the API-server frontend, and the engine and its
workers are separate processes whose output it never sees. So the launcher only settles the
tracking configuration -- validating the URI, naming the experiment, recording the command
the user actually typed -- and publishes it through the environment, the same way every
other setting in this example reaches the workers. Rank 0 opens the run and uploads the
recipe, the effective quantization config, the calibration log and the quantizer summary.

The run covers weight load through warm-up and closes ``FINISHED`` when the server is ready
to serve, rather than staying open for the server's whole lifetime.

Nothing here imports vLLM, so the tracking can be exercised without it.
"""

import argparse
import contextlib
import os
import shutil
import tempfile
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

import modelopt.torch.quantization as mtq
from modelopt.recipe import load_recipe
from modelopt.torch.utils.mlflow import (
    MlflowRunLogger,
    command_text,
    default_experiment_name,
    validate_tracking_uri,
)

TOOL_NAME = "vllm_serve_fakequant"

# Written by the launcher, read by the workers. The two MLflow-owned names are MLflow's own,
# so a shell that already exports them opts in without touching the command line.
TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"
RUN_NAME_ENV = "MODELOPT_MLFLOW_RUN_NAME"
REQUIRED_ENV = "MODELOPT_MLFLOW_REQUIRED"
COMMAND_ENV = "MODELOPT_MLFLOW_COMMAND"

# Everything the rank-0 worker needs in its environment to reach the tracking server. The
# credentials are never set here, only forwarded when the launching shell exported them --
# without that, a Ray worker authenticates as nobody and the run fails to open.
MLFLOW_ENV_VARS = frozenset(
    {
        TRACKING_URI_ENV,
        EXPERIMENT_ENV,
        RUN_NAME_ENV,
        REQUIRED_ENV,
        COMMAND_ENV,
        "MLFLOW_TRACKING_TOKEN",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_TRACKING_INSECURE_TLS",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
    }
)


def add_mlflow_args(parser: argparse.ArgumentParser) -> None:
    """Add the MLflow tracking flags to the launcher's parser.

    The multi-word flags are registered under both spellings. vLLM's
    ``FlexibleArgumentParser`` rewrites every ``--foo_bar`` on the command line to
    ``--foo-bar`` before matching, so the dashed spelling is the one that has to exist for
    the flag to be reachable at all; the underscored spelling is what ``hf_ptq`` uses and
    keeps these usable with a plain ``argparse.ArgumentParser``.
    """
    parser.add_argument(
        "--mlflow",
        default=None,
        help=(
            "Track this server's calibration on an MLflow server "
            "(e.g. https://<your-mlflow-server>/), uploading the command, the resolved "
            "recipe, the quantization config actually applied, the worker log and the "
            "quantizer summary. This is the quantization tracking server, which is "
            "unrelated to any tracking server an evaluation harness exports its scores to. "
            "MLflow's own $MLFLOW_TRACKING_URI enables tracking without this flag, which "
            "overrides it. A URI taken from the environment is best-effort: if it is "
            "unusable the server warns and serves untracked."
        ),
    )
    parser.add_argument(
        "--mlflow-experiment",
        "--mlflow_experiment",
        default=None,
        help=(
            "MLflow experiment name. Default: "
            f"$USER/{TOOL_NAME}/<model basename>-<recipe name, or the quantization config>."
        ),
    )
    parser.add_argument(
        "--mlflow-run-name",
        "--mlflow_run_name",
        default=None,
        help="MLflow run name. Default: the UTC start time as YYYYmmdd-HHMMSS.",
    )


def resolve_mlflow_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Settle the tracking configuration and publish it to the worker processes.

    Validating here rather than in the worker is what makes a typo in the URI fail at launch
    instead of after the weights are on the GPUs. As in ``hf_ptq``, only ``--mlflow`` is a
    deliberate request and therefore fatal when unusable; ``$MLFLOW_TRACKING_URI`` is
    commonly exported for unrelated tooling and must not take a serve down with it.
    """
    required = args.mlflow is not None
    uri = args.mlflow or os.environ.get(TRACKING_URI_ENV) or None
    if uri:
        try:
            uri = validate_tracking_uri(uri)
        except ValueError as e:
            if required:
                parser.error(f"--mlflow: {e}")
            warnings.warn(f"Ignoring ${TRACKING_URI_ENV}, continuing untracked: {e}")
            uri = None
    if not uri:
        # A rejected URI must not reach the workers, which would try it again and fail there.
        os.environ.pop(TRACKING_URI_ENV, None)
        return

    os.environ[TRACKING_URI_ENV] = uri
    os.environ[REQUIRED_ENV] = "1" if required else "0"
    # The workers' own sys.argv is vLLM's spawn plumbing; this is the command a user ran.
    os.environ[COMMAND_ENV] = command_text()
    os.environ[EXPERIMENT_ENV] = (
        args.mlflow_experiment
        or os.environ.get(EXPERIMENT_ENV)
        or default_experiment_name(TOOL_NAME, args.model, quant_variant())
    )
    if args.mlflow_run_name:
        os.environ[RUN_NAME_ENV] = args.mlflow_run_name
    print(
        f"[mlflow] tracking to {_without_credentials(uri)}, experiment {os.environ[EXPERIMENT_ENV]}"
    )


def _without_credentials(uri: str) -> str:
    """Strip any ``user:token@`` from *uri*, for printing.

    ``MlflowRunLogger`` masks the same thing in every URI it prints or uploads, and this
    line ends up in the worker log that the run itself uploads, so it has to match.
    """
    parsed = urlparse(uri)
    return parsed._replace(netloc=parsed.netloc.rpartition("@")[2]).geturl()


def quant_variant() -> str:
    """What distinguishes this serve of the model, for the default experiment name.

    Read from the environment rather than taken as an argument because that is where this
    example's quantization settings live, and both the launcher and a worker that was
    started directly (``vllm serve --worker-cls fakequant_worker.FakeQuantWorker``) need it.
    """
    if recipe := os.environ.get("RECIPE_PATH"):
        return Path(recipe.rstrip("/")).stem
    quant_cfg = os.environ.get("QUANT_CFG")
    kv_quant_cfg = os.environ.get("KV_QUANT_CFG")
    if quant_cfg or kv_quant_cfg:
        return "-".join(cfg for cfg in (quant_cfg, kv_quant_cfg) if cfg)
    if os.environ.get("MODELOPT_STATE_PATH"):
        return "modelopt_state"
    if os.environ.get("QUANT_FILE_PATH"):
        return "quantizer_state"
    return "unquantized"


class FakeQuantMlflowTracker:
    """Records one vLLM fake-quant worker's calibrate-and-serve as an MLflow run.

    Inert unless the launcher published a tracking URI *and* this is the global rank-0
    worker, so the worker needs no branching: every method is a no-op otherwise, and the
    server behaves exactly as it did before tracking existed.
    """

    def __init__(self, worker: Any, quant_config: dict[str, Any]):
        """Configure the run from the environment; nothing contacts the server yet."""
        uri = os.environ.get(TRACKING_URI_ENV) or None
        self._quant_config = quant_config
        self._staging: Path | None = None
        self._files: dict[str, Path] = {}
        self._closed = False
        self._worker = worker
        self._logger = MlflowRunLogger(
            uri or "",
            os.environ.get(EXPERIMENT_ENV) or _fallback_experiment(worker),
            run_name=os.environ.get(RUN_NAME_ENV) or None,
            enabled=bool(uri) and getattr(worker, "rank", 0) == 0,
            required=os.environ.get(REQUIRED_ENV) == "1",
        )

    @property
    def enabled(self) -> bool:
        """Whether this worker records the run."""
        return self._logger.enabled

    def start(self) -> None:
        """Open the run and upload what is already known.

        Called before the weights load so an unreachable server or a missing token fails in
        seconds rather than after a multi-minute load followed by a calibration.
        """
        if not self._logger.enabled:
            return
        self._staging = Path(tempfile.mkdtemp(prefix="modelopt-vllm-mlflow-"))
        self._files = {"summary/quant_summary.txt": self._staging / ".quant_summary.txt"}
        try:
            self._logger.start(
                params={**self._quant_config, **_vllm_params(self._worker)},
                tags=_run_tags(self._worker),
                texts=self._start_texts(),
                files=self._files,
            )
        except BaseException:
            # An explicit --mlflow is fatal here by design; leave no staging directory behind.
            self._discard_staging()
            raise
        if not self._logger.enabled:
            # A URI from the environment is best-effort: start() reports an unusable server
            # by disabling itself rather than raising, and every later method -- finish()
            # included -- returns before reaching the cleanup. So clean up here instead.
            self._discard_staging()

    def log_quant_config(self, quant_cfg: Any) -> None:
        """Upload the merged ``QUANT_CFG``/``KV_QUANT_CFG`` config, when that is what ran.

        Only for the preset path, where this is the sole record of what was applied: the
        params carry the preset *names*, while the config that reaches ``mtq.quantize`` is
        those two deep-copied, merged, and -- for an MLA model -- extended at runtime with
        ``*kv_c_bmm_quantizer`` / ``*k_pe_bmm_quantizer`` by inspecting the loaded model.

        A recipe run skips it: ``get_quant_config`` returns the recipe's ``quantize``
        section unchanged, which ``recipe/resolved_recipe.yaml`` already carries.

        Uploaded as soon as it is known rather than at the end, because a run that dies
        during calibration is exactly the one this artifact is wanted for.
        """
        if not self._logger.enabled or self._quant_config.get("recipe_path"):
            return
        try:
            text = _dump_yaml(quant_cfg)
        except Exception as e:
            # Never let an unserializable config take down a serve that would have worked.
            print(f"[mlflow] WARNING: could not serialize the quantization config: {e}")
            return
        self._logger.log_text("recipe/quant_cfg.yaml", text)

    def log_quant_summary(self, model: Any) -> None:
        """Stage the per-quantizer summary for upload; a no-op when untracked."""
        if not self._logger.enabled or self._staging is None:
            return
        # Writes .quant_summary.txt and prints only its path, so the console copy the caller
        # already printed is not repeated.
        mtq.print_quant_summary(model, output_dir=str(self._staging))

    def finish(self, status: str) -> None:
        """Upload the log and the summary, and close the run with *status*.

        Only the first call has an effect: vLLM drives the worker through several guarded
        steps, and a run already closed as ``FINISHED`` must not be reopened or downgraded
        by a failure in whatever the server does next.
        """
        if not self._logger.enabled or self._closed:
            return
        self._closed = True
        try:
            self._logger.finish(status, files=self._files)
        finally:
            self._discard_staging()

    def _discard_staging(self) -> None:
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None

    @contextlib.contextmanager
    def fail_on_error(self) -> Iterator[None]:
        """Close the run as ``FAILED`` if the wrapped step raises, then re-raise.

        vLLM drives the worker through several calls; without this a failure in any of them
        would leave the run ``RUNNING`` forever, with no log attached.
        """
        try:
            yield
        except BaseException:
            self.finish("FAILED")
            raise

    def _start_texts(self) -> dict[str, str]:
        texts = {}
        if command := os.environ.get(COMMAND_ENV):
            texts["command.txt"] = command
        if recipe_path := self._quant_config.get("recipe_path"):
            # The resolved recipe, not the source file: a recipe may be a directory or use
            # $imports, and only the resolved form stands alone.
            texts["recipe/resolved_recipe.yaml"] = _dump_yaml(
                load_recipe(recipe_path).model_dump(mode="json")
            )
        return texts


def _dump_yaml(value: Any) -> str:
    """YAML for an artifact, unwrapping a pydantic model first.

    A recipe's ``quantize`` section is a ``QuantizeConfig``, which ``yaml.safe_dump`` cannot
    represent. Raising here is deliberate: a ``repr`` fallback would upload an unparseable
    one-line blob under a ``.yaml`` name, which looks like a successful artifact until
    someone tries to read it. The caller turns the failure into a warning instead.
    """
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return yaml.safe_dump(value, sort_keys=False)


def _fallback_experiment(worker: Any) -> str:
    """Name the experiment from the worker itself, for a worker started without the launcher."""
    model = _model_config_value(worker, "model") or "unknown"
    return default_experiment_name(TOOL_NAME, str(model), quant_variant())


def _model_config_value(worker: Any, name: str, default: Any = None) -> Any:
    config = getattr(getattr(worker, "vllm_config", None), "model_config", None)
    return getattr(config, name, default)


def _vllm_params(worker: Any) -> dict[str, Any]:
    """Serving settings worth searching on, best-effort across vLLM versions.

    Every field is read through ``getattr`` with a default: these configs are reshuffled
    between vLLM releases, and a renamed attribute must not take down a serve that would
    otherwise have worked.
    """
    vllm_config = getattr(worker, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    params = {
        "model": _stringify(getattr(model_config, "model", None)),
        "served_model_name": _stringify(getattr(model_config, "served_model_name", None)),
        "dtype": _stringify(getattr(model_config, "dtype", None)),
        "max_model_len": getattr(model_config, "max_model_len", None),
        "enforce_eager": getattr(model_config, "enforce_eager", None),
        "quantization": _stringify(getattr(model_config, "quantization", None)),
        "tensor_parallel_size": getattr(parallel_config, "tensor_parallel_size", None),
        "pipeline_parallel_size": getattr(parallel_config, "pipeline_parallel_size", None),
        "data_parallel_size": getattr(parallel_config, "data_parallel_size", None),
        "world_size": getattr(parallel_config, "world_size", None),
        "kv_cache_dtype": _stringify(getattr(cache_config, "cache_dtype", None)),
        "vllm_version": _vllm_version(),
    }
    return {k: v for k, v in params.items() if v is not None}


def _run_tags(worker: Any) -> dict[str, str]:
    """Tags shared with ``hf_ptq``, so a checkpoint's PTQ run and the serves of it join up.

    ``checkpoint_path`` is the checkpoint being served, which is the same key ``hf_ptq``
    tags with the checkpoint it *writes* and the one an evaluation is pointed at
    (NEL's ``deployment.checkpoint_path``).
    """
    model = _stringify(_model_config_value(worker, "model")) or "unknown"
    tags = {
        "tool": TOOL_NAME,
        "model": Path(model).name,
        "checkpoint_path": _resolved(model),
        "vllm_version": _vllm_version(),
    }
    if served := _stringify(_model_config_value(worker, "served_model_name")):
        tags["served_model_name"] = served
    return tags


def _resolved(model: str) -> str:
    """The served checkpoint as an absolute path, or unchanged for a Hugging Face model id."""
    return str(Path(model).resolve()) if os.path.exists(model) else model


def _stringify(value: Any) -> str | None:
    """Flatten a config value to a string; vLLM stores some of these as one-element lists."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value) or None
    return str(value)


def _vllm_version() -> str:
    try:
        import vllm

        return str(vllm.__version__)
    except ImportError:
        return "unknown"
