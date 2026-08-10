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

"""Track a specdec_bench run on an MLflow server.

A benchmark's results otherwise exist only as JSON under ``--save_dir``,
which somebody has to upload before anyone else can see them. Logging to
MLflow makes the run visible as soon as it finishes, with its acceptance
length, throughput and full configuration queryable next to every other
run, and the log and result files attached.

Mirrors the flags ``examples/hf_ptq`` uses (``--mlflow``,
``--mlflow_experiment``, ``--mlflow_run_name``), so a shell that already
exports ``MLFLOW_TRACKING_URI`` opts in without changing the command.
"""

import argparse
import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path

from modelopt.torch.utils.mlflow import (
    MlflowRunLogger,
    default_experiment_name,
    validate_tracking_uri,
)

# Result files each metric writes into --save_dir, keyed by the artifact
# path they take on in MLflow. Missing entries are skipped by the logger,
# so listing every metric's output here is safe regardless of which ran.
_RESULT_FILES = {
    "results/acceptance_rate.json": "acceptance_rate.json",
    "results/specbench_results.json": "specbench_results.json",
    "results/timing.json": "timing.json",
    "results/aa_timing.json": "aa_timing.json",
    "results/configuration.json": "configuration.json",
    "results/acceptance_rate_analysis.png": "acceptance_rate_analysis.png",
}

# argparse fields that configure tracking itself rather than the benchmark.
_NON_PARAM_ARGS = frozenset({"mlflow", "mlflow_experiment", "mlflow_required", "mlflow_run_name"})


def add_mlflow_args(parser: argparse.ArgumentParser) -> None:
    """Add the MLflow tracking flags."""
    parser.add_argument(
        "--mlflow",
        default=None,
        help=(
            "Track this run on an MLflow server (e.g. https://<your-mlflow-server>/), "
            "uploading the configuration, the acceptance/timing results and the run log. "
            "MLflow's own $MLFLOW_TRACKING_URI enables tracking without this flag, which "
            "overrides it. A URI taken from the environment is best-effort: if it is "
            "unusable the run warns and continues untracked."
        ),
    )
    parser.add_argument(
        "--mlflow_experiment",
        default=None,
        help=(
            "MLflow experiment name. Default: "
            "$USER/specdec_bench/<model basename>-<speculative algorithm>."
        ),
    )
    parser.add_argument(
        "--mlflow_run_name",
        default=None,
        help="MLflow run name. Default: the UTC start time as YYYYmmdd-HHMMSS.",
    )


def resolve_mlflow_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Settle where tracking is configured from, and name the experiment."""
    # MLflow's own variable enables tracking on its own; --mlflow overrides it.
    # Only the flag is a deliberate request, so only the flag is fatal when the
    # URI is unusable: the variable is commonly exported for unrelated tooling
    # and must not fail a benchmark that is about to spend GPU hours.
    args.mlflow_required = args.mlflow is not None
    args.mlflow = args.mlflow or os.environ.get("MLFLOW_TRACKING_URI") or None
    if not args.mlflow:
        return
    try:
        args.mlflow = validate_tracking_uri(args.mlflow)
    except ValueError as e:
        if args.mlflow_required:
            parser.error(f"--mlflow: {e}")
        warnings.warn(f"Ignoring MLFLOW_TRACKING_URI, continuing untracked: {e}")
        args.mlflow = None
        return
    args.mlflow_experiment = args.mlflow_experiment or default_experiment_name(
        "specdec_bench",
        args.model_dir or "unknown",
        args.speculative_algorithm or "NONE",
    )


def _run_tags(args: argparse.Namespace) -> dict[str, str]:
    """Identity of what ran, as searchable tags.

    ``draft_model`` is what distinguishes two runs that share a verifier,
    which is the comparison this benchmark exists to make. The published Hub
    id is kept whole — the org is precisely what separates two drafters
    trained for the same base model — and only a local path is shortened to
    its leaf, since the leading directories say nothing about the drafter.
    """
    tags = {
        "model": Path(args.model_dir).name if args.model_dir else "",
        "algorithm": args.speculative_algorithm or "NONE",
        "engine": args.engine or "",
        "dataset": args.dataset or "",
    }
    hub_id = getattr(args, "draft_huggingface_model_id", None)
    if hub_id:
        tags["draft_model"] = str(hub_id)
    elif args.draft_model_dir:
        tags["draft_model"] = Path(args.draft_model_dir).name
    return {k: v for k, v in tags.items() if v}


def _headline_metrics(save_dir: str | None) -> dict[str, float]:
    """Acceptance length and throughput, read back from the metric JSONs.

    Read from disk rather than passed in because each metric owns its own
    output format, and the file is what a later consumer sees anyway. A
    missing or malformed file yields no metric rather than failing the run:
    tracking must never be why a completed benchmark reports failure.
    """
    if not save_dir:
        return {}
    out: dict[str, float] = {}

    def _first(path: Path):
        try:
            with path.open() as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(payload, list):
            return payload[0] if payload else None
        return payload

    directory = Path(save_dir)
    acceptance = _first(directory / "acceptance_rate.json") or {}
    for key in ("Average_AL", "Average_AR"):
        if isinstance(acceptance.get(key), (int, float)):
            out["avg_al"] = float(acceptance[key])
            break
    categories = acceptance.get("Category_AL") or acceptance.get("Category_AR") or {}
    if isinstance(categories, dict):
        for name, value in categories.items():
            if isinstance(value, (int, float)):
                out[f"category_al/{name}"] = float(value)

    timing = _first(directory / "timing.json") or {}
    for source, name in (("Output TPS", "output_tps"), ("Output TPS/gpu", "output_tps_per_gpu")):
        if isinstance(timing.get(source), (int, float)):
            out[name] = float(timing[source])
    return out


@contextmanager
def mlflow_run(args: argparse.Namespace):
    """Track this invocation for the duration of the block, or do nothing.

    The benchmark's headline metrics don't exist until the run has finished
    writing them, so unlike ``MlflowRunLogger.track`` this reads them out of
    ``--save_dir`` on the way out. The result files themselves are named
    upfront: the logger uploads only the ones the run actually wrote.
    """
    logger = MlflowRunLogger(
        args.mlflow,
        getattr(args, "mlflow_experiment", None) or "",
        run_name=getattr(args, "mlflow_run_name", None),
        enabled=bool(args.mlflow),
        required=bool(getattr(args, "mlflow_required", False)),
    )
    if not logger.enabled:
        yield None
        return

    params = {k: v for k, v in vars(args).items() if k not in _NON_PARAM_ARGS}
    files = (
        {artifact: Path(args.save_dir) / name for artifact, name in _RESULT_FILES.items()}
        if args.save_dir
        else {}
    )

    logger.start(params=params, tags=_run_tags(args))
    status = "FAILED"
    try:
        yield logger
        status = "FINISHED"
    finally:
        # Metrics are read here rather than passed to start(): the benchmark
        # writes them during the block. A failed run still uploads whatever it
        # managed to produce, which is usually the point of looking at it.
        logger.finish(status, files=files, metrics=_headline_metrics(args.save_dir))
