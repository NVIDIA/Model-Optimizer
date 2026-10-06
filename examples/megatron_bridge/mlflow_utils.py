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

"""Shared MLflow wiring for the Megatron-Bridge examples, mirroring ``examples/hf_ptq``.

Each script declares what it records as a :class:`~modelopt.torch.utils.mlflow.Tool` beside
its own flags; this module knows none of them, only how a run is opened and closed here. The
single-pass scripts open their own through :func:`~modelopt.torch.utils.mlflow.tracked_run`,
while ``distill.py`` opens one on the *last* rank for Megatron-Bridge to join, since that is
where it looks -- so one run carries both the invocation and the training metrics.

Every rank parses the same flags, so a typo in the URI fails identically everywhere rather
than on one rank while the others wait in a collective.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils.mlflow import (
    Tool,
    default_run_name,
    log_active_run_experiment_json,
    mask_tracking_uri,
    run_tags,
    split_tracking_credentials,
    tracked_run,
)
from modelopt.torch.utils.mlflow import add_mlflow_args as _add_mlflow_args
from modelopt.torch.utils.mlflow import resolve_mlflow_args as _resolve_mlflow_args

# These scripts' own bookkeeping, on top of the tracking settings the library already keeps
# out of the params. Every Tool here passes it as ``non_params``.
NON_PARAMS = frozenset({"checkpoint_exported", "mlflow_log_checkpoints", "prune_score"})


def checkpoint_name(path: str) -> str:
    """The directory name a checkpoint path ends in, for naming the experiment."""
    return Path(path.rstrip("/")).name


def checkpoint_root(path: str) -> str:
    """The directory the run that saved *path* tagged, given an ``iter_*`` dir or that root."""
    return str(Path(path).parent) if checkpoint_name(path).startswith("iter_") else path


# Re-exported so a script imports the whole tracking CLI from one module; add_mlflow_args
# below wraps because it has something to add.
resolve_mlflow_args = _resolve_mlflow_args


def add_mlflow_args(
    parser: argparse.ArgumentParser, tool: Tool, log_checkpoints: bool = False
) -> None:
    """Add the tracking flags, plus *log_checkpoints* for a script that saves repeatedly."""
    _add_mlflow_args(parser, tool)
    if log_checkpoints:
        parser.add_argument(
            "--mlflow_log_checkpoints",
            "--mlflow-log-checkpoints",
            action="store_true",
            help=(
                "Upload every saved checkpoint to the MLflow server as an artifact. Off by "
                "default, unlike Megatron-Bridge's own setting: a distilled checkpoint is "
                "tens to hundreds of GB, sent over HTTP on every save."
            ),
        )


@contextmanager
def mlflow_run(args: argparse.Namespace, tool: Tool) -> Iterator[None]:
    """Track this invocation for the duration of the block, for the single-pass scripts;
    ``distill.py`` uses :func:`distill_run` instead."""
    with tracked_run(
        args,
        tool,
        is_main=dist.is_master(),
        # Only a tool that settles its own pointer is asked, so only those scripts define the
        # flag -- an export that points each checkpoint at the run as it writes it does not.
        exported=(lambda: args.checkpoint_exported) if tool.settles_pointer else (lambda: False),
        world_size=dist.size(),
    ) as logger:
        # An unreachable server disables an optional logger, and it already said so; leaving
        # --mlflow set would make record_exported_checkpoint warn a second time, about a
        # missing pointer, as if that were a separate fault.
        if dist.is_master() and not logger.enabled:
            args.mlflow = None
        yield


@contextmanager
def distill_run(args: argparse.Namespace, tool: Tool) -> Iterator[bool]:
    """Open the run Megatron-Bridge will join, yielding whether this rank owns it.

    A distillation run gets what the single-pass scripts get; the metrics and the resolved
    config are Megatron-Bridge's, logged into this same run.
    """
    # Settled on one rank before anything reads it: the rank that opens the run is not the
    # rank that writes run_config.yaml, and their clocks cross second boundaries.
    if args.mlflow and not args.mlflow_run_name:
        args.mlflow_run_name = dist.broadcast(default_run_name())
    # Captured before the block: after any dist.cleanup() every rank reports itself as the
    # last one of a world of one, and they would all reach the pointer.
    owns_the_run = dist.is_last_process()
    with tracked_run(
        args,
        tool,
        # The last rank, which is where Megatron-Bridge looks for an active run.
        is_main=owns_the_run,
        exported=lambda: False,  # the pointer is record_checkpoint_provenance's
        world_size=dist.size(),
    ) as logger:
        # An unreachable server disables an optional logger, and logger_kwargs would
        # otherwise hand Megatron-Bridge the same URI -- fatally, since it opens its run from
        # inside the training loop. Settled by the rank that tried, for all of them.
        if args.mlflow:
            args.mlflow = dist.broadcast(
                args.mlflow if logger.enabled else None, src=dist.size() - 1
            )
        yield owns_the_run


def logger_kwargs(args: argparse.Namespace, tool: Tool) -> dict:
    """The MLflow half of Megatron-Bridge's ``LoggerConfig``, from these flags.

    Megatron-Bridge enters its MLflow path on ``mlflow_experiment`` and finds the run
    :func:`distill_run` already opened. Empty unless tracking was requested: these fields
    landed in ``LoggerConfig`` in 0.6, so passing them always would break an untracked run on
    an older one.
    """
    if not args.mlflow:
        return {}
    # Credentials moved into MLflow's own variables: Megatron-Bridge records this config as
    # params and writes it into the checkpoint as run_config.yaml, so a user:token@ left here
    # would be durable in both.
    recordable = split_tracking_credentials(args.mlflow)
    if recordable is None:
        # Every rank declines, or LoggerConfig would differ by rank and the last one -- where
        # Megatron-Bridge logs from -- would enter its MLflow path with no tracking URI.
        if dist.is_master():
            print(
                f"[mlflow] WARNING: {mask_tracking_uri(args.mlflow)} carries half a user:token, "
                "which MLflow's own variables cannot hold, so Megatron-Bridge is given no "
                "tracking settings: this run is recorded without its training metrics. Export "
                "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD instead to get both."
            )
        return {}
    return {
        "mlflow_tracking_uri": recordable,
        "mlflow_experiment": args.mlflow_experiment,
        # Always set, because distill_run defaults it: MLflow would otherwise invent a
        # random name, while the flag documents the UTC start time.
        "mlflow_run_name": args.mlflow_run_name,
        "mlflow_tags": run_tags(args, tool),
        "mlflow_log_artifacts": args.mlflow_log_checkpoints,
    }


def checkpoint_marker(args: argparse.Namespace, tool: Tool) -> str | None:
    """The iteration the checkpoint directory last recorded, or ``None`` if it has none.

    Read before training and handed back after: the directory existing proves nothing, since
    a resumed run finds the previous run's checkpoint already there.
    """
    checkpoint = tool.checkpoint(args)
    if checkpoint is None:
        return None
    try:
        return (Path(checkpoint) / "latest_checkpointed_iteration.txt").read_text().strip()
    except OSError:
        return None


def record_exported_checkpoint(
    args: argparse.Namespace, checkpoint_dir: Path | str, is_main: bool
) -> None:
    """Point one exported checkpoint at the run that wrote it, for a script that writes
    several. *is_main* has to be captured before any ``dist.cleanup()``, after which every
    rank reports itself as rank 0 and they would all write."""
    if not is_main:
        return
    if not log_active_run_experiment_json(checkpoint_dir) and args.mlflow:
        print(
            f"[mlflow] WARNING: --mlflow was given but no run was found, so "
            f"{checkpoint_dir} carries no provenance pointer."
        )


def record_checkpoint_provenance(
    args: argparse.Namespace,
    tool: Tool,
    saved_before: str | None = None,
    is_main: bool = True,
) -> None:
    """Point the trained checkpoint at the run Megatron-Bridge opened for it, from *is_main*
    -- the rank that owns it, captured before any ``dist.cleanup()``.

    *saved_before* is :func:`checkpoint_marker` from before training: only once this run has
    moved it is the pointer settled, so a run that died before its own first save leaves the
    checkpoint it resumed from alone. After a save the pointer is this run's or absent.
    """
    checkpoint = tool.checkpoint(args)
    if checkpoint is None or not is_main:
        return
    marker = checkpoint_marker(args, tool)
    if marker is None or marker == saved_before:
        return
    # Called whether or not this run is tracked: an untracked re-run into a reused
    # --output_dir must clear the pointer it inherits, which is what the helper does when it
    # finds no run.
    if not log_active_run_experiment_json(checkpoint) and args.mlflow:
        # Tracking was asked for, so finding no run here is a failure rather than an
        # untracked job -- Megatron-Bridge opened none on this rank -- and the inherited
        # pointer has just been cleared. Silence would be indistinguishable from success.
        print(
            f"[mlflow] WARNING: --mlflow was given but no run was found on rank "
            f"{dist.rank()}, so {checkpoint} carries no provenance pointer."
        )
