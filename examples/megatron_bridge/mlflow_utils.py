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

"""MLflow tracking for ``quantize.py``, mirroring ``examples/hf_ptq``.

Every rank parses and validates the same flags, so a typo in the URI fails identically
everywhere instead of on one rank while the others wait in a collective. Only the master rank
opens a run, so the log capture and the uploads happen once.

Nothing here imports Megatron, so the tracking can be exercised without it.
"""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import yaml

import modelopt.torch.utils.distributed as dist
from modelopt.recipe import load_recipe
from modelopt.torch.utils.mlflow import MlflowRunLogger, drop_experiment_json
from modelopt.torch.utils.mlflow import add_mlflow_args as _add_mlflow_args
from modelopt.torch.utils.mlflow import resolve_mlflow_args as _resolve_mlflow_args

TOOL_NAME = "megatron_bridge_quantize"

# The tracking settings describe the destination rather than the quantization, and
# checkpoint_exported is this script's own bookkeeping.
_NON_PARAM_ARGS = frozenset(
    {
        "checkpoint_exported",
        "mlflow",
        "mlflow_experiment",
        "mlflow_required",
        "mlflow_run_name",
    }
)


def add_mlflow_args(parser: argparse.ArgumentParser) -> None:
    """Add the MLflow tracking flags."""
    _add_mlflow_args(
        parser,
        TOOL_NAME,
        tracks=(
            "Track this run on an MLflow server (e.g. https://<your-mlflow-server>/), "
            "uploading the command, the resolved recipe, the run log and the quantizer "
            "summary, and writing .experiment.json into --export_megatron_path so the "
            "checkpoint names the run that produced it."
        ),
        variant_help="recipe name, or --quant_cfg if no --recipe",
    )


def resolve_mlflow_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Settle where tracking is configured from, and name the experiment."""
    _resolve_mlflow_args(
        args,
        parser,
        tool=TOOL_NAME,
        model=args.hf_model_name_or_path,
        # ``or "none"``: neither flag is required by the parser, and the run that reaches
        # get_quant_config without one fails there rather than while being named.
        variant=Path(args.recipe).stem if args.recipe else (args.quant_cfg or "none"),
    )


def _run_inputs(args: argparse.Namespace) -> tuple[dict, dict]:
    """Params and start-time artifacts describing this PTQ run."""
    params = {k: v for k, v in vars(args).items() if k not in _NON_PARAM_ARGS}
    # The parallelism flags say how the run was laid out but not how many GPUs it took:
    # data parallelism is implicit in the launcher's world size.
    params["world_size"] = dist.size()
    texts = {}
    if args.recipe:
        # The resolved recipe, not the source file: a recipe may be a directory or use
        # $imports, and only the resolved form is self-contained.
        resolved = load_recipe(args.recipe).model_dump(mode="json")
        texts["recipe/resolved_recipe.yaml"] = yaml.safe_dump(resolved, sort_keys=False)
    return params, texts


def _run_tags(args: argparse.Namespace) -> dict[str, str]:
    """Tags shared with ``hf_ptq`` and the evaluation side, so a PTQ run and whatever is
    later done with the checkpoint it produced can be found together on one server.

    ``checkpoint_path`` is the checkpoint this run *writes*, because that is what
    ``export_quantized_megatron_to_hf.py`` (and any QAD run) is later pointed at; the input is
    kept separately. It is resolved because a relative path is useless as a join key.
    """
    return {
        "model": Path(args.hf_model_name_or_path).name,
        "checkpoint_path": str(Path(args.export_megatron_path).resolve()),
        "source_checkpoint_path": args.hf_model_name_or_path,
    }


def _run_outputs(args: argparse.Namespace) -> dict[str, Path]:
    """Summaries written beside the checkpoint, keyed by artifact path.

    Uploaded without the leading dot, which is awkward to browse in the MLflow UI. A missing
    entry is skipped: the summary is written by the master rank only once quantization
    has finished.
    """
    return {"summary/quant_summary.txt": Path(args.export_megatron_path) / ".quant_summary.txt"}


@contextmanager
def mlflow_run(args: argparse.Namespace) -> Iterator[None]:
    """Track this invocation for the duration of the block, and keep the checkpoint's
    provenance pointer honest whether or not the run is tracked."""
    logger = MlflowRunLogger(
        args.mlflow or "",
        args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        enabled=bool(args.mlflow) and dist.is_master(),
        required=args.mlflow_required,
    )
    export_path = Path(args.export_megatron_path)
    if not logger.enabled:
        # Gathering the inputs re-reads the recipe, so keep it off the untracked path.
        try:
            yield
        finally:
            if args.checkpoint_exported and dist.is_master():
                drop_experiment_json(export_path)
        return
    params, texts = _run_inputs(args)
    with logger.track(
        params=params,
        tags=_run_tags(args),
        texts=texts,
        files=_run_outputs(args),
    ):
        try:
            yield
        finally:
            # Only a completed save may claim the checkpoint the pointer sits next to:
            # --export_megatron_path exists from print_quant_summary onwards, and may hold a
            # checkpoint from an earlier attempt whose weights this run never wrote.
            logger.log_experiment_json(export_path if args.checkpoint_exported else None)
