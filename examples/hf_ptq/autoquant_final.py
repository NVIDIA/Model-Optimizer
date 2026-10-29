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

"""Recover final layerwise calibration after the search model leaves scope."""

import hashlib
import json
import os
import shutil
from contextlib import ExitStack, nullcontext
from pathlib import Path

from autoquant_export import verify_export
from autoquant_pipeline import _workspace, publish_json

from modelopt.recipe import AutoQuantizeAssignment, load_recipe, plan_auto_quantize
from modelopt.torch.quantization._calibration_workers import _CalibrationSession
from modelopt.torch.quantization._layerwise_checkpoint import _assignment_checkpoint
from modelopt.torch.quantization._selected_model import _selected_model
from modelopt.torch.quantization._selected_search import (
    _load_selected_search,
    _selected_search_binding,
)
from modelopt.torch.quantization.model_calib import layerwise_calibrate

__all__ = []


def _read(path, args):
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > args.auto_quantize_resources["metadata_bytes"]
    ):
        raise ValueError("Finalization requires bounded regular metadata")
    return path.read_bytes()


def _binding_path(args):
    return Path(args.auto_quantize_assignment).with_suffix(".search.json")


def request_final_reload(args, assignment):
    """Publish a small binding after coordinator validation; never copy the search payload."""
    if not any(group["final_layerwise"] for group in assignment.as_dict()["groups"]):
        return
    binding = _selected_search_binding(
        args.auto_quantize_checkpoint, assignment, args.auto_quantize_resources["checkpoint_bytes"]
    )
    publish_json(_binding_path(args), binding, args)
    args.auto_quantize_reload = True


def reclaim_final_checkpoint(args):
    """Reclaim the sole owned final checkpoint only after verifying its complete export."""
    workspace = _workspace(args)
    owner_path, checkpoint, trash = (
        workspace.root / name for name in ("final.json", "final", ".final-reclaimed")
    )
    if not owner_path.exists():
        if checkpoint.exists() or trash.exists():
            raise ValueError("Refusing an unowned final-calibration directory")
        return
    owner = json.loads(_read(owner_path, args))
    if set(owner) != {"assignment_file", "export_path", "checkpoint"}:
        raise ValueError("Invalid final-calibration ownership metadata")
    assignment_path = Path(owner["assignment_file"])
    if assignment_path.parent != workspace.root:
        raise ValueError("Final assignment must belong to this workspace")
    assignment = hashlib.sha256(_read(assignment_path, args)).hexdigest()
    destination = Path(owner["export_path"])
    lease = (
        nullcontext()
        if destination == Path(args.export_path).resolve()
        else workspace._lock(destination.with_name(f".{destination.name}.autoquant.lock"))
    )
    with lease:
        verify_export(destination, assignment, args.auto_quantize_resources)
        if (
            checkpoint.is_symlink()
            or trash.is_symlink()
            or (checkpoint.exists() and trash.exists())
        ):
            raise ValueError("Refusing a conflicting final-calibration directory")
        if checkpoint.exists():
            with _assignment_checkpoint(checkpoint, **owner["checkpoint"]) as state:
                if state.start_layer != state.num_layers:
                    raise ValueError("Published export has an incomplete calibration checkpoint")
            # The owner survives an interruption during deletion of the renamed directory.
            os.replace(checkpoint, trash)
        if trash.exists():
            shutil.rmtree(trash)
        owner_path.unlink()


def finalize_selected(args, model, options):
    """Restore a frozen assignment on fresh weights, or return false when search is needed."""
    if not _binding_path(args).exists():
        return False
    assignment = AutoQuantizeAssignment(
        json.dumps(
            json.loads(_read(args.auto_quantize_assignment, args)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    steps = (options["num_calib_steps"], options["num_score_steps"])
    session = _CalibrationSession(
        options["calibration"],
        model,
        options["data_loader"],
        max(steps),
        [options["forward_step"], options["loss_func"], None],
    )
    plan = plan_auto_quantize(load_recipe(args.recipe))
    state = _load_selected_search(
        args.auto_quantize_checkpoint,
        json.loads(_read(_binding_path(args), args)),
        assignment,
        plan,
        session.identity,
        steps,
        args.auto_quantize_resources["checkpoint_bytes"],
    )
    if Path(args.export_path).exists():
        verify_export(
            Path(args.export_path),
            hashlib.sha256(_read(args.auto_quantize_assignment, args)).hexdigest(),
            args.auto_quantize_resources,
        )
        reclaim_final_checkpoint(args)
        return True
    settings, checkpoint_options = {}, {}
    workspace = _workspace(args)
    checkpoint = None

    def prepare_checkpoint(layerwise_settings, num_layers):
        nonlocal checkpoint
        settings.update(layerwise_settings)
        settings["calib_mutates_weights"] = False  # GGML finalization preserves source weights.
        settings.pop("checkpoint_dir")
        settings.pop("export_dir")
        checkpoint_options.update(
            assignment_digest=assignment.digest,
            provenance=session.identity,
            num_layers=num_layers,
            max_bytes=args.auto_quantize_resources["final_checkpoint_bytes"],
            save_every=settings["save_every"],
            calib_mutates_weights=False,
        )
        owner_path = workspace.root / "final.json"
        owner = {
            "assignment_file": str(Path(args.auto_quantize_assignment)),
            "export_path": str(Path(args.export_path).resolve()),
            "checkpoint": checkpoint_options,
        }
        # Match the persisted representation, including CUDA capability tuples.
        owner = json.loads(json.dumps(owner, allow_nan=False))
        if owner_path.exists() and json.loads(_read(owner_path, args)) != owner:
            reclaim_final_checkpoint(args)
        if not owner_path.exists() and any(
            (workspace.root / name).exists() for name in ("final", ".final-reclaimed")
        ):
            raise ValueError("Refusing unowned final-calibration data")
        publish_json(owner_path, owner, args)
        checkpoint = stack.enter_context(
            _assignment_checkpoint(workspace.root / "final", **checkpoint_options)
        )

    def replay(target):
        for batch in session:
            options["forward_step"](target, batch)

    with ExitStack() as stack:
        dispatch = _selected_model(
            model, assignment, plan, state, session.identity, before_conversion=prepare_checkpoint
        )
        del state
        assert checkpoint is not None
        layerwise_calibrate(
            model,
            replay,
            dispatch,
            _checkpoint_state=checkpoint,
            _max_activation_bytes=args.auto_quantize_resources["activation_bytes"],
            **settings,
        )
    return True
