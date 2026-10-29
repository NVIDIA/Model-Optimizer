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

"""Internal adapter for automatically generated format tasks."""

import base64
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml
from autoquant_export import export_lease
from autoquant_workspace import PipelineWorkspace

from modelopt.recipe import ModelOptAutoQuantizeRecipe, plan_auto_quantize
from modelopt.recipe.planner import (
    _pipeline_candidates,
    _pipeline_storage_bytes,
    _validate_execution_limits,
)
from modelopt.recipe.presets import KV_CACHE_NONE

__all__ = []


def publish_json(path, value, args):
    """Publish immutable metadata without exposing a partially written document."""
    _workspace(args).publish(path, json.dumps(value, sort_keys=True, allow_nan=False))


def _workspace(args):
    root = Path(args.auto_quantize_calibration_dir).parent
    return PipelineWorkspace(root.parent, root.name, args.auto_quantize_resources["metadata_bytes"])


@contextmanager
def pipeline_run(args):
    """Hold execution leases until all candidate or coordinator work has finished."""
    if not getattr(args, "auto_quantize_pipeline", False):
        yield
        return
    workspace = _workspace(args)
    with workspace.run(
        args.auto_quantize_calibration_candidate,
        args.auto_quantize_concurrency,
    ):
        if args.auto_quantize_calibration_candidate is None:
            with export_lease(args, workspace):
                yield
        else:
            yield


def configure_pipeline_args(args):
    """Resolve the frozen launch specification before any model loading."""
    raw = os.environ["MODELOPT_AUTOQUANT_PIPELINE"]
    if len(raw.encode()) > 64 * 1024:
        raise ValueError("Pipeline launch specification exceeds 64 KiB")
    spec = json.loads(base64.b64decode(raw, validate=True))
    if not isinstance(spec, dict) or set(spec) != {
        "recipe_yaml",
        "model",
        "dataset",
        "work_dir",
        "export_path",
        "resources",
        "calib_size",
        "calib_seq",
        "batch_size",
        "search_digest",
        "candidate",
        "candidate_signature",
    }:
        raise ValueError("Invalid automatic pipeline launch specification")
    recipe = ModelOptAutoQuantizeRecipe.model_validate(yaml.safe_load(spec["recipe_yaml"]))
    plan = plan_auto_quantize(recipe)
    plan.require_supported()
    args.auto_quantize_concurrency = _validate_execution_limits(plan, spec)
    allocated = os.environ.get("SLURM_GPUS_ON_NODE")
    if allocated is not None and (
        not allocated.isdecimal() or int(allocated) > spec["resources"]["allocated_gpus_per_worker"]
    ):
        raise ValueError("Actual Slurm GPU allocation exceeds the declared worker budget")
    candidates = _pipeline_candidates(recipe)
    args.auto_quantize_export_available_bytes = spec["resources"][
        "artifact_storage_bytes"
    ] - _pipeline_storage_bytes(len(candidates), spec["resources"])
    index = spec["candidate"]
    if index is not None and (type(index) is not int or not 0 <= index < len(candidates)):
        raise ValueError("Invalid format worker index")
    expected = candidates[index].checkpoint_signature if index is not None else None
    if spec["candidate_signature"] != expected:
        raise ValueError("Format worker identity differs from the resolved recipe")
    if plan.as_dict()["search_digest"] != spec["search_digest"]:
        raise ValueError("Automatic pipeline plan changed between submission and execution")
    if not Path(spec["model"]).is_dir():
        raise ValueError("Automatic workers require an existing shared source checkpoint")
    root = Path(spec["work_dir"]) / spec["search_digest"]
    recipe_id = hashlib.sha256(spec["recipe_yaml"].encode()).hexdigest()
    recipe_path = root / f"recipe-{recipe_id}.yaml"
    workspace = PipelineWorkspace(
        spec["work_dir"], spec["search_digest"], spec["resources"]["metadata_bytes"]
    )
    workspace.bind(spec)
    workspace.publish(recipe_path, spec["recipe_yaml"])
    args.recipe = str(recipe_path)
    args.pyt_ckpt_path = spec["model"]
    args.dataset = spec["dataset"]
    args.calib_size = str(spec["calib_size"])
    args.calib_seq = spec["calib_seq"]
    args.batch_size = spec["batch_size"]
    args.skip_generate = True
    args.verbose = False
    args.kv_cache_qformat = KV_CACHE_NONE
    args.export_path = spec["export_path"]
    args.auto_quantize_calibration_dir = str(root / "formats")
    args.auto_quantize_calibration_candidate = spec["candidate"]
    args.auto_quantize_calibration_max_input_bytes = spec["resources"]["input_bytes"]
    args.auto_quantize_checkpoint = str(root / "search") if spec["candidate"] is None else None
    args.auto_quantize_assignment = str(root / f"assignment-{recipe_id}.json")
    args.auto_quantize_resources = spec["resources"]
    args.auto_quantize_final_layerwise = any(
        f["final_layerwise"] for f in plan.as_dict()["search"]["formats"]
    )


def configure_pipeline_device(args):
    """Enforce the initial one-GPU layout and PyTorch allocator ceiling before loading weights."""
    if torch.cuda.device_count() != 1 or int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Automatic workers currently support exactly one visible GPU and rank")
    total = torch.cuda.get_device_properties(0).total_memory
    limit = args.auto_quantize_resources["gpu_bytes"]
    if not 0 < limit <= total:
        raise ValueError("Requested GPU memory ceiling is outside the visible GPU capacity")
    torch.cuda.set_per_process_memory_fraction(limit / total, 0)
