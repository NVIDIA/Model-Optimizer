# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Generate an ordinary launcher pipeline from an AutoQuant recipe and resource limits."""

import argparse
import base64
import json
import shlex
from pathlib import Path

import yaml

from modelopt.recipe import load_recipe, plan_auto_quantize
from modelopt.recipe.planner import _pipeline_candidates, _validate_execution_limits

__all__ = ["generate_auto_quantize_pipeline"]


def generate_auto_quantize_pipeline(
    recipe,
    model,
    dataset,
    work_dir,
    export_path,
    slurm_config,
    resources,
    calib_size=512,
    calib_seq=512,
    batch_size=1,
):
    """Expand independent formats and a validating coordinator; no stages are user-authored.

    Returns a launcher YAML dictionary. ``resources`` must set concurrency, total GPUs,
    per-worker GPU/host/input bytes, candidate/checkpoint bytes, total artifact bytes and retries.
    The initial rollout supports single-GPU workers and gradient, non-layerwise recipes.
    """
    resolved = load_recipe(recipe)
    plan = plan_auto_quantize(resolved)
    plan.require_supported()
    inputs = {
        "model": model,
        "dataset": dataset,
        "work_dir": work_dir,
        "export_path": export_path,
        "resources": resources,
        "calib_size": calib_size,
        "calib_seq": calib_seq,
        "batch_size": batch_size,
    }
    concurrency = _validate_execution_limits(plan, inputs)
    allocation_gpus = resources["allocated_gpus_per_worker"]
    if any(slurm_config.get(key, 1) != 1 for key in ("nodes", "ntasks_per_node")):
        raise ValueError("Automatic pipeline currently requires one node and rank per worker")
    if slurm_config.get("gpus_per_node", allocation_gpus) != allocation_gpus:
        raise ValueError("Slurm GPU allocation differs from the worker resource budget")
    srun_args = slurm_config.get("srun_args", ["--no-container-mount-home"]) or []
    if any(arg.startswith(("--gpu", "--gres", "-G")) for arg in srun_args):
        raise ValueError("Automatic workers own GPU step binding")
    if slurm_config.get("array") or slurm_config.get("local"):
        raise ValueError("Automatic workers require independent Slurm allocations")
    candidates = _pipeline_candidates(resolved)
    recipe_data = resolved.model_dump()
    recipe_data["metadata"]["recipe_type"] = resolved.recipe_type.value
    spec = {
        "recipe_yaml": yaml.safe_dump(recipe_data, sort_keys=False),
        **inputs,
        "search_digest": plan.as_dict()["search_digest"],
    }
    allocation = {
        **slurm_config,
        "nodes": 1,
        "ntasks_per_node": 1,
        "gpus_per_node": allocation_gpus,
        "srun_args": [*srun_args, "--gpus-per-task=1"],
        "mem": f"{resources['host_bytes'] // 1024**2}M",
    }
    command = shlex.join(
        ["python", "modules/Model-Optimizer/examples/hf_ptq/hf_ptq.py", "--auto_quantize_pipeline"]
    )
    tasks = []
    for index in [*range(len(candidates)), None]:
        payload = json.dumps(
            {
                **spec,
                "candidate": index,
                "candidate_signature": candidates[index].checkpoint_signature
                if index is not None
                else None,
            },
            sort_keys=True,
            allow_nan=False,
        )
        # NeMo embeds environment values in shell exports without quoting them.
        payload = base64.b64encode(payload.encode()).decode("ascii")
        if len(payload) > 64 * 1024:
            raise ValueError("Resolved pipeline launch specification exceeds 64 KiB")
        attempts = resources["retries"] + 1 if index is not None else 1
        tasks.append(
            {
                "inline": f"for attempt in $(seq {attempts}); do {command} && exit 0; done; exit 1",
                "environment": [{"MODELOPT_AUTOQUANT_PIPELINE": payload}],
                "slurm_config": dict(allocation),
                "wait_for_all": index is None,
            }
        )
    return {
        "pipeline": {"tasks": tasks, "max_concurrent_tasks": concurrency, "allow_to_fail": True}
    }


def main():
    """Generate a launcher YAML from a YAML containing recipe, data and resource parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    result = generate_auto_quantize_pipeline(**config)
    with Path(args.output).open("x") as stream:
        yaml.safe_dump(result, stream, sort_keys=False)


if __name__ == "__main__":
    main()
