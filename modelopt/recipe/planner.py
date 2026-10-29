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

"""Automatic AutoQuant planning from resolved recipes, without loading model weights."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from modelopt.torch.quantization.algorithms import QuantRecipe
from modelopt.torch.quantization.mode import (
    BaseCalibrateModeDescriptor,
    CalibrateModeRegistry,
    get_modelike_from_algo_cfg,
)

from .config import ModelOptAutoQuantizeRecipe

__all__ = [
    "AutoQuantizeAssignment",
    "AutoQuantizePlan",
    "freeze_auto_quantize_assignment",
    "plan_auto_quantize",
]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class AutoQuantizePlan:
    """Immutable planning snapshot; ``as_dict()`` returns a detached JSON-compatible view.

    A plan describes required work, not runtime validation of model layouts or resources.
    ``require_supported()`` must succeed before any model conversion or calibration.
    """

    _json: str

    def as_dict(self) -> dict:
        """Return the expanded plan without exposing mutable internal state."""
        return json.loads(self._json)

    def require_supported(self) -> None:
        """Reject unsupported execution or estimator contracts before work starts."""
        failures = self.as_dict()["unsupported"]
        if failures:
            raise ValueError("Unsupported automatic AutoQuant plan: " + "; ".join(failures))


@dataclass(frozen=True)
class AutoQuantizeAssignment:
    """Frozen group methods and mappings, bound to validated source and search provenance."""

    _json: str

    @property
    def digest(self) -> str:
        """Identity required by final-calibration and export recovery checkpoints."""
        return hashlib.sha256(self._json.encode()).hexdigest()

    def as_dict(self) -> dict:
        """Return detached assignment metadata; packed tensors stay in the checkpoint store."""
        return json.loads(self._json)


def freeze_auto_quantize_assignment(search_state, plan: AutoQuantizePlan, provenance: dict):
    """Freeze one complete solver result without flattening per-group calibration algorithms.

    The coordinator must validate provenance and packed artifacts before calling this function.
    This checks assignment coverage and ownership; it neither reads tensors nor mutates the model.
    """
    plan.require_supported()
    planned = plan.as_dict()
    for field in ("source", "inputs", "code"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            raise ValueError(f"Assignment requires validated {field} provenance")
    stats, best = search_state["candidate_stats"], search_state["best"]
    if not best["is_satisfied"] or not stats or set(stats) != set(best["recipe"]):
        raise ValueError("Cannot freeze an incomplete or infeasible assignment")
    if search_state["method"] != planned["search"]["method"]:
        raise ValueError("Assignment scoring method differs from the plan")
    setup = search_state.get("resolved_search_setup_signature")
    if not setup:
        raise ValueError("Assignment requires resolved runtime group provenance")
    runtime = {tuple(group[1]): group for group in setup}
    if len(runtime) != len(stats) or len(runtime) != len(setup):
        raise ValueError("Assignment runtime group coverage differs")
    formats = {f["id"]: f for f in planned["search"]["formats"]}
    groups, owned = [], set()
    for name in sorted(stats):
        stat, selected = stats[name], best["recipe"][name]
        if selected not in stat["formats"]:
            raise ValueError(f"Assignment selects an unavailable recipe for {name}")
        identity = (
            None
            if selected.is_no_quant
            else hashlib.sha256(selected.checkpoint_signature.encode()).hexdigest()
        )
        if identity is not None and identity not in formats:
            raise ValueError(f"Assignment recipe for {name} differs from the plan")
        modules, attrs = stat["module_names"], stat.get("quantizer_attrs", {})
        if (
            not modules
            or len(set(modules)) != len(modules)
            or owned.intersection(modules)
            or set(attrs) != set(modules)
            or any(not value or len(set(value)) != len(value) for value in attrs.values())
        ):
            raise ValueError(f"Incomplete or conflicting group mapping for {name}")
        owned.update(modules)
        expected = runtime.get(tuple(sorted(modules)))
        actual = (
            tuple((module, tuple(attrs[module])) for module in sorted(attrs)),
            tuple(sorted(r.checkpoint_signature for r in stat["formats"])),
            stat["allow_no_quant"],
            stat["is_fixed"],
            float(stat["cost_weight"]),
        )
        if expected is None or actual != (expected[2], *expected[4:]):
            raise ValueError(f"Assignment runtime mapping differs for {name}")
        groups.append(
            {
                "name": name,
                "format_id": identity,
                "configuration_yaml": yaml.safe_dump(selected.config.model_dump(), sort_keys=False),
                "module_names": sorted(modules),
                "quantizer_attrs": attrs,
                "is_fixed": stat["is_fixed"],
                "cost_weight": stat["cost_weight"],
                "final_layerwise": formats[identity]["final_layerwise"] if identity else False,
            }
        )
    return AutoQuantizeAssignment(
        _canonical(
            {
                "schema_version": 1,
                "plan_digest": planned["search_digest"],
                "provenance": provenance,
                "runtime_groups": setup,
                "groups": groups,
                "constraints": best["constraints"],
                "requested_constraints": planned["constraints"],
                "cost_denominator": search_state["cost_denominator"],
                "disabled_layers": search_state["disabled_layers"],
                "cost": search_state["cost"],
            }
        )
    )


def plan_auto_quantize(recipe: ModelOptAutoQuantizeRecipe) -> AutoQuantizePlan:
    """Expand an ordinary, already-loaded recipe into internal format tasks and dependencies.

    Example:
        ``plan = plan_auto_quantize(load_recipe(recipe_path))``
        ``plan.require_supported()``

    Each task retains the complete canonical quantization configuration and algorithm chain.
    BF16 is an internal scoring reference and does not create a calibration task. Layerwise
    search estimation is unsupported until the algorithm integration declares and validates it.
    This function neither launches workers nor changes legacy ``auto_quantize`` behavior.
    """
    if not isinstance(recipe, ModelOptAutoQuantizeRecipe):
        raise TypeError("Automatic AutoQuant planning requires an AutoQuantize recipe")
    aq = recipe.auto_quantize
    unsupported = []
    if aq.auto_quantize_method != "gradient" or aq.constraints.cost_model == "kv_cache":
        unsupported.append("initial automatic execution supports gradient weight search only")
    if aq.kv_cache is not None or recipe.kv_auto_quantize is not None:
        unsupported.append("automatic KV calibration/search is not integrated")

    formats = {}

    def include(config):
        candidate = QuantRecipe(config.model_dump(), name="FORMAT")
        if candidate.is_no_quant:
            return None
        signature = candidate.checkpoint_signature
        identity = hashlib.sha256(signature.encode()).hexdigest()
        if identity in formats:
            return identity
        algorithms, final_layerwise = [], False
        for mode_name, settings in get_modelike_from_algo_cfg(config.algorithm):
            mode = cast("BaseCalibrateModeDescriptor", CalibrateModeRegistry[mode_name])
            resolved = mode.config_class(**settings)
            needs_layerwise = resolved.layerwise.enable or mode._requires_final_layerwise
            final_layerwise |= needs_layerwise
            policy = (
                mode._auto_quantize_layerwise_estimator
                if needs_layerwise
                else mode._auto_quantize_search_policy
            )
            if needs_layerwise and not mode._supports_layerwise:
                unsupported.append(f"{identity}: {resolved.method} cannot run layerwise")
            if policy is None:
                unsupported.append(
                    f"{identity}: {resolved.method} has no validated "
                    + ("non-layerwise search estimator" if needs_layerwise else "search policy")
                )
            algorithms.append({"settings": resolved.model_dump(), "search_policy": policy})
        formats[identity] = {
            "id": identity,
            "signature": signature,
            "algorithms": algorithms,
            "final_layerwise": final_layerwise,
        }
        return identity

    default_formats = [include(config) for config in aq.candidate_formats]
    fixed_format = include(recipe.quantize) if recipe.quantize is not None else None
    spaces = [
        {
            "module_name_patterns": list(space.module_name_patterns),
            "format_ids": [include(config) for config in space.candidate_formats],
            "allow_no_quant": space.allow_no_quant,
        }
        for space in aq.module_search_spaces
    ]
    # Budgets and execution limits do not change calibration or raw sensitivity statistics.
    search = {
        "formats": [formats[key] for key in sorted(formats)],
        "default_format_ids": default_formats,
        "fixed_format_id": fixed_format,
        "module_search_spaces": spaces,
        "disabled_layers": list(aq.disabled_layers),
        "cost_excluded_layers": list(aq.cost_excluded_layers),
        "method": aq.auto_quantize_method,
        "method_options": aq.method_options,
        "score_size": aq.score_size,
        "cost_model": aq.constraints.cost_model,
        "cost": aq.constraints.cost.model_dump() if aq.constraints.cost else None,
    }
    tasks = [
        {"kind": "format_statistics", "format_id": key, "after": []} for key in sorted(formats)
    ]
    tasks.extend(
        [
            {"kind": "combine_raw_statistics", "after": sorted(formats)},
            {"kind": "solve", "after": ["combine_raw_statistics"]},
            {"kind": "freeze_assignment", "after": ["solve"]},
            {
                "kind": "finalize_selected",
                "after": ["freeze_assignment"],
                "layerwise_if_selected": [
                    key for key in sorted(formats) if formats[key]["final_layerwise"]
                ],
            },
            {"kind": "export", "after": ["finalize_selected"]},
        ]
    )
    return AutoQuantizePlan(
        _canonical(
            {
                "schema_version": 1,
                "search": search,
                "search_digest": hashlib.sha256(_canonical(search).encode()).hexdigest(),
                "constraints": aq.constraints.model_dump(exclude_none=True),
                "tasks": tasks,
                "unsupported": sorted(set(unsupported)),
            }
        )
    )


def _validate_execution_limits(plan, spec):
    """Validate the same resource/input contract at generation and worker entry."""
    resources = spec["resources"]
    calib_size, calib_seq, batch_size = (
        spec[key] for key in ("calib_size", "calib_seq", "batch_size")
    )
    model, work_dir, export_path = (spec[key] for key in ("model", "work_dir", "export_path"))
    required = {
        "max_concurrent_formats",
        "max_total_gpus",
        "allocated_gpus_per_worker",
        "gpu_bytes",
        "host_bytes",
        "input_bytes",
        "candidate_bytes",
        "checkpoint_bytes",
        "metadata_bytes",
        "export_bytes",
        "artifact_storage_bytes",
        "retries",
    }
    if (
        not isinstance(resources, dict)
        or set(resources) != required
        or any(
            type(value) is not int or value < (0 if key == "retries" else 1)
            for key, value in resources.items()
        )
    ):
        raise ValueError(
            "All automatic pipeline resource limits must be explicit positive integers"
        )
    if resources["retries"] > 3:
        raise ValueError("Automatic pipeline retries are limited to three per format")
    if any(type(value) is not int or value < 1 for value in (calib_size, calib_seq, batch_size)):
        raise ValueError("Calibration samples, sequence length and batch size must be positive")
    if calib_size % batch_size or plan.as_dict()["search"]["score_size"] % batch_size:
        raise ValueError("Fixed calibration and scoring counts must be divisible by batch size")
    if plan.as_dict()["search"]["score_size"] > calib_size:
        raise ValueError("Scoring samples must fit the calibration input snapshot")
    if any(not Path(value).is_absolute() for value in (model, work_dir, export_path)):
        raise ValueError("Model, work and export paths must be absolute shared paths")
    paths = [Path(value).resolve() for value in (model, work_dir, export_path)]
    if any(
        a == b or a in b.parents or b in a.parents
        for i, a in enumerate(paths)
        for b in paths[i + 1 :]
    ):
        raise ValueError("Model, artifact and export directories must not overlap")
    if not isinstance(spec["dataset"], str) or not spec["dataset"]:
        raise ValueError("Automatic pipeline requires a dataset name or local path")
    if resources["host_bytes"] < 1024**2 or resources["input_bytes"] > resources["host_bytes"]:
        raise ValueError("Input snapshot must fit the host-memory limit (at least one MiB)")
    count = len(plan.as_dict()["search"]["formats"])
    concurrency = min(resources["max_concurrent_formats"], count)
    if (
        not count
        or concurrency * resources["allocated_gpus_per_worker"] > resources["max_total_gpus"]
    ):
        raise ValueError("Format concurrency exceeds the total GPU allocation limit")
    if resources["export_bytes"] <= resources["metadata_bytes"]:
        raise ValueError("Export limit must leave space beyond the provenance manifest reservation")
    peak = _pipeline_storage_bytes(count, resources) + resources["export_bytes"]
    if peak > resources["artifact_storage_bytes"]:
        raise ValueError("Artifact storage limit cannot accommodate the bounded publication peak")
    return concurrency


def _pipeline_storage_bytes(count, resources):
    return (
        (count + min(count, resources["max_concurrent_formats"])) * resources["candidate_bytes"]
        + 2 * resources["checkpoint_bytes"]
        + resources["metadata_bytes"]
    )


def _pipeline_candidates(recipe):
    configs = [*recipe.auto_quantize.candidate_formats]
    for space in recipe.auto_quantize.module_search_spaces:
        configs.extend(space.candidate_formats)
    if recipe.quantize is not None:
        configs.append(recipe.quantize)
    return [
        candidate
        for candidate in sorted({QuantRecipe(c.model_dump(), name="FORMAT") for c in configs})
        if not candidate.is_no_quant
    ]
