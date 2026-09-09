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

"""Resolve human-authored recipes and seal immutable run bundles.

The recipe contract deliberately has no inheritance. It selects one maintained
model/workflow route and one site-owned resource profile.  The resolver then
materializes the established experiment, runner, and execution contracts for
the runtime, while retaining their complete provenance as generated evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from . import _source_identity
from ._recipe_inputs import DATA_FIELDS as _DATA_FIELDS
from ._recipe_inputs import Recipe as _Recipe
from ._recipe_inputs import Site as _Site
from ._recipe_inputs import location as _location
from ._recipe_inputs import parse_recipe as _parse_recipe
from ._recipe_inputs import parse_site as _parse_site
from ._recipe_inputs import recipe_template, site_template
from ._recipe_inputs import required_string as _required_string
from ._route_catalog import MODEL_IDS, ROUTES, ROUTES_BY_KEY, RouteProfile
from .compiler import compile_campaign_plan, load_execution_config, load_runner_config, plan_to_dict
from .config import _compose, _config_root
from .identity import stable_hash
from .stages import stage_ids

if TYPE_CHECKING:
    from .schema import CampaignPlan

__all__ = [
    "MODEL_IDS",
    "ROUTES",
    "ResolvedRecipeRun",
    "RouteProfile",
    "bundle_for_run_root",
    "explain_resolved_run",
    "materialize_resolved_bundle",
    "recipe_template",
    "resolve_recipe_run",
    "site_template",
]

_REPOSITORY_ROOT = _source_identity.REPOSITORY_ROOT
_CONFIG_ROOT = _REPOSITORY_ROOT / "examples" / "puzzletron" / "configs"
_BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResolvedRecipeRun:
    """Validated recipe inputs and generated runtime contracts."""

    recipe: _Recipe
    site: _Site
    route: RouteProfile
    resource: Mapping[str, Any]
    experiment: Mapping[str, Any]
    resolved_experiment: Mapping[str, Any]
    runner: Mapping[str, Any]
    execution: Mapping[str, Any]
    plan: Mapping[str, Any]
    provenance: Mapping[str, Any]
    code: Mapping[str, Any]
    bundle_id: str


def _route_for(recipe: _Recipe) -> RouteProfile:
    key = (recipe.model, recipe.workflow, recipe.mode)
    route = ROUTES_BY_KEY.get(key)
    if route is None:
        choices = ", ".join(sorted(item.route_id for item in ROUTES))
        raise ValueError(f"Unknown model/workflow/mode route {key!r}. Maintained routes: {choices}")
    return route


def _lookup_dotted(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def _set_dotted(payload: dict[str, Any], dotted: str, value: Any) -> Any:
    parts = dotted.split(".")
    target = payload
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise KeyError(dotted)
        target = child
    if parts[-1] not in target:
        raise KeyError(dotted)
    previous = target[parts[-1]]
    target[parts[-1]] = deepcopy(value)
    return previous


def _merge_explicit(
    base: dict[str, Any],
    update: Mapping[str, Any],
    *,
    path: str,
    source: Path,
    locations: Mapping[str, str],
    shadowed: list[dict[str, Any]],
) -> None:
    for key, value in update.items():
        dotted = f"{path}.{key}" if path else key
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            child = dict(base[key])
            _merge_explicit(
                child,
                value,
                path=dotted,
                source=source,
                locations=locations,
                shadowed=shadowed,
            )
            base[key] = child
            continue
        if key in base and base[key] == value:
            raise ValueError(
                f"Duplicate no-op advanced value {dotted} at "
                f"{_location(locations, f'advanced.{dotted}', source)}"
            )
        previous = deepcopy(base.get(key)) if key in base else None
        base[key] = deepcopy(value)
        shadowed.append(
            {
                "path": dotted,
                "previous": previous,
                "value": deepcopy(value),
                "winning_source": _location(locations, f"advanced.{dotted}", source),
            }
        )


def _source_guard_command(environment: Mapping[str, Any], worker_code: Mapping[str, Any]) -> str:
    expected = {key: worker_code.get(key) for key in ("revision", "dirty", "working_tree_sha256")}
    repository = str(environment["repository"])
    python = str(Path(str(environment["venv"])) / "bin" / "python")
    statement = (
        "from puzzletron_orchestrator.recipe_config import _assert_worker_source; "
        f"_assert_worker_source({repository!r}, {expected!r})"
    )
    return f"PYTHONPATH={shlex.quote(repository)} {shlex.quote(python)} -c {shlex.quote(statement)}"


def _runner_payload(
    site: _Site,
    resource: Mapping[str, Any],
    worker_code: Mapping[str, Any],
) -> dict[str, Any]:
    body = site.body
    kind = str(body["kind"])
    environment = deepcopy(dict(body["environment"]))
    environment.pop("source_revision", None)
    hf_home = dict(body.get("paths") or {}).get("hf_home")
    if hf_home:
        prerun = list(environment.get("prerun_commands") or ())
        prerun.insert(0, f"export HF_HOME={shlex.quote(str(hf_home))}")
    else:
        prerun = list(environment.get("prerun_commands") or ())
    prerun.append(_source_guard_command(environment, worker_code))
    environment["prerun_commands"] = prerun
    runner: dict[str, Any] = {
        "kind": kind,
        "execution_contract": environment,
    }
    if kind == "slurm":
        slurm = deepcopy(dict(body["slurm"]))
        slurm["max_nodes"] = resource["max_nodes"]
        if resource.get("partition") is not None:
            slurm["partition"] = resource["partition"]
        runner["slurm"] = slurm
    else:
        baremetal = deepcopy(dict(body["baremetal"]))
        baremetal["hosts"] = list(baremetal["hosts"])[: int(resource["max_nodes"])]
        runner["inventory"] = baremetal
    return {"runner": runner}


def _execution_payload(
    recipe: _Recipe,
    route: RouteProfile,
    resource: Mapping[str, Any],
    *,
    shadowed: list[dict[str, Any]],
) -> dict[str, Any]:
    execution: dict[str, Any] = {
        "schema_version": 1,
        "mode": resource["mode"],
        "defaults": {
            "failure_policy": "strict",
            "halt_policy": "drain",
            "gpus_per_node": resource["gpus_per_node"],
            **({"partition": resource["partition"]} if resource.get("partition") else {}),
        },
        "stages": deepcopy(dict(route.execution_stages)),
    }
    if recipe.advanced_execution:
        _merge_explicit(
            execution,
            recipe.advanced_execution,
            path="execution",
            source=recipe.source,
            locations=recipe.locations,
            shadowed=shadowed,
        )
    return {"execution": execution}


def _write_yaml(
    path: Path,
    payload: Mapping[str, Any],
    *,
    header: Sequence[str] = (),
) -> None:
    comments = "".join(f"# {line}\n" for line in header)
    path.write_text(comments + yaml.safe_dump(dict(payload), sort_keys=False))


def _compile_preview(
    experiment: Mapping[str, Any],
    runner: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> CampaignPlan:
    with tempfile.TemporaryDirectory(prefix="puzzletron-recipe-config-") as temporary:
        root = Path(temporary)
        experiment_path = root / "experiment.yaml"
        runner_path = root / "runner.yaml"
        execution_path = root / "execution.yaml"
        _write_yaml(experiment_path, experiment)
        _write_yaml(runner_path, runner)
        _write_yaml(execution_path, execution)
        return compile_campaign_plan(
            experiment_config_path=experiment_path,
            runner=load_runner_config(runner_path),
            execution=load_execution_config(execution_path),
        )


def _validate_usage(
    recipe: _Recipe,
    plan: CampaignPlan,
    resource: Mapping[str, Any],
) -> None:
    enabled = {node.stage_id for node in plan.stages}
    configured_execution = set(recipe.advanced_execution.get("stages", {}))
    unused_execution = sorted(configured_execution - enabled - {"final_report"})
    if unused_execution:
        raise ValueError(
            "Execution values target inactive stages and would be unused: "
            + ", ".join(unused_execution)
        )
    known_stages = set(stage_ids())
    for dotted in recipe.advanced_experiment:
        root = dotted.partition(".")[0]
        if root in known_stages and root not in enabled:
            if dotted == f"{root}.enabled" and recipe.advanced_experiment[dotted] is False:
                continue
            raise ValueError(
                f"advanced.experiment.{dotted} targets inactive stage {root!r} and is unused"
            )
    max_nodes = int(resource["max_nodes"])
    oversized = [
        f"{node.stage_id} ({node.nodes} nodes, {node.total_gpus} GPUs, mesh={dict(node.mesh)})"
        for node in plan.stages
        if node.nodes > max_nodes
    ]
    if oversized:
        raise ValueError(
            f"Resource profile {recipe.resource_profile!r} provides at most {max_nodes} node(s), "
            "but the resolved topology requires more: " + "; ".join(oversized)
        )


def _plan_without_preview_path(plan: CampaignPlan) -> dict[str, Any]:
    payload = plan_to_dict(plan)
    payload["experiment_config_path"] = "<generated resolved bundle>"
    return payload


def resolve_recipe_run(
    recipe_path: str | Path,
    site_path: str | Path,
    *,
    run_root: str | Path | None = None,
) -> ResolvedRecipeRun:
    """Resolve and validate one recipe + site pair without writing its run root."""

    recipe = _parse_recipe(recipe_path, run_root=run_root)
    site = _parse_site(site_path)
    route = _route_for(recipe)
    if (route.requires_data or recipe.data) and set(recipe.data) != _DATA_FIELDS:
        missing = ", ".join(sorted(_DATA_FIELDS - set(recipe.data)))
        raise ValueError(
            f"Route {route.route_id} requires explicit immutable data values; missing: {missing}"
        )
    input_placeholders = _placeholder_paths(
        {
            "recipe": recipe.as_dict(),
            "site": {"site": site.body},
        }
    )
    if input_placeholders:
        raise ValueError(
            "Unresolved example placeholders must be replaced before validation: "
            + ", ".join(input_placeholders)
        )
    controller_code = _code_revision()
    worker_code = _worker_code(site, controller_code)
    code = {"controller": controller_code, "worker": worker_code}
    try:
        resource = deepcopy(dict(site.resources[recipe.resource_profile]))
    except KeyError as error:
        choices = ", ".join(sorted(site.resources))
        raise ValueError(
            f"Unknown resource_profile {recipe.resource_profile!r} at "
            f"{_location(recipe.locations, 'resource_profile', recipe.source)}; "
            f"site profiles: {choices}"
        ) from error

    template_path = _CONFIG_ROOT / route.experiment_template
    experiment = _compose(
        template_path,
        root=_config_root(template_path),
        stack=(),
    )
    shadowed: list[dict[str, Any]] = []
    for section_name, raw_section in list(experiment.items()):
        if not isinstance(raw_section, Mapping) or "evaluator_revision" not in raw_section:
            continue
        section = dict(raw_section)
        previous = section["evaluator_revision"]
        if isinstance(previous, str) and "PUZZLETRON_SOURCE_REVISION" in previous:
            if worker_code.get("dirty"):
                fingerprint = worker_code.get("working_tree_sha256")
                revision = f"uncommitted:{str(fingerprint)[:12]}" if fingerprint else "uncommitted"
                winning_source = str(worker_code["source"])
            else:
                revision = str(worker_code["revision"])
                winning_source = str(worker_code["source"])
            section["evaluator_revision"] = revision
            experiment[section_name] = section
            shadowed.append(
                {
                    "path": f"experiment.{section_name}.evaluator_revision",
                    "previous": previous,
                    "value": revision,
                    "winning_source": winning_source,
                }
            )
    experiment = _freeze_environment_defaults(experiment, path="", shadowed=shadowed)
    if recipe.data.get("path"):
        previous = experiment.get("dataset_path")
        experiment["dataset_path"] = recipe.data["path"]
        shadowed.append(
            {
                "path": "experiment.dataset_path",
                "previous": previous,
                "value": recipe.data["path"],
                "winning_source": _location(recipe.locations, "data.path", recipe.source),
            }
        )
    if recipe.data.get("revision"):
        if (
            isinstance(experiment.get("prepare_dataset"), Mapping)
            and "revision" in experiment["prepare_dataset"]
        ):
            revision_section = "prepare_dataset"
        elif isinstance(experiment.get("data"), Mapping) and "revision" in experiment["data"]:
            revision_section = "data"
        else:
            raise ValueError(
                f"Route {route.route_id} cannot consume data.revision; "
                "the internal route template must define a revision field"
            )
        section_payload = dict(experiment[revision_section])
        previous = section_payload["revision"]
        section_payload["revision"] = recipe.data["revision"]
        experiment[revision_section] = section_payload
        shadowed.append(
            {
                "path": f"experiment.{revision_section}.revision",
                "previous": previous,
                "value": recipe.data["revision"],
                "winning_source": _location(recipe.locations, "data.revision", recipe.source),
            }
        )
    hf_home = dict(site.body.get("paths") or {}).get("hf_home")
    if hf_home and isinstance(experiment.get("prepare_dataset"), Mapping):
        prepare_dataset = dict(experiment["prepare_dataset"])
        if "evaluation_hf_home" in prepare_dataset:
            previous = prepare_dataset["evaluation_hf_home"]
            prepare_dataset["evaluation_hf_home"] = str(hf_home)
            experiment["prepare_dataset"] = prepare_dataset
            shadowed.append(
                {
                    "path": "experiment.prepare_dataset.evaluation_hf_home",
                    "previous": previous,
                    "value": str(hf_home),
                    "winning_source": _location(site.locations, "site.paths.hf_home", site.source),
                }
            )
    previous_root = experiment.get("puzzle_dir")
    experiment["puzzle_dir"] = str(recipe.run_root)
    shadowed.append(
        {
            "path": "experiment.puzzle_dir",
            "previous": previous_root,
            "value": str(recipe.run_root),
            "winning_source": recipe.run_root_source,
        }
    )
    for dotted, value in recipe.advanced_experiment.items():
        try:
            previous = _lookup_dotted(experiment, dotted)
        except KeyError as error:
            raise ValueError(
                f"Unknown advanced experiment path {dotted!r} at "
                f"{_location(recipe.locations, f'advanced.experiment.{dotted}', recipe.source)}"
            ) from error
        if previous == value:
            raise ValueError(
                f"Duplicate no-op advanced value {dotted!r} at "
                f"{_location(recipe.locations, f'advanced.experiment.{dotted}', recipe.source)}"
            )
        _set_dotted(experiment, dotted, value)
        shadowed.append(
            {
                "path": f"experiment.{dotted}",
                "previous": previous,
                "value": deepcopy(value),
                "winning_source": _location(
                    recipe.locations, f"advanced.experiment.{dotted}", recipe.source
                ),
            }
        )

    runner = _runner_payload(site, resource, worker_code)
    site_partition = dict(site.body.get("slurm") or {}).get("partition")
    if resource.get("partition") is not None and resource.get("partition") != site_partition:
        shadowed.append(
            {
                "path": "runner.slurm.partition",
                "previous": site_partition,
                "value": resource["partition"],
                "winning_source": _location(
                    site.locations,
                    f"resources.{recipe.resource_profile}.partition",
                    site.source,
                ),
            }
        )
    execution = _execution_payload(recipe, route, resource, shadowed=shadowed)
    hidden_environment = _string_paths_containing(experiment, "${oc.env:")
    if hidden_environment:
        raise ValueError(
            "Internal route left environment-dependent values unresolved: "
            + ", ".join(hidden_environment)
            + ". Move the choice into the recipe or site contract."
        )
    placeholders = _placeholder_paths(
        {
            "experiment": experiment,
            "runner": runner,
        }
    )
    if placeholders:
        raise ValueError(
            "Unresolved example placeholders must be replaced before validation: "
            + ", ".join(placeholders)
        )
    compiled_plan = _compile_preview(experiment, runner, execution)
    _validate_usage(recipe, compiled_plan, resource)
    enabled_stages = {node.stage_id for node in compiled_plan.stages}
    execution_body = deepcopy(dict(execution["execution"]))
    execution_body["stages"] = {
        stage_id: value
        for stage_id, value in dict(execution_body.get("stages", {})).items()
        if stage_id in enabled_stages or stage_id == "final_report"
    }
    execution = {"execution": execution_body}
    compiled_plan = _compile_preview(experiment, runner, execution)
    resolved_experiment = {
        key: value for key, value in compiled_plan.experiment_config.items() if key != "_runtime"
    }

    provenance = {
        "schema_version": 1,
        "route": {
            "value": route.route_id,
            "source": f"internal route catalog ({route.experiment_template})",
        },
        "recipe_values": {
            key: {
                "value": value,
                "source": (
                    recipe.run_root_source
                    if key == "run_root"
                    else _location(recipe.locations, key, recipe.source)
                ),
            }
            for key, value in recipe.as_dict().items()
            if key != "advanced"
        },
        "site": {
            "value": str(site.source),
            "resource_profile": recipe.resource_profile,
            "source": _location(
                site.locations, f"resources.{recipe.resource_profile}", site.source
            ),
        },
        "generated_values": {
            "experiment": {"source": f"internal route catalog ({route.experiment_template})"},
            "runner.execution_contract": {
                "source": _location(site.locations, "site.environment", site.source)
            },
            ("runner.slurm" if site.body["kind"] == "slurm" else "runner.inventory"): {
                "source": _location(
                    site.locations,
                    "site.slurm" if site.body["kind"] == "slurm" else "site.baremetal",
                    site.source,
                )
            },
            "execution.mode": {
                "source": _location(
                    site.locations,
                    f"resources.{recipe.resource_profile}.mode",
                    site.source,
                )
            },
            "execution.defaults.gpus_per_node": {
                "source": _location(
                    site.locations,
                    f"resources.{recipe.resource_profile}.gpus_per_node",
                    site.source,
                )
            },
            "execution.stages": {"source": f"internal route catalog ({route.route_id})"},
        },
        "shadowed": shadowed,
    }
    identity_payload = {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "recipe": recipe.as_dict(),
        "selected_site": runner,
        "route": route.route_id,
        "experiment_runtime": experiment,
        "experiment_resolved": resolved_experiment,
        "execution": execution,
        "provenance": provenance,
        "code": code,
    }
    bundle_id = stable_hash(identity_payload, prefix="resolved_bundle")
    return ResolvedRecipeRun(
        recipe=recipe,
        site=site,
        route=route,
        resource=resource,
        experiment=experiment,
        resolved_experiment=resolved_experiment,
        runner=runner,
        execution=execution,
        plan=_plan_without_preview_path(compiled_plan),
        provenance=provenance,
        code=code,
        bundle_id=bundle_id,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_working_tree_fingerprint = _source_identity.working_tree_fingerprint
_repository_revision = _source_identity.repository_revision


def _code_revision() -> dict[str, Any]:
    return _source_identity.code_revision()


def _worker_code(site: _Site, controller_code: Mapping[str, Any]) -> dict[str, Any]:
    return _source_identity.worker_code(site, controller_code, detect_revision=_repository_revision)


def _assert_worker_source(repository: str, expected: Mapping[str, Any]) -> None:
    _source_identity.assert_worker_source(
        repository, expected, detect_revision=_repository_revision
    )


def _verify_existing_bundle(
    path: Path, bundle_id: str, *, manifest_sha256: str | None = None
) -> None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Resolved bundle is incomplete: {path}")
    if manifest_sha256 is not None and _sha256(manifest_path) != manifest_sha256:
        raise RuntimeError(f"Resolved bundle manifest changed after activation: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("bundle_id") != bundle_id:
        raise RuntimeError(f"Resolved bundle identity mismatch: {path}")
    for name, digest in dict(manifest.get("files") or {}).items():
        candidate = path / name
        if not candidate.is_file() or _sha256(candidate) != digest:
            raise RuntimeError(f"Resolved bundle file changed after sealing: {candidate}")
    recipe = yaml.safe_load((path / "recipe.yaml").read_text())
    runner = yaml.safe_load((path / "runner.yaml").read_text())
    experiment_runtime = yaml.safe_load((path / "experiment.runtime.yaml").read_text())
    experiment_resolved = yaml.safe_load((path / "experiment.resolved.yaml").read_text())
    execution = yaml.safe_load((path / "execution.yaml").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    recomputed = stable_hash(
        {
            "schema_version": _BUNDLE_SCHEMA_VERSION,
            "recipe": recipe,
            "selected_site": runner,
            "route": provenance["route"]["value"],
            "experiment_runtime": experiment_runtime,
            "experiment_resolved": experiment_resolved,
            "execution": execution,
            "provenance": provenance,
            "code": manifest["code"],
        },
        prefix="resolved_bundle",
    )
    if recomputed != bundle_id:
        raise RuntimeError(f"Resolved bundle contents do not match its identity: {path}")


def _string_paths_containing(value: Any, needle: str, *, path: str = "") -> list[str]:
    if isinstance(value, str):
        return [path] if needle in value else []
    if isinstance(value, Mapping):
        return [
            candidate
            for key, item in value.items()
            for candidate in _string_paths_containing(
                item, needle, path=f"{path}.{key}" if path else str(key)
            )
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            candidate
            for index, item in enumerate(value)
            for candidate in _string_paths_containing(item, needle, path=f"{path}[{index}]")
        ]
    return []


_ENVIRONMENT_DEFAULT = re.compile(r"^\$\{oc\.env:([^,}]+),(.+)\}$")


def _freeze_environment_defaults(
    value: Any,
    *,
    path: str,
    shadowed: list[dict[str, Any]],
) -> Any:
    """Use a template's declared fallback without consulting process state."""

    if isinstance(value, Mapping):
        return {
            key: _freeze_environment_defaults(
                item,
                path=f"{path}.{key}" if path else str(key),
                shadowed=shadowed,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _freeze_environment_defaults(
                item,
                path=f"{path}[{index}]",
                shadowed=shadowed,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str) or not (match := _ENVIRONMENT_DEFAULT.fullmatch(value)):
        return value
    default = yaml.safe_load(match.group(2))
    shadowed.append(
        {
            "path": f"experiment.{path}",
            "previous": value,
            "value": deepcopy(default),
            "winning_source": f"declared fallback for {match.group(1)}",
        }
    )
    return default


def _placeholder_paths(value: Any, *, path: str = "") -> list[str]:
    return _string_paths_containing(value, "REPLACE_WITH_", path=path)


def _replace_plan_path(plan: CampaignPlan, experiment_path: Path) -> CampaignPlan:
    return replace(plan, experiment_config_path=str(experiment_path))


def materialize_resolved_bundle(
    resolved: ResolvedRecipeRun,
    *,
    activate: bool,
) -> Path:
    """Write one content-addressed bundle and optionally bind the run root to it."""

    orchestration_root = resolved.recipe.run_root / "orchestration"
    bundles_root = orchestration_root / "resolved_bundles"
    final = bundles_root / resolved.bundle_id
    current_path = orchestration_root / "current_bundle.json"
    if final.exists():
        _verify_existing_bundle(final, resolved.bundle_id)
    else:
        bundles_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{resolved.bundle_id}.", dir=str(bundles_root)))
        try:
            generated_header = (
                "GENERATED PUZZLETRON RUN ARTIFACT: DO NOT EDIT.",
                f"Resolved bundle: {resolved.bundle_id}",
                "Change the user recipe or site config, then launch under a new run root.",
            )
            _write_yaml(
                staging / "recipe.yaml",
                resolved.recipe.as_dict(),
                header=generated_header + ("Normalized snapshot of the human-authored recipe.",),
            )
            _write_yaml(
                staging / "experiment.runtime.yaml",
                resolved.experiment,
                header=generated_header
                + (
                    "Executable worker input; Hydra expressions are intentionally preserved.",
                    "Resume uses this sealed file. Inspect it, but do not use it as authoring input.",
                ),
            )
            _write_yaml(
                staging / "experiment.resolved.yaml",
                resolved.resolved_experiment,
                header=generated_header
                + (
                    "Fully resolved audit view for understanding the winning values.",
                    "This evidence-only view is not executable worker input.",
                ),
            )
            _write_yaml(
                staging / "runner.yaml",
                resolved.runner,
                header=generated_header + ("Resolved site and runner snapshot.",),
            )
            _write_yaml(
                staging / "execution.yaml",
                resolved.execution,
                header=generated_header + ("Resolved execution-policy snapshot.",),
            )
            plan = compile_campaign_plan(
                experiment_config_path=staging / "experiment.runtime.yaml",
                runner=load_runner_config(staging / "runner.yaml"),
                execution=load_execution_config(staging / "execution.yaml"),
            )
            plan = _replace_plan_path(plan, final / "experiment.runtime.yaml")
            (staging / "plan.json").write_text(json.dumps(plan_to_dict(plan), indent=2) + "\n")
            (staging / "provenance.json").write_text(
                json.dumps(resolved.provenance, indent=2, default=str) + "\n"
            )
            files = {
                path.name: _sha256(path) for path in sorted(staging.iterdir()) if path.is_file()
            }
            (staging / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": _BUNDLE_SCHEMA_VERSION,
                        "bundle_id": resolved.bundle_id,
                        "code": dict(resolved.code),
                        "files": files,
                    },
                    indent=2,
                )
                + "\n"
            )
            try:
                os.rename(staging, final)
            except OSError:
                if not final.is_dir():
                    raise
                _verify_existing_bundle(final, resolved.bundle_id)
                shutil.rmtree(staging)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    if activate:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{current_path.name}.",
            suffix=".tmp",
            dir=orchestration_root,
            delete=False,
        ) as stream:
            json.dump(
                {
                    "schema_version": _BUNDLE_SCHEMA_VERSION,
                    "bundle_id": resolved.bundle_id,
                    "manifest_sha256": _sha256(final / "manifest.json"),
                },
                stream,
                indent=2,
            )
            stream.write("\n")
            temporary = Path(stream.name)
        try:
            os.link(temporary, current_path)
        except FileExistsError:
            current = json.loads(current_path.read_text())
            if current.get("bundle_id") != resolved.bundle_id:
                raise ValueError(
                    f"Run root {resolved.recipe.run_root} is already bound to bundle "
                    f"{current.get('bundle_id')}; choose a new run_root or resume it"
                )
            _verify_existing_bundle(
                final,
                resolved.bundle_id,
                manifest_sha256=current.get("manifest_sha256"),
            )
        finally:
            temporary.unlink(missing_ok=True)
    return final


def bundle_for_run_root(run_root: str | Path) -> Path:
    """Return and integrity-check the active immutable bundle for one run."""

    root = Path(run_root).expanduser().resolve()
    current_path = root / "orchestration" / "current_bundle.json"
    if not current_path.is_file():
        raise FileNotFoundError(f"No active resolved bundle exists under {root}")
    current = json.loads(current_path.read_text())
    bundle_id = _required_string(
        current.get("bundle_id"), path="bundle_id", location=str(current_path)
    )
    bundle = root / "orchestration" / "resolved_bundles" / bundle_id
    _verify_existing_bundle(
        bundle,
        bundle_id,
        manifest_sha256=current.get("manifest_sha256"),
    )
    return bundle


def explain_resolved_run(resolved: ResolvedRecipeRun) -> str:
    """Render a compact, human-readable source and allocation explanation."""

    lines = [
        (f"route: {resolved.route.route_id} <- {resolved.provenance['route']['source']}"),
        (
            "intent: "
            f"search={resolved.route.search}, evaluation={resolved.route.evaluation}, "
            f"distillation={resolved.route.distillation}"
        ),
        f"bundle: {resolved.bundle_id}",
        "recipe values:",
    ]
    for key, item in resolved.provenance["recipe_values"].items():
        lines.append(f"  {key}: {item['value']!r} <- {item['source']}")
    lines.extend(
        [
            (
                f"site resource: {resolved.recipe.resource_profile} "
                f"({resolved.resource['mode']}, {resolved.resource['gpus_per_node']} GPUs/node, "
                f"up to {resolved.resource['max_nodes']} node(s)) <- "
                f"{resolved.provenance['site']['source']}"
            ),
            "stages:",
        ]
    )
    for node in resolved.plan["stages"]:
        mesh = node["mesh"]
        mesh_text = " ".join(
            f"{key}={mesh[key]}" for key in ("tp", "pp", "cp", "dp_shard", "dp_replicate", "ep")
        )
        lines.append(
            f"  {node['stage_id']}: {node['nodes']} node(s), {node['total_gpus']} GPU(s), "
            f"{node['instances']} instance(s), {mesh_text}"
        )
    shadows = list(resolved.provenance.get("shadowed") or ())
    lines.append(f"explicit resolved overrides: {len(shadows)}")
    for item in shadows:
        lines.append(f"  {item['path']} <- {item['winning_source']}")
    return "\n".join(lines)
