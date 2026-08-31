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


"""Render setup-v2 state into existing Puzzletron runtime contracts."""

from __future__ import annotations

import os
import shlex
import shutil
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped, unused-ignore]

from puzzletron_setup import SetupError
from puzzletron_setup.bundle import (
    BundleResult,
    dry_run_bundle,
    render_execution,
    render_experiment,
    render_runner,
    validate_bundle,
)

from .resolved import (
    ResolvedCampaignConfig,
    ResolvedParallelProfile,
    _plain,
    resolve_campaign_config,
)
from .validation import validate_state

if TYPE_CHECKING:
    from .state import WizardState

__all__ = [
    "build_bundles_v2",
    "render_execution_v2",
    "render_experiment_v2",
    "render_runner_v2",
]

_LEGACY_MESH_FIELDS = ("tp", "cp", "pp", "dp_shard", "dp_replicate", "ep")
_LEGACY_COMMON_CONSUMERS = (
    "width_importance",
    "depth_importance",
    "sort_sanity",
    "width_sanity",
    "replacement_scoring",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _post_mip_consumers(config: ResolvedCampaignConfig, node_type: str) -> tuple[str, ...]:
    consumers = []
    for flow_id, flow in config.post_mip_flows.items():
        for node_id, node in _mapping(_mapping(flow).get("nodes")).items():
            if _mapping(node).get("type") == node_type:
                consumers.append(f"post.{flow_id}.{node_id}")
    return tuple(sorted(consumers))


def _profile_for_consumers(
    config: ResolvedCampaignConfig,
    consumers: tuple[str, ...],
) -> ResolvedParallelProfile | None:
    for consumer in consumers:
        resource = config.stage_resources.get(consumer)
        if resource is not None and resource.profile_name is not None:
            profile = config.parallel_profiles.get(resource.profile_name)
            if profile is not None:
                return profile
    for consumer in consumers:
        for name in sorted(config.parallel_profiles):
            profile = config.parallel_profiles[name]
            if consumer in profile.consumers:
                return profile
    return None


def _legacy_meshes(config: ResolvedCampaignConfig) -> dict[str, dict[str, Any]]:
    default_mesh = dict.fromkeys(_LEGACY_MESH_FIELDS, 1)
    common_consumers = (*_LEGACY_COMMON_CONSUMERS, *_post_mip_consumers(config, "evaluation"))
    common = _profile_for_consumers(config, common_consumers)
    if common is None and config.parallel_profiles:
        common = config.parallel_profiles[sorted(config.parallel_profiles)[0]]

    profiles = {
        "common": common,
        "bypass": _profile_for_consumers(config, ("bypass", "bypass_sanity")) or common,
        "global_kd": _profile_for_consumers(
            config,
            ("global_kd", *_post_mip_consumers(config, "global_kd")),
        )
        or common,
    }
    return {
        name: (
            {key: getattr(profile, key) for key in _LEGACY_MESH_FIELDS}
            if profile is not None
            else deepcopy(default_mesh)
        )
        for name, profile in profiles.items()
    }


def _legacy_state_from_resolved(config: ResolvedCampaignConfig) -> dict[str, Any]:
    """Temporary compatibility adapter from the resolved snapshot to legacy renderers."""
    serving_workloads = _plain(config.serving_workloads)
    measurements = _plain(config.vllm_measurements)
    workload_id = str(next(iter(serving_workloads), "serving-default"))
    workload = _mapping(serving_workloads.get(workload_id))
    first_measurement_id = next(iter(measurements), None)
    runtime_measurement_id = (
        workload_id if _mapping(measurements.get(workload_id)) else first_measurement_id
    )
    measurement = _mapping(measurements.get(runtime_measurement_id))
    runtime = {
        "vllm_enabled": bool(measurements),
        "granularity": measurement.get("granularity", "subblock"),
        "workload_id": workload_id,
        "isl": int(workload.get("prefill_seq_len", config.data.sequence_length)),
        "osl": int(workload.get("generation_seq_len", 1024)),
        "concurrency": int(workload.get("max_num_seqs", 1)),
    }
    infrastructure = {
        "runner": {
            "kind": config.infrastructure.runner_kind,
            "slurm": _plain(config.infrastructure.slurm),
        },
        "execution_contract": _plain(config.infrastructure.execution_contract),
        "gpus_per_node": config.infrastructure.gpus_per_node,
        "meshes": _legacy_meshes(config),
        "workers": {
            "pool": config.infrastructure.gpus_per_node,
            "sharded": config.infrastructure.gpus_per_node,
        },
    }
    return {
        "schema_version": 1,
        "wizard_version": "1",
        "detailed": True,
        "model": config.model._legacy_model(),
        "inventory": config.model._legacy_inventory(),
        "answers": {
            "data": {
                "source": config.data.source,
                "selected_source": config.data.selected_source,
                "adapter": config.data.adapter,
                "modality": config.data.modality,
                "layout": config.data.layout,
                "sequence_length": config.data.sequence_length,
                "subsets": list(config.data.subsets),
                "subset_revision": config.data.subset_revision,
                "subset_weights": _plain(config.data.subset_weights),
                "acquisition": _plain(config.data.acquisition),
            },
            "pruning": _plain(config.pruning),
            "runtime": runtime,
            "mip": _plain(config.mip) or {"runs": {}},
            "post_mip": {"flows": _plain(config.post_mip_flows)},
            "infrastructure": infrastructure,
            "output": {"result_root": config.result_root},
        },
    }


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    current = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = deepcopy(value)


def _render_experiment_v2(
    config: ResolvedCampaignConfig,
    budget: str,
) -> dict[str, Any]:
    legacy = _legacy_state_from_resolved(config)
    rendered = render_experiment(legacy, budget)
    rendered = _deep_merge(rendered, _plain(config.compatibility.experiment))

    measurements = _plain(config.vllm_measurements)
    if measurements:
        rendered.setdefault("vllm_stats", {})["measurements"] = deepcopy(measurements)
        rendered["vllm_stats"]["enabled"] = True
        first = _mapping(measurements.get(next(iter(measurements))))
        rendered["vllm_stats"].update(
            {
                "prefill_seq_len": int(first.get("prefill_seq_len", 4096)),
                "generation_seq_len": int(first.get("generation_seq_len", 1024)),
                "batch_sizes": [int(first.get("batch_size", 1))],
            }
        )
        rendered["vllm_stats"].setdefault("runtime_stats", {}).update(
            deepcopy(_mapping(first.get("runtime_stats")))
        )

    mip = _plain(config.mip)
    if config.mip_configured:
        rendered["mip"] = _deep_merge(rendered.get("mip") or {}, mip)
        rendered["mip"]["enabled"] = True
    flows = _plain(config.post_mip_flows)
    if config.post_mip_flows_configured:
        rendered["post_mip"] = {"flows": deepcopy(flows)}
        if budget == "smoke":
            for flow in rendered["post_mip"]["flows"].values():
                for node in _mapping(flow.get("nodes")).values():
                    if node.get("type") == "downstream_evaluation":
                        node_config = _mapping(node.get("config"))
                        node_config["limit"] = min(int(node_config.get("limit", 8) or 8), 8)
                        node["config"] = node_config

    batch_mirrors = {
        "pruning.micro_batch_size": "data.calibration.micro_batch_size",
        "replacement_scoring.micro_batch_size": "data.replacement_scoring.micro_batch_size",
    }
    for dotted, value in config.stage_batches.items():
        _set_dotted(rendered, str(dotted), value)
        mirrored = batch_mirrors.get(str(dotted))
        if mirrored is not None:
            _set_dotted(rendered, mirrored, value)
    parallel_paths = {
        "depth_importance": "depth_importance.automodel.parallel",
        "width_importance": "pruning.automodel.parallel",
        "sort_sanity": "sort_sanity.automodel.parallel",
        "width_sanity": "width_sanity.automodel.parallel",
        "bypass": "bypass.automodel.parallel",
        "replacement_scoring": "replacement_scoring.automodel.parallel",
    }
    for stage_id, dotted in parallel_paths.items():
        resource = config.stage_resources.get(stage_id)
        profile = (
            config.parallel_profiles.get(resource.profile_name)
            if resource is not None and resource.profile_name is not None
            else None
        )
        if profile is None or not profile.source_nonempty:
            continue
        _set_dotted(
            rendered,
            dotted,
            profile._parallel(),
        )
    return rendered


def render_experiment_v2(state: WizardState, budget: str) -> dict[str, Any]:
    """Render algorithms, named measurements, MIP, and post-MIP flows."""
    return _render_experiment_v2(resolve_campaign_config(state), budget)


def _render_runner_v2(config: ResolvedCampaignConfig, budget: str) -> dict[str, Any]:
    rendered = render_runner(_legacy_state_from_resolved(config), budget)
    compatibility = _plain(config.compatibility.runner)
    return _deep_merge(rendered, compatibility)


def render_runner_v2(state: WizardState, budget: str) -> dict[str, Any]:
    """Render a canonical runner with explicit v2 overrides."""
    return _render_runner_v2(resolve_campaign_config(state), budget)


def _render_execution_v2(
    config: ResolvedCampaignConfig,
    budget: str,
    experiment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if experiment is None:
        experiment = _render_experiment_v2(config, budget)
    rendered = render_execution(_legacy_state_from_resolved(config), experiment, budget)
    stages = rendered["execution"]["stages"]
    default_gpus = config.infrastructure.gpus_per_node
    for stage_id, resource in config.stage_resources.items():
        entry = {
            "strategy": resource.strategy,
            "instances": resource.instances,
            "resource": resource.resource,
            "gpus_per_node": (
                resource.gpus_per_node if resource.gpus_per_node is not None else default_gpus
            ),
        }
        if resource.partition:
            entry["partition"] = resource.partition
        if resource.profile_name:
            profile = config.parallel_profiles.get(resource.profile_name)
            parallel = profile._parallel() if profile is not None else {}
            entry["parallel"] = {
                key: value for key, value in parallel.items() if key != "sequence_parallel"
            }
        elif resource.parallel is not None:
            entry["parallel"] = {
                key: value
                for key, value in _plain(resource.parallel).items()
                if key != "sequence_parallel"
            }
        stages[str(stage_id)] = entry
    return rendered


def render_execution_v2(state: WizardState, budget: str) -> dict[str, Any]:
    """Render every static and dynamic stage resource card independently."""
    return _render_execution_v2(resolve_campaign_config(state), budget)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_plain(payload), sort_keys=False, width=100))


def _bundle_readme(
    campaign_dir: Path,
    repository: str,
    acquisition: Mapping[str, Any] | None = None,
) -> str:
    orchestrator = Path(repository) / "examples/puzzletron/orchestrate.py"
    lines = [
        "# Puzzletron campaign",
        "",
        "Generated by `puzzletron_setup_v2.py`. The wizard did not launch any jobs.",
    ]
    acquisition = _mapping(acquisition)
    if acquisition:
        tool = Path(repository) / "examples/puzzletron/materialize_dataset.py"
        acquisition_command: list[str] = [
            "python",
            str(tool),
            str(acquisition["adapter"]),
            "--output",
            str(acquisition["output"]),
            "--seed",
            str(acquisition["seed"]),
        ]
        if acquisition["adapter"] == "puzzle_kd_v2":
            acquisition_command.extend(
                (
                    "--train-samples",
                    str(acquisition["train_samples"]),
                    "--validation-samples",
                    str(acquisition["validation_samples"]),
                )
            )
        else:
            acquisition_command.extend(
                ["--subsets", *[str(item) for item in acquisition["subsets"]]]
            )
            subset_rows = _mapping(acquisition.get("subset_rows"))
            if subset_rows:
                acquisition_command.extend(
                    [
                        "--subset-rows",
                        *[f"{name}={rows}" for name, rows in subset_rows.items()],
                    ]
                )
            acquisition_command.extend(
                (
                    "--num-samples",
                    str(acquisition["num_samples"]),
                    "--max-shards-per-subset",
                    str(acquisition["max_shards_per_subset"]),
                )
            )
            if acquisition.get("revision"):
                acquisition_command.extend(("--revision", str(acquisition["revision"])))
        lines.extend(
            (
                "",
                "## Prepare dataset",
                "",
                "The command is idempotent only when the existing manifest matches these answers.",
                "",
                "```bash",
                shlex.join(acquisition_command),
                "```",
            )
        )
    sections = (
        (
            "smoke",
            "Validate setup",
            "This bounded run checks the generated model, data, worker, and campaign wiring.",
            "After reviewing the plan and worker paths, launch the validation run:",
        ),
        (
            "production",
            "Run campaign",
            "Run this campaign after the setup validation succeeds.",
            "After reviewing the plan and worker paths, launch the campaign:",
        ),
    )
    for budget, heading, introduction, launch_introduction in sections:
        bundle = campaign_dir / budget
        orchestrator_args = [
            "python",
            str(orchestrator),
            "--experiment",
            str(bundle / "experiment.yaml"),
            "--runner",
            str(bundle / "runner.yaml"),
            "--execution",
            str(bundle / "execution.yaml"),
            "--stage",
            "full",
        ]
        inspect_command = shlex.join([*orchestrator_args, "--dry-run"])
        launch_command = shlex.join(orchestrator_args)
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                introduction,
                "",
                "Inspect the complete plan without submitting jobs:",
                "",
                "```bash",
                inspect_command,
                "```",
                "",
                launch_introduction,
                "",
                "```bash",
                launch_command,
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Resume setup",
            "",
            "```bash",
            shlex.join(
                [
                    "python",
                    str(Path(repository) / "examples/puzzletron/puzzletron_setup_v2.py"),
                    "--resume",
                    str(campaign_dir),
                ]
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_bundles_v2(campaign_dir: Path, state: WizardState) -> BundleResult:
    """Compile both budgets in a temporary tree, then publish them atomically."""
    issues = validate_state(state)
    if issues:
        details = "\n".join(f"- {issue.path}: {issue.message}" for issue in issues)
        raise SetupError(f"Setup v2 validation failed:\n{details}")

    config = resolve_campaign_config(state)
    campaign_dir = Path(campaign_dir).expanduser().resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".puzzletron-v2-", dir=str(campaign_dir.parent)))
    try:
        validations = {}
        for budget in ("smoke", "production"):
            bundle = temp_root / budget
            experiment = _render_experiment_v2(config, budget)
            runner = _render_runner_v2(config, budget)
            execution = _render_execution_v2(config, budget, experiment)
            _write_yaml(bundle / "experiment.yaml", experiment)
            _write_yaml(bundle / "runner.yaml", runner)
            _write_yaml(bundle / "execution.yaml", execution)
            validation = validate_bundle(bundle)
            if not validation.valid:
                raise SetupError(f"{budget} bundle is invalid: {validation.error}")
            validations[budget] = validation
            (bundle / "dry-run-plan.txt").write_text(dry_run_bundle(bundle))

        provenance = {
            path: {
                "value": _plain(record.value),
                "requested": _plain(record.requested),
                "effective": _plain(record.effective),
                "source": record.source,
            }
            for path, record in config.provenance.items()
        }
        _write_yaml(temp_root / "resolved_defaults.yaml", provenance)
        repository = str(config.infrastructure.execution_contract["repository"])
        (temp_root / "README.md").write_text(
            _bundle_readme(
                campaign_dir,
                repository,
                config.data.acquisition,
            )
        )

        for relative in ("smoke", "production"):
            source = temp_root / relative
            target = campaign_dir / relative
            backup = campaign_dir / f".{relative}.previous"
            if backup.exists():
                shutil.rmtree(backup)
            if target.exists():
                os.replace(target, backup)
            os.replace(source, target)
            if backup.exists():
                shutil.rmtree(backup)
        for name in ("resolved_defaults.yaml", "README.md"):
            os.replace(temp_root / name, campaign_dir / name)
        return BundleResult(
            campaign_dir=campaign_dir,
            smoke=validations["smoke"],
            production=validations["production"],
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
