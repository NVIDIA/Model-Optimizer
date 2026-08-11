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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

import yaml

from puzzletron_setup import SetupError
from puzzletron_setup.bundle import (
    BundleResult,
    dry_run_bundle,
    render_execution,
    render_experiment,
    render_runner,
    validate_bundle,
)

from .resolved import ResolvedCampaignConfig, resolve_campaign_config
from .resolved import _effective_default_value as _effective_default_value
from .validation import validate_state

if TYPE_CHECKING:
    from .state import WizardState

__all__ = [
    "build_bundles_v2",
    "render_execution_v2",
    "render_experiment_v2",
    "render_runner_v2",
]


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


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _legacy_state_from_resolved(config: ResolvedCampaignConfig) -> dict[str, Any]:
    """Temporary compatibility adapter from the resolved snapshot to legacy renderers."""
    serving_workloads = _plain(config.serving_workloads)
    measurements = _plain(config.vllm_measurements)
    projection = config.compatibility_projection
    workload_id = projection.workload_id
    workload = _mapping(serving_workloads.get(workload_id))
    measurement = _mapping(measurements.get(projection.runtime_measurement_id))
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
        "meshes": {
            "common": {
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "dp_shard": 1,
                "dp_replicate": 1,
                "ep": 1,
            },
            "bypass": {
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "dp_shard": 1,
                "dp_replicate": 1,
                "ep": 1,
            },
            "global_kd": {
                "tp": 1,
                "cp": 1,
                "pp": 1,
                "dp_shard": 1,
                "dp_replicate": 1,
                "ep": 1,
            },
        },
        "workers": {
            "pool": config.infrastructure.gpus_per_node,
            "sharded": config.infrastructure.gpus_per_node,
        },
    }
    first_profile_name = projection.first_parallel_profile_name
    first = (
        config.parallel_profiles.get(first_profile_name)
        if first_profile_name is not None
        else None
    )
    if first is not None:
        mesh = {
            key: getattr(first, key)
            for key in ("tp", "cp", "pp", "dp_shard", "dp_replicate", "ep")
        }
        infrastructure["meshes"] = {
            "common": deepcopy(mesh),
            "bypass": deepcopy(mesh),
            "global_kd": deepcopy(mesh),
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
        first = _mapping(
            measurements.get(config.compatibility_projection.first_measurement_id)
        )
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

    for dotted, value in config.stage_batches.items():
        _set_dotted(rendered, str(dotted), value)
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
    return _deep_merge(rendered, _plain(config.compatibility.runner))


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
                resource.gpus_per_node
                if resource.gpus_per_node is not None
                else default_gpus
            ),
        }
        if resource.partition:
            entry["partition"] = resource.partition
        if resource.profile_name:
            profile = config.parallel_profiles.get(resource.profile_name)
            entry["parallel"] = profile._parallel() if profile is not None else {}
        elif resource.parallel is not None:
            entry["parallel"] = _plain(resource.parallel)
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
                " ".join(shlex.quote(part) for part in acquisition_command),
                "```",
            )
        )
    for budget in ("smoke", "production"):
        bundle = campaign_dir / budget
        orchestrator_command = (
            f"python {orchestrator} --experiment {bundle / 'experiment.yaml'} "
            f"--runner {bundle / 'runner.yaml'} --execution {bundle / 'execution.yaml'} "
            "--stage full"
        )
        lines.extend(
            [
                "",
                f"## {budget.title()}",
                "",
                "```bash",
                f"{orchestrator_command} --dry-run",
                orchestrator_command,
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Resume setup",
            "",
            "```bash",
            f"python {Path(repository) / 'examples/puzzletron/puzzletron_setup_v2.py'} "
            f"--resume {campaign_dir}",
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
