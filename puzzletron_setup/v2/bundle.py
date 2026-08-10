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

from puzzletron_setup import WORKER_REPOSITORY_PLACEHOLDER, SetupError
from puzzletron_setup.bundle import (
    BundleResult,
    dry_run_bundle,
    render_execution,
    render_experiment,
    render_runner,
    validate_bundle,
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


def _mapping_path(value: Any, parts: list[str]) -> tuple[bool, Any]:
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _named_values(values: Any, parts: list[str]) -> dict[str, Any]:
    resolved = {}
    for name, value in _mapping(values).items():
        found, effective = _mapping_path(value, parts)
        if found:
            resolved[str(name)] = deepcopy(effective)
    return resolved


def _effective_default_value(state: WizardState, path: str, fallback: Any) -> Any:
    """Return the authored consumer value for one resolved default, if present."""
    record = state.records().get(path)
    if record is not None:
        return deepcopy(record.effective)

    root, *parts = path.split(".")
    direct_collection = state.collection(root)
    if direct_collection is not None:
        found, value = _mapping_path(direct_collection, parts)
        if found:
            return deepcopy(value)

    if path == "profiles":
        profiles = _mapping(state.collection("parallel_profiles"))
        return deepcopy(profiles) if profiles else fallback

    if root == "stages" and len(parts) == 2 and parts[1] == "instances":
        found, value = _mapping_path(state.collection("stage_resources"), parts)
        return deepcopy(value) if found else fallback

    if root == "mip" and len(parts) == 1:
        runs = _mapping(_mapping(state.collection("mip_config")).get("runs"))
        field = parts[0]
        if field == "num_solutions":
            values = _named_values(runs, ["solver", "num_solutions"])
        elif field == "objective":
            values = {
                str(name): [
                    item["metric"]
                    for item in _mapping(run).get("objectives", ())
                    if isinstance(item, Mapping) and "metric" in item
                ]
                for name, run in runs.items()
            }
        elif field == "goal_metric":
            values = {
                str(name): list(_mapping(_mapping(run).get("constraints")))
                for name, run in runs.items()
            }
        elif field == "goal_value":
            values = {
                str(name): deepcopy(_mapping(_mapping(run).get("constraints")))
                for name, run in runs.items()
            }
        else:
            values = {}
        return values or fallback

    if root == "vllm" and len(parts) == 1 and parts[0] == "enabled":
        return bool(_mapping(state.collection("vllm_measurements")))
    if root == "vllm":
        measurements = state.collection("vllm_measurements")
        if len(parts) == 1:
            workload_key = {
                "batch_size": "batch_size",
                "max_num_seqs": "max_num_seqs",
                "prefill_seq_len": "prefill_seq_len",
                "generation_seq_len": "generation_seq_len",
            }.get(parts[0])
            if workload_key:
                values = _named_values(state.collection("serving_workloads"), [workload_key])
            elif parts[0] == "granularity":
                values = _named_values(measurements, ["granularity"])
            else:
                values = {}
        elif len(parts) == 2 and parts[0] == "topology":
            values = _named_values(measurements, ["runtime_stats", "topology", parts[1]])
        else:
            values = {}
        return values or fallback

    return fallback


def _legacy_state(state: WizardState) -> Mapping[str, Any]:
    legacy = state.collection("legacy_state")
    if not isinstance(legacy, Mapping):
        raise SetupError(
            "The v2 authoring state is incomplete: no canonical campaign sections were recorded."
        )
    return legacy


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    current = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = deepcopy(value)


def render_experiment_v2(state: WizardState, budget: str) -> dict[str, Any]:
    """Render algorithms, named measurements, MIP, and post-MIP flows."""
    legacy = _legacy_state(state)
    rendered = render_experiment(legacy, budget)
    rendered = _deep_merge(rendered, _mapping(state.collection("experiment_overrides")))

    measurements = _mapping(state.collection("vllm_measurements"))
    if measurements:
        rendered.setdefault("vllm_stats", {})["measurements"] = deepcopy(measurements)
        rendered["vllm_stats"]["enabled"] = True
        first = next(iter(measurements.values()))
        first = _mapping(first)
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

    mip = state.collection("mip_config")
    if isinstance(mip, Mapping):
        rendered["mip"] = _deep_merge(rendered.get("mip") or {}, mip)
        rendered["mip"]["enabled"] = True
    flows = state.collection("post_mip_flows")
    if isinstance(flows, Mapping):
        rendered["post_mip"] = {"flows": deepcopy(dict(flows))}

    for dotted, value in _mapping(state.collection("stage_batches")).items():
        _set_dotted(rendered, str(dotted), value)
    resources = _mapping(state.collection("stage_resources"))
    profiles = _mapping(state.collection("parallel_profiles"))
    parallel_paths = {
        "depth_importance": "depth_importance.automodel.parallel",
        "width_importance": "pruning.automodel.parallel",
        "sort_sanity": "sort_sanity.automodel.parallel",
        "width_sanity": "width_sanity.automodel.parallel",
        "bypass": "bypass.automodel.parallel",
        "replacement_scoring": "replacement_scoring.automodel.parallel",
    }
    for stage_id, dotted in parallel_paths.items():
        profile_name = _mapping(resources.get(stage_id)).get("profile_name")
        profile = _mapping(profiles.get(str(profile_name)))
        if not profile:
            continue
        _set_dotted(
            rendered,
            dotted,
            {
                key: profile[key]
                for key in (
                    "tp",
                    "cp",
                    "pp",
                    "ep",
                    "dp_shard",
                    "dp_replicate",
                    "sequence_parallel",
                )
                if key in profile
            },
        )
    return rendered


def render_runner_v2(state: WizardState, budget: str) -> dict[str, Any]:
    """Render a canonical runner with explicit v2 overrides."""
    rendered = render_runner(_legacy_state(state), budget)
    return _deep_merge(rendered, _mapping(state.collection("runner_overrides")))


def render_execution_v2(state: WizardState, budget: str) -> dict[str, Any]:
    """Render every static and dynamic stage resource card independently."""
    experiment = render_experiment_v2(state, budget)
    rendered = render_execution(_legacy_state(state), experiment, budget)
    stages = rendered["execution"]["stages"]
    resources = _mapping(state.collection("stage_resources"))
    profiles = _mapping(state.collection("parallel_profiles"))
    default_gpus = int(
        _mapping(_mapping(_legacy_state(state).get("answers")).get("infrastructure")).get(
            "gpus_per_node", 8
        )
    )
    for stage_id, raw in resources.items():
        resource = _mapping(raw)
        entry = {
            "strategy": str(resource.get("strategy", "single")),
            "instances": int(resource.get("instances", 1)),
            "resource": str(resource.get("resource", "gpu")),
            "gpus_per_node": int(resource.get("gpus_per_node", default_gpus)),
        }
        if resource.get("partition"):
            entry["partition"] = str(resource["partition"])
        profile_name = resource.get("profile_name")
        if profile_name:
            profile = _mapping(profiles.get(str(profile_name)))
            entry["parallel"] = {
                key: profile[key]
                for key in (
                    "tp",
                    "cp",
                    "pp",
                    "ep",
                    "dp_shard",
                    "dp_replicate",
                    "sequence_parallel",
                )
                if key in profile
            }
        elif isinstance(resource.get("parallel"), Mapping):
            entry["parallel"] = deepcopy(dict(resource["parallel"]))
        stages[str(stage_id)] = entry
    return rendered


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

    campaign_dir = Path(campaign_dir).expanduser().resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".puzzletron-v2-", dir=str(campaign_dir.parent)))
    try:
        validations = {}
        for budget in ("smoke", "production"):
            bundle = temp_root / budget
            _write_yaml(bundle / "experiment.yaml", render_experiment_v2(state, budget))
            _write_yaml(bundle / "runner.yaml", render_runner_v2(state, budget))
            _write_yaml(bundle / "execution.yaml", render_execution_v2(state, budget))
            validation = validate_bundle(bundle)
            if not validation.valid:
                raise SetupError(f"{budget} bundle is invalid: {validation.error}")
            validations[budget] = validation
            (bundle / "dry-run-plan.txt").write_text(dry_run_bundle(bundle))

        resolved = {}
        for path, raw_record in dict(state.collection("default_resolutions") or {}).items():
            record = dict(raw_record)
            value = record.get("value")
            resolved[str(path)] = {
                "value": value,
                "requested": None,
                "effective": _effective_default_value(state, str(path), value),
                "source": record.get("source"),
            }
        resolved.update(
            {
                path: {
                    "value": record.value,
                    "requested": record.requested,
                    "effective": record.effective,
                    "source": record.source,
                }
                for path, record in state.records().items()
            }
        )
        _write_yaml(temp_root / "resolved_defaults.yaml", resolved)
        repository = str(
            state.get_field(
                "infrastructure.execution_contract.repository",
                WORKER_REPOSITORY_PLACEHOLDER,
            )
        )
        (temp_root / "README.md").write_text(
            _bundle_readme(
                campaign_dir,
                repository,
                state.collection("data_acquisition"),
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
