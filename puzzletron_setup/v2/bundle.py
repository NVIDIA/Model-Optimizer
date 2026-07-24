# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render setup-v2 state into existing Puzzletron runtime contracts."""

from __future__ import annotations

import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

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

from .state import WizardState
from .validation import validate_state

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


def _bundle_readme(campaign_dir: Path, repository: str) -> str:
    orchestrator = Path(repository) / "examples/puzzletron/orchestrate.py"
    lines = [
        "# Puzzletron campaign",
        "",
        "Generated by `puzzletron_setup_v2.py`. The wizard did not launch any jobs.",
    ]
    for budget in ("smoke", "production"):
        bundle = campaign_dir / budget
        command = (
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
                f"{command} --dry-run",
                command,
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
    temp_root = Path(
        tempfile.mkdtemp(prefix=".puzzletron-v2-", dir=str(campaign_dir.parent))
    )
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

        resolved = {
            path: {
                "value": record.value,
                "requested": record.requested,
                "effective": record.effective,
                "source": record.source,
            }
            for path, record in state.records().items()
        }
        _write_yaml(temp_root / "resolved_defaults.yaml", resolved)
        repository = str(
            state.get_field("infrastructure.execution_contract.repository", Path.cwd())
        )
        (temp_root / "README.md").write_text(_bundle_readme(campaign_dir, repository))

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
