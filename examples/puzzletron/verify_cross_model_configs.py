"""Verify generated cross-model configs before any checkpoint is loaded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from modelopt.torch.puzzletron.campaigns.schema import load_campaign
from modelopt.torch.puzzletron.pipeline_config import (
    load_runtime_hydra_config,
    pipeline_config_from_path,
)
from modelopt.torch.puzzletron.pruning.pruning_mixin import PruningMixIn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("examples/puzzletron/configs/clean/campaigns/cross_model_stage_matrix.yaml"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("puzzle_runs/clean/acceptance/2026-07-06-cross-model-stage-matrix"),
    )
    args = parser.parse_args()

    campaign = load_campaign(args.campaign)
    summary = {"version": 1, "campaign_fingerprint": campaign.fingerprint, "models": []}
    for model in campaign.models:
        path = args.root / "configs" / f"{model.model_id}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"missing generated config for {model.model_id}")
        raw = yaml.safe_load(path.read_text())
        if raw["model"]["force_hf"] is not False:
            raise ValueError(f"{model.model_id}: force_hf must remain false")
        expected_topology = {
            "tp": model.topology.tp,
            "cp": model.topology.cp,
            "pp": model.topology.pp,
            "ep": model.topology.ep,
            "dp_shard": model.topology.ep,
            "dp_replicate": model.topology.fsdp,
        }
        actual_topology = raw["pruning"]["automodel"]["parallel"]
        actual_axes = {key: actual_topology[key] for key in expected_topology}
        if actual_axes != expected_topology:
            raise ValueError(
                f"{model.model_id}: topology mismatch {actual_axes} != {expected_topology}"
            )
        axes = {entry["axis_id"] for entry in raw["pruning"]["activation_axes"]}
        passes = raw["pruning"]["activation_passes"]
        covered = [axis for item in passes for axis in item["axis_ids"]]
        if len(covered) != len(set(covered)) or set(covered) != axes:
            raise ValueError(
                f"{model.model_id}: activation pass coverage mismatch axes={sorted(axes)} "
                f"covered={covered}"
            )

        normalized = pipeline_config_from_path(path)
        runtime = load_runtime_hydra_config(normalized)
        instantiated = list(runtime.pruning.activation_passes)
        for item in instantiated:
            mixin = item.get("pruning_mixin", None)
            if mixin is not None and not isinstance(mixin, PruningMixIn):
                raise TypeError(
                    f"{model.model_id}/{item.name}: target did not instantiate: {type(mixin)}"
                )
        summary["models"].append(
            {
                "model_id": model.model_id,
                "force_hf": False,
                "topology": expected_topology,
                "axes": sorted(axes),
                "passes": [item["name"] for item in passes],
            }
        )

    output = args.root / "campaign" / "config_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
