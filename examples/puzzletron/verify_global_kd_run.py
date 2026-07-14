#!/usr/bin/env python3
"""Verify one native AutoModel global-KD run and summarize its loss trend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


OBJECTIVES = ("main_ce", "main_kd", "mtp_ce", "mtp_kd")


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected mapping in {path}")
    return value


def _weight(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("weight", 0.0)
    return float(value or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", choices=("llm", "vlm"), required=True)
    parser.add_argument("--mtp", choices=("on", "off"), required=True)
    parser.add_argument("--topology", required=True, help="TP,CP,PP,DP,EP")
    parser.add_argument("--min-records", type=int, default=3)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    recipe = _load_mapping(output_dir / "global_kd_recipe.yaml")
    stage = _load_mapping(output_dir / "stage_manifest.json")
    if stage.get("status") != "success":
        raise RuntimeError(f"global KD stage status is {stage.get('status')!r}")

    expected_topology = [int(value) for value in args.topology.split(",")]
    if len(expected_topology) != 5:
        raise RuntimeError("--topology must contain TP,CP,PP,DP,EP")
    distributed = dict(recipe.get("distributed") or {})
    observed_topology = [
        int(distributed.get(key, 1) or 1)
        for key in ("tp_size", "cp_size", "pp_size", "dp_size", "ep_size")
    ]
    if observed_topology != expected_topology:
        raise RuntimeError(
            f"global KD topology={observed_topology}, expected={expected_topology}"
        )
    if bool((recipe.get("model") or {}).get("force_hf", True)):
        raise RuntimeError("global KD student did not use force_hf=false")
    if bool((recipe.get("teacher_model") or {}).get("force_hf", True)):
        raise RuntimeError("global KD teacher did not use force_hf=false")

    training_log = output_dir / "checkpoints" / "training.jsonl"
    records = [
        json.loads(line)
        for line in training_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) < args.min_records:
        raise RuntimeError(
            f"global KD emitted {len(records)} training records; expected at least {args.min_records}"
        )
    objective_cfg = dict((recipe.get("objective") or {}))
    weights = {name: _weight(objective_cfg.get(name)) for name in OBJECTIVES}
    mtp_enabled = args.mtp == "on"
    if any((weights[name] > 0) != mtp_enabled for name in ("mtp_ce", "mtp_kd")):
        raise RuntimeError(f"global KD MTP weights do not match --mtp={args.mtp}: {weights}")

    totals: list[float] = []
    components: list[dict[str, float]] = []
    for record in records:
        row: dict[str, float] = {}
        for name in OBJECTIVES:
            value = record.get(name)
            if weights[name] > 0:
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError(f"missing/non-finite {name} in training record: {record}")
                row[name] = float(value)
        total = sum(weights[name] * row.get(name, 0.0) for name in OBJECTIVES)
        if not math.isfinite(total):
            raise RuntimeError(f"non-finite weighted global-KD loss: {record}")
        totals.append(total)
        components.append(row)

    summary = {
        "output_dir": str(output_dir),
        "domain": args.domain,
        "mtp_enabled": mtp_enabled,
        "topology": dict(zip(("tp", "cp", "pp", "dp", "ep"), observed_topology)),
        "weights": weights,
        "records": len(records),
        "weighted_loss": {
            "initial": totals[0],
            "final": totals[-1],
            "best": min(totals),
            "final_minus_initial": totals[-1] - totals[0],
            "values": totals,
        },
        "objective_components": components,
        "training_log": str(training_log),
    }
    output = output_dir / "global_kd_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
