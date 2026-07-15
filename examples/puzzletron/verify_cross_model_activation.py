"""Verify cross-model activation-score coverage and numerical health.

The scorer intentionally omits passes that matched no modules from its completed-pass
manifest.  This verifier distinguishes that explicit zero-target case from a partially
written pass: the former has ``args.json`` and no rank shards, while the latter is an
error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


_MAX_DIAGNOSTIC_QUANTILE_VALUES = 65_536


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _score_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value.detach().float().reshape(-1)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _score_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _score_tensors(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield torch.tensor([float(value)], dtype=torch.float32)


def _diagnostic_quantile_values(values: torch.Tensor) -> torch.Tensor:
    """Bound exact-quantile memory while sampling the full flattened score range."""
    if values.numel() <= _MAX_DIAGNOSTIC_QUANTILE_VALUES:
        return values
    indices = torch.linspace(
        0,
        values.numel() - 1,
        steps=_MAX_DIAGNOSTIC_QUANTILE_VALUES,
        device=values.device,
        dtype=torch.float64,
    ).round().to(dtype=torch.long)
    indices.clamp_(max=values.numel() - 1)
    return values.index_select(0, indices)


def _pass_summary(
    name: str,
    directory: Path,
    *,
    completed: bool,
    expected_eval_iters: int,
) -> dict[str, Any]:
    args = _load_json(directory / "args.json")
    if int(args.get("eval_iters", -1)) != expected_eval_iters:
        raise RuntimeError(
            f"{name}: eval_iters={args.get('eval_iters')} != {expected_eval_iters}"
        )
    rank_files = sorted(directory.glob("rank_*.pth"))
    if not completed:
        if rank_files:
            raise RuntimeError(
                f"{name}: has {len(rank_files)} rank shards but is absent from the pass manifest"
            )
        return {
            "status": "zero_targets",
            "eval_iters": expected_eval_iters,
            "rank_files": 0,
            "axes": [],
        }
    if not rank_files:
        raise RuntimeError(f"{name}: completed pass has no rank shards")
    expected_ranks = int(args.get("num_nodes", len(rank_files)))
    if len(rank_files) != expected_ranks:
        raise RuntimeError(
            f"{name}: rank shard count {len(rank_files)} != num_nodes {expected_ranks}"
        )

    modules: set[str] = set()
    values: list[torch.Tensor] = []
    for path in rank_files:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: expected a module-score mapping, got {type(raw).__name__}")
        modules.update(map(str, raw))
        values.extend(_score_tensors(raw))
    if not modules or not values:
        raise RuntimeError(f"{name}: completed pass has no modules or score values")
    flat = torch.cat(values)
    if not bool(torch.isfinite(flat).all()):
        raise RuntimeError(f"{name}: scores contain non-finite values")
    if not bool(torch.count_nonzero(flat)):
        raise RuntimeError(f"{name}: all score values are zero")
    quantile_values = _diagnostic_quantile_values(flat)

    fingerprints = list(args.get("observability", {}).get("batch_fingerprints") or [])
    return {
        "status": "passed",
        "eval_iters": expected_eval_iters,
        "rank_files": len(rank_files),
        "modules": len(modules),
        "values": int(flat.numel()),
        "nonzero_fraction": float(torch.count_nonzero(flat) / flat.numel()),
        "unique_values": int(torch.unique(flat).numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()) if flat.numel() > 1 else 0.0,
        "quantiles": {
            str(q): float(torch.quantile(quantile_values, q))
            for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "batch_fingerprints": fingerprints,
    }


def verify_model_activation(
    model_id: str,
    model_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    scoring = config["scoring"]
    expected_samples = int(scoring["eval_samples"])
    micro_batch_size = int(scoring["micro_batch_size"])
    expected_eval_iters = math.ceil(expected_samples / micro_batch_size)
    configured = {
        str(item["name"]): tuple(map(str, item.get("axis_ids") or ()))
        for item in config["pruning"]["activation_passes"]
    }

    base = model_root / "pruning/pruning_scores/automodel/all_axes"
    manifest = _load_json(base / "activation_passes_manifest.json")
    completed = list(map(str, manifest.get("passes") or ()))
    unknown = sorted(set(completed) - set(configured))
    if unknown:
        raise RuntimeError(f"{model_id}: pass manifest contains unconfigured passes {unknown}")

    passes: dict[str, dict[str, Any]] = {}
    fingerprints: set[tuple[str, ...]] = set()
    for name, axes in configured.items():
        summary = _pass_summary(
            name,
            base / name,
            completed=name in completed,
            expected_eval_iters=expected_eval_iters,
        )
        summary["axes"] = list(axes)
        pass_fingerprints = tuple(summary.pop("batch_fingerprints", ()))
        if pass_fingerprints:
            num_nodes = int(_load_json(base / name / "args.json").get("num_nodes", 1))
            if (
                len(pass_fingerprints) % expected_eval_iters
                or len(pass_fingerprints) > expected_eval_iters * num_nodes
                or len(set(pass_fingerprints)) != len(pass_fingerprints)
            ):
                raise RuntimeError(
                    f"{model_id}/{name}: invalid distributed batch-fingerprint count "
                    f"{len(pass_fingerprints)} for eval_iters={expected_eval_iters}, "
                    f"num_nodes={num_nodes}, unique={len(set(pass_fingerprints))}"
                )
            fingerprints.add(pass_fingerprints)
        passes[name] = summary
    if len(fingerprints) > 1:
        raise RuntimeError(f"{model_id}: activation passes consumed different batch fingerprints")

    return {
        "version": 1,
        "model_id": model_id,
        "status": "passed",
        "expected_samples": expected_samples,
        "micro_batch_size": micro_batch_size,
        "completed_passes": completed,
        "zero_target_passes": [
            name for name, value in passes.items() if value["status"] == "zero_targets"
        ],
        "batch_fingerprint_count": len(next(iter(fingerprints), ())),
        "passes": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("puzzle_runs/clean/acceptance/2026-07-06-cross-model-stage-matrix"),
    )
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()

    config_dir = args.root / "configs"
    models = args.models or sorted(path.stem for path in config_dir.glob("*.yaml"))
    summaries = [
        verify_model_activation(
            model_id,
            args.root / "models" / model_id,
            config_dir / f"{model_id}.yaml",
        )
        for model_id in models
    ]
    result = {"version": 1, "status": "passed", "models": summaries}
    output = args.root / "campaign/activation_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
