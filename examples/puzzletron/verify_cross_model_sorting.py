"""Verify cross-model sorted checkpoint structure before functional evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


_AXIS_FAMILY = {
    "hidden_width": "embedding",
    "ffn_intermediate": "ffn",
    "kv_groups": "attn",
    "q_heads_per_group": "attn",
    "gdn_key_groups": "gdn",
    "gdn_value_heads_per_group": "gdn",
    "gdn_key_head_dim": "gdn",
    "gdn_value_head_dim": "gdn",
    "moe_experts": "moe",
    "moe_expert_intermediate": "moe",
    "moe_shared_expert_intermediate": "moe",
    "mamba_heads": "mamba",
    "mamba_head_dim": "mamba",
}


def _load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _index(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    paths = sorted(checkpoint.glob("*.safetensors.index.json"))
    if len(paths) != 1:
        raise RuntimeError(f"{checkpoint}: expected one safetensors index, got {len(paths)}")
    return paths[0], _load(paths[0])


def verify_model_sorting(model_id: str, model_root: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    stage = _load(model_root / "manifests/sort.json")
    if stage.get("status") != "success":
        raise RuntimeError(f"{model_id}: sort stage status is {stage.get('status')!r}")
    teacher = model_root / "ckpts/teacher"
    sorted_dir = model_root / "ckpts/sorted_teacher"
    _, teacher_index = _index(teacher)
    _, sorted_index = _index(sorted_dir)
    if teacher_index.get("weight_map") != sorted_index.get("weight_map"):
        raise RuntimeError(f"{model_id}: sorted weight map differs from the teacher")
    weight_files = sorted(set(teacher_index["weight_map"].values()))
    for name in weight_files:
        source, target = teacher / name, sorted_dir / name
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(f"{model_id}: missing teacher/sorted shard {name}")
        if source.stat().st_size != target.stat().st_size:
            raise RuntimeError(f"{model_id}: shard size changed for {name}")

    permutations = _load(sorted_dir / "sorted_permutations.json")
    families = {str(key).split(".", 1)[0] for key in permutations}
    active_axes = {
        str(entry["axis_id"])
        for entry in config["pruning"]["activation_axes"]
    }
    required_families = {
        _AXIS_FAMILY[axis]
        for axis in active_axes
        if axis in _AXIS_FAMILY
    }
    missing = sorted(required_families - families)
    if missing:
        raise RuntimeError(f"{model_id}: missing permutation families {missing}")

    sort_manifest = _load(sorted_dir / "parallel_sort_manifest.json")
    if sort_manifest.get("status") != "complete":
        raise RuntimeError(f"{model_id}: parallel sort manifest is incomplete")
    if sort_manifest.get("protected_tensor_equality_verified") is not True:
        raise RuntimeError(f"{model_id}: protected tensor equality was not verified")
    inventory = sort_manifest.get("tensor_inventory") or {}
    if int(inventory.get("embedding", 0)) < 1 or int(inventory.get("lm_head", 0)) < 1:
        raise RuntimeError(f"{model_id}: embedding/lm-head inventory is incomplete")
    if config["data"]["modality"] == "multimodal" and int(inventory.get("vision", 0)) < 1:
        raise RuntimeError(f"{model_id}: multimodal sort did not inventory the ViT")
    if config["distillation"].get("mtp_enabled") and int(inventory.get("mtp", 0)) < 1:
        raise RuntimeError(f"{model_id}: MTP sort inventory is empty")
    if any(axis.startswith("moe_") for axis in active_axes) and int(inventory.get("moe", 0)) < 1:
        raise RuntimeError(f"{model_id}: MoE sort inventory is empty")

    expected_deferred = sorted((config.get("sort") or {}).get("deferred_axes") or [])
    if sorted(sort_manifest.get("deferred_axes") or []) != expected_deferred:
        raise RuntimeError(f"{model_id}: deferred-axis manifest does not match config")
    return {
        "model_id": model_id,
        "status": "passed",
        "weight_files": len(weight_files),
        "permutations": len(permutations),
        "permutation_families": sorted(families),
        "deferred_axes": expected_deferred,
        "tensor_inventory": inventory,
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
    models = args.models or sorted(path.stem for path in (args.root / "configs").glob("*.yaml"))
    rows = [
        verify_model_sorting(
            model,
            args.root / "models" / model,
            args.root / "configs" / f"{model}.yaml",
        )
        for model in models
    ]
    result = {"version": 1, "status": "passed", "models": rows}
    output = args.root / "campaign/sort_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
