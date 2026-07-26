#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create isolated physical parents and Cartesian libraries for every width."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.block_config import maybe_cast_block_configs
from modelopt.torch.puzzletron.candidates import discover_bypass_checkpoints
from modelopt.torch.puzzletron.distributed_eval.config import checkpoint_identity
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.pruning.materialize import materialize_hidden_width_checkpoint
from modelopt.torch.puzzletron.replacement_library.build_replacement_library import (
    build_replacement_library_from_sorted_teacher,
)
from modelopt.torch.puzzletron.scoring_parent import ensure_scoring_parent
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _prepare_scenario_destination(
    scenario_dir: Path,
    *,
    source_checkpoint_fingerprint: str,
    bypass_source_fingerprint: str | None = None,
    overwrite_stale: bool,
) -> bool:
    """Return whether a width scenario must be built, removing it only when authorized."""
    if not scenario_dir.exists():
        return True
    manifest_path = scenario_dir / "scenario_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        checkpoint = scenario_dir / "ckpts" / "sorted_teacher"
        bypass_overlay = scenario_dir / "ckpts" / "bypass_overlay"
        complete = all(
            path.is_file()
            for path in (
                checkpoint / "config.json",
                scenario_dir / "replacement_library.json",
                scenario_dir / "single_sequence_replacement_solutions.json",
            )
        ) and (
            bypass_source_fingerprint is None
            or (bypass_overlay / "config.json").is_file()
        )
        if (
            manifest.get("status") == "complete"
            and manifest.get("source_checkpoint_fingerprint")
            == source_checkpoint_fingerprint
            and manifest.get("bypass_source_fingerprint")
            == bypass_source_fingerprint
            and complete
        ):
            return False
    if not overwrite_stale:
        raise FileExistsError(
            "width scenario exists with a different parent identity: "
            f"{scenario_dir}; pass --overwrite-stale to rebuild it"
        )
    shutil.rmtree(scenario_dir)
    return True


def _resolve_source_checkpoint(
    config: dict, *, explicit: str | None
) -> Path:
    parent = ensure_scoring_parent(config)
    if explicit is None:
        return parent.path
    requested = Path(explicit).resolve()
    requested_identity = checkpoint_identity(requested)
    if requested_identity["fingerprint"] != parent.fingerprint:
        raise RuntimeError(
            "explicit width-scenario source does not match the resolved scoring parent: "
            f"{requested} != {parent.path}"
        )
    return requested


def _resolve_bypass_checkpoint(config: dict, puzzle_dir: Path) -> Path | None:
    """Resolve the optional nested-bypass checkpoint used for one-block overlays."""
    configured = (config.get("replacement_scoring") or {}).get("bypass_checkpoint_dir")
    if configured is not None:
        checkpoint = Path(configured).resolve()
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(
                f"bypass overlay checkpoint is incomplete: {checkpoint}"
            )
        return checkpoint

    library = (
        config.get("build_library")
        or config.get("build_replacement_library")
        or {}
    )
    include_bypass = bool(
        library.get(
            "include_bypass",
            (config.get("bypass") or {}).get("enabled", False),
        )
    )
    if not include_bypass:
        return None

    checkpoints = discover_bypass_checkpoints(puzzle_dir)
    if not checkpoints:
        raise FileNotFoundError(
            "build_library.include_bypass is enabled, but no realized bypass "
            f"checkpoint was found under {puzzle_dir / 'ckpts'}"
        )
    if len(checkpoints) > 1:
        choices = ", ".join(str(checkpoint) for checkpoint in checkpoints)
        raise RuntimeError(
            "multiple realized bypass checkpoints are available; set "
            f"replacement_scoring.bypass_checkpoint_dir explicitly: {choices}"
        )
    return checkpoints[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--overwrite-stale", action="store_true")
    args = parser.parse_args()

    cfg = pipeline_config_from_path(args.config)
    puzzle_dir = Path(cfg["puzzle_dir"])
    source = _resolve_source_checkpoint(cfg, explicit=args.source_checkpoint)
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"width-scenario parent is incomplete: {source}")
    source_identity = checkpoint_identity(source)
    bypass_source = _resolve_bypass_checkpoint(cfg, puzzle_dir)
    bypass_identity = checkpoint_identity(bypass_source) if bypass_source is not None else None
    model_cfg = cfg.get("model") or {}
    descriptor = resolve_descriptor_from_pretrained(
        str(source),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    ).descriptor
    source_config = load_model_config(
        source,
        trust_remote_code=descriptor.requires_trust_remote_code(),
    )
    teacher_width = int(descriptor.get_language_model_config(source_config).hidden_size)
    teacher_blocks = list(maybe_cast_block_configs(source_config.block_configs))
    embedding = dict(cfg.get("embedding_pruning") or {})
    widths = tuple(int(width) for width in embedding.get("widths", ()))

    summary = {
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_identity": source_identity,
        "bypass_source": (
            str(bypass_source.resolve()) if bypass_source is not None else None
        ),
        "bypass_source_identity": bypass_identity,
        "scenarios": [],
    }
    for width in widths:
        scenario_dir = puzzle_dir / "scenarios" / f"width-{width:04d}" / "depth-00"
        if not _prepare_scenario_destination(
            scenario_dir,
            source_checkpoint_fingerprint=source_identity["fingerprint"],
            bypass_source_fingerprint=(
                bypass_identity["fingerprint"] if bypass_identity is not None else None
            ),
            overwrite_stale=bool(args.overwrite_stale),
        ):
            summary["scenarios"].append(
                json.loads((scenario_dir / "scenario_manifest.json").read_text())
            )
            continue
        parent_dir = scenario_dir / "ckpts" / "sorted_teacher"
        if width == teacher_width:
            parent_dir.parent.mkdir(parents=True, exist_ok=True)
            if parent_dir.is_symlink() or parent_dir.exists():
                if parent_dir.resolve() != source.resolve():
                    raise FileExistsError(
                        f"full-width parent exists with a different target: {parent_dir}"
                    )
            else:
                parent_dir.symlink_to(source.resolve(), target_is_directory=True)
        else:
            materialize_hidden_width_checkpoint(
                source,
                descriptor,
                width,
                parent_dir,
                alignment=int(embedding.get("alignment", 1)),
            )

        bypass_overlay_dir = None
        if bypass_source is not None:
            bypass_overlay_dir = scenario_dir / "ckpts" / "bypass_overlay"
            if width == teacher_width:
                bypass_overlay_dir.parent.mkdir(parents=True, exist_ok=True)
                bypass_overlay_dir.symlink_to(bypass_source.resolve(), target_is_directory=True)
            else:
                materialize_hidden_width_checkpoint(
                    bypass_source,
                    descriptor,
                    width,
                    bypass_overlay_dir,
                    alignment=int(embedding.get("alignment", 1)),
                )

        build_replacement_library_from_sorted_teacher(
            master_puzzle_dir=scenario_dir,
            sorted_teacher_dir=parent_dir,
            descriptor=descriptor,
            search_space=cfg.get("search_space") or {},
            include_noops=bool(
                (cfg.get("build_replacement_library") or {}).get(
                    "include_noops", False
                )
            ),
            hidden_width=width,
        )
        library = json.loads((scenario_dir / "replacement_library.json").read_text())
        solutions = json.loads(
            (scenario_dir / "single_sequence_replacement_solutions.json").read_text()
        )
        teacher_entries = sum(
            1
            for entry in library["entries"]
            if entry["child_block_configs"][0]
            == teacher_blocks[entry["parent_layer_indices"][0]].to_dict()
        )
        num_layers = len(teacher_blocks)
        if teacher_entries != num_layers:
            raise RuntimeError(
                f"width {width} library has {teacher_entries} teacher entries, "
                f"expected {num_layers}"
            )
        if len(solutions) != len(library["entries"]) - num_layers:
            raise RuntimeError(
                f"width {width} solution/library cardinality mismatch: "
                f"solutions={len(solutions)} entries={len(library['entries'])} layers={num_layers}"
            )
        scenario = {
            "status": "complete",
            "hidden_width": width,
            "removed_sublayers": 0,
            "scenario_dir": str(scenario_dir),
            "parent_checkpoint": str(parent_dir.resolve()),
            "parent_checkpoint_identity": checkpoint_identity(parent_dir),
            "source_checkpoint_fingerprint": source_identity["fingerprint"],
            "bypass_checkpoint": (
                str(bypass_overlay_dir.resolve()) if bypass_overlay_dir is not None else None
            ),
            "bypass_checkpoint_identity": (
                checkpoint_identity(bypass_overlay_dir)
                if bypass_overlay_dir is not None
                else None
            ),
            "bypass_source_fingerprint": (
                bypass_identity["fingerprint"] if bypass_identity is not None else None
            ),
            "model_revision": (cfg.get("model") or {}).get("revision"),
            "descriptor": cfg.get("descriptor"),
            "alignment": int(embedding.get("alignment", 1)),
            "library_entries": len(library["entries"]),
            "teacher_entries": teacher_entries,
            "replacement_solutions": len(solutions),
        }
        _atomic_json(scenario_dir / "scenario_manifest.json", scenario)
        summary["scenarios"].append(scenario)

    _atomic_json(puzzle_dir / "scenarios" / "width_scenarios.json", summary)


if __name__ == "__main__":
    main()
