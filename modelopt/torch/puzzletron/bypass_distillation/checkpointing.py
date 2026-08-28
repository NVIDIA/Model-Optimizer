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

"""Checkpoint discovery and publication for AutoModel local distillation."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import torch
from safetensors import safe_open

if TYPE_CHECKING:
    from omegaconf import DictConfig

from ..tools.logger import mprint
from .bypass_utils import load_bypass_state

__all__ = [
    "copy_hf_auxiliary_assets",
    "find_latest_completed_checkpoint",
    "publish_elastic_checkpoint",
    "quarantine_incomplete_checkpoint",
    "realize_bypass_checkpoints",
    "require_distributed_path_consensus",
    "save_ranked_state_checkpoint",
    "validate_automodel_bypass_checkpoint",
    "validate_consolidated_hf_checkpoint",
    "validate_ranked_state_checkpoint",
]


_HF_WEIGHT_FILENAMES = {
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}


def copy_hf_auxiliary_assets(source_dir: str | Path, consolidated_dir: str | Path) -> None:
    """Copy absent non-weight HF processor assets into a consolidated checkpoint."""

    source_dir = Path(source_dir)
    consolidated_dir = Path(consolidated_dir)
    if not source_dir.is_dir() or not consolidated_dir.is_dir():
        return
    for source in source_dir.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(source_dir)
        if source.name in _HF_WEIGHT_FILENAMES or source.suffix in {".safetensors", ".bin"}:
            continue
        destination = consolidated_dir / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def require_distributed_path_consensus(path: str | Path, purpose: str) -> None:
    """Reject rank-local path divergence before entering storage collectives."""

    resolved = str(Path(path).expanduser().resolve())
    if not torch.distributed.is_initialized():
        return
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, resolved)
    unique = sorted({str(value) for value in gathered})
    if len(unique) != 1:
        raise RuntimeError(f"distributed {purpose} path mismatch across ranks: {unique}")


def quarantine_incomplete_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Move an interrupted same-step checkpoint aside before a clean retry."""

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    if (checkpoint_dir / "saving_completed").is_file():
        raise FileExistsError(f"refusing to replace completed checkpoint: {checkpoint_dir}")
    quarantine = checkpoint_dir.with_name(f".{checkpoint_dir.name}.quarantine.{uuid.uuid4().hex}")
    checkpoint_dir.replace(quarantine)
    return quarantine


def save_ranked_state_checkpoint(
    checkpoint_dir: str | Path,
    *,
    state_name: str,
    rank: int,
    state: object,
) -> Path:
    """Atomically save one rank's state into an AutoModel checkpoint."""

    state_dir = Path(checkpoint_dir) / state_name
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / f"{state_name}_dp_rank_{int(rank)}.pt"
    temporary = target.with_name(f"{target.name}.tmp.{uuid.uuid4().hex}")
    try:
        torch.save(state, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def validate_ranked_state_checkpoint(
    checkpoint_dir: str | Path,
    *,
    state_name: str,
    expected_ranks: Iterable[int],
) -> dict[str, object]:
    """Validate that every expected ranked state file exists and is loadable."""

    state_dir = Path(checkpoint_dir) / state_name
    ranks = sorted({int(rank) for rank in expected_ranks})
    paths = {rank: state_dir / f"{state_name}_dp_rank_{rank}.pt" for rank in ranks}
    missing = [rank for rank, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"{state_name} checkpoint is missing rank file(s): {missing}")

    corrupt: dict[int, str] = {}
    for rank, path in paths.items():
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - report corrupt checkpoint payloads uniformly
            corrupt[rank] = str(error)
    if corrupt:
        raise RuntimeError(f"{state_name} checkpoint has unreadable rank file(s): {corrupt}")

    return {
        "state_name": state_name,
        "expected_ranks": ranks,
        "files": len(paths),
        "status": "complete",
    }


def validate_automodel_bypass_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_rng_ranks: Iterable[int],
) -> dict[str, object]:
    """Validate the artifacts AutoModel needs to resume bypass training."""

    checkpoint_dir = Path(checkpoint_dir)
    for name in ("config.yaml", "losses.json"):
        path = checkpoint_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"AutoModel checkpoint is missing {name}: {checkpoint_dir}")

    for name in ("grad_scaler.pt", "step_scheduler.pt"):
        path = checkpoint_dir / name
        if not path.is_file():
            raise RuntimeError(f"AutoModel checkpoint is missing tracked state {name}")
        try:
            torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - preserve checkpoint error context
            raise RuntimeError(
                f"AutoModel checkpoint has unreadable tracked state {name}: {error}"
            ) from error

    model_dir = checkpoint_dir / "model"
    model_shards = sorted(model_dir.glob("*.safetensors"))
    if not model_shards:
        raise RuntimeError(f"AutoModel checkpoint has no model shards: {model_dir}")
    for shard in model_shards:
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                if not list(handle.keys()):
                    raise RuntimeError("empty tensor inventory")
        except Exception as error:  # noqa: BLE001 - normalize corrupt shard failures
            raise RuntimeError(f"unreadable AutoModel model shard {shard}: {error}") from error

    optim_dir = checkpoint_dir / "optim"
    metadata = optim_dir / ".metadata"
    if not metadata.is_file() or metadata.stat().st_size == 0:
        raise RuntimeError(f"AutoModel checkpoint is missing optimizer metadata: {metadata}")
    optimizer_shards = sorted(optim_dir.glob("*.distcp"))
    empty_optimizer_shards = [path.name for path in optimizer_shards if path.stat().st_size == 0]
    if not optimizer_shards or empty_optimizer_shards:
        raise RuntimeError(
            "AutoModel checkpoint has missing or empty optimizer shards: "
            f"count={len(optimizer_shards)} empty={empty_optimizer_shards}"
        )

    rng = validate_ranked_state_checkpoint(
        checkpoint_dir,
        state_name="rng",
        expected_ranks=expected_rng_ranks,
    )
    return {
        "checkpoint": str(checkpoint_dir.resolve()),
        "model_shards": len(model_shards),
        "optimizer_shards": len(optimizer_shards),
        "rng": rng,
        "status": "complete",
    }


def validate_consolidated_hf_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_layer_prefixes: tuple[str, ...] = (),
) -> dict[str, object]:
    """Reject incomplete HF exports before Puzzletron publishes them.

    Distributed PP/EP saves can produce a syntactically valid index even when
    one rank's shard is absent.  Validate both sides of the index and require a
    tensor for every descriptor-declared decoder-layer prefix.
    """

    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(
            f"consolidated checkpoint has no model.safetensors.index.json: {checkpoint_dir}"
        )
    payload = json.loads(index_path.read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"consolidated checkpoint has an empty weight map: {index_path}")

    shard_names = set(weight_map.values())
    missing_shards = sorted(name for name in shard_names if not (checkpoint_dir / name).is_file())
    if missing_shards:
        raise RuntimeError(
            f"consolidated checkpoint is missing indexed shard file(s): {missing_shards[:10]}"
        )

    actual_keys: set[str] = set()
    duplicate_keys: set[str] = set()
    for shard_name in sorted(shard_names):
        with safe_open(checkpoint_dir / shard_name, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in actual_keys:
                    duplicate_keys.add(key)
                actual_keys.add(key)
    if duplicate_keys:
        raise RuntimeError(
            "consolidated checkpoint contains duplicate tensor key(s): "
            f"{sorted(duplicate_keys)[:10]}"
        )

    indexed_keys = set(weight_map)
    missing_indexed_keys = sorted(indexed_keys - actual_keys)
    unindexed_keys = sorted(actual_keys - indexed_keys)
    if missing_indexed_keys or unindexed_keys:
        raise RuntimeError(
            "consolidated checkpoint index/tensor inventory mismatch: "
            f"missing_indexed={missing_indexed_keys[:10]} unindexed={unindexed_keys[:10]}"
        )

    missing_prefixes = [
        prefix
        for prefix in expected_layer_prefixes
        if not any(key == prefix or key.startswith(prefix + ".") for key in indexed_keys)
    ]
    if missing_prefixes:
        raise RuntimeError(
            f"consolidated checkpoint is missing expected layer prefix(es): {missing_prefixes[:10]}"
        )

    return {
        "checkpoint": str(checkpoint_dir.resolve()),
        "indexed_keys": len(indexed_keys),
        "actual_keys": len(actual_keys),
        "shards": len(shard_names),
        "expected_layer_prefixes": len(expected_layer_prefixes),
        "status": "complete",
    }


def find_latest_completed_checkpoint(run_parent_dir: str | Path) -> Path | None:
    """Return the newest resumable checkpoint, never a stale best/start snapshot."""
    run_parent_dir = Path(run_parent_dir)
    state = load_bypass_state(run_parent_dir)
    if state is not None:
        checkpoints = state.get("checkpoints", {})
        for role in ("final", "resume"):
            candidate = checkpoints.get(role)
            if candidate and (Path(candidate) / "saving_completed").exists():
                return Path(candidate)

    latest = run_parent_dir / "latest"
    if latest.exists():
        resolved = latest.resolve()
        if re.match(r"^step-\d+-ckpt$", resolved.name) and (resolved / "saving_completed").exists():
            return resolved
    if not run_parent_dir.exists():
        return None

    pattern = re.compile(r"^step-(\d+)-ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in run_parent_dir.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for _, path in sorted(candidates, reverse=True):
        if (path / "saving_completed").exists():
            return path
    return None


def realize_bypass_checkpoints(cfg: DictConfig) -> tuple[Path, Path]:
    """Publish the selected AutoModel checkpoint under the canonical ckpts alias."""
    state = load_bypass_state(cfg.bypass.experiment_dir) or {}
    checkpoints = state.get("checkpoints", {})
    mode = cfg.bypass.get("realize_best_or_latest", "latest")
    if mode == "best":
        roles = ("best", "final", "resume")
    elif mode == "latest":
        roles = ("final", "resume", "best")
    else:
        raise ValueError(f"Invalid bypass.realize_best_or_latest={mode!r}")

    checkpoint_dir = next(
        (
            Path(checkpoints[role]).resolve()
            for role in roles
            if checkpoints.get(role) and Path(checkpoints[role]).exists()
        ),
        None,
    )
    if checkpoint_dir is None:
        fallback = Path(cfg.bypass.experiment_dir) / "latest"
        if not fallback.exists():
            raise FileNotFoundError(
                f"Could not find a bypass checkpoint to realize in {cfg.bypass.experiment_dir}"
            )
        checkpoint_dir = fallback.resolve()

    realized = checkpoint_dir
    consolidated = checkpoint_dir / "model" / "consolidated"
    if not (realized / "config.json").exists() and (consolidated / "config.json").exists():
        realized = consolidated
    if not (realized / "config.json").exists():
        raise FileNotFoundError(f"Realized AutoModel checkpoint has no config.json: {realized}")

    ckpts_dir = Path(cfg.puzzle_dir) / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    alias = ckpts_dir / cfg.bypass.experiment_id
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    alias.symlink_to(realized.resolve(), target_is_directory=True)
    mprint(f"Created symlink: {alias} -> {realized}")
    return realized, alias


def publish_elastic_checkpoint(cfg: DictConfig) -> Path:
    """Point the canonical elastic parent at the realized native local-KD model."""
    ckpts_dir = Path(cfg.puzzle_dir) / "ckpts"
    realized = ckpts_dir / cfg.bypass.experiment_id
    target = realized.resolve() if realized.exists() else None
    if target is None or not (target / "config.json").exists():
        raise RuntimeError(
            f"Elastic bypass checkpoint {realized} is not a usable HF/AnyModel checkpoint"
        )
    alias = ckpts_dir / "elastic_sorted_teacher"
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    alias.symlink_to(target, target_is_directory=True)
    mprint(f"Published elastic sorted teacher: {alias} -> {target}")
    return alias
