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

"""Atomic, strict PDD checkpoint publication around the stock AutoModel Checkpointer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.distributed as dist

from modelopt.torch.fastgen import PDDMetadata
from modelopt.torch.fastgen.plugins.qwen_image_pdd import QWEN_IMAGE_PDD_EXECUTION

if TYPE_CHECKING:
    from collections.abc import Sequence

_CHECKPOINT_SCHEMA_VERSION = 5
_COMPLETE_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal.") from error
    return value.lower()


def _require_qwen_image_execution(identity: Any) -> None:
    if not isinstance(identity, Mapping) or identity.get("qwen_image") != {
        "execution": QWEN_IMAGE_PDD_EXECUTION
    }:
        raise RuntimeError("PDD checkpoint has an incompatible Qwen execution identity.")


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _broadcast_rank0_payload(value: Any) -> Any:
    payload = [value]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _gather_objects(value: Any) -> list[Any]:
    if not dist.is_available() or not dist.is_initialized():
        return [value]
    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda candidate: len(candidate.parts), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"PDD checkpoint staging contains a symlink: {path}.")
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            _fsync_directory(path)
    _fsync_directory(root)


def _dcp_payload_hashes(checkpoint: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for component in ("model", "optim"):
        root = checkpoint / component
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"PDD checkpoint is missing the {component} DCP directory.")
        files: list[Path] = []
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"PDD {component} DCP tree contains a symlink: {path}.")
            if path.is_file():
                files.append(path)
        relative_files = {path.relative_to(checkpoint).as_posix() for path in files}
        if f"{component}/.metadata" not in relative_files:
            raise RuntimeError(f"PDD checkpoint is missing strict {component} DCP metadata.")
        if not any(relative != f"{component}/.metadata" for relative in relative_files):
            raise RuntimeError(f"PDD checkpoint is missing {component} DCP payload shards.")
        for path in files:
            hashes[path.relative_to(checkpoint).as_posix()] = _sha256(path)
    return hashes


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read PDD checkpoint JSON {path}.") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"PDD checkpoint JSON {path} must contain an object.")
    return value


def build_pdd_checkpoint_identity(
    *,
    qwen_image_execution: str,
    metadata: PDDMetadata,
    model_id: str,
    model_revision: str | None,
    guidance_scale: float | None,
    ordered_train_id_sha256: str,
    ordered_heldout_id_sha256: str,
    dataset_snapshot_sha256: str,
    local_batch_size: int,
    grad_accumulation_steps: int,
    training_seed: int,
    validation_seed: int,
    validation_every_steps: int,
    max_grad_norm: float,
    zero_grad_warmup_steps: int,
    activation_checkpointing: bool,
    dtype: str,
    optimizer: Any,
    scheduler: Any,
) -> dict[str, Any]:
    """Build the strict, path-independent compatibility identity for PDD resume."""
    if qwen_image_execution != QWEN_IMAGE_PDD_EXECUTION:
        raise ValueError("qwen_image_execution must identify the bound FastGen MR210 forward.")
    if not isinstance(metadata, PDDMetadata):
        raise TypeError("metadata must be PDDMetadata.")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string.")
    if (
        not isinstance(model_revision, str)
        or len(model_revision) != 40
        or any(character not in "0123456789abcdef" for character in model_revision)
    ):
        raise ValueError("model_revision must be an exact lowercase 40-character commit.")
    if guidance_scale is not None and (
        isinstance(guidance_scale, bool) or not isinstance(guidance_scale, int | float)
    ):
        raise TypeError("guidance_scale must be a real number or null.")
    if type(local_batch_size) is not int or local_batch_size < 1:
        raise ValueError("local_batch_size must be an integer >= 1.")
    if grad_accumulation_steps != 1:
        raise ValueError("PDD v1 exact resume requires grad_accumulation_steps=1.")
    for name, value, minimum in (
        ("training_seed", training_seed, 0),
        ("validation_seed", validation_seed, 0),
        ("validation_every_steps", validation_every_steps, 1),
        ("zero_grad_warmup_steps", zero_grad_warmup_steps, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if isinstance(max_grad_norm, bool) or not isinstance(max_grad_norm, int | float):
        raise TypeError("max_grad_norm must be a real number.")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and > 0.")
    if type(activation_checkpointing) is not bool:
        raise TypeError("activation_checkpointing must be bool.")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("dtype must be a non-empty string.")
    if type(optimizer).__module__ != "torch.optim.adamw" or type(optimizer).__name__ != "AdamW":
        raise TypeError("PDD checkpoint identity requires the stock torch.optim.AdamW optimizer.")
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "qwen_image": {"execution": qwen_image_execution},
        "model": {"id": model_id, "revision": model_revision, "dtype": dtype},
        "pdd_metadata": metadata.to_dict(),
        "guidance": {"scale": None if guidance_scale is None else float(guidance_scale)},
        "data": {
            "ordered_train_id_sha256": _require_sha256(
                ordered_train_id_sha256,
                name="ordered_train_id_sha256",
            ),
            "ordered_heldout_id_sha256": _require_sha256(
                ordered_heldout_id_sha256,
                name="ordered_heldout_id_sha256",
            ),
            "dataset_snapshot_sha256": _require_sha256(
                dataset_snapshot_sha256,
                name="dataset_snapshot_sha256",
            ),
            "local_batch_size": local_batch_size,
            "grad_accumulation_steps": grad_accumulation_steps,
        },
        "topology": {"world_size": _world_size(), "pure_data_parallel": True},
        "training": {
            "seed": training_seed,
            "validation_seed": validation_seed,
            "validation_every_steps": validation_every_steps,
            "max_grad_norm": float(max_grad_norm),
            "zero_grad_warmup_steps": zero_grad_warmup_steps,
            "activation_checkpointing": activation_checkpointing,
        },
        "optimizer": {
            "class": "torch.optim.AdamW",
            "param_groups": [
                {
                    "lr": float(group["lr"]),
                    "betas": [float(beta) for beta in group["betas"]],
                    "eps": float(group["eps"]),
                    "weight_decay": float(group["weight_decay"]),
                    "amsgrad": bool(group["amsgrad"]),
                    "maximize": bool(group["maximize"]),
                    "capturable": bool(group["capturable"]),
                    "differentiable": bool(group["differentiable"]),
                    "foreach": bool(group["foreach"]),
                    "fused": bool(group["fused"]),
                }
                for group in optimizer.param_groups
            ],
        },
        "scheduler": {
            "class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
            "base_lrs": [float(value) for value in scheduler.base_lrs],
            "policy": "constant",
        },
    }


@dataclass(frozen=True)
class PDDResumeState:
    """Restored progress plus the first logical IDs that must be served next."""

    checkpoint_path: Path
    completed_steps: int
    sample_slots_consumed: int
    expected_next_sample_ids: tuple[str, ...]
    parent_checkpoint: str | None

    def verify_first_batch(self, sample_ids: Sequence[str]) -> None:
        if tuple(sample_ids) != self.expected_next_sample_ids:
            raise RuntimeError(
                "first resumed sample IDs do not match the checkpoint: "
                f"expected={self.expected_next_sample_ids}, actual={tuple(sample_ids)}."
            )


class _StepSchedulerCheckpointState:
    """Rank-local carrier for the normalized next-data StepScheduler cursor."""

    def __init__(self) -> None:
        self.state: dict[str, int] = {"step": 0, "epoch": 0}

    def state_dict(self) -> dict[str, int]:
        return dict(self.state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"step", "epoch"}:
            raise ValueError("PDD StepScheduler state must contain step and epoch.")
        step = state["step"]
        epoch = state["epoch"]
        if type(step) is not int or step < 0 or type(epoch) is not int or epoch < 0:
            raise ValueError("PDD StepScheduler step and epoch must be nonnegative integers.")
        self.state = {"step": step, "epoch": epoch}


def _checkpoint_sidecar_paths(checkpoint: Path, world_size: int) -> list[Path]:
    paths: list[Path] = []
    for rank in range(world_size):
        paths.extend(
            (
                checkpoint / "rng" / f"rng_dp_rank_{rank}.pt",
                checkpoint / "sampler" / f"sampler_dp_rank_{rank}.pt",
                checkpoint / "step_scheduler" / f"step_scheduler_dp_rank_{rank}.pt",
                checkpoint / "trainer" / f"trainer_dp_rank_{rank}.pt",
            )
        )
    return paths


def validate_pdd_training_checkpoint(
    checkpoint: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    """Validate a complete training checkpoint without deserializing pickle sidecars."""
    unresolved_checkpoint = Path(checkpoint)
    if unresolved_checkpoint.is_symlink():
        raise RuntimeError(f"PDD checkpoint cannot be a symlink: {unresolved_checkpoint}.")
    checkpoint = unresolved_checkpoint.resolve()
    if not checkpoint.is_dir():
        raise RuntimeError(f"PDD checkpoint is not a regular directory: {checkpoint}.")
    symlinks = [path for path in checkpoint.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"PDD checkpoint contains a symlink: {symlinks[0]}.")
    marker_path = checkpoint / "COMPLETE"
    manifest_path = checkpoint / "manifest.json"
    if (
        not marker_path.is_file()
        or marker_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise RuntimeError(f"PDD checkpoint is incomplete: {checkpoint}.")
    marker = _read_json(marker_path)
    if (
        set(marker) != {"schema_version", "manifest_sha256"}
        or marker.get("schema_version") != _COMPLETE_SCHEMA_VERSION
    ):
        raise RuntimeError("PDD COMPLETE marker is incompatible.")
    if marker["manifest_sha256"] != _sha256(manifest_path):
        raise RuntimeError("PDD COMPLETE marker does not match manifest content.")
    manifest = _read_json(manifest_path)
    manifest_keys = {
        "schema_version",
        "identity",
        "completed_steps",
        "learning_rates",
        "step_scheduler",
        "parent_checkpoint",
        "rank_progress",
        "dcp_sha256",
        "sidecar_sha256",
    }
    if set(manifest) != manifest_keys or manifest["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("PDD checkpoint manifest schema is incompatible.")
    if expected_identity is not None and manifest["identity"] != expected_identity:
        raise RuntimeError("PDD checkpoint identity does not match the current run.")
    identity = manifest["identity"]
    _require_qwen_image_execution(identity)
    topology = identity.get("topology") if isinstance(identity, Mapping) else None
    world_size = topology.get("world_size") if isinstance(topology, Mapping) else None
    if type(world_size) is not int or world_size < 1:
        raise RuntimeError("PDD checkpoint identity has an invalid world size.")
    if expected_world_size is not None and world_size != expected_world_size:
        raise RuntimeError(
            f"PDD checkpoint world size {world_size} does not match {expected_world_size}."
        )
    if _read_json(checkpoint / "pdd_config.json") != identity:
        raise RuntimeError("PDD checkpoint config sidecar does not match the manifest.")
    trainer_state = _read_json(checkpoint / "trainer_state.json")
    if trainer_state != {
        "completed_steps": manifest["completed_steps"],
        "learning_rates": manifest["learning_rates"],
        "step_scheduler": manifest["step_scheduler"],
        "parent_checkpoint": manifest["parent_checkpoint"],
    }:
        raise RuntimeError("PDD trainer-state sidecar does not match the manifest.")
    step_scheduler_state = manifest["step_scheduler"]
    if (
        not isinstance(step_scheduler_state, dict)
        or set(step_scheduler_state) != {"step", "epoch"}
        or step_scheduler_state.get("step") != manifest["completed_steps"]
        or type(step_scheduler_state.get("epoch")) is not int
        or step_scheduler_state["epoch"] < 0
    ):
        raise RuntimeError("PDD checkpoint StepScheduler state is invalid.")
    rank_progress = manifest["rank_progress"]
    if not isinstance(rank_progress, list) or len(rank_progress) != world_size:
        raise RuntimeError("PDD checkpoint rank progress does not match its topology.")
    expected_dcp = manifest["dcp_sha256"]
    if not isinstance(expected_dcp, dict) or any(
        not isinstance(path, str) or not isinstance(digest, str)
        for path, digest in expected_dcp.items()
    ):
        raise RuntimeError("PDD checkpoint DCP hash inventory is malformed.")
    if _dcp_payload_hashes(checkpoint) != expected_dcp:
        raise RuntimeError("PDD checkpoint DCP payload inventory or hash does not match.")
    expected_sidecars = _checkpoint_sidecar_paths(checkpoint, world_size)
    expected_relative = {path.relative_to(checkpoint).as_posix() for path in expected_sidecars}
    if (
        not isinstance(manifest["sidecar_sha256"], dict)
        or set(manifest["sidecar_sha256"]) != expected_relative
    ):
        raise RuntimeError("PDD checkpoint sidecar inventory does not match the topology.")
    for path in expected_sidecars:
        relative = path.relative_to(checkpoint).as_posix()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"PDD checkpoint sidecar is missing: {relative}.")
        if _sha256(path) != manifest["sidecar_sha256"][relative]:
            raise RuntimeError(f"PDD checkpoint sidecar hash mismatch: {relative}.")
    return manifest


def resolve_pdd_training_checkpoint(
    root: str | Path,
    restore_from: str | Path,
    *,
    expected_world_size: int,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve an explicit checkpoint or the newest compatible complete LATEST candidate."""
    unresolved_root = Path(root)
    if unresolved_root.is_symlink():
        raise ValueError("PDD checkpoint_dir cannot be a symlink.")
    root = unresolved_root.resolve()
    if str(restore_from).upper() != "LATEST":
        candidate = Path(restore_from)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_symlink():
            raise ValueError("explicit PDD checkpoint cannot be a symlink.")
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("explicit PDD checkpoint must be beneath checkpoint_dir.") from error
        manifest = validate_pdd_training_checkpoint(
            candidate,
            expected_world_size=expected_world_size,
        )
        if expected_identity is not None and not _identity_contains(
            manifest.get("identity"), expected_identity
        ):
            raise RuntimeError("explicit PDD checkpoint identity does not match the selector.")
        return candidate, manifest

    if not isinstance(expected_identity, Mapping) or not expected_identity:
        raise ValueError("LATEST resolution requires a non-empty expected_identity selector.")

    pointed: tuple[int, Path, dict[str, Any]] | None = None
    pointer = root / "LATEST"
    if pointer.is_file() and not pointer.is_symlink():
        candidate = (root / pointer.read_text().strip()).resolve()
        try:
            candidate.relative_to(root)
            manifest = validate_pdd_training_checkpoint(
                candidate,
                expected_world_size=expected_world_size,
            )
            completed = manifest["completed_steps"]
            if type(completed) is not int or completed < 0:
                raise RuntimeError("PDD checkpoint completed_steps is invalid.")
            if not _identity_contains(manifest.get("identity"), expected_identity):
                raise RuntimeError("pointed PDD checkpoint identity does not match the selector.")
            pointed = (completed, candidate, manifest)
        except (ValueError, RuntimeError):
            pass

    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    if root.is_dir():
        for path in root.iterdir():
            suffix = path.name.removeprefix("step_")
            if not path.is_dir() or not path.name.startswith("step_") or not suffix.isdigit():
                continue
            if pointed is not None and int(suffix) <= pointed[0]:
                continue
            try:
                manifest = validate_pdd_training_checkpoint(
                    path,
                    expected_world_size=expected_world_size,
                )
            except RuntimeError:
                continue
            if not _identity_contains(manifest.get("identity"), expected_identity):
                continue
            completed = manifest["completed_steps"]
            if type(completed) is int and completed >= 0:
                candidates.append((completed, path.resolve(), manifest))
    if pointed is not None:
        candidates.append(pointed)
    if not candidates:
        raise FileNotFoundError(f"no complete compatible PDD checkpoint exists beneath {root}.")
    return max(candidates, key=lambda item: (item[0], item[1].name))[1:]


def _identity_contains(actual: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(actual, Mapping):
        return False
    return all(key in actual and actual[key] == value for key, value in expected.items())


class PDDCheckpointManager:
    """Publish and restore complete, metadata-compatible PDD checkpoints only."""

    def __init__(
        self,
        *,
        root: str | Path,
        checkpointer: Any,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        step_scheduler: Any,
        trainer: Any,
        sampler: Any,
        rng: Any,
        identity: Mapping[str, Any],
    ) -> None:
        self.root = Path(root).resolve()
        self.checkpointer = checkpointer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_scheduler = step_scheduler
        self._step_scheduler_checkpoint_state = _StepSchedulerCheckpointState()
        self.trainer = trainer
        self.sampler = sampler
        self.rng = rng
        self._last_checkpoint: Path | None = None
        self.identity = json.loads(json.dumps(identity, sort_keys=True))
        if self.identity.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("PDD checkpoint identity has an unsupported schema version.")
        _require_qwen_image_execution(self.identity)
        topology = self.identity.get("topology")
        if not isinstance(topology, dict) or topology.get("world_size") != _world_size():
            raise ValueError("PDD checkpoint identity world size does not match the process group.")
        if bool(getattr(checkpointer.config, "is_async", False)):
            raise ValueError("PDD v1 atomic publication requires synchronous checkpoint saves.")

    def _rank_summary(self) -> dict[str, Any]:
        sampler_state = self.sampler.state_dict()
        return {
            "rank": _rank(),
            "epoch": sampler_state["epoch"],
            "committed_batches": sampler_state["committed_batches"],
            "sample_slots_consumed": sampler_state["sample_slots_consumed"],
            "plan_sha256": sampler_state["plan_sha256"],
            "next_sample_ids": list(sampler_state["next_sample_ids"]),
        }

    def _sidecar_paths(self, checkpoint: Path) -> list[Path]:
        return _checkpoint_sidecar_paths(checkpoint, _world_size())

    def _manifest(self, checkpoint: Path) -> dict[str, Any]:
        manifest = _read_json(checkpoint / "manifest.json")
        expected = {
            "schema_version",
            "identity",
            "completed_steps",
            "learning_rates",
            "step_scheduler",
            "parent_checkpoint",
            "rank_progress",
            "dcp_sha256",
            "sidecar_sha256",
        }
        if set(manifest) != expected:
            raise RuntimeError("PDD checkpoint manifest has incompatible keys.")
        if manifest["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("PDD checkpoint manifest schema is unsupported.")
        return manifest

    def _validate_checkpoint(self, checkpoint: Path, *, require_identity: bool) -> dict[str, Any]:
        return validate_pdd_training_checkpoint(
            checkpoint,
            expected_identity=self.identity if require_identity else None,
            expected_world_size=_world_size(),
        )

    def _compatible_candidates(
        self,
        *,
        after_completed_steps: int | None = None,
    ) -> list[tuple[int, Path]]:
        candidates: list[tuple[int, Path]] = []
        if not self.root.is_dir():
            return candidates
        for path in self.root.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            if after_completed_steps is not None:
                prefix = "step_"
                suffix = path.name.removeprefix(prefix)
                if not path.name.startswith(prefix) or not suffix.isdigit():
                    continue
                if int(suffix) <= after_completed_steps:
                    continue
            try:
                manifest = self._validate_checkpoint(path, require_identity=True)
            except RuntimeError:
                continue
            completed = manifest["completed_steps"]
            if type(completed) is int and completed >= 0:
                candidates.append((completed, path.resolve()))
        return sorted(candidates, key=lambda item: (item[0], item[1].name), reverse=True)

    def resolve(self, restore_from: str | Path | None) -> Path | None:
        """Resolve LATEST by scanning only complete, identity-compatible checkpoints."""
        if restore_from is None:
            return None
        if str(restore_from).upper() == "LATEST":
            pointer = self.root / "LATEST"
            pointed: tuple[int, Path] | None = None
            if pointer.is_file() and not pointer.is_symlink():
                name = pointer.read_text().strip()
                candidate = (self.root / name).resolve()
                try:
                    candidate.relative_to(self.root)
                    manifest = self._validate_checkpoint(candidate, require_identity=True)
                    completed = manifest["completed_steps"]
                    if type(completed) is not int or completed < 0:
                        raise RuntimeError("PDD checkpoint completed_steps is invalid.")
                    pointed = (completed, candidate)
                except (ValueError, RuntimeError):
                    pass
            candidates = self._compatible_candidates(
                after_completed_steps=None if pointed is None else pointed[0]
            )
            if pointed is not None:
                candidates.append(pointed)
                candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
            return candidates[0][1] if candidates else None

        candidate = Path(restore_from)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ValueError("explicit PDD checkpoint must be beneath checkpoint_dir.") from error
        self._validate_checkpoint(candidate, require_identity=True)
        return candidate

    def _collective_resolve(self, restore_from: str | Path | None) -> Path | None:
        if _world_size() == 1:
            return self.resolve(restore_from)
        status: dict[str, Any] | None = None
        if _rank() == 0:
            try:
                resolved = self.resolve(restore_from)
                status = {
                    "ok": True,
                    "path": None if resolved is None else str(resolved),
                }
            except BaseException as error:
                status = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        status = _broadcast_rank0_payload(status)
        if not isinstance(status, dict) or type(status.get("ok")) is not bool:
            raise RuntimeError("rank 0 broadcast a malformed checkpoint resolution status.")
        if not status["ok"]:
            raise RuntimeError(f"rank-0 checkpoint resolution failed: {status.get('error')}.")
        resolved_path = status.get("path")
        if resolved_path is None:
            return None
        if not isinstance(resolved_path, str):
            raise RuntimeError("rank 0 broadcast a malformed checkpoint path.")
        return Path(resolved_path)

    def _prepare_staging(self, final: Path) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        if final.exists():
            try:
                self._validate_checkpoint(final, require_identity=False)
            except RuntimeError:
                shutil.rmtree(final)
            else:
                raise FileExistsError(f"complete PDD checkpoint already exists: {final}.")
        staging_name = f".{final.name}.{uuid.uuid4().hex}.staging"
        (self.root / staging_name).mkdir()
        return staging_name

    def _publish_staging(
        self,
        *,
        staging: Path,
        final: Path,
        completed_steps: int,
        learning_rates: list[float],
        step_scheduler_state: Mapping[str, int],
        parent: Path | None,
        rank_summaries: list[dict[str, Any]],
    ) -> None:
        sidecars = self._sidecar_paths(staging)
        sidecar_sha256 = {path.relative_to(staging).as_posix(): _sha256(path) for path in sidecars}
        manifest = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "identity": self.identity,
            "completed_steps": completed_steps,
            "learning_rates": learning_rates,
            "step_scheduler": dict(step_scheduler_state),
            "parent_checkpoint": None if parent is None else parent.name,
            "rank_progress": sorted(rank_summaries, key=lambda summary: summary["rank"]),
            "dcp_sha256": _dcp_payload_hashes(staging),
            "sidecar_sha256": sidecar_sha256,
        }
        _atomic_json(staging / "pdd_config.json", self.identity)
        _atomic_json(
            staging / "trainer_state.json",
            {
                "completed_steps": completed_steps,
                "learning_rates": learning_rates,
                "step_scheduler": dict(step_scheduler_state),
                "parent_checkpoint": manifest["parent_checkpoint"],
            },
        )
        _atomic_json(staging / "manifest.json", manifest)
        _fsync_tree(staging)
        staging.rename(final)
        _fsync_directory(self.root)
        _atomic_json(
            final / "COMPLETE",
            {
                "schema_version": _COMPLETE_SCHEMA_VERSION,
                "manifest_sha256": _sha256(final / "manifest.json"),
            },
        )
        self._validate_checkpoint(final, require_identity=True)
        _atomic_text(self.root / "LATEST", final.name + "\n")

    def save(self) -> Path:
        """Save into staging, publish atomically, mark complete, then update LATEST."""
        completed_steps = self.trainer.completed_steps
        if type(completed_steps) is not int or completed_steps <= 0:
            raise ValueError("PDD checkpoint requires at least one completed optimizer step.")
        rank_summaries = _gather_objects(self._rank_summary())
        if len({summary["sample_slots_consumed"] for summary in rank_summaries}) != 1:
            raise RuntimeError("PDD ranks disagree on consumed sample slots.")
        learning_rates = [float(group["lr"]) for group in self.optimizer.param_groups]
        live_step_scheduler_state = self.step_scheduler.state_dict()
        if live_step_scheduler_state.get("step") != completed_steps:
            raise RuntimeError("PDD StepScheduler state does not match the completed update.")
        sampler_epoch = self.sampler.state_dict()["epoch"]
        live_epoch = live_step_scheduler_state.get("epoch")
        if sampler_epoch not in {live_epoch, live_epoch + 1}:
            raise RuntimeError("PDD sampler epoch is incompatible with the StepScheduler epoch.")
        step_scheduler_state = {"step": completed_steps, "epoch": sampler_epoch}
        self._step_scheduler_checkpoint_state.load_state_dict(step_scheduler_state)
        rank_scheduler_states = _gather_objects(step_scheduler_state)
        if any(state != step_scheduler_state for state in rank_scheduler_states):
            raise RuntimeError("PDD ranks disagree on StepScheduler checkpoint state.")
        final = self.root / f"step_{completed_steps:08d}"
        parent = self._last_checkpoint

        prepare_status = None
        if _rank() == 0:
            try:
                prepare_status = {"ok": True, "staging": self._prepare_staging(final)}
            except BaseException as error:
                prepare_status = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        prepare_status = _broadcast_rank0_payload(prepare_status)
        if not isinstance(prepare_status, dict) or type(prepare_status.get("ok")) is not bool:
            raise RuntimeError("rank 0 broadcast a malformed checkpoint preparation status.")
        if not prepare_status["ok"]:
            raise RuntimeError(
                f"rank-0 checkpoint preparation failed: {prepare_status.get('error')}."
            )
        staging = self.root / prepare_status["staging"]

        self.checkpointer.save_model(self.model, str(staging))
        self.checkpointer.save_optimizer(
            self.optimizer,
            self.model,
            str(staging),
            self.scheduler,
        )
        sidecar_error = None
        try:
            self.checkpointer.save_on_dp_ranks(self.rng, "rng", str(staging))
            self.checkpointer.save_on_dp_ranks(self.sampler, "sampler", str(staging))
            self.checkpointer.save_on_dp_ranks(
                self._step_scheduler_checkpoint_state,
                "step_scheduler",
                str(staging),
            )
            self.checkpointer.save_on_dp_ranks(self.trainer, "trainer", str(staging))
        except BaseException as error:
            sidecar_error = f"{type(error).__name__}: {error}"
        sidecar_errors = _gather_objects(sidecar_error)
        sidecar_failures = [
            f"rank {rank}: {message}"
            for rank, message in enumerate(sidecar_errors)
            if message is not None
        ]
        if sidecar_failures:
            raise RuntimeError("PDD checkpoint sidecar save failed; " + "; ".join(sidecar_failures))
        _barrier()

        publish_status: dict[str, Any] | None = None
        if _rank() == 0:
            try:
                self._publish_staging(
                    staging=staging,
                    final=final,
                    completed_steps=completed_steps,
                    learning_rates=learning_rates,
                    step_scheduler_state=step_scheduler_state,
                    parent=parent,
                    rank_summaries=rank_summaries,
                )
                publish_status = {"ok": True}
            except BaseException as error:
                publish_status = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
        publish_status = _broadcast_rank0_payload(publish_status)
        if not isinstance(publish_status, dict) or type(publish_status.get("ok")) is not bool:
            raise RuntimeError("rank 0 broadcast a malformed checkpoint publication status.")
        if not publish_status["ok"]:
            raise RuntimeError(
                f"rank-0 checkpoint publication failed: {publish_status.get('error')}."
            )
        self._last_checkpoint = final
        return final

    def load(self, restore_from: str | Path | None) -> PDDResumeState | None:
        """Strictly restore model, optimizer/scheduler, cursor/trainer, and RNG last."""
        checkpoint = self._collective_resolve(restore_from)
        if checkpoint is None:
            return None
        manifest = self._manifest(checkpoint)
        self.checkpointer.load_model(self.model, str(checkpoint / "model"))
        self.checkpointer.load_optimizer(
            self.optimizer,
            self.model,
            str(checkpoint),
            self.scheduler,
        )
        self.checkpointer.load_on_dp_ranks(self.trainer, "trainer", str(checkpoint))
        self.checkpointer.load_on_dp_ranks(self.sampler, "sampler", str(checkpoint))
        self.checkpointer.load_on_dp_ranks(
            self._step_scheduler_checkpoint_state,
            "step_scheduler",
            str(checkpoint),
        )
        rank_progress = manifest["rank_progress"]
        if not isinstance(rank_progress, list) or len(rank_progress) != _world_size():
            raise RuntimeError("PDD checkpoint rank progress does not match world size.")
        progress = rank_progress[_rank()]
        if progress.get("rank") != _rank():
            raise RuntimeError("PDD checkpoint rank progress is not ordered by rank.")
        sampler_state = self.sampler.state_dict()
        for key in (
            "epoch",
            "committed_batches",
            "sample_slots_consumed",
            "plan_sha256",
            "next_sample_ids",
        ):
            if sampler_state[key] != progress[key]:
                raise RuntimeError(f"PDD restored sampler {key} does not match the manifest.")
        if self.trainer.completed_steps != manifest["completed_steps"]:
            raise RuntimeError("PDD restored trainer step does not match the manifest.")
        step_scheduler_state = self._step_scheduler_checkpoint_state.state_dict()
        if step_scheduler_state != manifest["step_scheduler"]:
            raise RuntimeError("PDD restored StepScheduler state does not match the manifest.")
        if step_scheduler_state["step"] != self.trainer.completed_steps:
            raise RuntimeError("PDD restored StepScheduler step does not match the trainer.")
        if step_scheduler_state["epoch"] != sampler_state["epoch"]:
            raise RuntimeError("PDD restored StepScheduler epoch does not match the sampler.")
        self.step_scheduler.load_state_dict(step_scheduler_state)
        current_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        if current_lrs != manifest["learning_rates"]:
            raise RuntimeError("PDD restored learning rate does not match the manifest.")
        self.checkpointer.load_on_dp_ranks(self.rng, "rng", str(checkpoint))
        self._last_checkpoint = checkpoint
        return PDDResumeState(
            checkpoint_path=checkpoint,
            completed_steps=manifest["completed_steps"],
            sample_slots_consumed=progress["sample_slots_consumed"],
            expected_next_sample_ids=tuple(progress["next_sample_ids"]),
            parent_checkpoint=manifest["parent_checkpoint"],
        )
