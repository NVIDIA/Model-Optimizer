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

"""Committed rank-local repair generations, without duplicating frozen models."""

import hashlib
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

__all__ = ["LocalRepairCheckpoint"]


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(digest, value):
    if isinstance(value, DTensor):
        digest.update(str((value.device_mesh.mesh.tolist(), value.placements)).encode())
        value = value.to_local()
    if hasattr(value, "get_qtensor"):
        native = value.get_qtensor()
        digest.update(str(native.metadata).encode())
        value = native._quantized_data
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
    for start in range(0, raw.numel(), 8 * 1024 * 1024):
        digest.update(raw[start : start + 8 * 1024 * 1024].cpu().numpy().tobytes())


def _model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            digest.update(str((name, tuple(value.shape), value.dtype)).encode())
            _tensor_hash(digest, value)
        else:
            digest.update(json.dumps([name, value], sort_keys=True, default=str).encode())
    for name, module in model.named_modules():
        if hasattr(module, "num_bits") and hasattr(module, "backend"):
            digest.update(str((name, module.extra_repr())).encode())
    return digest.hexdigest()


def _batch_hash(batch):
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            _tensor_hash(digest, value)
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                update(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(str((type(value).__name__, len(value))).encode())
            for item in value:
                update(item)
        elif value is None or isinstance(value, (str, int, float, bool)):
            digest.update(json.dumps(value).encode())
        else:
            raise ValueError(
                "Checkpointed batch Sequences require tensor or JSON-compatible leaves."
            )

    update(batch)
    return digest.hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value):
    with path.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


class _CheckpointRanks:
    def __init__(self, group=None, *, distributed=False):
        self.group = group
        self.distributed = distributed
        self.rank = dist.get_rank(group) if distributed else 0
        self.size = dist.get_world_size(group) if distributed else 1
        self.coordinator = dist.get_global_rank(group, 0) if group is not None else 0

    def _gather(self, value):
        if self.size == 1:
            return [value]
        values: list[Any] = [None] * self.size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def _coordinate(self, operation):
        result: list[dict[str, Any]] = [{}]
        if self.rank == 0:
            try:
                result[0] = {"value": operation()}
            except Exception as exc:
                result[0] = {"error": f"{type(exc).__name__}: {exc}"}
        if self.size > 1:
            dist.broadcast_object_list(result, src=self.coordinator, group=self.group)
        if "error" in result[0]:
            raise ValueError("Repair checkpoint transaction failed: " + result[0]["error"])
        return result[0]["value"]


class LocalRepairCheckpoint(_CheckpointRanks):
    """Commit every rank's auxiliary state before retiring the prior generation."""

    def __init__(self, student, teacher, config, runtime=None):
        """Bind resume to the immutable PTQ inputs, recipe, and realized topology."""
        super().__init__(
            runtime.checkpoint_group() if runtime is not None else None,
            distributed=runtime is not None,
        )
        settings = config.model_dump(exclude={"checkpoint_dir", "resume_from", "output_dir"})
        self.identity = {
            "student": _model_hash(student),
            "teacher": _model_hash(teacher),
            "config": settings,
            "topology": runtime.checkpoint_topology() if runtime is not None else None,
        }

    def restore_data_cursor(self, restore):
        """Validate every rank's next-sample cursor before any training collective."""
        result, error = None, None
        try:
            result = restore()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [value for value in self._gather(error) if value is not None]
        if errors:
            raise ValueError("Repair checkpoint sample order differs: " + "; ".join(errors))
        return result

    def save(self, directory, step, state):
        """Write/fsync all shards, commit a manifest, then retire the old state."""
        root = Path(directory)

        def begin():
            root.mkdir(parents=True, exist_ok=True)
            name = f"step-{step:06d}-{uuid.uuid4().hex}"
            (root / (".pending-" + name)).mkdir()
            return name

        name = self._coordinate(begin)
        pending = root / (".pending-" + name)
        shard = pending / f"rank-{self.rank}.pt"
        receipt = None
        try:
            payload = {**state, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}
            if torch.cuda.is_available():
                payload["cuda_rng"] = torch.cuda.get_rng_state()
            with shard.open("wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            receipt = {"step": step, "identity": self.identity, "sha256": _file_hash(shard)}
            _write_json(pending / f"rank-{self.rank}.json", receipt)
            _sync_directory(pending)
        except Exception as exc:
            receipt = {"error": f"rank {self.rank}: {type(exc).__name__}: {exc}"}
        receipts = self._gather(receipt)
        errors = [value["error"] for value in receipts if "error" in value]
        if errors:
            raise ValueError("Repair checkpoint shards failed: " + "; ".join(errors))

        def commit():
            for rank, expected in enumerate(receipts):
                actual = json.loads((pending / f"rank-{rank}.json").read_text())
                if (
                    json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True)
                    or actual["step"] != step
                ):
                    raise ValueError("Repair checkpoint rank receipts disagree.")
            manifest = {
                "step": step,
                "world_size": self.size,
                "identities": {str(rank): value["identity"] for rank, value in enumerate(receipts)},
                "shards": {
                    f"rank-{rank}.pt": value["sha256"] for rank, value in enumerate(receipts)
                },
            }
            _write_json(pending / "COMMITTED.json", manifest)
            _sync_directory(pending)
            os.replace(pending, root / name)
            _sync_directory(root)
            previous = (root / "latest").read_text().strip() if (root / "latest").exists() else None
            temporary = root / (".latest-" + uuid.uuid4().hex)
            with temporary.open("w") as stream:
                stream.write(name + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, root / "latest")
            _sync_directory(root)
            if previous and previous != name and Path(previous).name == previous:
                old = root / previous
                if not old.is_symlink() and (old / "COMMITTED.json").is_file():
                    shutil.rmtree(old)
            return str(root / name)

        return Path(self._coordinate(commit))

    def load(self, directory):
        """Validate the committed generation collectively before restoring any rank."""

        def resolve():
            root = Path(directory)
            if (root / "latest").is_file():
                name = (root / "latest").read_text().strip()
                if Path(name).name != name:
                    raise ValueError("Invalid repair checkpoint latest pointer.")
                root = root / name
            if not (root / "COMMITTED.json").is_file():
                raise ValueError("Repair checkpoint generation is incomplete.")
            return str(root)

        root = Path(self._coordinate(resolve))
        error = None
        state = None
        manifest = None
        try:
            manifest = json.loads((root / "COMMITTED.json").read_text())
            identities: dict[str, Any] = manifest["identities"]
            if manifest["world_size"] != self.size or json.dumps(
                identities[str(self.rank)], sort_keys=True
            ) != json.dumps(self.identity, sort_keys=True):
                raise ValueError(
                    "Repair checkpoint input, topology or configuration identity differs."
                )
            if set(manifest["shards"]) != {f"rank-{rank}.pt" for rank in range(self.size)}:
                raise ValueError("Repair checkpoint does not contain every rank.")
            shard = root / f"rank-{self.rank}.pt"
            if _file_hash(shard) != manifest["shards"][shard.name]:
                raise ValueError("Repair checkpoint shard failed integrity validation.")
            # Trusted optimizer/RNG state generated by this runner, not imported models.
            state = torch.load(shard, map_location="cpu", weights_only=False)
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [value for value in self._gather(error) if value is not None]
        if errors:
            raise ValueError("Repair checkpoint validation failed: " + "; ".join(errors))
        assert manifest is not None
        return manifest["step"], state
