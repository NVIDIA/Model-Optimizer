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
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_map

__all__ = ["BypassCheckpoint"]


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
        digest.update(f"{type(native).__module__}.{type(native).__qualname__}".encode())
        digest.update(_batch_hash(_semantic_config(native.metadata)).encode())
        value = native._quantized_data
    value = value.detach()
    if value.numel() * value.element_size() <= 8 * 1024 * 1024:
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        return
    rows = max(1, (8 * 1024 * 1024) // (value[0].numel() * value.element_size()))
    for chunk in value.split(rows, dim=0):
        _tensor_hash(digest, chunk.squeeze(0) if chunk.shape[0] == 1 else chunk)


def _model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict(keep_vars=True).items():
        if isinstance(value, torch.Tensor):
            digest.update(str((name, tuple(value.shape), value.dtype)).encode())
            _tensor_hash(digest, value)
        else:
            digest.update(json.dumps([name, value], sort_keys=True, default=str).encode())
    for name, module in model.named_modules():
        native = getattr(module, "_native_weight_meta", None)
        if native is not None:
            digest.update(name.encode())
            digest.update(_batch_hash(_semantic_config(native)).encode())
        if hasattr(module, "num_bits") and hasattr(module, "backend"):
            digest.update(str((name, module.extra_repr())).encode())
    config = getattr(model, "config", None)
    if callable(getattr(config, "to_dict", None)):
        digest.update(b"model_config:")
        digest.update(json.dumps(_semantic_config(config), sort_keys=True).encode())
    return digest.hexdigest()


def _semantic_config(value):
    if callable(getattr(value, "to_dict", None)):
        original, value = value, value.to_dict()
        # Keep live subconfigs so their omitted execution choices are included too.
        for name in value:
            nested = getattr(original, name, None)
            if callable(getattr(nested, "to_dict", None)):
                value[name] = nested
        # HF omits these when saving, although they affect execution precision/math.
        for name in ("distributed_config", "_attn_implementation", "_experts_implementation"):
            if hasattr(original, name):
                value[name] = getattr(original, name)
    if isinstance(value, dict):
        return {
            key: _semantic_config(item)
            for key, item in value.items()
            if key not in {"_name_or_path", "_commit_hash", "transformers_version"}
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_config(item) for item in value]
    if isinstance(value, (torch.dtype, Path)):
        return str(value)
    return value


def _batch_hash(batch):
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            _tensor_hash(digest, value)
        elif isinstance(value, dict):
            digest.update(str(("dict", len(value))).encode())
            for key in sorted(value, key=lambda key: (type(key).__name__, str(key))):
                update(key)
                update(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(str((type(value).__name__, len(value))).encode())
            for item in value:
                update(item)
        elif value is None or isinstance(value, (str, int, float, bool)):
            digest.update(json.dumps([type(value).__name__, value]).encode())
        else:
            raise ValueError(
                "Checkpointed batch Sequences require tensor or JSON-compatible leaves."
            )

    update(batch)
    return digest.hexdigest()


def _data_identity(loader):
    if isinstance(loader, Sequence):
        digest = hashlib.sha256(str(len(loader)).encode())
        for index in range(len(loader)):
            digest.update(_batch_hash(loader[index]).encode())
        return {"kind": "sequence", "length": len(loader), "sha256": digest.hexdigest()}
    if (
        callable(getattr(loader, "state_dict", None))
        and callable(getattr(loader, "load_state_dict", None))
        and getattr(loader, "data_identity", None) is not None
    ):
        return {"kind": "stateful", "identity": _semantic_config(loader.data_identity)}
    raise ValueError(
        "Checkpointing needs a replayable Sequence or a stateful loader with data_identity"
    )


def _rng_state(devices):
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), *state[2:]),
        "cuda": {device: torch.cuda.get_rng_state(device) for device in devices},
    }


def _restore_rng_state(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.array(numpy[1], dtype=np.uint32), *numpy[2:]))
    for device, value in state["cuda"].items():
        torch.cuda.set_rng_state(value, device)


def _optimizer_state(optimizer, fields):
    names = {id(value): name for name, value in fields.items()}
    groups = [
        {
            **{key: value for key, value in group.items() if key != "params"},
            "params": [names[id(parameter)] for parameter in group["params"]],
        }
        for group in optimizer.param_groups
    ]
    state = {
        names[id(parameter)]: tree_map(
            lambda v: v.detach().cpu() if isinstance(v, torch.Tensor) else v, values
        )
        for parameter, values in optimizer.state.items()
    }
    return {"groups": groups, "state": state}


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
            raise ValueError("Bypass checkpoint transaction failed: " + result[0]["error"])
        return result[0]["value"]


class BypassCheckpoint(_CheckpointRanks):
    """Commit every rank's auxiliary state before retiring the prior generation."""

    def __init__(self, student, teacher, config, runtime=None, *, dataloader):
        """Bind committed state to immutable inputs, the recipe and realized owners."""
        super().__init__(
            runtime.checkpoint_group() if runtime is not None else None,
            distributed=runtime is not None,
        )
        self.directory = config.checkpoint_dir
        self.identity = None
        error = None
        try:
            self.identity = {
                "student": _model_hash(student),
                "teacher": _model_hash(teacher),
                "config": config.model_dump(
                    exclude={"checkpoint_dir", "resume_from", "checkpoint_interval"}
                ),
                "topology": runtime.checkpoint_topology() if runtime is not None else None,
                "data": _data_identity(dataloader),
                "runtime": {
                    "bypass_recipe": 1,
                    "torch": str(torch.__version__),
                    "matmul_precision": torch.get_float32_matmul_precision(),
                    "tf32": torch.backends.cuda.matmul.allow_tf32,
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                },
            }
            if any(
                getattr(getattr(model, "config", None), "model_type", None)
                for model in (student, teacher)
            ):
                self.identity["runtime"]["transformers"] = version("transformers")
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [value for value in self._gather(error) if value]
        if errors:
            raise ValueError("Bypass checkpoint identity failed: " + "; ".join(errors))

    def restore_data_cursor(self, restore):
        """Validate every rank's next-sample cursor before any training collective."""
        result, error = None, None
        try:
            result = restore()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [value for value in self._gather(error) if value is not None]
        if errors:
            raise ValueError("Bypass checkpoint sample order differs: " + "; ".join(errors))
        return result

    def save(self, step, *, fields, optimizers, data_position, rng_state):
        """Write/fsync all shards, commit a manifest, then retire the old state."""
        root = Path(self.directory)
        state, error = None, None
        try:
            state = {
                "step": step,
                "fields": {name: value.detach().cpu() for name, value in fields.items()},
                "optimizers": {
                    name: _optimizer_state(optimizer, fields)
                    for name, optimizer in optimizers.items()
                },
                "data_position": data_position,
                "rng_state": rng_state,
            }
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        reports = self._gather({"error": error, "identity": self.identity})
        errors = [value["error"] for value in reports if value["error"]]
        if errors:
            raise ValueError("; ".join(errors))

        def begin():
            root.mkdir(parents=True, exist_ok=True)
            if (root / "latest").exists():
                previous = (root / "latest").read_text().strip()
                if Path(previous).name != previous:
                    raise ValueError("Invalid Bypass latest pointer")
                manifest = json.loads((root / previous / "COMMITTED.json").read_text())
                identities = {str(rank): value["identity"] for rank, value in enumerate(reports)}
                if manifest["step"] >= step or json.dumps(
                    manifest["identities"], sort_keys=True
                ) != json.dumps(identities, sort_keys=True):
                    raise ValueError(
                        "Checkpoint directory contains another or newer run; resume it or choose a new directory"
                    )
            name = f"step-{step:06d}-{uuid.uuid4().hex}"
            (root / (".pending-" + name)).mkdir()
            return name

        name = self._coordinate(begin)
        pending = root / (".pending-" + name)
        shard = pending / f"rank-{self.rank}.pt"
        receipt = None
        try:
            with shard.open("wb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            receipt = {"step": step, "identity": self.identity, "sha256": _file_hash(shard)}
            _write_json(pending / f"rank-{self.rank}.json", receipt)
            _sync_directory(pending)
        except Exception as exc:
            receipt = {"error": f"rank {self.rank}: {type(exc).__name__}: {exc}"}
        receipts = self._gather(receipt)
        errors = [value["error"] for value in receipts if "error" in value]

        def discard_pending():
            if pending.exists():
                shutil.rmtree(pending)
                _sync_directory(root)

        if errors:
            self._coordinate(discard_pending)
            raise ValueError("Bypass checkpoint shards failed: " + "; ".join(errors))

        def commit():
            for rank, expected in enumerate(receipts):
                actual = json.loads((pending / f"rank-{rank}.json").read_text())
                if (
                    json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True)
                    or actual["step"] != step
                ):
                    raise ValueError("Bypass checkpoint rank receipts disagree.")
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

        try:
            return Path(self._coordinate(commit))
        except Exception:
            self._coordinate(discard_pending)
            raise

    def load(self, directory):
        """Validate the committed generation collectively before restoring any rank."""

        def resolve():
            root = Path(directory)
            if (root / "latest").is_file():
                name = (root / "latest").read_text().strip()
                if Path(name).name != name:
                    raise ValueError("Invalid Bypass checkpoint latest pointer.")
                root = root / name
            if not (root / "COMMITTED.json").is_file():
                raise ValueError("Bypass checkpoint generation is incomplete.")
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
                    "Bypass checkpoint input, topology or configuration identity differs."
                )
            if set(manifest["shards"]) != {f"rank-{rank}.pt" for rank in range(self.size)}:
                raise ValueError("Bypass checkpoint does not contain every rank.")
            shard = root / f"rank-{self.rank}.pt"
            if _file_hash(shard) != manifest["shards"][shard.name]:
                raise ValueError("Bypass checkpoint shard failed integrity validation.")
            state = torch.load(shard, map_location="cpu", weights_only=True)
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [value for value in self._gather(error) if value is not None]
        if errors:
            raise ValueError("Bypass checkpoint validation failed: " + "; ".join(errors))
        assert manifest is not None
        return manifest["step"], state

    def restore_training_state(self, state, fields, optimizers):
        """Validate every owner before changing masters or name-keyed Lion state."""
        error = None
        names = {id(parameter): name for name, parameter in fields.items()}
        try:
            if (
                state["fields"].keys() != fields.keys()
                or state["optimizers"].keys() != optimizers.keys()
            ):
                raise ValueError("Bypass native field ownership differs")
            for name, parameter in fields.items():
                saved = state["fields"][name]
                if saved.shape != parameter.shape or saved.dtype != torch.float32:
                    raise ValueError("Bypass native field shape or precision differs")
            for name, optimizer in optimizers.items():
                saved = state["optimizers"][name]
                current = optimizer.param_groups
                if len(current) != len(saved["groups"]):
                    raise ValueError("Bypass optimizer groups differ")
                for group, previous in zip(current, saved["groups"], strict=True):
                    if {names[id(p)] for p in group["params"]} != set(previous["params"]):
                        raise ValueError("Bypass optimizer field ownership differs")
                for field, values in saved["state"].items():
                    if field not in fields or set(values) != {"exp_avg"}:
                        raise ValueError("Bypass Lion state contains an unknown native field")
                    if (
                        values["exp_avg"].shape != fields[field].shape
                        or values["exp_avg"].dtype != torch.float32
                    ):
                        raise ValueError("Bypass Lion moment shape or precision differs")
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [value for value in self._gather(error) if value]
        if errors:
            raise ValueError("; ".join(errors))
        with torch.no_grad():
            for name, parameter in fields.items():
                parameter.copy_(state["fields"][name])
        for name, optimizer in optimizers.items():
            saved = state["optimizers"][name]
            current = optimizer.state_dict()
            restored: dict[str, Any] = {"state": {}, "param_groups": []}
            for group, index_group, previous in zip(
                optimizer.param_groups, current["param_groups"], saved["groups"], strict=True
            ):
                restored["param_groups"].append({**previous, "params": index_group["params"]})
                for parameter, index in zip(group["params"], index_group["params"], strict=True):
                    field = names[id(parameter)]
                    if field in saved["state"]:
                        restored["state"][index] = saved["state"][field]
            optimizer.load_state_dict(restored)
