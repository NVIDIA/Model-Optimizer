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

"""Collectively restore a PDD checkpoint and publish a safe Qwen-Image export."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

sys.dont_write_bytecode = True

_THIS_DIR = Path(__file__).resolve().parent
_FASTGEN_DIR = _THIS_DIR.parent
_REPO_ROOT = _FASTGEN_DIR.parents[2]
for path in (_REPO_ROOT, _FASTGEN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=_THIS_DIR / "configs" / "qwen_image.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint basename/path beneath checkpoint_dir, or LATEST; defaults to config.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size-gib", type=float, default=5.0)
    parser.add_argument("--memory-headroom", type=float, default=1.25)
    return parser.parse_args()


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def host_available_bytes() -> int:
    """Return the strictest visible host/cgroup memory availability estimate."""
    candidates: list[int] = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    for limit_path, used_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ):
        limit = _read_integer(limit_path)
        used = _read_integer(used_path)
        if limit is not None and used is not None and 0 < limit < (1 << 62):
            candidates.append(max(0, limit - used))
    if not candidates:
        raise RuntimeError("cannot determine host memory availability.")
    return min(candidates)


def _state_sizes(model: torch.nn.Module) -> tuple[int, int]:
    sizes = [value.numel() * value.element_size() for value in model.state_dict().values()]
    if not sizes:
        raise RuntimeError("PDD export model has an empty state dictionary.")
    return sum(sizes), max(sizes)


def collective_export_memory_preflight(
    model: torch.nn.Module,
    *,
    max_shard_bytes: int,
    headroom: float,
    device: torch.device,
) -> tuple[int, int]:
    """Abort collectively before full-state gathering when host/GPU headroom is insufficient."""
    if not math.isfinite(headroom) or headroom < 1.0:
        raise ValueError("memory_headroom must be finite and >= 1.")
    full_state_bytes, largest_tensor_bytes = _state_sizes(model)
    local_error = None
    try:
        required_gpu = math.ceil(largest_tensor_bytes * headroom)
        if device.type == "cuda":
            free_gpu, _total_gpu = torch.cuda.mem_get_info(device)
            if free_gpu < required_gpu:
                raise MemoryError(
                    f"rank {dist.get_rank()} has {free_gpu} free GPU bytes; "
                    f"full-state gather requires at least {required_gpu}."
                )
        if dist.get_rank() == 0:
            required_host = math.ceil((full_state_bytes + max_shard_bytes) * headroom)
            available_host = host_available_bytes()
            if available_host < required_host:
                raise MemoryError(
                    f"rank 0 has {available_host} available host bytes; export requires at "
                    f"least {required_host}."
                )
    except BaseException as error:
        local_error = f"{type(error).__name__}: {error}"
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        raise RuntimeError("PDD export memory preflight failed; " + "; ".join(failures))
    return full_state_bytes, largest_tensor_bytes


def _git_source_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if dirty:
        raise RuntimeError("PDD export requires a clean ModelOpt source checkout.")
    return {"commit": commit, "dirty": False}


def _collective_publication_preflight(output_dir: Path) -> Mapping[str, Any]:
    status = None
    if dist.get_rank() == 0:
        try:
            if output_dir.is_symlink() or output_dir.resolve().exists():
                raise FileExistsError(f"PDD export output already exists: {output_dir}.")
            status = {"ok": True, "modelopt_source": _git_source_identity()}
        except BaseException as error:
            status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    payload = [status]
    dist.broadcast_object_list(payload, src=0)
    status = payload[0]
    if not isinstance(status, Mapping) or type(status.get("ok")) is not bool:
        raise RuntimeError("rank 0 broadcast malformed PDD publication preflight status.")
    if not status["ok"]:
        raise RuntimeError(f"PDD publication preflight failed: {status.get('error')}.")
    modelopt_source = status.get("modelopt_source")
    if not isinstance(modelopt_source, Mapping):
        raise RuntimeError("rank 0 broadcast malformed ModelOpt source identity.")
    return modelopt_source


def _require_checkpoint_identity(config: Any, setup: Any, manifest: Mapping[str, Any]) -> None:
    from modelopt.torch.fastgen import PDDMetadata

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("PDD checkpoint has no identity mapping.")
    pdd_metadata = identity.get("pdd_metadata")
    if not isinstance(pdd_metadata, Mapping):
        raise RuntimeError("PDD checkpoint has no PDD metadata mapping.")
    if PDDMetadata.from_dict(pdd_metadata) != setup.metadata:
        raise RuntimeError("PDD checkpoint metadata does not match the configured student.")
    if identity.get("model") != {
        "id": config.model_id,
        "revision": config.model_revision,
        "dtype": str(config.dtype).removeprefix("torch."),
    }:
        raise RuntimeError("PDD checkpoint model identity does not match the export config.")
    checkpoint_automodel = identity.get("automodel")
    if not isinstance(checkpoint_automodel, Mapping):
        raise RuntimeError("PDD checkpoint has no AutoModel identity.")
    for key in (
        "distribution",
        "version",
        "package_tree_sha256",
        "wheel_sha256",
        "runtime_versions",
    ):
        if checkpoint_automodel.get(key) != setup.automodel_snapshot.get(key):
            raise RuntimeError(f"PDD checkpoint AutoModel identity mismatch for {key}.")
    topology = identity.get("topology")
    if not isinstance(topology, Mapping) or topology.get("world_size") != dist.get_world_size():
        raise RuntimeError("PDD checkpoint topology does not match the export process group.")


def _collective_checkpoint_identity(config: Any, setup: Any, manifest: Mapping[str, Any]) -> None:
    local_error = None
    try:
        _require_checkpoint_identity(config, setup, manifest)
    except BaseException as error:
        local_error = f"{type(error).__name__}: {error}"
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failures:
        raise RuntimeError("PDD checkpoint identity validation failed; " + "; ".join(failures))


def _checkpoint_selector_identity(config: Any, setup: Any) -> dict[str, Any]:
    return {
        "model": {
            "id": config.model_id,
            "revision": config.model_revision,
            "dtype": str(config.dtype).removeprefix("torch."),
        },
        "pdd_metadata": setup.metadata.to_dict(),
        "guidance": {
            "scale": config.pdd.guidance_scale,
            "rescale": config.guidance.rescale,
            "eps": config.guidance.eps,
        },
        "automodel": {
            key: setup.automodel_snapshot[key]
            for key in (
                "distribution",
                "version",
                "package_tree_sha256",
                "wheel_sha256",
                "runtime_versions",
            )
        },
        "topology": {"world_size": dist.get_world_size(), "pure_data_parallel": True},
    }


def _collective_checkpoint_resolution(
    config: Any, setup: Any, restore_from: str
) -> tuple[Path, Mapping[str, Any]]:
    from pdd.checkpoint import resolve_pdd_training_checkpoint

    status = None
    if dist.get_rank() == 0:
        try:
            checkpoint, manifest = resolve_pdd_training_checkpoint(
                config.checkpoint.checkpoint_dir,
                restore_from,
                expected_world_size=dist.get_world_size(),
                expected_identity=_checkpoint_selector_identity(config, setup),
            )
            status = {"ok": True, "checkpoint": str(checkpoint), "manifest": manifest}
        except BaseException as error:
            status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    payload = [status]
    dist.broadcast_object_list(payload, src=0)
    status = payload[0]
    if not isinstance(status, Mapping) or type(status.get("ok")) is not bool:
        raise RuntimeError("rank 0 broadcast malformed PDD checkpoint resolution status.")
    if not status["ok"]:
        raise RuntimeError(f"PDD checkpoint resolution failed: {status.get('error')}.")
    return Path(status["checkpoint"]), status["manifest"]


def main() -> None:
    args = _parse_args()
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    from pdd.artifacts import sha256_file
    from pdd.export import write_pdd_export
    from pdd.recipe import (
        build_pdd_export_setup,
        initialize_pdd_distributed,
        resolve_pdd_recipe_config,
    )

    raw = yaml.safe_load(args.config.read_text())
    config = resolve_pdd_recipe_config(raw)
    if Path(config.model_id).is_dir() or config.model_revision is None:
        raise ValueError(
            "PDD inference export requires a remote model ID and pinned 40-character revision; "
            "mutable local model directories are training-only inputs."
        )
    if not math.isfinite(args.max_shard_size_gib) or args.max_shard_size_gib <= 0:
        raise ValueError("max_shard_size_gib must be finite and > 0.")
    if not math.isfinite(args.memory_headroom) or args.memory_headroom < 1.0:
        raise ValueError("memory_headroom must be finite and >= 1.")
    max_shard_bytes = int(args.max_shard_size_gib * (1 << 30))
    initialize_pdd_distributed(
        backend="nccl" if config.device.type == "cuda" else "gloo",
        timeout_minutes=60,
    )
    modelopt_source = _collective_publication_preflight(args.output_dir)
    restore_from = args.checkpoint or config.checkpoint.restore_from
    if not restore_from:
        raise ValueError("PDD export requires --checkpoint or checkpoint.restore_from.")
    setup = build_pdd_export_setup(config)
    try:
        checkpoint, checkpoint_manifest = _collective_checkpoint_resolution(
            config, setup, restore_from
        )
        _collective_checkpoint_identity(config, setup, checkpoint_manifest)
        setup.checkpointer.load_model(setup.student, str(checkpoint / "model"))
        full_state_bytes, largest_tensor_bytes = collective_export_memory_preflight(
            setup.student,
            max_shard_bytes=max_shard_bytes,
            headroom=args.memory_headroom,
            device=config.device,
        )
        state_dict = get_model_state_dict(
            setup.student,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        local_error = None
        try:
            if dist.get_rank() == 0:
                if set(state_dict) != set(setup.checkpoint_keys):
                    raise RuntimeError("gathered PDD full-state keys do not match the model.")
                if (
                    sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
                    != full_state_bytes
                ):
                    raise RuntimeError(
                        "gathered PDD full-state byte count changed after preflight."
                    )
                gathered_largest = max(
                    tensor.numel() * tensor.element_size() for tensor in state_dict.values()
                )
                if gathered_largest != largest_tensor_bytes:
                    raise RuntimeError("gathered PDD largest tensor changed after preflight.")
            elif state_dict:
                raise RuntimeError("nonzero rank received a full CPU state dictionary.")
        except BaseException as error:
            local_error = f"{type(error).__name__}: {error}"
        gather_errors: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gather_errors, local_error)
        failures = [f"rank {rank}: {error}" for rank, error in enumerate(gather_errors) if error]
        if failures:
            raise RuntimeError("PDD full-state gather validation failed; " + "; ".join(failures))

        publication = None
        if dist.get_rank() == 0:
            try:
                output = write_pdd_export(
                    args.output_dir,
                    state_dict,
                    metadata=setup.metadata,
                    transformer_config=setup.transformer_config,
                    identity=checkpoint_manifest["identity"],
                    source_checkpoint={
                        "name": checkpoint.name,
                        "manifest_sha256": sha256_file(checkpoint / "manifest.json"),
                        "completed_steps": checkpoint_manifest["completed_steps"],
                    },
                    modelopt_source=modelopt_source,
                    max_shard_bytes=max_shard_bytes,
                )
                publication = {"ok": True, "output": str(output)}
            except BaseException as error:
                publication = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        payload = [publication]
        dist.broadcast_object_list(payload, src=0)
        publication = payload[0]
        if not isinstance(publication, Mapping) or type(publication.get("ok")) is not bool:
            raise RuntimeError("rank 0 broadcast malformed PDD publication status.")
        if not publication["ok"]:
            raise RuntimeError(f"PDD export publication failed: {publication.get('error')}.")
        if dist.get_rank() == 0:
            print(json.dumps(publication, indent=2, sort_keys=True))
    finally:
        setup.checkpointer.close()


if __name__ == "__main__":
    main()
