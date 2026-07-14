# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated, bounded safetensors export and strict PDD reconstruction helpers."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pdd_artifacts import (
    load_canonical_json,
    require_sha256,
    resolve_relative_artifact,
    sha256_file,
    write_canonical_json,
)
from safetensors import safe_open
from safetensors.torch import save_file

from modelopt.torch.fastgen import PDDConfig, PDDMetadata
from modelopt.torch.fastgen.plugins.qwen_image_pdd import QWEN_IMAGE_PDD_LAYER_SPEC

_EXPORT_SCHEMA_VERSION = 1
_COMPLETE_SCHEMA_VERSION = 1
_EXPORT_FORMAT = "modelopt-pdd-safetensors"
_CONFIG_FILE = "config.json"
_METADATA_FILE = "pdd_metadata.json"
_INDEX_FILE = "diffusion_pytorch_model.safetensors.index.json"
_MANIFEST_FILE = "manifest.json"
_COMPLETE_FILE = "COMPLETE"
_SHARD_PATTERN = "diffusion_pytorch_model-{index:05d}-of-{count:05d}.safetensors"

PDD_INFERENCE_SCHEDULES: Mapping[str, tuple[int, ...]] = {
    "pdd-2": (64, 64),
    "pdd-4": (32, 32, 32, 32),
    "pdd-8": (16, 16, 16, 16, 16, 16, 16, 16),
}


@dataclass(frozen=True)
class PDDExportDescriptor:
    """Validated non-tensor export metadata."""

    root: Path
    manifest: Mapping[str, Any]
    metadata: PDDMetadata
    transformer_config: Mapping[str, Any]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"PDD export staging contains a symlink: {path}.")
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        elif path.is_dir():
            _fsync_directory(path)
    _fsync_directory(root)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _validate_state_dict(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("PDD export state_dict must be a non-empty mapping.")
    tensors: dict[str, torch.Tensor] = {}
    for key in sorted(state_dict):
        tensor = state_dict[key]
        if not isinstance(key, str) or not key:
            raise ValueError("PDD export tensor keys must be non-empty strings.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"PDD export value {key!r} is not a tensor.")
        if tensor.device.type != "cpu" or tensor.is_meta:
            raise ValueError(f"PDD export tensor {key!r} must be a materialized CPU tensor.")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise TypeError(f"PDD export tensor {key!r} must be a dense, unquantized tensor.")
        if tensor.dtype.is_floating_point and not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"PDD export tensor {key!r} is non-finite.")
        tensors[key] = tensor.detach().contiguous()
    return tensors


def _save_probe(staging: Path, keys: Sequence[str], tensors: Mapping[str, torch.Tensor]) -> int:
    probe = staging / f".probe-{uuid.uuid4().hex}.safetensors"
    payload = {key: tensors[key].clone() for key in keys}
    try:
        save_file(payload, str(probe), metadata={"format": "pt"})
        return probe.stat().st_size
    finally:
        probe.unlink(missing_ok=True)


def _bounded_shard_groups(
    staging: Path,
    tensors: Mapping[str, torch.Tensor],
    max_shard_bytes: int,
) -> list[tuple[str, ...]]:
    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be a positive integer.")
    initial: list[tuple[str, ...]] = []
    current: list[str] = []
    current_bytes = 0
    for key, tensor in tensors.items():
        size = _tensor_nbytes(tensor)
        if size >= max_shard_bytes:
            raise ValueError(
                f"tensor {key!r} has {size} bytes and cannot fit beneath the physical "
                f"shard bound {max_shard_bytes}."
            )
        if current and current_bytes + size >= max_shard_bytes:
            initial.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += size
    if current:
        initial.append(tuple(current))

    bounded: list[tuple[str, ...]] = []
    pending = list(initial)
    while pending:
        keys = pending.pop(0)
        if _save_probe(staging, keys, tensors) <= max_shard_bytes:
            bounded.append(keys)
            continue
        if len(keys) == 1:
            raise ValueError(
                f"tensor {keys[0]!r} plus safetensors metadata exceeds max_shard_bytes."
            )
        midpoint = len(keys) // 2
        pending[0:0] = [keys[:midpoint], keys[midpoint:]]
    return bounded


def _validate_identity(identity: Mapping[str, Any], metadata: PDDMetadata) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise TypeError("PDD export identity must be a mapping.")
    required = {"automodel", "guidance", "model", "pdd_metadata", "topology"}
    missing = sorted(required.difference(identity))
    if missing:
        raise ValueError(f"PDD export identity is missing keys: {missing}.")
    if identity["pdd_metadata"] != metadata.to_dict():
        raise ValueError("PDD export metadata does not match the checkpoint identity.")
    model = _require_exact_mapping(
        identity["model"], {"id", "revision", "dtype"}, name="identity.model"
    )
    if not isinstance(model["id"], str) or not model["id"] or not isinstance(model["dtype"], str):
        raise ValueError("PDD export checkpoint identity has an invalid model ID or dtype.")
    revision = model["revision"]
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("PDD export requires a pinned 40-character model revision.")
    try:
        int(revision, 16)
    except ValueError as error:
        raise ValueError("PDD export model revision must be hexadecimal.") from error
    automodel = _require_exact_mapping(
        identity["automodel"],
        {"distribution", "version", "package_tree_sha256", "wheel_sha256", "runtime_versions"},
        name="identity.automodel",
    )
    if (
        not isinstance(automodel["distribution"], str)
        or not automodel["distribution"]
        or not isinstance(automodel["version"], str)
        or not isinstance(automodel["runtime_versions"], Mapping)
    ):
        raise ValueError("PDD export AutoModel identity is malformed.")
    require_sha256(automodel["package_tree_sha256"], name="AutoModel package tree SHA-256")
    require_sha256(automodel["wheel_sha256"], name="AutoModel wheel SHA-256")
    guidance = _require_exact_mapping(
        identity["guidance"], {"scale", "rescale", "eps"}, name="identity.guidance"
    )
    for name, value in guidance.items():
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not torch.isfinite(torch.tensor(float(value))).item()
        ):
            raise ValueError(f"PDD export guidance {name} must be finite or null.")
    topology = identity["topology"]
    if (
        not isinstance(topology, Mapping)
        or type(topology.get("world_size")) is not int
        or topology["world_size"] < 1
        or topology.get("pure_data_parallel") is not True
    ):
        raise ValueError("PDD export checkpoint identity has invalid pure-DP topology.")
    return dict(identity)


def _validate_modelopt_source(source: Mapping[str, Any]) -> dict[str, Any]:
    source = _require_exact_mapping(source, {"commit", "dirty"}, name="modelopt_source")
    commit = source["commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("modelopt_source.commit must be a 40-character Git commit.")
    try:
        int(commit, 16)
    except ValueError as error:
        raise ValueError("modelopt_source.commit must be hexadecimal.") from error
    if source["dirty"] is not False:
        raise ValueError("modelopt_source.dirty must be false.")
    return dict(source)


def write_pdd_export(
    output_dir: str | Path,
    state_dict: Mapping[str, Any],
    *,
    metadata: PDDMetadata,
    transformer_config: Mapping[str, Any],
    identity: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    modelopt_source: Mapping[str, Any],
    max_shard_bytes: int,
) -> Path:
    """Publish a complete PDD export into a previously absent directory."""
    if not isinstance(metadata, PDDMetadata):
        raise TypeError("metadata must be PDDMetadata.")
    if metadata.layer_spec != QWEN_IMAGE_PDD_LAYER_SPEC:
        raise ValueError("PDD export supports only the fixed Qwen-Image layer specification.")
    if not isinstance(transformer_config, Mapping):
        raise TypeError("transformer_config must be a mapping.")
    checkpoint_keys = {"name", "manifest_sha256", "completed_steps"}
    if not isinstance(source_checkpoint, Mapping) or set(source_checkpoint) != checkpoint_keys:
        raise ValueError(f"source_checkpoint must contain exactly {sorted(checkpoint_keys)}.")
    if (
        not isinstance(source_checkpoint["name"], str)
        or not source_checkpoint["name"]
        or Path(source_checkpoint["name"]).name != source_checkpoint["name"]
    ):
        raise ValueError("source_checkpoint.name must be a basename.")
    require_sha256(source_checkpoint["manifest_sha256"], name="source manifest SHA-256")
    if (
        type(source_checkpoint["completed_steps"]) is not int
        or source_checkpoint["completed_steps"] < 1
    ):
        raise ValueError("source_checkpoint.completed_steps must be an integer >= 1.")
    resolved_modelopt_source = _validate_modelopt_source(modelopt_source)
    resolved_identity = _validate_identity(identity, metadata)
    tensors = _validate_state_dict(state_dict)

    unresolved_output = Path(output_dir)
    if unresolved_output.is_symlink():
        raise ValueError("PDD export output cannot be a symlink.")
    output = unresolved_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"PDD export output already exists: {output}.")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir()
    published = False
    try:
        groups = _bounded_shard_groups(staging, tensors, max_shard_bytes)
        shard_names = [
            _SHARD_PATTERN.format(index=index, count=len(groups))
            for index in range(1, len(groups) + 1)
        ]
        weight_map: dict[str, str] = {}
        for name, keys in zip(shard_names, groups):
            payload = {key: tensors[key].clone() for key in keys}
            save_file(payload, str(staging / name), metadata={"format": "pt"})
            if (staging / name).stat().st_size > max_shard_bytes:
                raise RuntimeError(f"PDD safetensors shard exceeds its physical bound: {name}.")
            weight_map.update(dict.fromkeys(keys, name))

        total_tensor_bytes = sum(_tensor_nbytes(tensor) for tensor in tensors.values())
        write_canonical_json(staging / _CONFIG_FILE, dict(transformer_config))
        write_canonical_json(staging / _METADATA_FILE, metadata.to_dict())
        write_canonical_json(
            staging / _INDEX_FILE,
            {"metadata": {"total_size": total_tensor_bytes}, "weight_map": weight_map},
        )
        file_names = [_CONFIG_FILE, _METADATA_FILE, _INDEX_FILE, *shard_names]
        files = {
            name: {
                "sha256": sha256_file(staging / name),
                "size": (staging / name).stat().st_size,
            }
            for name in file_names
        }
        tensor_specs = {
            key: {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "nbytes": _tensor_nbytes(tensor),
                "shape": list(tensor.shape),
                "shard": weight_map[key],
            }
            for key, tensor in tensors.items()
        }
        manifest = {
            "schema_version": _EXPORT_SCHEMA_VERSION,
            "format": _EXPORT_FORMAT,
            "identity": resolved_identity,
            "source_checkpoint": dict(source_checkpoint),
            "modelopt_source": resolved_modelopt_source,
            "max_shard_bytes": max_shard_bytes,
            "total_tensor_bytes": total_tensor_bytes,
            "tensors": tensor_specs,
            "files": files,
        }
        write_canonical_json(staging / _MANIFEST_FILE, manifest)
        write_canonical_json(
            staging / _COMPLETE_FILE,
            {
                "schema_version": _COMPLETE_SCHEMA_VERSION,
                "manifest_sha256": sha256_file(staging / _MANIFEST_FILE),
            },
        )
        _fsync_tree(staging)
        inspect_pdd_export(staging)
        staging.rename(output)
        published = True
        _fsync_directory(output.parent)
        return output
    except BaseException:
        target = output if published else staging
        if target.exists() and not target.is_symlink():
            shutil.rmtree(target)
        raise


def _require_exact_mapping(value: Any, keys: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{name} keys mismatch: expected={sorted(keys)}, actual={actual}.")
    return value


def inspect_pdd_export(export_dir: str | Path) -> PDDExportDescriptor:
    """Authenticate a PDD export without loading its tensor payloads."""
    unresolved_root = Path(export_dir)
    if unresolved_root.is_symlink():
        raise RuntimeError(f"PDD export cannot be a symlink: {unresolved_root}.")
    root = unresolved_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"PDD export is not a regular directory: {root}.")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"PDD export contains a symlink: {path}.")
        if path.is_dir():
            raise RuntimeError(f"PDD export must be flat, found directory: {path}.")

    complete = _require_exact_mapping(
        load_canonical_json(root / _COMPLETE_FILE),
        {"schema_version", "manifest_sha256"},
        name="PDD COMPLETE",
    )
    if complete["schema_version"] != _COMPLETE_SCHEMA_VERSION:
        raise ValueError("PDD COMPLETE schema version is unsupported.")
    if require_sha256(
        complete["manifest_sha256"], name="PDD COMPLETE manifest SHA-256"
    ) != sha256_file(root / _MANIFEST_FILE):
        raise RuntimeError("PDD COMPLETE does not match the export manifest.")
    manifest = _require_exact_mapping(
        load_canonical_json(root / _MANIFEST_FILE),
        {
            "schema_version",
            "format",
            "identity",
            "source_checkpoint",
            "modelopt_source",
            "max_shard_bytes",
            "total_tensor_bytes",
            "tensors",
            "files",
        },
        name="PDD export manifest",
    )
    if manifest["schema_version"] != _EXPORT_SCHEMA_VERSION or manifest["format"] != _EXPORT_FORMAT:
        raise ValueError("PDD export manifest schema or format is unsupported.")
    _validate_modelopt_source(manifest["modelopt_source"])
    if type(manifest["max_shard_bytes"]) is not int or manifest["max_shard_bytes"] <= 0:
        raise ValueError("PDD export max_shard_bytes is invalid.")
    if type(manifest["total_tensor_bytes"]) is not int or manifest["total_tensor_bytes"] <= 0:
        raise ValueError("PDD export total_tensor_bytes is invalid.")
    source = _require_exact_mapping(
        manifest["source_checkpoint"],
        {"name", "manifest_sha256", "completed_steps"},
        name="source_checkpoint",
    )
    if (
        not isinstance(source["name"], str)
        or not source["name"]
        or Path(source["name"]).name != source["name"]
    ):
        raise ValueError("source_checkpoint.name must be a basename.")
    require_sha256(source["manifest_sha256"], name="source checkpoint manifest SHA-256")
    if type(source["completed_steps"]) is not int or source["completed_steps"] < 1:
        raise ValueError("source_checkpoint.completed_steps is invalid.")

    files = manifest["files"]
    if not isinstance(files, Mapping) or not files:
        raise ValueError("PDD export file inventory must be a non-empty mapping.")
    expected_names = set(files) | {_MANIFEST_FILE, _COMPLETE_FILE}
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(
            f"PDD export file inventory mismatch: expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}."
        )
    for name, record in files.items():
        if Path(name).name != name:
            raise ValueError(f"PDD export file name must be a basename: {name!r}.")
        record = _require_exact_mapping(record, {"sha256", "size"}, name=f"files[{name!r}]")
        path = resolve_relative_artifact(root, name)
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError(f"PDD export file size is invalid for {name!r}.")
        if path.stat().st_size != record["size"]:
            raise RuntimeError(f"PDD export file size mismatch for {name!r}.")
        if sha256_file(path) != require_sha256(record["sha256"], name=f"files[{name!r}].sha256"):
            raise RuntimeError(f"PDD export file SHA-256 mismatch for {name!r}.")
        if name.endswith(".safetensors") and record["size"] > manifest["max_shard_bytes"]:
            raise RuntimeError(f"PDD export shard exceeds max_shard_bytes: {name!r}.")
    mandatory = {_CONFIG_FILE, _METADATA_FILE, _INDEX_FILE}
    if not mandatory.issubset(files):
        raise RuntimeError(
            f"PDD export is missing mandatory files: {sorted(mandatory - set(files))}."
        )

    metadata_data = load_canonical_json(root / _METADATA_FILE)
    metadata = PDDMetadata.from_dict(metadata_data)
    if metadata.layer_spec != QWEN_IMAGE_PDD_LAYER_SPEC:
        raise ValueError("PDD export carries a non-Qwen layer specification.")
    identity = manifest["identity"]
    _validate_identity(identity, metadata)
    transformer_config = load_canonical_json(root / _CONFIG_FILE)
    if not isinstance(transformer_config, Mapping):
        raise ValueError("PDD transformer config must contain an object.")

    tensors = manifest["tensors"]
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("PDD export tensor inventory must be a non-empty mapping.")
    index = _require_exact_mapping(
        load_canonical_json(root / _INDEX_FILE),
        {"metadata", "weight_map"},
        name="safetensors index",
    )
    index_metadata = _require_exact_mapping(
        index["metadata"], {"total_size"}, name="index metadata"
    )
    if index_metadata["total_size"] != manifest["total_tensor_bytes"]:
        raise RuntimeError("PDD safetensors index total size does not match the manifest.")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, Mapping) or set(weight_map) != set(tensors):
        raise RuntimeError("PDD safetensors index keys do not match the tensor inventory.")
    shard_names = {name for name in files if name.endswith(".safetensors")}
    shard_pattern = re.compile(r"diffusion_pytorch_model-(\d{5})-of-(\d{5})\.safetensors")
    shard_numbers = []
    for name in shard_names:
        match = shard_pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"PDD export has a noncanonical shard name: {name!r}.")
        shard_numbers.append((int(match.group(1)), int(match.group(2))))
    if not shard_numbers:
        raise RuntimeError("PDD export has no safetensors shards.")
    shard_count = len(shard_numbers)
    if sorted(shard_numbers) != [(index, shard_count) for index in range(1, shard_count + 1)]:
        raise RuntimeError("PDD export safetensors shard numbering is inconsistent.")
    if set(weight_map.values()) != shard_names:
        raise RuntimeError("PDD safetensors index shard inventory does not match export files.")

    total = 0
    for key, spec in tensors.items():
        if not isinstance(key, str) or not key:
            raise ValueError("PDD tensor inventory keys must be non-empty strings.")
        spec = _require_exact_mapping(
            spec,
            {"dtype", "nbytes", "shape", "shard"},
            name=f"tensors[{key!r}]",
        )
        if (
            not isinstance(spec["dtype"], str)
            or type(spec["nbytes"]) is not int
            or spec["nbytes"] <= 0
            or not isinstance(spec["shape"], list)
            or any(type(size) is not int or size < 0 for size in spec["shape"])
            or spec["shard"] != weight_map[key]
        ):
            raise ValueError(f"PDD tensor specification is malformed for {key!r}.")
        total += spec["nbytes"]
    if total != manifest["total_tensor_bytes"]:
        raise RuntimeError("PDD tensor byte inventory does not match total_tensor_bytes.")
    return PDDExportDescriptor(root, manifest, metadata, transformer_config)


def pdd_config_from_metadata(
    metadata: PDDMetadata,
    *,
    blocks: Sequence[int] | None = None,
    schedule: str | None = None,
    guidance_scale: float | None = None,
) -> PDDConfig:
    """Build a fresh validated inference config from authenticated metadata."""
    if (blocks is None) == (schedule is None):
        raise ValueError("Specify exactly one of blocks or schedule.")
    if schedule is not None:
        try:
            resolved = PDD_INFERENCE_SCHEDULES[schedule]
        except KeyError as error:
            raise ValueError(
                f"Unknown PDD schedule {schedule!r}; expected {sorted(PDD_INFERENCE_SCHEDULES)}."
            ) from error
    else:
        if isinstance(blocks, str | bytes) or not isinstance(blocks, Sequence):
            raise TypeError("blocks must be a sequence of integers.")
        resolved = tuple(blocks)
    return PDDConfig(
        grid_size=metadata.grid_size,
        grid_max_t=metadata.grid_max_t,
        flow_shift=metadata.flow_shift,
        block_size_min=metadata.block_size_min,
        block_size_max=metadata.block_size_max,
        inference_blocks=list(resolved),
        student_sample_steps=len(resolved),
        teacher_integrator=metadata.teacher_integrator,
        guidance_scale=guidance_scale,
        num_train_timesteps=None,
    )


def _load_shard(
    descriptor: PDDExportDescriptor,
    shard_name: str,
) -> dict[str, torch.Tensor]:
    expected = {
        key: spec
        for key, spec in descriptor.manifest["tensors"].items()
        if spec["shard"] == shard_name
    }
    path = descriptor.root / shard_name
    loaded: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
        if len(keys) != len(set(keys)) or set(keys) != set(expected):
            raise RuntimeError(f"PDD safetensors keys do not match the index for {shard_name!r}.")
        for key in keys:
            tensor = stream.get_tensor(key)
            spec = expected[key]
            if list(tensor.shape) != spec["shape"]:
                raise RuntimeError(f"PDD tensor shape mismatch for {key!r}.")
            if str(tensor.dtype).removeprefix("torch.") != spec["dtype"]:
                raise RuntimeError(f"PDD tensor dtype mismatch for {key!r}.")
            if _tensor_nbytes(tensor) != spec["nbytes"]:
                raise RuntimeError(f"PDD tensor byte-size mismatch for {key!r}.")
            if tensor.dtype.is_floating_point and not torch.isfinite(tensor).all().item():
                raise FloatingPointError(f"PDD tensor {key!r} is non-finite.")
            loaded[key] = tensor
    return loaded


def load_pdd_export_into_model(
    export_dir: str | Path,
    model: torch.nn.Module,
) -> PDDExportDescriptor:
    """Strictly stream authenticated safetensors shards into a converted CPU model."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an nn.Module.")
    descriptor = inspect_pdd_export(export_dir)
    expected_state = model.state_dict()
    specs = descriptor.manifest["tensors"]
    if set(expected_state) != set(specs):
        missing = sorted(set(expected_state) - set(specs))
        extra = sorted(set(specs) - set(expected_state))
        raise RuntimeError(f"PDD model/export keys mismatch: missing={missing}, extra={extra}.")
    for key, expected in expected_state.items():
        spec = specs[key]
        if list(expected.shape) != spec["shape"]:
            raise RuntimeError(f"PDD model/export shape mismatch for {key!r}.")
        if str(expected.dtype).removeprefix("torch.") != spec["dtype"]:
            raise RuntimeError(f"PDD model/export dtype mismatch for {key!r}.")

    shard_names = sorted({spec["shard"] for spec in specs.values()})
    loaded_keys: set[str] = set()
    for shard_name in shard_names:
        shard = _load_shard(descriptor, shard_name)
        if loaded_keys.intersection(shard):
            raise RuntimeError("PDD tensor appears in more than one safetensors shard.")
        incompatible = model.load_state_dict(shard, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"PDD shard has unexpected keys: {incompatible.unexpected_keys}.")
        loaded_keys.update(shard)
    if loaded_keys != set(expected_state):
        raise RuntimeError("PDD safe load did not assign every expected tensor.")
    if any(parameter.is_meta for parameter in model.parameters()) or any(
        buffer.is_meta for buffer in model.buffers()
    ):
        raise RuntimeError("PDD safe load left meta tensors in the reconstructed model.")
    model.eval().requires_grad_(False)
    return descriptor
