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


"""Atomic topology-native Torch DCP publication into a matching PTQ skeleton.

This is not an unsharded HF deployment export. All ranks must use a shared
filesystem and the same caller-provided topology, converted graph and parameter
layouts at restore. Native weights remain DTensors; ordinary and quantizer
fields carry rank namespaces so DCP cannot deduplicate distinct TP/EP payloads.
"""

import copy
import json
import os
import shutil
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.tensor import DTensor

import modelopt.torch.opt as mto
from modelopt.torch.opt.conversion import ModeloptStateManager
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.quantization.conversion import maybe_promote_nvfp4_static_quantizer
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils.native_weight import _VALIDATED
from modelopt.torch.utils import get_unwrapped_name

from ...local_checkpoint import _CheckpointRanks, _file_hash, _sync_directory, _write_json
from ...local_parallel import _model_group

__all__ = ["publish", "restore"]

_FORMAT = "modelopt_transformers_dcp_v1"
_PENDING_PREFIX = ".pending-bypass-"
_OPTIONS = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)


def _canonical(value):
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _same(left, right):
    return json.dumps(_canonical(left), sort_keys=True) == json.dumps(
        _canonical(right), sort_keys=True
    )


def _topology(context):
    return {"axes": context.device_mesh.mesh_dim_names, "ranks": context.device_mesh.mesh.tolist()}


def _tensor_schema(value):
    if not isinstance(value, torch.Tensor):
        return {"object": type(value).__module__ + "." + type(value).__qualname__}
    schema = {"shape": tuple(value.shape), "dtype": value.dtype}
    if isinstance(value, DTensor):
        schema.update(
            mesh=value.device_mesh.mesh.tolist(),
            axes=value.device_mesh.mesh_dim_names,
            placements=[str(item) for item in value.placements],
            local_shape=tuple(value.to_local().shape),
        )
    if hasattr(value, "get_qtensor"):
        schema["packed"] = _canonical(value.get_state()["metadata"])
    return schema


@contextmanager
def _raw_attributes(model):
    with ExitStack() as stack:
        for module in model.modules():
            if isinstance(module, DynamicModule):
                stack.enter_context(module.reset_dynamic_attributes())
        yield


def _native_state(model):
    with _raw_attributes(model):
        return get_model_state_dict(model, options=_OPTIONS)


def _quantizers(model):
    return {
        get_unwrapped_name(name, model): module
        for name, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, TensorQuantizer)
    }


def _quantized_field(name, names):
    return name.rsplit(".", 1)[0] in names


def _aliases(model, keys):
    result = {}
    for kind, iterator in (
        ("parameters", model.named_parameters),
        ("buffers", model.named_buffers),
    ):
        owners, aliases = {}, {}
        for name, value in iterator(remove_duplicate=False):
            name = get_unwrapped_name(name, model)
            if name not in keys:
                continue
            owner = owners.setdefault(id(value), name)
            if owner != name:
                aliases[name] = owner
        result[kind] = aliases
    owners, aliases = {}, {}
    for name, module in _quantizers(model).items():
        owner = owners.setdefault(id(module), name)
        if owner != name:
            aliases[name] = owner
    result["modules"] = aliases
    return result


def _payload(flat, names, rank):
    weights, owned = {}, {}
    for key, value in flat.items():
        if isinstance(value, DTensor) and not _quantized_field(key, names):
            weights[key] = value
        else:
            owned[key] = value.to_local() if isinstance(value, DTensor) else value
    return {"weights": weights, f"rank_{rank}": owned}


def _modes(state):
    return [(name, entry["config"]) for name, entry in state["modelopt_state_dict"]]


def _capture(model):
    if not ModeloptStateManager.is_converted(model):
        raise ValueError("HF native DCP requires an already converted ModelOpt PTQ skeleton.")
    flat = _native_state(model)
    aliases = _aliases(model, flat)
    state = {
        "modelopt": mto.modelopt_state(model),
        "aliases": aliases,
        "quantizers": sorted(_quantizers(model)),
        # Mode history contains layouts from earlier conversion stages. Capture
        # live properties and fields after packed staging/partitioning separately.
        "quantizer_state": {
            name: module.get_modelopt_state()
            for name, module in _quantizers(model).items()
            if name not in aliases["modules"]
        },
        "schema": {name: _tensor_schema(value) for name, value in flat.items()},
        "parameters": {
            get_unwrapped_name(name, model): _tensor_schema(value)
            for name, value in model.named_parameters()
        },
        "objects": {
            name: value for name, value in flat.items() if not isinstance(value, torch.Tensor)
        },
    }
    return flat, state


def _errors(ranks, error, label):
    failures = [value for value in ranks._gather(error) if value]
    if failures:
        raise ValueError(label + ": " + "; ".join(failures))


def _report_paths(size):
    return [f"reports/rank-{rank}-metrics.json" for rank in range(size)]


def _receipt_files(pending, ranks, reports):
    paths = sorted(
        path
        for path in pending.rglob("*")
        if path.is_file() and str(path.relative_to(pending)) not in reports
    )
    return {
        str(path.relative_to(pending)): {"sha256": _file_hash(path), "bytes": path.stat().st_size}
        for index, path in enumerate(paths)
        if index % ranks.size == ranks.rank
    }


def publish(student, context, config, metrics):
    """Commit every rank's physical weights, native fields and metadata atomically.

    ``output_dir`` must be new or empty. DCP uses sharded, device-resident state;
    no full-state gathering, consolidated CPU copy, or deployment export occurs.
    The returned metrics path is a mutable diagnostic report; the committed
    manifest lists its exact name separately from immutable, hashed model state.
    """
    ranks = _CheckpointRanks(_model_group(context), distributed=True)
    requested = Path(config.output_dir).absolute()
    reports = _report_paths(ranks.size)

    def begin():
        if requested.name.startswith(_PENDING_PREFIX):
            raise ValueError("Repair output_dir uses the reserved pending-checkpoint prefix.")
        if requested.is_symlink() or (requested.exists() and any(requested.iterdir())):
            raise ValueError("Repair output_dir must be a new or empty directory.")
        destination = requested.resolve()
        pending = destination.parent / (_PENDING_PREFIX + uuid.uuid4().hex)
        pending.mkdir(parents=True)
        (pending / "reports").mkdir()
        return str(destination), str(pending)

    destination_name, pending_name = ranks._coordinate(begin)
    destination, pending = Path(destination_name), Path(pending_name)
    try:
        flat, state, error = None, None, None
        try:
            flat, state = _capture(student)
            with (pending / f"rank-{ranks.rank}-modelopt.pt").open("wb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            _write_json(pending / reports[ranks.rank], metrics)
        except Exception as exc:
            error = f"rank {ranks.rank}: {type(exc).__name__}: {exc}"
        _errors(ranks, error, "HF checkpoint metadata failed")
        assert flat is not None and state is not None
        payload = _payload(flat, state["quantizers"], ranks.rank)
        error = None
        try:
            dcp.save(
                payload,
                storage_writer=dcp.FileSystemWriter(pending / "weights", sync_files=True),
                process_group=ranks.group,
            )
        except Exception as exc:
            error = f"rank {ranks.rank}: {type(exc).__name__}: {exc}"
        _errors(ranks, error, "HF checkpoint DCP save failed")
        error, receipt = None, None
        try:
            receipt = _receipt_files(pending, ranks, reports)
        except Exception as exc:
            error = f"rank {ranks.rank}: {type(exc).__name__}: {exc}"
        _errors(ranks, error, "HF checkpoint receipts failed")
        receipts = ranks._gather(receipt)

        def commit():
            files = {name: value for receipt in receipts for name, value in receipt.items()}
            actual = {
                str(path.relative_to(pending)) for path in pending.rglob("*") if path.is_file()
            }
            if actual != set(files) | set(reports) or any(
                (pending / name).stat().st_size != value["bytes"] for name, value in files.items()
            ):
                raise ValueError("HF checkpoint all-rank receipts disagree.")
            _write_json(pending / "config.json", config.model_dump())
            files["config.json"] = {
                "sha256": _file_hash(pending / "config.json"),
                "bytes": (pending / "config.json").stat().st_size,
            }
            manifest = {
                "format": _FORMAT,
                "world_size": ranks.size,
                "generation": pending.name.removeprefix(_PENDING_PREFIX),
                "topology": _topology(context),
                "files": files,
                "reports": reports,
            }
            _write_json(pending / "COMMITTED.json", manifest)
            _sync_directory(pending / "weights")
            _sync_directory(pending / "reports")
            _sync_directory(pending)
            os.replace(pending, destination)
            _sync_directory(destination.parent)
            return str(destination)

        ranks._coordinate(commit)
    except BaseException:
        ranks._coordinate(lambda: shutil.rmtree(pending) if pending.exists() else None)
        raise
    return {
        "checkpoint": str(destination),
        "metrics": str(destination / reports[ranks.rank]),
    }


def _validated_manifest(root, context, ranks):
    def read():
        if (
            root.is_symlink()
            or root.resolve().name.startswith(_PENDING_PREFIX)
            or not (root / "COMMITTED.json").is_file()
        ):
            raise ValueError("HF checkpoint is incomplete or unpublished.")
        manifest = json.loads((root / "COMMITTED.json").read_text())
        if manifest.get("format") != _FORMAT or not manifest.get("generation"):
            raise ValueError("HF checkpoint is not a published destination.")
        if manifest["world_size"] != ranks.size or not _same(
            manifest["topology"], _topology(context)
        ):
            raise ValueError("HF checkpoint topology differs; use the saved parallel layout.")
        reports = _report_paths(ranks.size)
        if manifest.get("reports") != reports or set(reports) & set(manifest["files"]):
            raise ValueError("HF checkpoint mutable diagnostic report paths differ.")
        if any((root / name).is_symlink() for name in reports):
            raise ValueError("HF checkpoint diagnostic reports must be ordinary files.")
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        if actual != set(manifest["files"]) | set(reports) | {"COMMITTED.json"}:
            raise ValueError("HF checkpoint integrity validation found missing or extra files.")
        return manifest

    manifest = ranks._coordinate(read)
    error = None
    try:
        for index, (name, receipt) in enumerate(sorted(manifest["files"].items())):
            if index % ranks.size != ranks.rank:
                continue
            relative = Path(name)
            path = root / relative
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or path.is_symlink()
                or path.stat().st_size != receipt["bytes"]
                or _file_hash(path) != receipt["sha256"]
            ):
                raise ValueError(f"HF checkpoint integrity validation failed for {name}.")
    except Exception as exc:
        error = f"rank {ranks.rank}: {type(exc).__name__}: {exc}"
    _errors(ranks, error, "HF checkpoint integrity validation failed")
    return manifest


def _prepare_restore(model, state):
    current, captured = _capture(model)
    if not _same(state["parameters"], captured["parameters"]):
        raise ValueError("HF checkpoint parameter shapes, packed format or DTensor layouts differ.")
    if not _same(_modes(state["modelopt"]), _modes(captured["modelopt"])):
        raise ValueError("HF checkpoint ModelOpt modes/configurations differ.")
    if state["quantizers"] != captured["quantizers"]:
        raise ValueError("HF checkpoint quantizer graph differs from the converted skeleton.")
    if state["aliases"]["parameters"] != captured["aliases"]["parameters"]:
        raise ValueError("HF checkpoint tied-parameter ownership differs.")
    if state["aliases"]["modules"] != captured["aliases"]["modules"]:
        raise ValueError("HF checkpoint tied-quantizer-module ownership differs.")
    for name, schema in captured["schema"].items():
        if not _quantized_field(name, state["quantizers"]) and not _same(
            schema, state["schema"].get(name)
        ):
            raise ValueError(f"HF checkpoint non-quantizer state layout differs: {name}")
    device = next(model.parameters()).device
    target = {}
    for name, schema in state["schema"].items():
        if "object" in schema:
            target[name] = copy.deepcopy(state["objects"][name])
        elif name in current and _same(schema, captured["schema"].get(name)):
            target[name] = current[name]
        elif _quantized_field(name, state["quantizers"]) and "mesh" not in schema:
            target[name] = torch.empty(schema["shape"], dtype=schema["dtype"], device=device)
        else:
            raise ValueError(f"HF checkpoint requires a matching physical state skeleton: {name}")
    modules = {
        get_unwrapped_name(name, model): module
        for name, module in model.named_modules(remove_duplicate=False)
    }
    for alias, owner in state["aliases"]["buffers"].items():
        if (
            alias not in target
            or owner not in target
            or alias.rpartition(".")[0] not in modules
            or owner.rpartition(".")[0] not in modules
            or not _same(state["schema"][alias], state["schema"][owner])
        ):
            raise ValueError("HF checkpoint buffer-alias graph is inconsistent.")
    canonical_names = set(state["quantizers"]) - set(state["aliases"]["modules"])
    if set(state["quantizer_state"]) != canonical_names:
        raise ValueError("HF checkpoint current quantizer metadata graph differs.")
    layouts = {
        name: values.get("_pytorch_state_metadata", {})
        for name, values in state["quantizer_state"].items()
    }
    for name in state["quantizers"]:
        canonical_name = state["aliases"]["modules"].get(name, name)
        if canonical_name not in layouts:
            raise ValueError(f"HF checkpoint quantizer metadata is absent: {name}")
        for kind in ("params", "buffers"):
            for leaf, layout in layouts[canonical_name].get(kind, {}).items():
                key = f"{name}.{leaf}" if name else leaf
                schema = state["schema"].get(key, {})
                if not _same(layout, {field: schema.get(field) for field in ("shape", "dtype")}):
                    raise ValueError(f"HF checkpoint quantizer layout metadata differs: {key}")
    return target


def _apply_metadata(model, state, flat):
    quantizers = _quantizers(model)
    for name, saved in state["quantizer_state"].items():
        quantizer = quantizers[name]
        maybe_promote_nvfp4_static_quantizer(quantizer, saved)
        quantizer.set_from_modelopt_state(saved, properties_only=True)
        buffers = saved.get("_pytorch_state_metadata", {}).get("buffers", {})
        for leaf in set(quantizer._buffers) - set(buffers) - quantizer._non_persistent_buffers_set:
            del quantizer._buffers[leaf]
        for leaf in buffers:
            quantizer._buffers[leaf] = flat[f"{name}.{leaf}" if name else leaf]
        quantizer._quantizer_cache = None
        _VALIDATED.pop(quantizer, None)
    ModeloptStateManager(model).state_dict()[:] = copy.deepcopy(
        state["modelopt"]["modelopt_state_dict"]
    )


def _restore_aliases(model, aliases):
    modules = {
        get_unwrapped_name(name, model): module
        for name, module in model.named_modules(remove_duplicate=False)
    }
    for alias, owner in aliases["buffers"].items():
        parent, _, leaf = alias.rpartition(".")
        owner_parent, _, owner_leaf = owner.rpartition(".")
        modules[parent]._buffers[leaf] = modules[owner_parent]._buffers[owner_leaf]


def restore(model, context, directory):
    """Restore native DCP into the same caller-owned converted graph and topology.

    Integrity, modes, parameter geometry and every rank's schema are collectively
    checked before loading live tensors. This does not construct, re-quantize,
    repartition, consolidate, or change the topology of a model. As with native
    DCP, runtime I/O/device failures during the validated load are not rolled back.
    """
    ranks = _CheckpointRanks(_model_group(context), distributed=True)
    root = Path(directory).absolute()
    _validated_manifest(root, context, ranks)
    flat, state, error = None, None, None
    try:
        state = torch.load(
            root / f"rank-{ranks.rank}-modelopt.pt", map_location="cpu", weights_only=True
        )
        flat = _prepare_restore(model, state)
        payload = _payload(flat, state["quantizers"], ranks.rank)
        metadata = dcp.FileSystemReader(root / "weights").read_metadata()
        for key, value in flatten_state_dict(payload)[0].items():
            if key not in metadata.state_dict_metadata:
                raise ValueError(f"HF checkpoint DCP metadata is missing {key}.")
            saved = metadata.state_dict_metadata[key]
            if isinstance(value, torch.Tensor) and (
                tuple(saved.size) != tuple(value.shape) or saved.properties.dtype != value.dtype
            ):
                raise ValueError(f"HF checkpoint DCP tensor metadata differs for {key}.")
    except Exception as exc:
        error = f"rank {ranks.rank}: {type(exc).__name__}: {exc}"
    _errors(ranks, error, "HF checkpoint validation failed")
    assert flat is not None and state is not None
    key_sets = ranks._gather(sorted(flatten_state_dict(payload)[0]))
    expected_keys = {key for keys in key_sets for key in keys}
    if set(metadata.state_dict_metadata) != expected_keys:
        raise ValueError("HF checkpoint DCP metadata differs from the all-rank state union.")
    dcp.load(
        payload, storage_reader=dcp.FileSystemReader(root / "weights"), process_group=ranks.group
    )
    flat.update(payload["weights"])
    for name, value in payload[f"rank_{ranks.rank}"].items():
        if isinstance(flat[name], DTensor):
            with torch.no_grad():
                flat[name].to_local().copy_(value)
        else:
            flat[name] = value
    _apply_metadata(model, state, flat)
    with _raw_attributes(model):
        set_model_state_dict(model, flat, options=_OPTIONS)
    _restore_aliases(model, state["aliases"])
    return model
