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

"""Native AutoModel DCP weights with topology-owned ModelOpt quantizer state."""

import copy
import json
import os
import uuid
from contextlib import ExitStack
from pathlib import Path

import torch

import modelopt.torch.opt as mto
from modelopt.torch.opt.conversion import ModeloptStateManager
from modelopt.torch.opt.dynamic import DynamicModule
from modelopt.torch.quantization.conversion import restore_quantizer_state
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils import get_unwrapped_name

from ...local_checkpoint import _CheckpointRanks, _file_hash, _sync_directory, _write_json
from .parallel import _model_group

__all__ = []


def _topology(context):
    return {
        "axes": context.device_mesh.mesh_dim_names,
        "ranks": context.device_mesh.mesh.tolist(),
        "expert_ranks": context.moe_mesh.mesh.tolist() if context.moe_mesh is not None else None,
    }


def _schema(model):
    return {
        get_unwrapped_name(name, model): [list(value.shape), str(value.dtype)]
        for name, value in model.named_parameters()
    }


def _aliases(model):
    owners, aliases = {}, {}
    for name, value in model.named_buffers(remove_duplicate=False):
        name = get_unwrapped_name(name, model)
        previous = owners.setdefault(id(value), name)
        if previous != name:
            aliases[name] = previous
    return aliases


def _checkpointer(context):
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig

    def rank(axis):
        try:
            return context.device_mesh[axis].get_local_rank()
        except (KeyError, RuntimeError):
            return 0

    return CheckpointingConfig(
        model_save_format="torch_save",
        save_consolidated=False,
        is_async=False,
        load_metadata_with_gloo=False,
    ).build(
        dp_rank=rank("dp"),
        tp_rank=rank("tp"),
        pp_rank=rank("pp"),
        moe_mesh=context.moe_mesh,
        process_group=_model_group(context),
    )


class _NativeStudentState:
    def __init__(self, model, rank, context):
        from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState

        self.model = model
        self.native = ModelState(model, has_expert_parallelism=context.ep_size > 1)
        self.rank_key = f"quantizers_rank_{rank}"
        self.quantizers = {
            get_unwrapped_name(name, model)
            for name, module in model.named_modules()
            if isinstance(module, TensorQuantizer)
        }

    def state_dict(self):
        native = self.native.state_dict()
        # DCP may deduplicate ordinary tensors as replicas. Expert/TP-owned
        # quantizer buffers can differ by rank even when their FQNs are equal.
        quantized = {
            name: value
            for name, value in native.items()
            if name.rsplit(".", 1)[0] in self.quantizers
        }
        return {
            "weights": {name: value for name, value in native.items() if name not in quantized},
            self.rank_key: quantized,
        }

    def load_state_dict(self, state):
        # DCP verifies parameter FQNs through getattr before loading. Use the
        # same raw-attribute contract as ModelOpt's native state-dict hooks so
        # this metadata walk never quantizes partially sharded expert weights.
        with ExitStack() as stack:
            for module in self.model.modules():
                if isinstance(module, DynamicModule):
                    stack.enter_context(module.reset_dynamic_attributes())
            self.native.load_state_dict({**state["weights"], **state[self.rank_key]})


def _restore_metadata(model, state):
    saved = state["modelopt"]
    if not ModeloptStateManager.is_converted(model):
        skeleton = copy.deepcopy(saved)
        # As in native sharded ModelOpt restore, create the graph first without
        # reading/calibrating sharded weights. DCP restores exact buffers below.
        for _, entry in skeleton["modelopt_state_dict"]:
            entry["metadata"].pop("quantizer_state", None)
        model = mto.restore_from_modelopt_state(model, skeleton)
    manager = ModeloptStateManager(model)
    modes = list(manager.modes_with_states())
    if [str(mode) for mode, _, _ in modes] != [name for name, _ in saved["modelopt_state_dict"]]:
        raise ValueError("Native checkpoint ModelOpt modes differ from the target model.")
    device = next(model.parameters()).device
    for (_, config, _), (_, entry) in zip(modes, saved["modelopt_state_dict"], strict=True):
        if "quantizer_state" in entry["metadata"]:
            restore_quantizer_state(model, config, entry["metadata"], post_restore=False)
    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            module.to(device=device)
            module._quantizer_cache = None
    modules = {get_unwrapped_name(name, model): module for name, module in model.named_modules()}
    for name, owner in state["aliases"].items():
        parent, _, leaf = name.rpartition(".")
        owner_parent, _, owner_leaf = owner.rpartition(".")
        modules[parent]._buffers[leaf] = modules[owner_parent]._buffers[owner_leaf]
    return model


def publish(student, context, config, metrics):
    """Atomically publish native weights and rank-owned fields on the realized mesh."""
    ranks = _CheckpointRanks(_model_group(context), distributed=True)
    destination = Path(config.output_dir).resolve()

    def begin():
        if destination.exists() and (destination.is_symlink() or any(destination.iterdir())):
            raise ValueError("Repair output_dir must be a new or empty directory.")
        pending = destination.parent / ("." + destination.name + "-" + uuid.uuid4().hex)
        pending.mkdir(parents=True)
        return str(pending)

    pending = Path(ranks._coordinate(begin))
    metadata = pending / f"rank-{ranks.rank}-modelopt.pt"
    error = None
    try:
        state = {
            "modelopt": mto.modelopt_state(student),
            "aliases": _aliases(student),
            "schema": _schema(student),
        }
        with metadata.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _write_json(pending / f"rank-{ranks.rank}-metrics.json", metrics)
    except Exception as exc:
        error = f"rank {ranks.rank}: {type(exc).__name__}: {exc}"
    errors = [value for value in ranks._gather(error) if value]
    if errors:
        raise ValueError("Native checkpoint metadata failed: " + "; ".join(errors))
    checkpointer = _checkpointer(context)
    try:
        checkpointer.save_distributed_state(
            _NativeStudentState(student, ranks.rank, context), "model", str(pending)
        )
    finally:
        checkpointer.close()
    receipts = ranks._gather({metadata.name: _file_hash(metadata)})

    def commit():
        _write_json(pending / "config.json", config.model_dump())
        manifest = {
            "format": "modelopt_automodel_dcp_v1",
            "topology": _topology(context),
            "metadata": {name: digest for receipt in receipts for name, digest in receipt.items()},
            "dcp_metadata": _file_hash(pending / "model" / ".metadata"),
        }
        _write_json(pending / "COMMITTED.json", manifest)
        _sync_directory(pending)
        os.replace(pending, destination)
        _sync_directory(destination.parent)
        return str(destination)

    ranks._coordinate(commit)
    return {
        "checkpoint": str(destination),
        "metrics": str(destination / f"rank-{ranks.rank}-metrics.json"),
    }


def restore(model, context, directory):
    """Restore native fields and weights into a matching distributed model skeleton."""
    ranks = _CheckpointRanks(_model_group(context), distributed=True)
    root = Path(directory)
    state, error = None, None
    try:
        manifest = json.loads((root / "COMMITTED.json").read_text())
        if manifest["format"] != "modelopt_automodel_dcp_v1" or json.dumps(
            manifest["topology"], sort_keys=True
        ) != json.dumps(_topology(context), sort_keys=True):
            raise ValueError(
                "Native checkpoint topology differs; restore using the saved parallel layout."
            )
        metadata = root / f"rank-{ranks.rank}-modelopt.pt"
        if (
            _file_hash(metadata) != manifest["metadata"][metadata.name]
            or _file_hash(root / "model" / ".metadata") != manifest["dcp_metadata"]
        ):
            raise ValueError("Native checkpoint metadata failed integrity validation.")
        state = torch.load(metadata, map_location="cpu", weights_only=True)
        if state["schema"] != _schema(model):
            raise ValueError("Native checkpoint parameter shapes or compute dtype differ.")
    except Exception as exc:
        error = f"rank {ranks.rank}: {type(exc).__name__}: {exc}"
    errors = [value for value in ranks._gather(error) if value]
    if errors:
        raise ValueError("Native checkpoint validation failed: " + "; ".join(errors))
    model = _restore_metadata(model, state)
    checkpointer = _checkpointer(context)
    try:
        checkpointer.load_distributed_state(
            _NativeStudentState(model, ranks.rank, context), "model", str(root)
        )
    finally:
        checkpointer.close()
    from modelopt.torch.quantization.utils.native_weight import _VALIDATED

    for module in model.modules():
        if isinstance(module, TensorQuantizer):
            _VALIDATED.pop(module, None)
    return model
