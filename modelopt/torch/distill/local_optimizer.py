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

"""Coalesced unit optimizers with grouping-independent checkpoint field order."""

import torch
from torch.distributed.tensor import DTensor

__all__ = ["load_optimizer_state_dict", "make_optimizer", "optimizer_state_dict"]

# Bound foreach scratch storage rather than batching an entire MoE's fields.
_MAX_GROUP_BYTES = 64 * 1024 * 1024


def make_optimizer(groups, config, partition=None):
    """Group compatible native fields without combining independent repair units."""
    fields, combined, ordered = [], {}, []
    for group in groups:
        for parameter in group.params:
            fields.append(parameter)
            key = (
                group.lr,
                group.fields,
                type(parameter),
                parameter.device,
                parameter.dtype,
                parameter.layout,
                partition(parameter) if partition is not None else None,
                # DTensor foreach support depends on its mesh and placements.
                # Keep its existing execution until that combination is validated.
                id(parameter) if isinstance(parameter, DTensor) else None,
            )
            size = parameter.numel() * parameter.element_size()
            entry, used = combined.get(key, (None, 0))
            if entry is None or used + size > _MAX_GROUP_BYTES:
                entry = {"params": [], "lr": group.lr, "initial_lr": group.lr}
                ordered.append(entry)
                used = 0
            entry["params"].append(parameter)
            combined[key] = (entry, used + size)
    optimizer = torch.optim.Adagrad if config.optimizer == "adagrad" else torch.optim.AdamW
    return optimizer(ordered, weight_decay=0), fields


def _field_entries(optimizer):
    state = optimizer.state_dict()
    entries = {}
    for actual, saved in zip(optimizer.param_groups, state["param_groups"], strict=True):
        options = {key: value for key, value in saved.items() if key != "params"}
        for parameter, index in zip(actual["params"], saved["params"], strict=True):
            entries[id(parameter)] = (index, options)
    return state, entries


def optimizer_state_dict(optimizer, fields):
    """Keep the original per-field checkpoint layout, independently of runtime grouping."""
    state, entries = _field_entries(optimizer)
    values, groups = {}, []
    for index, parameter in enumerate(fields):
        previous, options = entries[id(parameter)]
        if previous in state["state"]:
            values[index] = state["state"][previous]
        groups.append({**options, "params": [index]})
    return {"state": values, "param_groups": groups}


def load_optimizer_state_dict(optimizer, fields, state):
    """Restore old per-field checkpoints into compatible coalesced runtime groups."""
    indices, options = [], []
    for group in state["param_groups"]:
        indices.extend(group["params"])
        options.extend(
            [{key: value for key, value in group.items() if key != "params"}] * len(group["params"])
        )
    if indices != list(range(len(fields))):
        raise ValueError("Repair optimizer checkpoint field ownership differs.")
    current, entries = _field_entries(optimizer)
    positions = {id(parameter): index for index, parameter in enumerate(fields)}
    values, groups = {}, []
    for actual, saved in zip(optimizer.param_groups, current["param_groups"], strict=True):
        members = [positions[id(parameter)] for parameter in actual["params"]]
        settings = options[members[0]]
        if any(options[index] != settings for index in members):
            raise ValueError("Repair optimizer checkpoint has incompatible field hyperparameters.")
        if settings["initial_lr"] != actual["initial_lr"]:
            raise ValueError("Repair optimizer checkpoint learning rate differs.")
        groups.append({**settings, "params": saved["params"]})
        for parameter, position in zip(actual["params"], members, strict=True):
            if position in state["state"]:
                values[entries[id(parameter)][0]] = state["state"][position]
    optimizer.load_state_dict({"state": values, "param_groups": groups})
