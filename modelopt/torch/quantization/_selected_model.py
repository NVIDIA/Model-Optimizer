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

"""Construct a frozen selected mixture on freshly loaded source weights."""

import copy
import itertools
from typing import cast

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode

from ._calibration_workers import _digest
from ._selected_calibration import _SelectedLayerCalibrator
from ._source_estimation import _validate_source_layout
from .config import QuantizeConfig, QuantizerCfgEntry
from .conversion import (
    _normalize_fused_experts_quantizer_name,
    quantizer_state,
    restore_quantizer_state,
    set_quantizer_by_cfg,
)
from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry, get_modelike_from_algo_cfg
from .nn import TensorQuantizer
from .utils.core_utils import set_quantizer_state_dict
from .utils.layerwise_calib import LayerActivationCollector

__all__ = []


def _selected_model(model, assignment, plan, search_state, identity):
    """Prepare final calibration from an immutable assignment and validated coordinator state.

    The caller validates the search checkpoint and replay inputs before entering. No estimation
    payload is restored for a layerwise group. All selected quantizers exist before traversal.
    """
    # Recipe planning imports quantization, so import its contracts after package initialization.
    from modelopt.recipe.planner import freeze_auto_quantize_assignment
    from modelopt.torch.quantization.model_quant import quantize

    _validate_source_layout(model)
    if any(isinstance(module, TensorQuantizer) for module in model.modules()):
        raise ValueError("Selected finalization requires unconverted original source weights")
    if (
        _digest(dict(itertools.chain(model.named_parameters(), model.named_buffers())))
        != identity["source"]
    ):
        raise ValueError("Selected finalization source differs from the frozen assignment")
    frozen = freeze_auto_quantize_assignment(search_state, plan, identity)
    if frozen.digest != assignment.digest:
        raise ValueError("Selected finalization assignment or provenance differs")
    groups = assignment.as_dict()["groups"]
    layers = LayerActivationCollector.get_decoder_layers(model)
    if not layers:
        raise ValueError("Selected finalization requires supported decoder layers")
    modules = dict(model.named_modules())
    owners = {}
    for index, layer in enumerate(layers):
        for module in layer.modules():
            if owners.setdefault(module, index) != index:
                raise ValueError("Selected finalization cannot share modules across layers")

    metadata, buffers, methods, owned, indexes = {}, {}, [], set(), {}
    traversal = None
    for group in groups:
        names = group["module_names"]
        if any(name not in modules for name in names):
            raise ValueError("Selected finalization module mapping differs")
        recipe = search_state["best"]["recipe"][group["name"]]
        group_keys = [f"{name}.{attr}" for name in names for attr in group["quantizer_attrs"][name]]
        if owned.intersection(group_keys):
            raise ValueError("Selected finalization has overlapping quantizer mappings")
        owned.update(group_keys)
        if recipe.is_no_quant:
            continue
        if group["final_layerwise"]:
            locations = {owners.get(modules[name]) for name in names}
            if None in locations or len(locations) != 1:
                raise ValueError(
                    "Selected final-calibration groups must lie within one decoder layer"
                )
            for mode_name, settings in get_modelike_from_algo_cfg(recipe.config.algorithm):
                mode = cast("BaseCalibrateModeDescriptor", CalibrateModeRegistry[mode_name])
                algorithm = mode.config_class(**settings)
                if (
                    not mode._supports_layerwise
                    or mode._auto_quantize_search_policy != "same_algorithm"
                ):
                    raise ValueError("Selected algorithm lacks source-preserving layer dispatch")
                if algorithm.layerwise.enable or mode._requires_final_layerwise:
                    options = algorithm.layerwise.model_dump(exclude={"enable"})
                    if options["checkpoint_dir"] is not None or options["export_dir"] is not None:
                        raise ValueError(
                            "Automatic finalization owns its checkpoint and export destinations"
                        )
                    if traversal is not None and traversal != options:
                        raise ValueError(
                            "Selected algorithms require incompatible layer traversals"
                        )
                    traversal = options
                if algorithm._mutates_weights:
                    weight_cfgs = [
                        entry["cfg"]
                        for entry in recipe.config.quant_cfg
                        if "weight_quantizer" in entry["quantizer_name"]
                        and entry.get("enable", True)
                        and entry.get("cfg") is not None
                    ]
                    if (
                        algorithm.method not in {"gptq", "imatrix"}
                        or not weight_cfgs
                        or any(
                            isinstance(cfg, list) or cfg.backend != "ggml" for cfg in weight_cfgs
                        )
                    ):
                        raise ValueError(
                            "Selected finalization requires source-preserving weight algorithms"
                        )
            methods.append((names, recipe.config.algorithm))
            continue
        state = search_state["quantizer_states"].get(recipe)
        if state is None:
            raise ValueError("Selected finalization is missing a completed candidate")
        if recipe not in indexes:
            quantizer_index = {}
            for name, value in state["metadata"].items():
                quantizer_index.setdefault(
                    _normalize_fused_experts_quantizer_name(name), []
                ).append((name, value))
            indexes[recipe] = quantizer_index
        for key in group_keys:
            if key not in indexes[recipe]:
                raise ValueError("Selected candidate quantizer mapping is incomplete")
            for name, value in indexes[recipe][key]:
                if "_pytorch_state_metadata" not in value:
                    raise ValueError("Selected finalization currently requires tensor quantizers")
                description = value["_pytorch_state_metadata"]
                tensors = {**description["buffers"], **description["params"]}
                payload = state["state_dict"].get(name)
                if payload is None or set(payload) != set(tensors):
                    raise ValueError("Selected candidate quantizer buffers are incomplete")
                for field, tensor in payload.items():
                    spec = tensors[field]
                    if (
                        not isinstance(tensor, torch.Tensor)
                        or tensor.is_meta
                        or tensor.dtype != spec["dtype"]
                        or tuple(tensor.shape) != tuple(spec["shape"])
                    ):
                        raise ValueError("Selected candidate buffer layout differs")
                metadata[name], buffers[name] = value, payload
    if not methods:
        raise ValueError("Selected finalization requires at least one layerwise group")
    baseline = search_state.get("fixed_quantization_config") or {
        "quant_cfg": [{"quantizer_name": "*", "enable": False}]
    }
    resolved = QuantizeConfig(**dict(baseline, algorithm=None))
    disabled = search_state.get("disabled_layers") or []
    disabled = [disabled] if isinstance(disabled, str) else disabled
    resolved.quant_cfg.extend(
        QuantizerCfgEntry(quantizer_name=name, enable=False) for name in disabled
    )

    def configure(target):
        quantize(target, resolved)
        for group in groups:
            if not group["is_fixed"]:
                recipe = search_state["best"]["recipe"][group["name"]]
                for name in group["module_names"]:
                    set_quantizer_by_cfg(target.get_submodule(name), recipe.config.quant_cfg)

    # Exercise the registered conversion and dispatch on structure only. Never duplicate source
    # storage or materialize candidate payloads while checking plugin-generated quantizer names.
    with (
        FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False) as mode,
        FakeCopyMode(mode),
    ):
        probe = copy.deepcopy(model)
        configure(probe)
    if not set(metadata) <= quantizer_state(probe).keys():
        raise ValueError("Selected candidate quantizers differ from the source layout")
    inside = {
        module
        for layer in LayerActivationCollector.get_decoder_layers(probe)
        for module in layer.modules()
    }
    if any(
        isinstance(module, TensorQuantizer)
        and module.is_enabled
        and (module not in inside or _normalize_fused_experts_quantizer_name(name) not in owned)
        for name, module in probe.named_modules()
    ):
        raise ValueError("Selected finalization requires quantizers in frozen decoder groups")
    _SelectedLayerCalibrator(probe, methods)
    del probe, mode

    configure(model)
    merged = quantizer_state(model)
    merged.update(metadata)
    restore_quantizer_state(model, resolved, {"quantizer_state": merged})
    set_quantizer_state_dict(model, buffers)
    dispatch = _SelectedLayerCalibrator(model, methods)
    return dispatch
