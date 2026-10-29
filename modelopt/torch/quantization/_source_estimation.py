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

"""Bounded original-activation estimates for recipes requiring final layerwise calibration."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.utils._pytree import tree_leaves

from modelopt.torch.opt.searcher import ForwardLoop

from ._selected_calibration import _SelectedLayerCalibrator
from .config import QuantizeAlgoCfgType
from .ggml.common import offload_packed_weights
from .nn import TensorQuantizer
from .utils.core_utils import has_accelerate_offload
from .utils.layerwise_calib import LayerActivationCollector

__all__ = []

_SOURCE_ACTIVATION_ESTIMATOR = "source_activation_v1"


def _input_storages(inputs):
    storages = {}
    for value in tree_leaves(inputs):
        if isinstance(value, torch.Tensor):
            if isinstance(value, DTensor) or value.is_meta or value.layout != torch.strided:
                raise ValueError("Source estimates require materialized, strided inputs")
            storage = value.untyped_storage()
            storages[value.device, storage.data_ptr()] = storage.nbytes()
        elif value is not None and not isinstance(
            value, (str, bool, int, float, torch.dtype, torch.device)
        ):
            raise ValueError("Source estimates cannot account for opaque input state")
    return storages


@torch.no_grad()
def _estimate_source_activations(
    model: nn.Module,
    groups: Sequence[tuple[Sequence[str], QuantizeAlgoCfgType]],
    forward_loop: ForwardLoop,
    *,
    max_activation_bytes: int,
):
    """Estimate configured candidate groups from original-source layer inputs.

    This is not final calibration: it omits the eventual selected quantized prefix. The caller
    owns source/input identity, fresh candidate quantizers and worker memory/artifact limits.
    No checkpoints or exports are written; only a complete candidate may be published.
    """
    if type(max_activation_bytes) is not int or max_activation_bytes < 1:
        raise ValueError("Source estimates require a positive activation-cache byte limit")
    if not callable(forward_loop):
        raise ValueError("Source estimates require a replayable forward loop")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("Source estimates currently require one process")
    if any(module.training or isinstance(module, FSDPModule) for module in model.modules()):
        raise ValueError("Source estimates require an evaluation-mode, unsharded model")
    if has_accelerate_offload(model):
        raise ValueError("Source estimates do not yet support offloaded weights")
    seen, devices = set(), set()
    for _, parameter in model.named_parameters(remove_duplicate=False):
        if isinstance(parameter, DTensor) or parameter.is_meta or not parameter.is_contiguous():
            raise ValueError("Source estimates require materialized contiguous weights")
        storage = parameter.untyped_storage()
        key = (parameter.device, storage.data_ptr())
        if storage.nbytes() and key in seen:
            raise ValueError("Source estimates do not yet support shared weight storage")
        seen.add(key)
        devices.add(parameter.device)
    if len(devices) != 1:
        raise ValueError("Source estimates currently require one materialized model device")

    dispatch = _SelectedLayerCalibrator(model, groups)
    owned = set()
    for methods in dispatch.methods.values():
        for quantizers, chain in methods.values():
            owned.update(quantizers)
            if any(mode._auto_quantize_search_policy != "same_algorithm" for mode, _ in chain):
                raise ValueError("Algorithm has no validated source-preserving leaf estimator")
    quantizers = [q for q in model.modules() if isinstance(q, TensorQuantizer)]
    flags = {q: q.is_enabled for q in quantizers}
    enabled = {q for q, was_enabled in flags.items() if was_enabled}
    if owned != enabled:
        raise ValueError("Source estimation groups must cover every enabled candidate quantizer")

    layers = LayerActivationCollector.get_decoder_layers(model)
    assert layers is not None
    original_layers = list(layers)
    collector = LayerActivationCollector(model)
    peak_bytes = 0

    def capture(layer, retained):
        nonlocal peak_bytes
        storages = _input_storages(retained)

        def check_inputs(_module, args, kwargs):
            nonlocal peak_bytes
            proposed = storages | _input_storages((args, kwargs))
            size = sum(proposed.values())
            if size > max_activation_bytes:
                raise ValueError("Source estimate activation cache exceeds its byte limit")
            storages.update(proposed)
            peak_bytes = max(peak_bytes, size)

        handle = layer.register_forward_pre_hook(check_inputs, with_kwargs=True)
        try:
            return collector.get_input_activations(layer, forward_loop)
        finally:
            handle.remove()

    try:
        for q in flags:
            q.disable()
        collector._patch_all_layers(decoder_layers=layers)
        inputs = capture(original_layers[0], [])
        for index, layer in enumerate(original_layers):
            # Capture the unquantized successor before any candidate update in this layer.
            following = (
                capture(original_layers[index + 1], inputs)
                if index + 1 < len(original_layers)
                else []
            )
            layer._layerwise_calib.mode = "original"

            def replay(module):
                for args, kwargs in inputs:
                    module(*args, **kwargs)

            current_quantizers = [q for q in layer.modules() if isinstance(q, TensorQuantizer)]
            try:
                for q in current_quantizers:
                    q._disabled = not flags[q]
                dispatch(layer, replay)
            finally:
                for q in current_quantizers:
                    q.disable()
            offload_packed_weights(layer)
            inputs = following
    finally:
        collector._unpatch_all_layers()
        # Restore by identity: the collector temporarily removes layers from named_modules().
        for q, was_enabled in flags.items():
            q._disabled = not was_enabled
    return {"policy": _SOURCE_ACTIVATION_ESTIMATOR, "peak_activation_bytes": peak_bytes}
