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

"""Selected-method dispatch inside an existing sequential layer traversal."""

import json
from collections.abc import Sequence
from typing import cast

from torch import nn

from modelopt.torch.opt.searcher import ForwardLoop

from .config import QuantizeAlgoCfgType
from .ggml.common import preserve_source_weights
from .mode import (
    BaseCalibrateModeDescriptor,
    CalibrateModeRegistry,
    get_modelike_from_algo_cfg,
    wrapped_calib_func,
)
from .nn import TensorQuantizer
from .utils.layerwise_calib import LayerActivationCollector

__all__ = []


class _SelectedLayerCalibrator:
    """Prepare scoped methods for already-configured selected quantizers.

    The caller owns assignment/source validation, fresh quantizer construction and checkpoint
    publication. This callback never chooses formats or starts a layer traversal itself.
    """

    def __init__(
        self, model: nn.Module, groups: Sequence[tuple[Sequence[str], QuantizeAlgoCfgType]]
    ):
        layers = LayerActivationCollector.get_decoder_layers(model)
        if not layers or not groups:
            raise ValueError("Selected calibration requires decoder layers and selected groups")
        owners = {}
        for layer in layers:
            for module in layer.modules():
                if owners.setdefault(module, layer) is not layer:
                    raise ValueError("Selected calibration cannot share modules across layers")
        modules = dict(model.named_modules())
        assigned = set()
        self.methods = {}
        self.layerwise_config: dict | None = None
        for names, algorithm in groups:
            if not names or len(set(names)) != len(names):
                raise ValueError("Selected calibration requires unique, nonempty module groups")
            targets = [modules[name] for name in names]
            if any(target not in owners for target in targets):
                raise ValueError("Selected calibration groups must be inside decoder layers")
            layer = owners[targets[0]]
            if any(owners[target] is not layer for target in targets):
                raise ValueError("A selected calibration group cannot cross decoder layers")
            quantizers = {
                quantizer
                for target in targets
                for quantizer in target.modules()
                if isinstance(quantizer, TensorQuantizer) and quantizer.is_enabled
            }
            if not quantizers or assigned.intersection(quantizers):
                raise ValueError(
                    "Selected calibration has empty or overlapping quantizer ownership"
                )
            chain = []
            final_layerwise = False
            for name, settings in get_modelike_from_algo_cfg(algorithm):
                mode = cast("BaseCalibrateModeDescriptor", CalibrateModeRegistry[name])
                config = mode.config_class(**settings)
                if not mode._supports_layerwise or mode._calib_func is None:
                    raise ValueError("Selected algorithm does not support decoder-layer dispatch")
                if config.layerwise.enable or mode._requires_final_layerwise:
                    final_layerwise = True
                    options = config.layerwise.model_dump(exclude={"enable"})
                    if self.layerwise_config is not None and options != self.layerwise_config:
                        raise ValueError(
                            "Selected algorithms require incompatible layer traversals"
                        )
                    self.layerwise_config = options
                if config._mutates_weights:
                    weights = {
                        q
                        for target in targets
                        for name, q in target.named_modules()
                        if "weight_quantizer" in name and q in quantizers
                    }
                    if (
                        config.method not in {"gptq", "imatrix"}
                        or not weights
                        or any(q.backend != "ggml" for q in weights)
                    ):
                        raise ValueError("Selected dispatch requires source-preserving algorithms")
                # The outer runner owns these settings; preserve the original method/settings.
                config.layerwise.enable = False
                chain.append((mode, config))
            if not final_layerwise:
                raise ValueError("Only layerwise methods belong in selected final calibration")
            key = json.dumps([config.model_dump() for _, config in chain], sort_keys=True)
            methods = self.methods.setdefault(layer, {})
            if key not in methods:
                methods[key] = (set(), chain)
            methods[key][0].update(quantizers)
            assigned.update(quantizers)

    def __call__(self, layer: nn.Module, forward_loop: ForwardLoop):
        """Calibrate selected groups from this layer's inputs, then restore the selected mixture."""
        if layer not in self.methods:
            return
        enabled = {q: q.is_enabled for q in layer.modules() if isinstance(q, TensorQuantizer)}
        ratios = {
            module: module._moe_calib_experts_ratio
            for module in layer.modules()
            if hasattr(module, "_moe_calib_experts_ratio")
        }
        try:
            for key in sorted(self.methods[layer]):
                selected, chain = self.methods[layer][key]
                for q in enabled:
                    q._disabled = not (enabled[q] and q in selected)
                for mode, config in chain:
                    try:
                        with preserve_source_weights():
                            wrapped_calib_func(
                                layer, config, forward_loop, func=type(mode)._calib_func
                            )
                    finally:
                        for module, ratio in ratios.items():
                            module._moe_calib_experts_ratio = ratio
        finally:
            for q, was_enabled in enabled.items():
                q._disabled = not was_enabled
