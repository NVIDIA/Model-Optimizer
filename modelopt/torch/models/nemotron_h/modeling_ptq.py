# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Nemotron-H PTQ modeling (HF model type ``nemotron_h``)."""

import torch.nn as nn

from modelopt.torch.quantization.plugins.huggingface import _is_supported_hf_model
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

__all__: list[str] = []


def is_nemotron_h_model(model: nn.Module) -> bool:
    return get_nemotron_h_decoder_layers(model) is not None


def get_nemotron_h_decoder_layers(model: nn.Module) -> nn.ModuleList | None:
    if not _is_supported_hf_model(model):
        return None

    # Custom remote-code checkpoint uses model.backbone.layers;
    # native transformers NemotronHModel uses model.model.layers.
    for container_attr in ("backbone", "model"):
        container = getattr(model, container_attr, None)
        if container is not None and hasattr(container, "layers"):
            layers = container.layers
            if layers and hasattr(layers[0], "block_type"):
                return layers

    return None


# Order matters: more specific predicates must be registered first because
# the first matching entry wins.  Nemotron-H must precede the generic
# homogeneous HF discoverer (which explicitly rejects Nemotron-H); the HF plugin
# imports this module before registering that discoverer.
LayerActivationCollector.register_decoder_layer_support(
    is_nemotron_h_model, get_nemotron_h_decoder_layers
)
