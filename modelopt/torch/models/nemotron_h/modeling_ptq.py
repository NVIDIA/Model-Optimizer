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
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.huggingface import (
    _is_supported_hf_model,
    _QuantFusedExperts,
)
from modelopt.torch.quantization.utils.layerwise_calib import LayerActivationCollector

__all__: list[str] = []


class _QuantNemotronHExperts(_QuantFusedExperts):
    """Quantized ``NemotronHExperts``: non-gated fused experts, ``down_proj(act_fn(up_proj(x)))``.

    The container holds 3-D ``up_proj`` and ``down_proj`` parameters instead of the gated
    ``gate_up_proj``, and its forward runs ``up_proj`` then ``down_proj`` once per expert on the
    eager path, or once per expert-sorted batch on ``grouped_mm``. So only the first projection's
    name differs from :class:`_QuantFusedExperts`: its per-expert weight quantizers and shared input
    quantizer live under ``up_proj_*``. Registered explicitly, these experts skip the
    transposed-layout check of :func:`register_fused_experts_on_the_fly`: the wrapper relies on
    transformers storing their weights ``(num_experts, out_dim, in_dim)``, as every supported
    release does, and refuses experts transformers marks ``is_transposed``.
    """

    _first_proj_attr = "up_proj"
    # Read by the export: a non-gated up_proj is exported whole, not split into gate|up halves.
    _is_gated = False

    def _setup(self):
        # The per-expert slicing reads (out_dim, in_dim); a transposed layout would slice every
        # expert's weight wrong without any shape mismatch to catch it.
        if getattr(self, "is_transposed", False):
            raise NotImplementedError(
                f"{type(self).__name__} quantizes expert weights stored (num_experts, out_dim, "
                "in_dim), but these experts are marked is_transposed."
            )
        super()._setup()


if NemotronHExperts not in QuantModuleRegistry:
    QuantModuleRegistry.register({NemotronHExperts: "hf.NemotronHExperts"})(_QuantNemotronHExperts)


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
