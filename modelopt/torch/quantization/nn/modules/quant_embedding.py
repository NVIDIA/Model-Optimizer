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

"""Quantized Embedding."""

import contextlib

import torch
import torch.nn as nn

from ...tensor_quant import QUANT_DESC_8BIT_PER_TENSOR
from ...utils import is_torch_export_mode
from .quant_module import QuantModule, QuantModuleRegistry
from .tensor_quantizer import HardDisabledTensorQuantizer, SequentialQuantizer, TensorQuantizer

__all__ = ["QuantEmbedding"]


@QuantModuleRegistry.register({nn.Embedding: "nn.Embedding"})
class _QuantEmbedding(QuantModule):
    """Quantized version of ``nn.Embedding``.

    The literal input to ``nn.Embedding`` is integer indices, which cannot be
    fake-quantized. The ``input_quantizer`` attribute is kept (for symmetry with
    other quant modules and for introspection by export/calibration code) but is
    a :class:`HardDisabledTensorQuantizer` — direct ``enable*()`` calls raise and
    wildcard configs are silently absorbed-then-force-disabled. Only the embedding
    table (weight) and the lookup output (an activation feeding downstream layers)
    are quantizable.

    Quantizer roles:
        - ``weight_quantizer``: quantizes the embedding table (``self.weight``).
        - ``input_quantizer``: permanently disabled placeholder
          (:class:`HardDisabledTensorQuantizer`).
        - ``output_quantizer``: optional activation quantizer for the lookup output,
          disabled by default.
    """

    weight_quantizer: TensorQuantizer | SequentialQuantizer
    input_quantizer: HardDisabledTensorQuantizer
    output_quantizer: TensorQuantizer
    _enable_weight_quantization: bool
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_output = QUANT_DESC_8BIT_PER_TENSOR

    @contextlib.contextmanager
    def quantize_weight(self):
        """Context in which ``self.weight`` is quantized via the dynamic attribute."""
        self._enable_weight_quantization = True
        try:
            yield
        finally:
            self._enable_weight_quantization = False

    @staticmethod
    def _get_quantized_weight(module: "_QuantEmbedding", weight: torch.Tensor) -> torch.Tensor:
        if module._enable_weight_quantization or is_torch_export_mode():
            return module.weight_quantizer(weight)
        return weight

    def _setup(self):
        """Register weight, (locked) input, and output quantizers."""
        self._register_temp_attribute(
            "weight_quantizer", TensorQuantizer(self.default_quant_desc_weight)
        )
        # Build the input quantizer disabled. HardDisabledTensorQuantizer's mutators raise,
        # so we disable it once at construction via direct attribute assignment.
        input_quantizer = HardDisabledTensorQuantizer(self.default_quant_desc_input)
        input_quantizer._disabled = True
        self._register_temp_attribute("input_quantizer", input_quantizer)
        self._register_temp_attribute(
            "output_quantizer", TensorQuantizer(self.default_quant_desc_output)
        )
        self.output_quantizer.disable()
        self._register_temp_attribute("_enable_weight_quantization", False)
        self._register_dynamic_attribute("weight", self._get_quantized_weight)

    def forward(self, input, *args, **kwargs):
        """Quantize the embedding table, look up, then optionally quantize the output.

        ``input_quantizer`` is intentionally never invoked — embedding inputs are
        integer indices. ``HardDisabledTensorQuantizer.set_from_attribute_config``
        keeps that quantizer disabled regardless of what configs target it, so a
        runtime check in this hot path is unnecessary.
        """
        if is_torch_export_mode():
            # quantize_weight()'s attribute write is not allowed under torch.export;
            # weight quantization is still applied inline via _get_quantized_weight's
            # is_torch_export_mode() branch. Apply output_quantizer in this path too
            # so users who opt into output activation quantization don't silently
            # lose it during export — matches QuantInputBase.forward's behavior.
            output = super().forward(input, *args, **kwargs)
        else:
            with self.quantize_weight():
                output = super().forward(input, *args, **kwargs)
        return self.output_quantizer(output)


# Public alias consistent with quant_linear / quant_conv naming.
QuantEmbedding = _QuantEmbedding
