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

"""GPT-OSS PTQ modeling (HF model type ``gpt_oss``)."""

from contextlib import contextmanager
from functools import partial

import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from modelopt.torch.quantization.nn import QuantModuleRegistry, TensorQuantizer
from modelopt.torch.quantization.plugins.custom import _QuantFunctionalMixin
from modelopt.torch.quantization.plugins.huggingface import (
    _transposed_quantize,
    _TransposedExpertsCalibMixin,
)

__all__: list[str] = []


class _QuantGptOssExperts(_TransposedExpertsCalibMixin, _QuantFunctionalMixin):
    """Quantized wrapper for `transformers.GptOssExperts`.

    Quantizes `gate_up_proj` and `down_proj` weights via dynamic attributes inside `quantize_weight()`.
    Activations into `gate_up_proj` are quantized by `gate_up_proj_input_quantizer`. For `down_proj`
    activation quantization, we intercept `torch.Tensor.__matmul__`/`torch.bmm` and quantize inputs
    on every second call (since the first call computes `gate_up_proj` outputs and second call
    computes `down_proj` outputs).
    """

    @staticmethod
    def _get_quantized_weight(quantizer, module, weight):
        # MoE weight is accessed for each expert in one forward pass. so lets cache it
        if module._enable_weight_quantization:
            if hasattr(quantizer, "_cached_quant_val"):
                return getattr(quantizer, "_cached_quant_val")
            quantizer._cached_quant_val = _transposed_quantize(weight, quantizer)
            return quantizer._cached_quant_val
        return weight

    def _setup_for_weight_quantization(self):
        self._register_dynamic_attribute(
            "gate_up_proj", partial(self._get_quantized_weight, self.gate_up_proj_weight_quantizer)
        )
        self._register_dynamic_attribute(
            "down_proj", partial(self._get_quantized_weight, self.down_proj_weight_quantizer)
        )

    def _setup(self):
        assert not hasattr(self, "kernel_layer_name"), (
            "ModelOpt quantization does not support patched forward for kernel_hub"
        )
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.gate_up_proj_weight_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer = TensorQuantizer()
        self.down_proj_weight_quantizer = TensorQuantizer()

        self._register_temp_attribute("_enable_weight_quantization", False)
        self._register_temp_attribute("_down_proj_mul", False)
        self._setup_for_weight_quantization()

    @property
    def functionals_to_replace(self):
        # Use torch.ops.aten to bypass Python dispatch and avoid RecursionError
        # (torch.matmul / __matmul__ can dispatch to each other)
        _aten_bmm = torch.ops.aten.bmm
        _aten_matmul = torch.ops.aten.matmul

        def _quantized_bmm(batch1, batch2, *, out=None):
            batch1 = self.down_proj_input_quantizer(batch1) if self._down_proj_mul else batch1
            self._down_proj_mul = not self._down_proj_mul  # toggle the flag
            if out is not None:
                return torch.ops.aten.bmm.out(batch1, batch2, out=out)
            return _aten_bmm(batch1, batch2)

        def _tensor_matmul(self_t, other):
            self_t = self.down_proj_input_quantizer(self_t) if self._down_proj_mul else self_t
            self._down_proj_mul = not self._down_proj_mul
            return _aten_matmul(self_t, other)

        return [
            (torch, "bmm", _quantized_bmm),
            (torch.Tensor, "__matmul__", _tensor_matmul),
        ]

    @contextmanager
    def quantize_weight(self):
        """Context in which MoE weight is quantized."""
        self._enable_weight_quantization = True
        try:
            yield
        finally:
            for module in self.modules():
                if isinstance(module, TensorQuantizer) and hasattr(module, "_cached_quant_val"):
                    delattr(module, "_cached_quant_val")
        self._enable_weight_quantization = False

    def forward(
        self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None
    ) -> torch.Tensor:
        """Forward method to add quantization."""
        hidden_states = self.gate_up_proj_input_quantizer(hidden_states)
        with self.quantize_weight():
            return super().forward(hidden_states, router_indices, routing_weights)


if GptOssExperts not in QuantModuleRegistry:
    QuantModuleRegistry.register({GptOssExperts: "hf.GptOssExperts"})(_QuantGptOssExperts)
