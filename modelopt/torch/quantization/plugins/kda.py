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

"""QAT integration for the optional FLA KimiDeltaAttention layer."""

import importlib.metadata
import inspect
from functools import lru_cache
from types import FunctionType

from ..linear_attention.kda import matmul_kda
from ..nn import QuantModuleRegistry
from .custom import CUSTOM_MODEL_PLUGINS
from .linear_attention import _LinearAttentionQuantMixin

__all__ = []


@lru_cache(maxsize=8)
def _forward_signature(function):
    return inspect.signature(function)


class _QuantKimiDeltaAttention(_LinearAttentionQuantMixin):
    linear_attention_quantizer_names = ("kda_state_quantizer", "kda_w_quantizer")

    def validate_linear_attention(self):
        super().validate_linear_attention()
        if self.linear_attention_is_enabled and self.linear_attention_config.backend != "matmul":
            raise ValueError("KDA numerical emulation requires backend='matmul'")

    def forward(self, *args, **kwargs):
        self.validate_linear_attention()
        if not self.linear_attention_is_enabled:
            return super().forward(*args, **kwargs)
        original = super().forward.__func__
        if self.linear_attention_config.decode is not None:
            arguments = _forward_signature(original).bind(self, *args, **kwargs).arguments
            if arguments.get("use_cache", False) or arguments.get("past_key_values") is not None:
                raise NotImplementedError(
                    "Decode-aware QAT requires use_cache=False; use explicit numerical carry for continuation"
                )
        # Bind this invocation so copied modules cannot retain another module's quantizers.
        namespace = {
            **original.__globals__,
            "chunk_kda": self._quantized_chunk,
            "fused_recurrent_kda": self._unsupported_recurrent,
        }
        forward = FunctionType(
            original.__code__,
            namespace,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        forward.__kwdefaults__ = original.__kwdefaults__
        return forward(self, *args, **kwargs)

    def _quantized_chunk(self, *args, **kwargs):
        return matmul_kda(
            *args,
            policy=self.linear_attention_config,
            state_qdq=self.kda_state_quantizer.is_enabled and self.kda_state_quantizer._if_quant,
            state_format=self._linear_attn_state_format,
            prefill_lengths=self._linear_attention_prefill_lengths,
            **kwargs,
        )

    def _unsupported_recurrent(self, *args, **kwargs):
        if self.linear_attention_config.decode is not None:
            return self._quantized_chunk(*args, **kwargs)
        raise NotImplementedError(
            "KDA prefill QAT requires the chunk path; recurrent decode is not yet supported"
        )


def _register_fla_kda(model):
    for module in model.modules():
        cls = type(module)
        if cls.__module__ != "fla.layers.kda" or cls.__name__ != "KimiDeltaAttention":
            continue
        if cls in QuantModuleRegistry:
            continue
        if importlib.metadata.version("flash-linear-attention") != "0.5.1":
            raise RuntimeError("KDA QAT is qualified with flash-linear-attention==0.5.1")
        if importlib.metadata.version("fla-core") != "0.5.1":
            raise RuntimeError("KDA QAT requires fla-core==0.5.1")
        QuantModuleRegistry.register({cls: "fla_KimiDeltaAttention"})(_QuantKimiDeltaAttention)


CUSTOM_MODEL_PLUGINS.add(_register_fla_kda)
