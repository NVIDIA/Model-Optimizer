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

"""Shared module policy and checkpoint support for linear-attention QAT."""

from ..config import QuantizerAttributeConfig
from ..linear_attention.config import LinearAttentionConfig
from ..linear_attention.utils import validate_gdn_quantizer
from ..nn import QuantModule, TensorQuantizer

__all__ = []


class _LinearAttentionQuantMixin(QuantModule):
    linear_attention_quantizer_names = ("gdn_state_quantizer", "gdn_w_quantizer")

    def _setup(self):
        for name in self.linear_attention_quantizer_names:
            self._register_temp_attribute(
                name, TensorQuantizer(QuantizerAttributeConfig(enable=False))
            )
        self._register_temp_attribute("linear_attention_config", LinearAttentionConfig())
        self._register_temp_attribute("_linear_attention_prefill_lengths", None)

    @property
    def _linear_attn_state(self):
        return getattr(self, self.linear_attention_quantizer_names[0])

    @property
    def _linear_attn_state_format(self):
        return "int8" if self._linear_attn_state.num_bits == 8 else "fp8_e4m3"

    @property
    def _linear_attn_w(self):
        return getattr(self, self.linear_attention_quantizer_names[1])

    @property
    def linear_attention_is_enabled(self):
        """Whether an operand, state, or arithmetic policy changes the computation."""
        return (
            self._linear_attn_state.is_enabled
            or self._linear_attn_w.is_enabled
            or self.linear_attention_config.decode is not None
        )

    def validate_linear_attention(self):
        """Validate quantizer contracts shared by GDN and KDA."""
        if self._linear_attn_state.is_enabled:
            validate_gdn_quantizer(
                self._linear_attn_state,
                name=self.linear_attention_quantizer_names[0],
                num_bits=((4, 3), 8),
            )
            if self._linear_attn_state.axis != (0, 1):
                raise ValueError(
                    f"{self.linear_attention_quantizer_names[0]} supports only axis=(0, 1) "
                    "with state.block_v tiling"
                )
            decode = self.linear_attention_config.decode
            if decode is not None and decode.state_codec == "int8_hadamard32":
                if self._linear_attn_state_format != "int8":
                    raise ValueError("int8_hadamard32 requires INT8 state quantization")
        if self._linear_attn_w.is_enabled:
            validate_gdn_quantizer(
                self._linear_attn_w, name=self.linear_attention_quantizer_names[1]
            )
            if self.linear_attention_config.decode is not None:
                raise ValueError("Decode's exact prefix does not support WY operand QDQ")

    def modelopt_post_restore(self, prefix=""):
        """Validate the restored numerical policy and quantizers."""
        super().modelopt_post_restore(prefix)
        self.validate_linear_attention()
