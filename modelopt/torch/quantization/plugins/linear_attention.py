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
from ..linear_attention.utils import state_quantizer_config
from ..nn import QuantModule, TensorQuantizer

__all__ = []


class _LinearAttentionQuantMixin(QuantModule):
    linear_attention_quantizer_names: tuple[str, ...] = ()
    replay_quantizer_names = ("replay_key_quantizer", "replay_update_quantizer")

    def _setup(self):
        for name in (*self.linear_attention_quantizer_names, *self.replay_quantizer_names):
            self._register_temp_attribute(
                name, TensorQuantizer(QuantizerAttributeConfig(enable=False))
            )
        self._register_temp_attribute("linear_attention_config", LinearAttentionConfig())
        self._register_temp_attribute("_linear_attention_prefill_lengths", None)

    @property
    def _linear_attn_state(self):
        return getattr(self, self.linear_attention_quantizer_names[0])

    @property
    def linear_attention_is_enabled(self):
        """Whether an operand, state, or arithmetic policy changes the computation."""
        return (
            any(
                getattr(self, name).is_enabled
                for name in (*self.linear_attention_quantizer_names, *self.replay_quantizer_names)
            )
            or self.linear_attention_config.decode is not None
        )

    def _migrate_legacy_replay_quantizers(self):
        decode = self.linear_attention_config.decode
        replay = decode.replay if decode is not None else None
        legacy = getattr(replay, "_legacy_factor_qdq", None)
        if (
            replay is not None
            and legacy is None
            and self.linear_attention_config.schema_version == 1
        ):
            legacy = replay.__dict__.pop("factor_qdq", True)
        if replay is not None and legacy is not None:
            for name in self.replay_quantizer_names:
                grouping = (
                    {"axis": (0,)}
                    if name == "replay_key_quantizer"
                    else {"block_sizes": {-1: self.linear_attention_config.state.block_v}}
                )
                getattr(self, name).set_from_attribute_config(
                    QuantizerAttributeConfig(
                        num_bits=(4, 3), type="dynamic", enable=legacy, **grouping
                    )
                )
            replay._legacy_factor_qdq = None
        self.linear_attention_config.schema_version = 2

    def validate_linear_attention(self):
        """Validate quantizer contracts shared by GDN and KDA."""
        decode = self.linear_attention_config.decode
        for name in self.replay_quantizer_names:
            quantizer = getattr(self, name)
            if quantizer.is_enabled:
                if decode is None or decode.mode != "replay":
                    raise ValueError(f"{name} requires a replay execution policy")
                if not quantizer.fake_quant:
                    raise ValueError(f"{name} requires fake quantization for training")
        if self._linear_attn_state.is_enabled:
            state_format, group_size = state_quantizer_config(
                self._linear_attn_state,
                name=self.linear_attention_quantizer_names[0],
            )
            if decode is not None and decode.state_codec == "int8_hadamard32":
                if state_format != "int8":
                    raise ValueError("int8_hadamard32 requires INT8 state quantization")
                if group_size:
                    raise ValueError("TensorQuantizer block_sizes requires state_codec='tile'")

    def modelopt_post_restore(self, prefix=""):
        """Validate the restored numerical policy and quantizers."""
        super().modelopt_post_restore(prefix)
        self.validate_linear_attention()
