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

"""Kernel routing for Megatron Kimi Delta Attention quantization."""

from functools import partial

from ..linear_attention.kda import matmul_kda
from .linear_attention import _LinearAttentionQuantMixin

__all__ = ["KimiDeltaAttentionStateQuantMixin"]


def _discard_legacy_w_quantizer_state(quantizer_state):
    # Early KDA checkpoints included a disabled placeholder; WY QDQ was never supported.
    for name in list(quantizer_state):
        if name.rsplit(".", 1)[-1] == "kda_w_quantizer":
            if not quantizer_state[name].get("_disabled", False):
                raise ValueError("KDA checkpoints with enabled kda_w_quantizer are unsupported")
            del quantizer_state[name]


class KimiDeltaAttentionStateQuantMixin(_LinearAttentionQuantMixin):
    """Adds state quantizers and decode-aware kernel routing to Kimi Delta Attention."""

    linear_attention_quantizer_names = ("kda_state_quantizer",)

    def validate_linear_attention(self):
        """Require the materialized backend for KDA numerical emulation."""
        super().validate_linear_attention()
        if self.linear_attention_is_enabled and self.linear_attention_config.backend == "fla":
            raise ValueError("KDA numerical emulation requires backend='serving'")

    def _state_quantized_chunk_kda(self, kernel, *args, **kwargs):
        self.validate_linear_attention()
        if not self.linear_attention_is_enabled:
            return kernel(*args, **kwargs)
        # FLA kernels are an optional dependency; the training layer belongs to Megatron.
        from fla.ops.kda import chunk_kda

        while isinstance(kernel, partial):
            args = (*kernel.args, *args)
            kwargs = {**kernel.keywords, **kwargs}
            kernel = kernel.func
        if kernel is not chunk_kda:
            raise NotImplementedError("KDA quantization requires FLA's chunk_kda callable")
        return matmul_kda(
            *args,
            policy=self.linear_attention_config,
            state_quantizer=self.kda_state_quantizer,
            replay_key_quantizer=self.replay_key_quantizer,
            replay_update_quantizer=self.replay_update_quantizer,
            prefill_lengths=self._linear_attention_prefill_lengths,
            **kwargs,
        )
