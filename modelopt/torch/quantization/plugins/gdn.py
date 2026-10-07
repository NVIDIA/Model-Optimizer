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

"""GatedDeltaNet recurrent-state fake quantization.

State QAT uses an explicit prefix/suffix policy with native serving arithmetic.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from ..linear_attention.gdn import gdn_state_qat
from .linear_attention import _LinearAttentionQuantMixin

__all__ = ["GatedDeltaNetStateQuantMixin"]

GatedDeltaRuleFn = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


def _fla_chunk_gated_delta_rule() -> GatedDeltaRuleFn:
    # FLA is an optional, heavy dependency needed only when quantization is enabled.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


class GatedDeltaNetStateQuantMixin(_LinearAttentionQuantMixin):
    """Adds recurrent-state fake quantization to a GatedDeltaNet module.

    Subclasses route the module's chunked gated-delta-rule call through
    :meth:`_state_quantized_chunk_gated_delta_rule`. Enable ``*gdn_state_quantizer``
    through ``quant_cfg`` and select an explicit serving policy. State supports dynamic
    E4M3 or signed narrow-range INT8 with identity STE. The execution policy is saved in
    ModelOpt metadata. ``gdn_w_quantizer`` is a disabled legacy checkpoint handle;
    enabling WY operand quantization is no longer supported.
    """

    linear_attention_quantizer_names = ("gdn_state_quantizer", "gdn_w_quantizer")

    def validate_linear_attention(self):
        """Reject retired W QAT and validate the state quantizer."""
        if self.gdn_w_quantizer.is_enabled:
            raise ValueError(
                "GDN W quantization is no longer supported. Disable gdn_w_quantizer; "
                "use gdn_state_quantizer with an explicit serving policy for state QAT."
            )
        super().validate_linear_attention()

    @property
    def gdn_state_qdq_block_v(self) -> int:
        """Execution tile width; also sets grouping for legacy tile quantizers."""
        return self.linear_attention_config.state_block_v

    def _state_quantized_chunk_gated_delta_rule(
        self, gated_delta_rule: GatedDeltaRuleFn, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Route enabled state quantization to its training implementation."""
        self.validate_linear_attention()
        if not self.linear_attention_is_enabled:
            return gated_delta_rule(*args, **kwargs)
        while isinstance(gated_delta_rule, partial):
            args = (*gated_delta_rule.args, *args)
            kwargs = {**gated_delta_rule.keywords, **kwargs}
            gated_delta_rule = gated_delta_rule.func
        if gated_delta_rule is not _fla_chunk_gated_delta_rule():
            raise NotImplementedError(
                "GatedDeltaNet quantization supports only FLA's chunk_gated_delta_rule "
                f"callable or a functools.partial of it; got {gated_delta_rule!r}."
            )
        chunk_size = kwargs.pop("chunk_size", 64)
        if chunk_size != 64:
            raise ValueError("GDN fake quantization supports only chunk_size=64")
        return gdn_state_qat(
            *args,
            policy=self.linear_attention_config,
            state_quantizer=self.gdn_state_quantizer,
            chunk_size=chunk_size,
            prefill_lengths=self._linear_attention_prefill_lengths,
            **kwargs,
        )
