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

"""GatedDeltaNet state and WY activation fake quantization.

State QAT uses an explicit prefix/suffix policy with native serving arithmetic.
W-only QAT applies TensorQuantizer to the WY-transformed keys and reuses FLA's
state kernels. Both paths retain gradients through quantization with identity STE.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from ..linear_attention.prefill import matmul_gdn
from ..linear_attention.utils import validate_gdn_quantizer
from .linear_attention import _LinearAttentionQuantMixin

__all__ = ["GatedDeltaNetStateQuantMixin"]

GatedDeltaRuleFn = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


def _fla_chunk_gated_delta_rule() -> GatedDeltaRuleFn:
    # FLA is an optional, heavy dependency needed only when quantization is enabled.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def _w_qdq_chunk_gated_delta_rule() -> GatedDeltaRuleFn:
    # Imported on first use: flash-linear-attention is a heavy optional dependency that only the
    # enabled quantizer needs, and importing it warns on machines without a GPU.
    try:
        from modelopt.torch.kernels.quantization.linear_attention.fla_chunk_gated_delta_rule import (
            chunk_gated_delta_rule,
        )
    except ImportError as e:
        raise RuntimeError(
            "GDN fake quantization needs Triton and fla-core==0.5.1 on a CUDA "
            f"device; importing the W-quantizing kernel failed with {e!r}."
        ) from e
    return chunk_gated_delta_rule


class GatedDeltaNetStateQuantMixin(_LinearAttentionQuantMixin):
    """Adds ``gdn_state_quantizer`` and ``gdn_w_quantizer`` to a GatedDeltaNet module.

    Subclasses route the module's chunked gated-delta-rule call through
    :meth:`_state_quantized_chunk_gated_delta_rule`. Both quantizers start disabled; enable them
    with ``quant_cfg`` entries on ``*gdn_state_quantizer`` / ``*gdn_w_quantizer`` such as the
    ``configs/ptq/units/gdn_state_fp8_dynamic`` and ``gdn_w_fp8_dynamic`` recipe units. The state
    quantizer carries the QDQ configuration. State supports dynamic E4M3 or signed
    narrow-range INT8; W supports dynamic E4M3.
    Both sites require identity STE; state QAT uses an explicit serving policy.
    The execution policy is saved in ModelOpt metadata.
    """

    linear_attention_quantizer_names = ("gdn_state_quantizer", "gdn_w_quantizer")

    def validate_linear_attention(self):
        """Validate state and GDN WY operand quantization."""
        super().validate_linear_attention()
        if self.gdn_w_quantizer.is_enabled:
            validate_gdn_quantizer(self.gdn_w_quantizer, name="gdn_w_quantizer")
            if self.linear_attention_config.decode is not None:
                raise ValueError("Decode's exact prefix does not support WY operand QDQ")

    @property
    def gdn_state_qdq_block_v(self) -> int:
        """Execution tile width; also sets grouping for legacy tile quantizers."""
        return self.linear_attention_config.state.block_v

    def _state_quantized_chunk_gated_delta_rule(
        self, gated_delta_rule: GatedDeltaRuleFn, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Route enabled state or W quantization to its training implementation."""
        self.validate_linear_attention()
        quantize_w = self.gdn_w_quantizer.is_enabled
        if not self.linear_attention_is_enabled:
            return gated_delta_rule(*args, **kwargs)
        if self.linear_attention_config.backend == "fla" and self.gdn_state_quantizer.is_enabled:
            raise ValueError(
                "Chunk-only state QAT is retired. Select backend='serving' with an explicit "
                "decode policy and supply prefill lengths through linear_attention_training_phase. "
                "Existing state checkpoints are not silently migrated to a different QDQ schedule."
            )
        while isinstance(gated_delta_rule, partial):
            args = (*gated_delta_rule.args, *args)
            kwargs = {**gated_delta_rule.keywords, **kwargs}
            gated_delta_rule = gated_delta_rule.func
        if gated_delta_rule is not _fla_chunk_gated_delta_rule():
            raise NotImplementedError(
                "GatedDeltaNet quantization supports only FLA's chunk_gated_delta_rule "
                f"callable or a functools.partial of it; got {gated_delta_rule!r}."
            )
        chunk_size = kwargs.pop("chunk_size", self.linear_attention_config.chunk_size)
        if chunk_size != self.linear_attention_config.chunk_size:
            raise ValueError("GDN fake quantization supports only chunk_size=64")
        if self.linear_attention_config.backend != "fla":
            return matmul_gdn(
                *args,
                policy=self.linear_attention_config,
                state_quantizer=self.gdn_state_quantizer,
                replay_key_quantizer=self.replay_key_quantizer,
                replay_update_quantizer=self.replay_update_quantizer,
                chunk_size=chunk_size,
                prefill_lengths=self._linear_attention_prefill_lengths,
                **kwargs,
            )
        return _w_qdq_chunk_gated_delta_rule()(
            *args,
            chunk_size=chunk_size,
            w_quantizer=self.gdn_w_quantizer if quantize_w else None,
            **kwargs,
        )
