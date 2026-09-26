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

"""Fake quantization of the GatedDeltaNet (GDN) recurrent state.

The chunked gated-delta-rule kernel keeps each head's ``[K, V]`` recurrent state in fp32 inside
one Triton launch and carries it from chunk to chunk. To emulate a deployment that stores that
state in FP8, ModelOpt runs an adapted copy of the kernel
(:mod:`modelopt.torch.kernels.quantization.linear_attention`) that fake-quantizes the state to
E4M3 at the end of every chunk, with a scale computed inside the kernel from the state itself.
The backward pass recomputes the same quantized states and passes the state gradient straight
through the quantization, so QAT and QAD train against the quantized recurrence. A second
quantizer covers ``w``, the WY-transformed keys that multiply the state; ``w`` is a regular tensor,
so it is fake-quantized by the ``TensorQuantizer`` itself before the kernel reads it.
"""

from collections.abc import Callable
from typing import Any

import torch

from ..config import QuantizerAttributeConfig
from ..linear_attention.utils import validate_gdn_quantizer
from ..nn import QuantModule, TensorQuantizer

__all__ = ["GatedDeltaNetStateQuantMixin"]

GatedDeltaRuleFn = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


def _state_qdq_chunk_gated_delta_rule() -> GatedDeltaRuleFn:
    # Imported on first use: flash-linear-attention is a heavy optional dependency that only the
    # enabled quantizer needs, and importing it warns on machines without a GPU.
    try:
        from modelopt.torch.kernels.quantization.linear_attention.fla_chunk_gated_delta_rule import (
            chunk_gated_delta_rule,
        )
    except ImportError as e:
        raise RuntimeError(
            "GDN fake quantization needs Triton and fla-core==0.5.1 on a CUDA "
            f"device; importing the state-quantizing kernel failed with {e!r}."
        ) from e
    return chunk_gated_delta_rule


class GatedDeltaNetStateQuantMixin(QuantModule):
    """Adds ``gdn_state_quantizer`` and ``gdn_w_quantizer`` to a GatedDeltaNet module.

    Subclasses route the module's chunked gated-delta-rule call through
    :meth:`_state_quantized_chunk_gated_delta_rule`. Both quantizers start disabled; enable them
    with ``quant_cfg`` entries on ``*gdn_state_quantizer`` / ``*gdn_w_quantizer`` such as the
    ``configs/ptq/units/gdn_state_fp8_dynamic`` and ``gdn_w_fp8_dynamic`` recipe units. The state
    quantizer carries the fused QDQ configuration. Both sites currently require dynamic
    E4M3 and identity STE. State QDQ uses fixed 64-column value tiles and 64-token chunks.
    """

    def _setup(self):
        for name in ("gdn_state_quantizer", "gdn_w_quantizer"):
            self._register_temp_attribute(
                name, TensorQuantizer(QuantizerAttributeConfig(enable=False))
            )

    def validate_linear_attention(self) -> None:
        """Reject numerical settings that the fused training path cannot implement."""
        for name in ("gdn_state_quantizer", "gdn_w_quantizer"):
            quantizer = getattr(self, name)
            if quantizer.is_enabled:
                validate_gdn_quantizer(quantizer, name=name)
        # The state handle configures fused QDQ; W's grouping is executed by TensorQuantizer.
        if self.gdn_state_quantizer.is_enabled and self.gdn_state_quantizer.axis != (0, 1):
            raise ValueError(
                "gdn_state_quantizer supports only axis=(0, 1) with 64-column value tiling"
            )

    def modelopt_post_restore(self, prefix: str = ""):
        """Validate restored quantizers against the fused training kernel requirements."""
        super().modelopt_post_restore(prefix)
        self.validate_linear_attention()

    def _state_quantized_chunk_gated_delta_rule(
        self, gated_delta_rule: GatedDeltaRuleFn, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Call ``gated_delta_rule`` or, if a quantizer is on, the vendored quantizing copy."""
        self.validate_linear_attention()
        quantize_state = self.gdn_state_quantizer.is_enabled and self.gdn_state_quantizer._if_quant
        quantize_w = self.gdn_w_quantizer.is_enabled
        if not (quantize_state or quantize_w):
            return gated_delta_rule(*args, **kwargs)
        if getattr(gated_delta_rule, "__name__", None) != "chunk_gated_delta_rule":
            raise NotImplementedError(
                "GatedDeltaNet quantizers require the fla chunked kernel; the deterministic torch "
                f"kernel ({gated_delta_rule!r}) is not supported."
            )
        chunk_size = kwargs.pop("chunk_size", 64)
        if chunk_size != 64:
            raise ValueError("GDN fake quantization supports only chunk_size=64")
        return _state_qdq_chunk_gated_delta_rule()(
            *args,
            chunk_size=chunk_size,
            state_qdq=int(quantize_state),
            state_qdq_block_v=64,
            w_quantizer=self.gdn_w_quantizer if quantize_w else None,
            **kwargs,
        )
