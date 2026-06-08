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

"""Megatron-Core TE GroupedLinear LoRA plugin for MoE expert layers.

Registers LoRA adapters against ``TEColumnParallelGroupedLinear`` and
``TERowParallelGroupedLinear`` -- the grouped-GEMM modules used by
Megatron-Core when ``moe_grouped_gemm=True`` (the default for Nemotron-3
Hybrid MoE). Per-expert ``(A_e, B_e)`` factors are stored as stacked
tensors of shape ``[num_experts, ...]`` and dispatched in ``forward`` by
splitting the input along ``tokens_per_expert``.

The plugin also honors ``PEFTAttributeConfig.lora_dtype`` so the LoRA
sidecar can be pinned to BF16 independently of the base layer's storage
dtype (e.g. on top of a fake-quantized low-bit base for QAD).
"""

import warnings
from typing import Any

import torch
import torch.nn as nn

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )

    HAVE_TE_GROUPED = (
        TEColumnParallelGroupedLinear is not None and TERowParallelGroupedLinear is not None
    )
except ImportError:
    TEColumnParallelGroupedLinear = None  # type: ignore[assignment]
    TERowParallelGroupedLinear = None  # type: ignore[assignment]
    HAVE_TE_GROUPED = False

# Quantized TE-grouped classes registered by modelopt.torch.quantization.plugins.megatron.
# Used at module bottom to register the LoRA class against the quantized base too, so the
# quantize -> LoRA flow ("mtq.quantize then mtpeft.update_model") works on the grouped path.
try:
    from modelopt.torch.quantization.plugins.megatron import (
        _MegatronTEGroupedColumnParallelLinear as _QuantTEColumnParallelGroupedLinear,
    )
    from modelopt.torch.quantization.plugins.megatron import (
        _MegatronTEGroupedRowParallelLinear as _QuantTERowParallelGroupedLinear,
    )

    HAVE_QUANT_TE_GROUPED = (
        _QuantTEColumnParallelGroupedLinear is not None
        and _QuantTERowParallelGroupedLinear is not None
    )
except ImportError:
    _QuantTEColumnParallelGroupedLinear = None  # type: ignore[assignment]
    _QuantTERowParallelGroupedLinear = None  # type: ignore[assignment]
    HAVE_QUANT_TE_GROUPED = False

from ...config import PEFTAttributeConfig
from ..layer import LoRAModule, LoRAModuleRegistry

__all__ = []

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_lora_dtype(
    attr_config: PEFTAttributeConfig, fallback: torch.dtype
) -> torch.dtype:
    """Map ``attr_config.lora_dtype`` to a ``torch.dtype``; ``None`` inherits ``fallback``."""
    if attr_config.lora_dtype is None:
        return fallback
    return _DTYPE_MAP[attr_config.lora_dtype]


class _StackedLoRAFactor(nn.Module):
    """Carrier ``nn.Module`` that holds a single stacked per-expert LoRA factor.

    The wrapped ``nn.Parameter`` has shape ``[num_experts, *suffix]``. ``forward``
    is intentionally unused -- the parent ``LoRAModule`` subclass dispatches
    per-expert against ``self.weight`` directly. Inheriting from ``nn.Module``
    keeps the factor visible in ``state_dict`` and reachable by ``add_module``,
    which is what ``LoRAModule._register_adapter`` expects.
    """

    def __init__(self, num_experts: int, *suffix: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, *suffix, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise RuntimeError(
            "_StackedLoRAFactor.forward should not be called directly; "
            "the parent LoRAModule subclass dispatches per-expert against self.weight."
        )


if HAVE_TE_GROUPED:

    class _LoRATEGroupedBase(LoRAModule):
        """Shared base for column- and row-parallel TE-grouped LoRA wrappers.

        Subclasses exist only to bind the registration to a specific TE grouped
        linear class -- column or row parallel. All allocation and dispatch
        logic lives here.

        Tensor layout (matches the TE grouped weight convention ``[out, in]`` per expert):

        * ``lora_a.weight``: ``[num_gemms, rank, in_features]``
        * ``lora_b.weight``: ``[num_gemms, out_features, rank]``

        For column-parallel grouped, ``out_features`` is already the per-rank
        TP-sharded value (TE's ``explicit_expert_comm`` divides ``output_size``
        by ``tp_size`` at construction). For row-parallel grouped, ``in_features``
        is the per-rank value. Sharded-state-dict integration is deferred to a
        later commit; this PR carries the forward path only.
        """

        def _register_stacked_adapter(
            self,
            adapter_name: str,
            lora_a: _StackedLoRAFactor,
            lora_b: _StackedLoRAFactor,
            rank: int,
            scale: float,
            enable: bool,
        ) -> None:
            """Register stacked factors on the parent's device, preserving their dtype.

            Differs from ``_MegatronParallelLoRABase._register_adapter_with_device``
            (the non-grouped variant) in that we do not copy the parent layer's
            dtype onto the factors -- their dtype was set at construction per
            ``attr_config.lora_dtype``. That decoupling is what makes a BF16 LoRA
            sidecar over a fake-quant low-bit base possible.
            """
            base_param = next(iter(self.parameters()), None)
            if base_param is not None:
                lora_a = lora_a.to(base_param.device)
                lora_b = lora_b.to(base_param.device)
            super()._register_adapter(adapter_name, lora_a, lora_b, rank, scale, enable)

        def update_layer_lora(
            self, adapter_name: str, attr_config: PEFTAttributeConfig
        ) -> None:
            """Allocate and initialize per-expert stacked LoRA factors."""
            num_experts = self.num_gemms
            rank = attr_config.rank

            base_param = next(iter(self.parameters()), None)
            fallback_dtype = base_param.dtype if base_param is not None else torch.bfloat16
            dtype = _resolve_lora_dtype(attr_config, fallback_dtype)

            lora_a = _StackedLoRAFactor(num_experts, rank, self.in_features, dtype=dtype)
            lora_b = _StackedLoRAFactor(num_experts, self.out_features, rank, dtype=dtype)

            with torch.no_grad():
                if attr_config.lora_init_method == "svdquant":
                    self._init_factors_svdquant(lora_a.weight, lora_b.weight, rank)
                else:
                    # "kaiming_zeros" (default): honor the user's lora_a_init / lora_b_init
                    # initializers (defaults: Kaiming on A, zeros on B).
                    attr_config.lora_a_init(lora_a.weight)
                    attr_config.lora_b_init(lora_b.weight)

            self._register_stacked_adapter(
                adapter_name, lora_a, lora_b, rank, attr_config.scale, attr_config.enable
            )

        def _init_factors_svdquant(
            self,
            lora_a_weight: torch.Tensor,
            lora_b_weight: torch.Tensor,
            rank: int,
        ) -> None:
            """Per-expert SVD of the quantization residual ``W - quant(W)``.

            For each expert ``e`` in range(num_gemms): compute the residual between the
            full-precision per-expert weight and its fake-quantized image, run rank-``r``
            SVD via ``modelopt.torch.quantization.model_calib.svd``, then write the
            factors into the stacked carrier tensors:

            - ``lora_a_weight[e]`` <- ``vt`` (shape ``[r, in_features]``)
            - ``lora_b_weight[e]`` <- ``us`` (shape ``[out_features, r]``)

            so that ``B_e @ A_e == us @ vt`` approximates the quantization residual at
            init time. Falls back to zero-init on both factors (with a warning) if the
            wrapped layer has no attached/enabled ``weight_quantizer`` -- in that case the
            residual is undefined and ``mtq.quantize`` must run before ``mtpeft.update_model``.
            """
            quantizer = getattr(self, "weight_quantizer", None)
            quant_enabled = quantizer is not None and getattr(quantizer, "is_enabled", True)
            if not quant_enabled:
                warnings.warn(
                    "lora_init_method='svdquant' requested but no enabled weight_quantizer "
                    "is attached to this TE-grouped linear. The quantization residual is "
                    "undefined; falling back to zero-init on both LoRA factors. To get a "
                    "meaningful SVDQuant init, run mtq.quantize() before mtpeft.update_model().",
                    stacklevel=2,
                )
                lora_a_weight.zero_()
                lora_b_weight.zero_()
                return

            # Local import keeps the dependency on quantization optional at module-load
            # time -- only callers using lora_init_method='svdquant' pay the import cost.
            from modelopt.torch.quantization.model_calib import svd as _svd_helper

            for e in range(self.num_gemms):
                w_e = getattr(self, f"weight{e}")  # [out_features, in_features]
                w_q = quantizer(w_e)
                residual = (w_e - w_q).detach()
                us, vt = _svd_helper(residual, rank)
                # vt: [rank, in_features], us: [out_features, rank]
                lora_a_weight[e].copy_(vt.to(lora_a_weight.dtype))
                lora_b_weight[e].copy_(us.to(lora_b_weight.dtype))

        def forward(
            self, x: torch.Tensor, tokens_per_expert, *args, **kwargs
        ) -> Any:
            """Add per-expert LoRA outputs to the grouped GEMM's base output.

            Bypasses ``LoRAModule.forward`` via ``super(LoRAModule, self)`` because
            its adapter loop calls ``lora_a(x)``, which would invoke
            ``_StackedLoRAFactor.forward`` and raise. The stacked layout requires
            per-expert dispatch keyed by ``tokens_per_expert``.
            """
            base_out, base_bias = super(LoRAModule, self).forward(
                x, tokens_per_expert, *args, **kwargs
            )

            if not self._lora_adapters:
                return base_out, base_bias

            # tokens_per_expert is normally list[int] by the time it reaches the
            # grouped linear (TEGroupedMLP.forward converts via .tolist() before
            # calling fc1/fc2). Accept a tensor too for direct-call use cases.
            if isinstance(tokens_per_expert, torch.Tensor):
                tpe_list = tokens_per_expert.tolist()
            else:
                tpe_list = list(tokens_per_expert)

            x_splits = torch.split(x, tpe_list, dim=0)

            result = base_out
            for adapter in self._lora_adapters.values():
                if not adapter["enable"]:
                    continue
                a_weight = adapter["lora_a"].weight  # [E, r, in]
                b_weight = adapter["lora_b"].weight  # [E, out, r]
                scale = adapter["scale"]

                per_expert_outputs = []
                for e, x_e in enumerate(x_splits):
                    # matmul handles n_e == 0; output stays [0, out_features].
                    intermediate = x_e.to(a_weight.dtype) @ a_weight[e].T
                    out_e = (intermediate @ b_weight[e].T).to(base_out.dtype) * scale
                    per_expert_outputs.append(out_e)

                lora_out = torch.cat(per_expert_outputs, dim=0)
                result = result + lora_out

            return result, base_bias

    @LoRAModuleRegistry.register(
        {TEColumnParallelGroupedLinear: "megatron_TEColumnParallelGroupedLinear"}
    )
    class _LoRATEGroupedColumnParallelLinear(_LoRATEGroupedBase):
        """LoRA wrapper for ``TEColumnParallelGroupedLinear``.

        ``out_features`` is the per-rank TP-sharded value;
        ``in_features`` is the full hidden size.
        """

    @LoRAModuleRegistry.register(
        {TERowParallelGroupedLinear: "megatron_TERowParallelGroupedLinear"}
    )
    class _LoRATEGroupedRowParallelLinear(_LoRATEGroupedBase):
        """LoRA wrapper for ``TERowParallelGroupedLinear``.

        ``in_features`` is the per-rank TP-sharded value;
        ``out_features`` is the full hidden size.
        """

    # Register the LoRA classes against the *quantized* TE-grouped classes too, so the
    # ``mtq.quantize`` then ``mtpeft.update_model`` flow finds them on the grouped path.
    # Mirrors the pattern in peft/lora/plugins/megatron.py:244-250 for non-grouped linears.
    # Required for lora_init_method='svdquant' to find a weight_quantizer on self.
    if HAVE_QUANT_TE_GROUPED:
        LoRAModuleRegistry.register(
            {
                _QuantTEColumnParallelGroupedLinear: (
                    "quant_megatron_TEColumnParallelGroupedLinear"
                )
            }
        )(_LoRATEGroupedColumnParallelLinear)
        LoRAModuleRegistry.register(
            {
                _QuantTERowParallelGroupedLinear: (
                    "quant_megatron_TERowParallelGroupedLinear"
                )
            }
        )(_LoRATEGroupedRowParallelLinear)
