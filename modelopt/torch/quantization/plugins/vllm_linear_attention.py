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

"""Recurrent-state QDQ at native vLLM prefill and decode call boundaries."""

from functools import partial
from types import FunctionType

import torch
import vllm
from packaging.version import Version
from vllm.forward_context import get_forward_context

from ..linear_attention.config import LinearAttentionConfig
from ..nn import QuantModuleRegistry
from .custom import CUSTOM_MODEL_PLUGINS
from .linear_attention import _LinearAttentionQuantMixin

__all__ = ["bind_vllm_linear_attention"]


def bind_vllm_linear_attention(model, model_runner):
    """Validate state-only adapters before calibration or generation."""
    layers = [m for m in model.modules() if isinstance(m, _QuantVllmLinearAttention)]
    for layer in layers:
        layer.validate_linear_attention()
    if any(layer.linear_attention_is_enabled for layer in layers):
        _validate_runtime(model_runner)
    return layers


def _validate_runtime(runner):
    version = Version(vllm.__version__)
    if version.release[:2] != (0, 15):
        raise NotImplementedError("Linear-attention fakequant currently requires vLLM 0.15.x V1")
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    shape = MambaStateShapeCalculator.gated_delta_net_state_shape(1, 2, 2, 32, 64, 4)[-1]
    if shape[-2:] != (32, 64):
        raise NotImplementedError(
            "Linear-attention fakequant requires the key-first vLLM state ABI"
        )
    config = runner.vllm_config
    if not config.model_config.enforce_eager or config.scheduler_config.async_scheduling:
        raise ValueError("State fakequant requires --enforce-eager --no-async-scheduling")
    if config.speculative_config is not None or config.cache_config.enable_prefix_caching:
        raise ValueError("Disable speculative decoding and prefix caching for state fakequant")
    if config.cache_config.mamba_cache_mode != "none":
        raise ValueError("State fakequant requires mamba_cache_mode='none'")
    if config.kv_transfer_config is not None or config.ec_transfer_config is not None:
        raise ValueError("State fakequant does not support state transfer")
    parallel = config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "pipeline_parallel_size",
            "data_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    ):
        raise ValueError("State fakequant supports TP with PP=1, DP=1, and CP=1")


class _QuantVllmLinearAttention(_LinearAttentionQuantMixin):
    def validate_linear_attention(self):
        super().validate_linear_attention()
        if (
            self._linear_attn_w.is_enabled
            or self.linear_attention_config != LinearAttentionConfig()
        ):
            raise ValueError(
                "vLLM supports only state TensorQuantizer at native call boundaries; "
                "prefill operand, arithmetic, and decode policies are unsupported"
            )

    def _quantized_state_call(self, native, *args, **kwargs):
        state = kwargs.get("initial_state")
        if state is not None and self._linear_attn_state.is_enabled:
            if state.dtype != torch.float32:
                raise ValueError("State fakequant requires an FP32 recurrent cache")
            indices = kwargs.get("ssm_state_indices")
            if indices is None:
                # Native prefill has already initialized fresh requests to zero.
                kwargs["initial_state"] = self._linear_attn_state(state)
            else:
                # Eager, non-speculative decode has one cache slot per active sequence.
                indices = indices[: kwargs["cu_seqlens"].numel() - 1].long()
                state.index_copy_(
                    0, indices, self._linear_attn_state(state.index_select(0, indices))
                )
        return native(*args, **kwargs)

    def _forward_with_state_qdq(self, original, kernel_names, *args, **kwargs):
        if not self.linear_attention_is_enabled:
            return original(*args, **kwargs)
        context = get_forward_context()
        if context.attn_metadata is None:
            return original(*args, **kwargs)
        if context.attn_metadata[self.prefix].spec_sequence_masks is not None:
            raise ValueError("Speculative metadata is unsupported for state fakequant")
        function = original.__func__
        replacements = {
            name: partial(self._quantized_state_call, function.__globals__[name])
            for name in kernel_names
        }
        # Bind wrappers for this invocation only; every wrapper calls the original kernel.
        forward = FunctionType(
            function.__code__,
            {**function.__globals__, **replacements},
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        forward.__kwdefaults__ = function.__kwdefaults__
        return forward(self, *args, **kwargs)


class _QuantVllmGDN(_QuantVllmLinearAttention):
    def _forward_core(self, *args, **kwargs):
        return self._forward_with_state_qdq(
            super()._forward_core,
            ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"),
            *args,
            **kwargs,
        )


class _QuantVllmKDA(_QuantVllmLinearAttention):
    linear_attention_quantizer_names = ("kda_state_quantizer", "kda_w_quantizer")

    def _forward(self, *args, **kwargs):
        return self._forward_with_state_qdq(
            super()._forward,
            ("chunk_kda", "fused_recurrent_kda"),
            *args,
            **kwargs,
        )


def _register_vllm_linear_attention(model):
    adapters = {
        ("vllm.model_executor.models.qwen3_next", "Qwen3NextGatedDeltaNet"): _QuantVllmGDN,
        ("vllm.model_executor.layers.kda", "KimiDeltaAttention"): _QuantVllmKDA,
    }
    for module in model.modules():
        cls = type(module)
        adapter = adapters.get((cls.__module__, cls.__name__))
        if adapter is not None and cls not in QuantModuleRegistry:
            QuantModuleRegistry.register({cls: f"vllm_{cls.__name__}"})(adapter)


CUSTOM_MODEL_PLUGINS.add(_register_vllm_linear_attention)
