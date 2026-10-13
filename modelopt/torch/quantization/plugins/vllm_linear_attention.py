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

import torch
import vllm
from packaging.version import Version
from vllm.forward_context import get_forward_context

from ..linear_attention.utils import _state_qdq, state_quantizer_config
from ..nn import QuantModuleRegistry
from .custom import CUSTOM_MODEL_PLUGINS
from .linear_attention import _LinearAttentionQuantMixin
from .vllm import create_parallel_state

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
    if version.release[:2] != (0, 30):
        raise NotImplementedError("Linear-attention fakequant currently requires vLLM 0.30.x")
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    shape = MambaStateShapeCalculator.gated_delta_net_state_shape(1, 2, 2, 32, 64, 4)[-1]
    if shape[-2:] != (64, 32):
        raise NotImplementedError(
            "Linear-attention fakequant requires the value-first vLLM state ABI"
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
    linear_attention_quantizer_names = ("gdn_state_quantizer", "gdn_w_quantizer")

    def _setup(self):
        super()._setup()
        self.parallel_state = create_parallel_state()

    def validate_linear_attention(self):
        # vLLM owns execution and phase boundaries; validate only the state-QDQ policy.
        if self._linear_attn_state.is_enabled:
            state_quantizer_config(self._linear_attn_state)
        if (
            any(
                getattr(self, name).is_enabled for name in self.linear_attention_quantizer_names[1:]
            )
            or self.linear_attention_config.precision == "replayssm"
        ):
            raise ValueError(
                "vLLM state fakequant supports plain state QDQ; W and ReplaySSM are unsupported"
            )

    def warn_linear_attention_profile(self):
        """Native vLLM selects the kernels; training profiles do not override them."""

    def _quantize_cache(self, metadata):
        if metadata.spec_sequence_masks is not None:
            raise ValueError("Speculative metadata is unsupported for state fakequant")
        count = metadata.num_decodes + metadata.num_prefills
        indices = metadata.non_spec_state_indices_tensor[:count].long()
        if metadata.has_initial_state is not None:
            # Fresh prefill starts from zero in vLLM; do not touch its recycled cache slot.
            indices = indices[metadata.has_initial_state[:count]]
        if indices.numel() == 0:
            return
        state = self.kv_cache[1]
        if state.dtype != torch.float32:
            raise ValueError("State fakequant requires an FP32 recurrent cache")
        # vLLM stores [slot, head, V, K]; ModelOpt groups along the value dimension.
        selected = state.index_select(0, indices).transpose(-1, -2).contiguous()
        rounded = _state_qdq(
            selected, self._linear_attn_state, block_v=self.linear_attention_config.state_block_v
        )
        state.index_copy_(0, indices, rounded.transpose(-1, -2))

    def forward(self, *args, **kwargs):
        if self._linear_attn_state.is_enabled:
            metadata = get_forward_context().attn_metadata
            if metadata is not None and (layer_metadata := metadata.get(self.prefix)) is not None:
                self._quantize_cache(layer_metadata)
        return super().forward(*args, **kwargs)


class _QuantVllmKDA(_QuantVllmLinearAttention):
    linear_attention_quantizer_names = ("kda_state_quantizer",)


def _register_vllm_linear_attention(model):
    adapters = {
        ("vllm.models.kimi_k3.nvidia.kda", "KimiK3DeltaAttention"): _QuantVllmKDA,
        (
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
            "QwenGatedDeltaNetAttention",
        ): _QuantVllmLinearAttention,
        (
            "vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn",
            "KimiGatedDeltaNetAttention",
        ): _QuantVllmKDA,
    }
    for module in model.modules():
        cls = type(module)
        adapter = adapters.get((cls.__module__, cls.__name__))
        if adapter is not None and cls not in QuantModuleRegistry:
            QuantModuleRegistry.register({cls: f"vllm_{cls.__name__}"})(adapter)


CUSTOM_MODEL_PLUGINS.add(_register_vllm_linear_attention)
